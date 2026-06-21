#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT="/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/nhanguyen"
REPO_ROOT="${WORKSPACE_ROOT}/SPAN-rtmri-artspeech"
ENV_ROOT="${REPO_ROOT}/.venv"
CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/configs/speech2rtmri_artspeech/full_database_grouille_a100_10h.yaml}"
SKIP_MANIFEST="${SKIP_MANIFEST:-1}"
SKIP_AUDIO_CACHE="${SKIP_AUDIO_CACHE:-0}"
RUN_DEMO="${RUN_DEMO:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
JOB_LABEL="${JOB_LABEL:-full_database_grouille_a100_10h}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_LOG_DIR="${REPO_ROOT}/logs/gpu_runs/${JOB_LABEL}_${OAR_JOB_ID:-manual}_${RUN_STAMP}"
PHASE_TRACE="${RUN_LOG_DIR}/phase_trace.jsonl"

cd "${REPO_ROOT}"

if ! type module >/dev/null 2>&1; then
  source /etc/profile
fi

module purge
module load cuda/12.1.1

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export XDG_CACHE_HOME="${WORKSPACE_ROOT}/paper2_aat_eval_server/cache"
export PIP_CACHE_DIR="${WORKSPACE_ROOT}/paper2_aat_eval_server/cache/pip"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

PYTHON_BIN="${ENV_ROOT}/bin/python"
if [ ! -x "${PYTHON_BIN}" ]; then
  echo "Missing python env at ${PYTHON_BIN}" >&2
  exit 1
fi

mkdir -p "${RUN_LOG_DIR}"
cp "${CONFIG_PATH}" "${RUN_LOG_DIR}/config_used.yaml"
exec > >(tee -a "${RUN_LOG_DIR}/stdout.log") 2> >(tee -a "${RUN_LOG_DIR}/stderr.log" >&2)

record_phase_event() {
  local phase="$1"
  local event="$2"
  local ts
  ts="$(date --iso-8601=seconds)"
  printf '{"phase":"%s","event":"%s","ts":"%s"}\n' "${phase}" "${event}" "${ts}" >> "${PHASE_TRACE}"
}

run_phase() {
  local phase="$1"
  shift
  echo "PHASE=${phase}"
  record_phase_event "${phase}" "start"
  if "$@"; then
    record_phase_event "${phase}" "end"
  else
    local status=$?
    record_phase_event "${phase}" "fail"
    return "${status}"
  fi
}

write_run_metadata() {
  "${PYTHON_BIN}" - <<PY
import json
from pathlib import Path

payload = {
    "job_label": ${JOB_LABEL@Q},
    "job_id": ${OAR_JOB_ID:-0},
    "config_path": ${CONFIG_PATH@Q},
    "run_log_dir": ${RUN_LOG_DIR@Q},
    "phase_trace": ${PHASE_TRACE@Q},
    "python_bin": ${PYTHON_BIN@Q},
    "cuda_visible_devices": ${CUDA_VISIBLE_DEVICES@Q},
    "skip_manifest": ${SKIP_MANIFEST@Q},
    "skip_audio_cache": ${SKIP_AUDIO_CACHE@Q},
    "run_demo": ${RUN_DEMO@Q},
    "run_eval": ${RUN_EVAL@Q},
}
Path(${RUN_LOG_DIR@Q}).mkdir(parents=True, exist_ok=True)
Path(${RUN_LOG_DIR@Q}, "run_metadata.json").write_text(json.dumps(payload, indent=2) + "\\n", encoding="utf-8")
PY
}

finalize_phase_report() {
  "${PYTHON_BIN}" -m speech2rtmri_artspeech.report --phase-trace "${PHASE_TRACE}" --output-dir "${RUN_LOG_DIR}/report" >/dev/null 2>&1 || true
}

trap finalize_phase_report EXIT

OUTPUT_ROOT="$("${PYTHON_BIN}" - <<PY
from speech2rtmri_artspeech.config import load_config
cfg = load_config(r"${CONFIG_PATH}")
print(cfg["runtime"]["output_dir"])
PY
)"
write_run_metadata
echo "CONFIG_PATH=${CONFIG_PATH}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "RUN_LOG_DIR=${RUN_LOG_DIR}"

if [ "${SKIP_MANIFEST}" != "1" ]; then
  run_phase build_manifest "${PYTHON_BIN}" -m speech2rtmri_artspeech.build_manifest --config "${CONFIG_PATH}"
fi

if [ "${SKIP_AUDIO_CACHE}" != "1" ]; then
  run_phase extract_audio_embeddings "${PYTHON_BIN}" -m speech2rtmri_artspeech.extract_audio_embeddings --config "${CONFIG_PATH}"
fi

run_phase train "${PYTHON_BIN}" -m speech2rtmri_artspeech.train --config "${CONFIG_PATH}"

LAST_RUN_DIR="$(ls -1dt "${OUTPUT_ROOT}"/train/train_* | head -n 1)"
LAST_CHECKPOINT=""
for candidate in \
  "${LAST_RUN_DIR}/checkpoints/best_visual.pt" \
  "${LAST_RUN_DIR}/checkpoints/best_valid_loss.pt" \
  "${LAST_RUN_DIR}/checkpoints/best.pt" \
  "${LAST_RUN_DIR}/checkpoints/last.pt"; do
  if [ -f "${candidate}" ]; then
    LAST_CHECKPOINT="${candidate}"
    break
  fi
done
if [ ! -f "${LAST_CHECKPOINT}" ]; then
  echo "Missing checkpoint after training under ${LAST_RUN_DIR}" >&2
  exit 1
fi

if [ "${RUN_DEMO}" = "1" ]; then
  run_phase demo "${PYTHON_BIN}" -m speech2rtmri_artspeech.demo --config "${CONFIG_PATH}" --checkpoint "${LAST_CHECKPOINT}" --split test --num_samples 5
fi
if [ "${RUN_EVAL}" = "1" ]; then
  run_phase evaluate "${PYTHON_BIN}" -m speech2rtmri_artspeech.evaluate --config "${CONFIG_PATH}" --checkpoint "${LAST_CHECKPOINT}" --split test
fi
