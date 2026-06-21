#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT="/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/nhanguyen"
REPO_ROOT="${WORKSPACE_ROOT}/SPAN-rtmri-artspeech"
ENV_ROOT="${REPO_ROOT}/.venv"
CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/configs/speech2rtmri_artspeech/full_database_grouille_a100_10h.yaml}"
RUN_AUDIO_CACHE="${RUN_AUDIO_CACHE:-0}"
FORCE_CPU_AUDIO_CACHE="${FORCE_CPU_AUDIO_CACHE:-0}"
JOB_LABEL="${JOB_LABEL:-$(basename "${CONFIG_PATH}" .yaml)_cpu_prepare}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_LOG_DIR="${REPO_ROOT}/logs/cpu_prepare/${JOB_LABEL}_${RUN_STAMP}"
PHASE_TRACE="${RUN_LOG_DIR}/phase_trace.jsonl"

cd "${REPO_ROOT}"

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

finalize_phase_report() {
  "${PYTHON_BIN}" -m speech2rtmri_artspeech.report --phase-trace "${PHASE_TRACE}" --output-dir "${RUN_LOG_DIR}/report" >/dev/null 2>&1 || true
}

trap finalize_phase_report EXIT

mapfile -t CONFIG_VALUES < <("${PYTHON_BIN}" - <<PY
from speech2rtmri_artspeech.config import load_config
cfg = load_config(r"${CONFIG_PATH}")
print(cfg["runtime"]["output_dir"])
print(cfg["audio"]["encoder"])
print(cfg["audio"].get("device", "cpu"))
PY
)
OUTPUT_ROOT="${CONFIG_VALUES[0]}"
AUDIO_ENCODER="${CONFIG_VALUES[1]}"
AUDIO_DEVICE="${CONFIG_VALUES[2]}"

echo "CONFIG_PATH=${CONFIG_PATH}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "RUN_LOG_DIR=${RUN_LOG_DIR}"
echo "AUDIO_ENCODER=${AUDIO_ENCODER}"
echo "AUDIO_DEVICE=${AUDIO_DEVICE}"

run_phase build_manifest "${PYTHON_BIN}" -m speech2rtmri_artspeech.build_manifest --config "${CONFIG_PATH}"

if [ "${RUN_AUDIO_CACHE}" = "1" ]; then
  if [ "${AUDIO_ENCODER}" != "mfcc" ] && [ "${FORCE_CPU_AUDIO_CACHE}" != "1" ]; then
    echo "Refusing CPU audio cache for encoder ${AUDIO_ENCODER}. Use the GPU runtime phase or set FORCE_CPU_AUDIO_CACHE=1." >&2
    exit 1
  fi
  if [ "${AUDIO_DEVICE}" != "cpu" ] && [ "${FORCE_CPU_AUDIO_CACHE}" != "1" ]; then
    echo "Config requests audio.device=${AUDIO_DEVICE}. Keep audio cache on the GPU lane or set FORCE_CPU_AUDIO_CACHE=1." >&2
    exit 1
  fi
  run_phase extract_audio_embeddings "${PYTHON_BIN}" -m speech2rtmri_artspeech.extract_audio_embeddings --config "${CONFIG_PATH}"
fi
