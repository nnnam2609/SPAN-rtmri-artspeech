#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT="/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/nhanguyen"
JOB_SCRIPT="${WORKSPACE_ROOT}/SPAN-rtmri-artspeech/scripts/job_train_speech2rtmri_artspeech_full_database_grouille_a100_10h.sh"
CONFIG_PATH="${CONFIG_PATH:-${WORKSPACE_ROOT}/SPAN-rtmri-artspeech/configs/speech2rtmri_artspeech/full_database_grouille_a100_10h.yaml}"
LAUNCH_DIR="/home/nhanguyen"
WORKSPACE_LOG_ROOT="${WORKSPACE_ROOT}/_logs"
LOG_LINK_ROOT="/home/nhanguyen/speech2rtmri_artspeech_logs"
STDOUT_PATH="${LOG_LINK_ROOT}/%jobid%.out"
STDERR_PATH="${LOG_LINK_ROOT}/%jobid%.err"
SKIP_MANIFEST="${SKIP_MANIFEST:-1}"
SKIP_AUDIO_CACHE="${SKIP_AUDIO_CACHE:-0}"
RUN_DEMO="${RUN_DEMO:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
JOB_LABEL="${JOB_LABEL:-full_database_grouille_a100_10h}"
WRAPPER_DIR="${WORKSPACE_ROOT}/SPAN-rtmri-artspeech/scripts/.generated_oar"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
WRAPPER_SCRIPT="${WRAPPER_DIR}/${JOB_LABEL}_${RUN_STAMP}.sh"

mkdir -p "${WORKSPACE_LOG_ROOT}"
mkdir -p "${WRAPPER_DIR}"
ln -sfn "${WORKSPACE_LOG_ROOT}" "${LOG_LINK_ROOT}"

cat > "${WRAPPER_SCRIPT}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export CONFIG_PATH=${CONFIG_PATH@Q}
export SKIP_MANIFEST=${SKIP_MANIFEST@Q}
export SKIP_AUDIO_CACHE=${SKIP_AUDIO_CACHE@Q}
export RUN_DEMO=${RUN_DEMO@Q}
export RUN_EVAL=${RUN_EVAL@Q}
export JOB_LABEL=${JOB_LABEL@Q}
exec ${JOB_SCRIPT@Q}
EOF
chmod +x "${WRAPPER_SCRIPT}"

oarsub \
  -q default \
  -t exotic \
  -p "cluster='grouille' AND gpu_model='A100-PCIE-40GB' AND gpu_compute_capability_major >= 8" \
  -l /host=1/gpu=1,walltime=10:00:00 \
  -d "${LAUNCH_DIR}" \
  -O "${STDOUT_PATH}" \
  -E "${STDERR_PATH}" \
  "${WRAPPER_SCRIPT}"
