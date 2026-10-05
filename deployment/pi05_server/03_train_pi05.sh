#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}"
export OPENPI_DELTA_CHECKPOINT="${OPENPI_DELTA_CHECKPOINT:-1}"
export OPENPI_DELTA_BASE_PARAMS="${OPENPI_DELTA_BASE_PARAMS:-${BASE_PARAMS}}"

cd "${POLICY_ROOT}"
"${PYTHON_BIN}" scripts/train.py pi05_lora \
  --exp-name="${EXP_NAME:-teleop_pi05}" \
  --batch-size="${BATCH_SIZE:-64}" \
  --num-workers="${NUM_WORKERS:-4}" \
  --fsdp-devices="${FSDP_DEVICES:-1}" \
  --data.repo-id=teleop_pi05 \
  --weight-loader.params-path="${BASE_PARAMS}" \
  --dataset-path="${PIPELINE_ROOT}/02_pi05_dataset"
