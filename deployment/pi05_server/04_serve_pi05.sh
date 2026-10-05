#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

CHECKPOINT_FILE="${PIPELINE_ROOT}/03_training/SELECTED_CHECKPOINT.txt"
if [[ $# -gt 0 ]]; then
  CHECKPOINT_PATH="$1"
elif [[ -s "${CHECKPOINT_FILE}" ]]; then
  CHECKPOINT_PATH="$(grep -vE '^[[:space:]]*(#|$)' "${CHECKPOINT_FILE}" | head -n 1)"
else
  echo "Missing checkpoint. Pass its path or write it to ${CHECKPOINT_FILE}." >&2
  exit 2
fi

if [[ ! -d "${CHECKPOINT_PATH}" ]]; then
  echo "Checkpoint directory does not exist: ${CHECKPOINT_PATH}" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
cd "${POLICY_ROOT}"
exec "${PYTHON_BIN}" scripts/serve_policy.py \
  --host="${POLICY_HOST:-127.0.0.1}" \
  --port="${POLICY_PORT:-8000}" \
  --seed="${POLICY_SEED:-7}" \
  policy:checkpoint \
  --policy.dir="${CHECKPOINT_PATH}" \
  --policy.config=pi05_lora
