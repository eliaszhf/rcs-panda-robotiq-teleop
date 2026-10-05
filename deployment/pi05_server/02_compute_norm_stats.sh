#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

cd "${POLICY_ROOT}"
"${PYTHON_BIN}" scripts/compute_norm_stats.py \
  --config-name pi05_lora \
  --repo-id teleop_pi05 \
  --dataset-path "${PIPELINE_ROOT}/02_pi05_dataset"
