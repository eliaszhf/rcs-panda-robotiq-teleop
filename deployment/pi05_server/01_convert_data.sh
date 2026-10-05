#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

cd "${POLICY_ROOT}"
"${PYTHON_BIN}" "${PIPELINE_ROOT}/scripts/convert_parquet_to_pi05.py" \
  --input-dir "${PIPELINE_ROOT}/01_raw_upload/incoming" \
  --output-dir "${PIPELINE_ROOT}/02_pi05_dataset" \
  --overwrite
