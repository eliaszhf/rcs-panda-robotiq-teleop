#!/usr/bin/env bash
set -euo pipefail

PIPELINE_ROOT="${PIPELINE_ROOT:-/home/zhanghf/robomme/teleop_pi05_pipeline}"
POLICY_ROOT="${POLICY_ROOT:-${PIPELINE_ROOT}/code/robomme_policy_learning}"
PYTHON_BIN="${PYTHON_BIN:-/home/zhanghf/robomme/robomme_policy_learning/.venv/bin/python}"
BASE_PARAMS="${BASE_PARAMS:-/data/public/openpi/openpi-assets/checkpoints/pi05_base/params}"

test -d "${PIPELINE_ROOT}"
test -f "${POLICY_ROOT}/pyproject.toml"
test -x "${PYTHON_BIN}"
test -d "${BASE_PARAMS}"

POLICY_PYTHONPATH="${POLICY_ROOT}/src:${POLICY_ROOT}/packages/openpi-client/src:${POLICY_ROOT}"
export PYTHONPATH="${POLICY_PYTHONPATH}${PYTHONPATH:+:${PYTHONPATH}}"
