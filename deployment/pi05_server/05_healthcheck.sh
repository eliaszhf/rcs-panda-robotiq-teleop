#!/usr/bin/env bash
set -euo pipefail

POLICY_HOST="${POLICY_HOST:-127.0.0.1}"
POLICY_PORT="${POLICY_PORT:-8000}"
curl --fail --silent --show-error "http://${POLICY_HOST}:${POLICY_PORT}/healthz"

