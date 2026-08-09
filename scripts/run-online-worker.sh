#!/usr/bin/env bash
set -euo pipefail
: "${VOLLYAI_INTEGRATION_ID:?set VOLLYAI_INTEGRATION_ID}"
: "${VOLLYAI_TOKEN:?set VOLLYAI_TOKEN}"
cd "$(dirname "$0")/.."
uv run volleyball-analysis worker
