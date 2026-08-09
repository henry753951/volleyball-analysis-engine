#!/usr/bin/env bash
set -euo pipefail
: "${VOLLYAI_TOKEN:?set VOLLYAI_TOKEN}"
cd "$(dirname "$0")/.."
uv run volleyball-analysis worker
