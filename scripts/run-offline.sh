#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 3 ]]; then
  echo "usage: $0 AI_JOB_JSON CLIP_MP4 OUTPUT_DIR [KEYPOINTS_JSON]" >&2
  exit 2
fi
cd "$(dirname "$0")/.."
args=(offline --job "$1" --clip "$2" --output "$3")
if [[ $# -ge 4 ]]; then
  args+=(--keypoints "$4")
fi
uv run --no-sync volleyball-analysis "${args[@]}"
