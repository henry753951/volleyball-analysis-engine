#!/usr/bin/env bash
set -euo pipefail

# Ubuntu Docker entrypoint. It can be piped directly from GitHub; it downloads tar archives,
# not Git checkouts, and then runs the local copy.
REPO_OWNER="henry753951"
ENGINE_REPO="volleyball-analysis-engine"
CENTRAL_REPO="volleyball-monitoring-ai"
ENGINE_ARCHIVE="https://github.com/${REPO_OWNER}/${ENGINE_REPO}/archive/refs/heads/main.tar.gz"
CENTRAL_ARCHIVE="https://github.com/${REPO_OWNER}/${CENTRAL_REPO}/archive/refs/heads/main.tar.gz"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]:-}")" && pwd)"
ORIGINAL_ARGS=("$@")

MODE="docker"
SERVER_URL="https://volleyai.hsulab.net/"
CENTRAL_HTTP_URL="https://volleyai.hsulab.net/"
TOKEN="${VOLLYAI_TOKEN:-}"
CREATE_LOCAL_TOKEN=0
INSTALL_DIR="${VOLLYAI_ENGINE_DIR:-$HOME/volleyball-analysis-engine}"
INSTANCE_PREFIX="analysis-worker"
ASSETS_ROOT=""
MULTITASK_SDK_ROOT="${VOLLYAI_MULTITASK_SDK_ROOT:-}"
MULTITASK_SDK_URL="${VOLLYAI_MULTITASK_SDK_URL:-}"
OSNET_URL="${VOLLYAI_OSNET_URL:-https://huggingface.co/datasets/holma91/SAM-Deep-EIoU/resolve/main/checkpoints/osnet_sports.pth.tar?download=true}"
TORCH_BACKEND="auto"
GPU_IDS=()
SKIP_MODEL_DOWNLOAD=0
NO_START=0
REFRESH_SOURCE=0

die() { printf 'error: %s\n' "$*" >&2; exit 1; }
require_command() { command -v "$1" >/dev/null 2>&1 || die "$1 is required"; }
show_help() {
  cat <<'EOF'
Usage: install-ubuntu.sh [options]
  --mode docker                    Docker is the only supported deployment mode
  --server-url URL                 Default: https://volleyai.hsulab.net/
  --central-http-url URL           Token API base URL for --create-local-token
  --token TOKEN                    Worker access token
  --create-local-token             Create a development token through the token API
  --multitask-sdk-root PATH        Existing private SDK directory
  --multitask-sdk-url URL          Private SDK zip URL
  --osnet-url URL                  Sports OSNet checkpoint URL
  --gpu-ids 0,1                    Explicit physical GPU IDs
  --gpu-id 2                       Add one physical GPU ID
  --torch-backend auto|cu130|cpu   PyTorch backend passed to the Docker build
  --skip-model-download             Reuse already prepared assets
  --no-start                       Build/configure without starting workers
  --refresh-source                 Redownload the engine archive
EOF
}

while (($#)); do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --server-url) SERVER_URL="$2"; shift 2 ;;
    --central-http-url) CENTRAL_HTTP_URL="$2"; shift 2 ;;
    --token) TOKEN="$2"; shift 2 ;;
    --create-local-token) CREATE_LOCAL_TOKEN=1; shift ;;
    --install-dir) INSTALL_DIR="$2"; shift 2 ;;
    --instance-prefix) INSTANCE_PREFIX="$2"; shift 2 ;;
    --assets-root) ASSETS_ROOT="$2"; shift 2 ;;
    --multitask-sdk-root) MULTITASK_SDK_ROOT="$2"; shift 2 ;;
    --multitask-sdk-url) MULTITASK_SDK_URL="$2"; shift 2 ;;
    --osnet-url) OSNET_URL="$2"; shift 2 ;;
    --torch-backend) TORCH_BACKEND="$2"; shift 2 ;;
    --gpu-ids) IFS=',' read -r -a GPU_IDS <<<"$2"; shift 2 ;;
    --gpu-id) GPU_IDS+=("$2"); shift 2 ;;
    --skip-model-download) SKIP_MODEL_DOWNLOAD=1; shift ;;
    --no-start) NO_START=1; shift ;;
    --refresh-source) REFRESH_SOURCE=1; shift ;;
    --with-reid|--kpr-checkpoint|--kpr-checkpoint-url|--dino-url)
      die "$1 is not part of the Docker deployment; use the base multitask worker assets"
      ;;
    -h|--help) show_help; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

require_command curl
require_command tar
require_command python3
require_command find
require_command realpath
[[ "$MODE" == docker ]] || die '--mode docker is the only supported Ubuntu deployment mode'
[[ "$TORCH_BACKEND" == auto || "$TORCH_BACKEND" == cu130 || "$TORCH_BACKEND" == cpu ]] || die '--torch-backend must be auto, cu130 or cpu'

if [[ ! -f "$SCRIPT_DIR/../pyproject.toml" ]]; then
  TARGET="$(realpath -m "$INSTALL_DIR")"
  if [[ -f "$TARGET/pyproject.toml" && "$REFRESH_SOURCE" != 1 ]]; then
    exec "$TARGET/scripts/install-ubuntu.sh" "${ORIGINAL_ARGS[@]}"
  fi
  mkdir -p "$TARGET"
  TMP_ROOT="$(mktemp -d)"
  trap 'rm -rf "$TMP_ROOT"' EXIT
  curl -fsSL "$ENGINE_ARCHIVE" | tar -xzf - -C "$TMP_ROOT"
  SOURCE_ROOT="$(find "$TMP_ROOT" -mindepth 1 -maxdepth 1 -type d -print -quit)"
  [[ -n "$SOURCE_ROOT" ]] || die 'unable to unpack the engine archive'
  cp -a "$SOURCE_ROOT/." "$TARGET/"
  chmod +x "$TARGET/scripts/install-ubuntu.sh" "$TARGET/scripts/uninstall-ubuntu.sh"
  exec "$TARGET/scripts/install-ubuntu.sh" "${ORIGINAL_ARGS[@]}"
fi

PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
if [[ -z "$ASSETS_ROOT" ]]; then ASSETS_ROOT="$PROJECT_ROOT/.models"; fi
if [[ "$ASSETS_ROOT" != /* ]]; then ASSETS_ROOT="$PROJECT_ROOT/$ASSETS_ROOT"; fi
ASSETS_ROOT="$(realpath -m "$ASSETS_ROOT")"
CENTRAL_ROOT="$(cd -- "$PROJECT_ROOT/.." && pwd)/$CENTRAL_REPO"

if [[ ! -f "$CENTRAL_ROOT/sdk/pyproject.toml" ]]; then
  mkdir -p "$CENTRAL_ROOT"
  TMP_CENTRAL="$(mktemp -d)"
  trap 'rm -rf "$TMP_CENTRAL"' EXIT
  curl -fsSL "$CENTRAL_ARCHIVE" | tar -xzf - -C "$TMP_CENTRAL"
  CENTRAL_SOURCE="$(find "$TMP_CENTRAL" -mindepth 1 -maxdepth 1 -type d -print -quit)"
  [[ -n "$CENTRAL_SOURCE" ]] || die 'unable to unpack the central repository archive'
  cp -a "$CENTRAL_SOURCE/." "$CENTRAL_ROOT/"
fi

cd "$PROJECT_ROOT"
if [[ "$TORCH_BACKEND" == auto ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    mapfile -t DETECTED_GPU_IDS < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr -d '[:space:]' | grep -E '^[0-9]+$' || true)
    TORCH_BACKEND=$([[ ${#DETECTED_GPU_IDS[@]} -gt 0 ]] && echo cu130 || echo cpu)
  else
    DETECTED_GPU_IDS=()
    TORCH_BACKEND=cpu
  fi
else
  DETECTED_GPU_IDS=()
  if command -v nvidia-smi >/dev/null 2>&1; then
    mapfile -t DETECTED_GPU_IDS < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr -d '[:space:]' | grep -E '^[0-9]+$' || true)
  fi
fi
if ((${#GPU_IDS[@]} == 0)) && [[ "$TORCH_BACKEND" == cu130 ]]; then GPU_IDS=("${DETECTED_GPU_IDS[@]}"); fi
if [[ "$TORCH_BACKEND" == cu130 && ${#GPU_IDS[@]} -eq 0 ]]; then die 'CUDA selected but no NVIDIA GPU was detected'; fi
if [[ "$TORCH_BACKEND" == cpu && ${#GPU_IDS[@]} -gt 0 ]]; then die 'GPU IDs cannot be used with CPU backend'; fi

if [[ "$SERVER_URL" == https://* ]]; then WS_URL="wss://${SERVER_URL#https://}"; elif [[ "$SERVER_URL" == http://* ]]; then WS_URL="ws://${SERVER_URL#http://}"; else WS_URL="$SERVER_URL"; fi
WS_URL="${WS_URL%/}"
WS_URL="${WS_URL/\/api\/v1\/ai\/providers\/ws/\/api\/v2\/ai\/providers\/ws}"
[[ "$WS_URL" == */api/v2/ai/providers/ws ]] || WS_URL="$WS_URL/api/v2/ai/providers/ws"

if [[ -z "$TOKEN" && $CREATE_LOCAL_TOKEN -eq 1 ]]; then
  RESPONSE="$(curl -fsS -X POST "${CENTRAL_HTTP_URL%/}/api/v1/operations/ai-worker-tokens" \
    -H 'x-dev-role: ADMIN' -H 'x-dev-user-id: 00000000-0000-4000-8000-000000000001' \
    -H 'x-dev-display-name: Dev Operator' -H 'Content-Type: application/json' \
    --data "{\"name\":\"${INSTANCE_PREFIX}-$(date +%Y%m%d-%H%M%S)\"}")"
  TOKEN="$(printf '%s' "$RESPONSE" | python3 -c 'import json,sys; print(json.load(sys.stdin)["token"])')"
elif [[ -z "$TOKEN" ]]; then
  read -r -s -p 'Worker Access Token: ' TOKEN
  printf '\n'
fi
[[ "$TOKEN" =~ ^[^[:space:]]{16,}$ ]] || die 'worker token must be one line with at least 16 characters'

if ((SKIP_MODEL_DOWNLOAD == 0)); then
  DOWNLOAD_ARGS=(scripts/download_worker_models.py --assets-root "$ASSETS_ROOT" --osnet-url "$OSNET_URL")
  [[ -n "$MULTITASK_SDK_ROOT" ]] && DOWNLOAD_ARGS+=(--multitask-sdk-root "$MULTITASK_SDK_ROOT")
  [[ -n "$MULTITASK_SDK_URL" ]] && DOWNLOAD_ARGS+=(--multitask-sdk-url "$MULTITASK_SDK_URL")
  python3 "${DOWNLOAD_ARGS[@]}"
fi

umask 077
ENV_FILE="$PROJECT_ROOT/.env"
[[ -f "$ENV_FILE" ]] || cp "$PROJECT_ROOT/.env.example" "$ENV_FILE"
chmod 600 "$ENV_FILE"
set_env() {
  local key="$1" value="$2" escaped
  escaped="$(printf '%s' "$value" | sed 's/[&|]/\\&/g')"
  if grep -q "^${key}=" "$ENV_FILE"; then sed -i -E "s|^${key}=.*|${key}=${escaped}|" "$ENV_FILE"; else printf '%s=%s\n' "$key" "$value" >>"$ENV_FILE"; fi
}
SDK_HOST_ROOT="${MULTITASK_SDK_ROOT:-$ASSETS_ROOT/volleyball_inference_sdk}"
set_env VOLLYAI_SERVER_WS_URL "$WS_URL"
set_env VOLLYAI_TOKEN "$TOKEN"
set_env VOLLYAI_INSTANCE_ID "$INSTANCE_PREFIX-docker"
set_env VOLLYAI_DEVICE "$([[ "$TORCH_BACKEND" == cpu ]] && echo cpu || echo cuda:0)"
set_env VOLLYAI_TORCH_BACKEND "$TORCH_BACKEND"
set_env VOLLYAI_WORKER_IMAGE volleyball-analysis-engine:local
set_env VOLLYAI_MULTITASK_SDK_HOST_ROOT "$SDK_HOST_ROOT"
set_env VOLLYAI_SMP_HOST_ROOT "$ASSETS_ROOT/selective-mask-propagation"
COMPOSE_BASE=(--env-file "$ENV_FILE" -f "$PROJECT_ROOT/compose.yaml")
require_command docker
docker compose "${COMPOSE_BASE[@]}" build analysis-worker
if ((NO_START == 0)); then
  if ((${#GPU_IDS[@]})); then
    for gpu_id in "${GPU_IDS[@]}"; do
      VOLLYAI_GPU_ID="$gpu_id" VOLLYAI_INSTANCE_ID="$INSTANCE_PREFIX-gpu$gpu_id" docker compose --project-name "$INSTANCE_PREFIX-gpu$gpu_id" --env-file "$ENV_FILE" -f "$PROJECT_ROOT/compose.yaml" -f "$PROJECT_ROOT/compose.gpu.yaml" up -d --no-build
    done
  else
    docker compose "${COMPOSE_BASE[@]}" --project-name "$INSTANCE_PREFIX-cpu" up -d --no-build
  fi
fi
printf 'Ubuntu Docker installation complete: %s\n' "$PROJECT_ROOT"
