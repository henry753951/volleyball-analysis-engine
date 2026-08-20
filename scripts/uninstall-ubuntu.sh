#!/usr/bin/env bash
set -euo pipefail

# Can be executed directly from GitHub. It only targets the named worker projects/processes.
PROJECT_ROOT="${VOLLYAI_ENGINE_DIR:-$HOME/volleyball-analysis-engine}"
MODE=all
PROJECT_PREFIX=analysis-worker
ASSETS_ROOT=""
PURGE_MODELS=0
PURGE_VOLUMES=0
PURGE_IMAGE=0
YES=0

die() { printf 'error: %s\n' "$*" >&2; exit 1; }
show_help() {
  cat <<'EOF'
Usage: uninstall-ubuntu.sh [options]
  --mode all|docker                Docker runtime to remove (default: all)
  --engine-dir PATH                Engine install directory
  --project-prefix NAME            Prefix used by the installer
  --assets-root PATH               Model directory when purging
  --purge-models                   Remove downloaded model assets
  --purge-docker-volumes           Remove named Docker workspace/cache volumes
  --purge-docker-image             Remove volleyball-analysis-engine:local
  --yes                            Skip purge confirmation
EOF
}
while (($#)); do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --engine-dir) PROJECT_ROOT="$2"; shift 2 ;;
    --project-prefix) PROJECT_PREFIX="$2"; shift 2 ;;
    --assets-root) ASSETS_ROOT="$2"; shift 2 ;;
    --purge-models) PURGE_MODELS=1; shift ;;
    --purge-docker-volumes) PURGE_VOLUMES=1; shift ;;
    --purge-docker-image) PURGE_IMAGE=1; shift ;;
    --yes) YES=1; shift ;;
    -h|--help) show_help; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done
[[ "$MODE" == all || "$MODE" == docker ]] || die '--mode must be all or docker'
if [[ "$PROJECT_ROOT" != /* ]]; then PROJECT_ROOT="$PWD/$PROJECT_ROOT"; fi
PROJECT_ROOT="$(realpath -m "$PROJECT_ROOT")"
if [[ -z "$ASSETS_ROOT" ]]; then ASSETS_ROOT="$PROJECT_ROOT/.models"; fi
if [[ "$ASSETS_ROOT" != /* ]]; then ASSETS_ROOT="$PROJECT_ROOT/$ASSETS_ROOT"; fi
ASSETS_ROOT="$(realpath -m "$ASSETS_ROOT")"
confirm() { ((YES)) && return 0; read -r -p "$1 [y/N] " answer; [[ "$answer" == y || "$answer" == Y ]]; }

if [[ "$MODE" == all || "$MODE" == docker ]]; then
  if command -v docker >/dev/null 2>&1; then
    mapfile -t CONTAINER_NAMES < <(docker ps -a --format '{{.Names}}')
    PROJECT_PATTERN="^${PROJECT_PREFIX}-(gpu[0-9]+|cpu)-analysis-worker-[0-9]+$"
    PROJECTS=()
    for name in "${CONTAINER_NAMES[@]}"; do
      if [[ "$name" =~ $PROJECT_PATTERN ]]; then
        project="${name%-analysis-worker-*}"
        [[ " ${PROJECTS[*]} " == *" $project "* ]] || PROJECTS+=("$project")
      fi
    done
    for project in "${PROJECTS[@]}"; do
      if [[ -f "$PROJECT_ROOT/compose.yaml" ]]; then
        DOWN_ARGS=(--project-name "$project" -f "$PROJECT_ROOT/compose.yaml" -f "$PROJECT_ROOT/compose.gpu.yaml" down --remove-orphans)
        ((PURGE_VOLUMES)) && DOWN_ARGS+=(--volumes)
        docker compose "${DOWN_ARGS[@]}"
      else
        for name in "${CONTAINER_NAMES[@]}"; do
          if [[ "$name" == "$project-analysis-worker-"* ]]; then docker rm -f "$name" >/dev/null; fi
        done
      fi
    done
    if ((PURGE_IMAGE)); then docker image rm volleyball-analysis-engine:local 2>/dev/null || true; fi
  fi
fi

if ((PURGE_MODELS)); then
  confirm "Remove downloaded model assets at $ASSETS_ROOT?" || die 'model removal cancelled'
  rm -rf -- "$ASSETS_ROOT"
fi
printf 'Ubuntu worker uninstall complete; source code and .env were preserved\n'
