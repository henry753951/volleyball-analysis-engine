# Ubuntu Deployment Prompt

You are deploying Volleyball Analysis Engine workers on an Ubuntu host.

Complete the deployment and verification steps directly on the host. Do not ask the operator to
manually install Docker, NVIDIA tooling, models, or worker containers unless a command fails and
requires operator intervention.

Never ask the operator to paste a Worker Token into chat. Ask them to enter it directly in their
terminal with hidden input, then reply only `done`.

## Required server URL

Use this complete WebSocket endpoint as the recommended value:

```text
wss://volleyai.hsulab.net/api/v2/ai/providers/ws
```

The installer also accepts this base URL and converts it automatically:

```text
https://volleyai.hsulab.net/
```

Do not use `wss://volleyai.hsulab.net/api/v1/ai/providers/ws`. The `/api/v1/operations/...`
routes are REST management endpoints and are not the worker WebSocket endpoint.

## Verify the host

```bash
cat /etc/os-release
uname -m
docker --version || true
docker compose version || true
nvidia-smi || true
```

If Docker is already working, do not reinstall it. The host does not need `uv`; `uv` is used only
inside the Docker image during the build.

For GPU deployment, this must succeed before continuing:

```bash
docker run --rm --gpus all \
  nvidia/cuda:13.0.0-base-ubuntu24.04 \
  nvidia-smi
```

## Enter credentials securely

```bash
read -rp "VolleyAI Server URL [wss://volleyai.hsulab.net/api/v2/ai/providers/ws]: " VOLLEYAI_SERVER_URL
export VOLLEYAI_SERVER_URL="${VOLLEYAI_SERVER_URL:-wss://volleyai.hsulab.net/api/v2/ai/providers/ws}"

read -rsp "VolleyAI Worker Token: " VOLLEYAI_WORKER_TOKEN
echo
export VOLLEYAI_WORKER_TOKEN
```

Never print `VOLLEYAI_WORKER_TOKEN`, `env`, or `printenv`.

## Deploy

For GPU 0 and GPU 1:

```bash
curl -fsSL \
  https://raw.githubusercontent.com/henry753951/volleyball-analysis-engine/main/scripts/install-ubuntu.sh \
  | bash -s -- \
      --mode docker \
      --server-url "$VOLLEYAI_SERVER_URL" \
      --token "$VOLLEYAI_WORKER_TOKEN" \
      --multitask-checkpoint-url \
      'https://huggingface.co/Henry753951/volleyball-analysis-multitask-v2/resolve/main/best.pth?download=true' \
      --osnet-url \
      'https://huggingface.co/datasets/holma91/SAM-Deep-EIoU/resolve/main/checkpoints/osnet_sports.pth.tar?download=true' \
      --gpu-ids 0,1
```

Change `--gpu-ids` to the physical GPUs selected for this host. The installer starts one worker per
selected GPU; each container sees its assigned device as `cuda:0`.

The SDK is already bundled at `src/volleyball_sdk/`. Do not provide an SDK URL, manifest URL, SHA,
Git LFS checkout, or `uv` command on the host.

For CPU-only deployment, omit `--gpu-ids` and add:

```text
--torch-backend cpu
```

## Verify

```bash
docker ps --filter name=analysis-worker
docker logs --tail 100 analysis-worker-gpu0-analysis-worker-1
docker exec analysis-worker-gpu0-analysis-worker-1 nvidia-smi
```

Confirm that containers are running, CUDA and models initialize, the expected GPU is used, and the
worker connects without authentication errors. Do not print the Worker Token while debugging.

## Uninstall

Preserve source, model assets, credentials, and Docker volumes:

```bash
curl -fsSL \
  https://raw.githubusercontent.com/henry753951/volleyball-analysis-engine/main/scripts/uninstall-ubuntu.sh \
  | bash -s -- --mode all
```

Only use `--purge-models`, `--purge-docker-volumes`, or `--purge-docker-image --yes` when the
operator explicitly requests destructive cleanup.
