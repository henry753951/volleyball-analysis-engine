# Ubuntu deployment

This is the operator-facing path for a real Ubuntu host. The installer is executable through a
GitHub raw URL and downloads tar archives; it does not require `git clone`.

## 1. Host prerequisites

For a Docker deployment:

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl tar python3
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker "$USER"
```

Log out and back in after adding the user to the `docker` group. For NVIDIA GPUs, install a
matching NVIDIA driver and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html),
then verify:

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu24.04 nvidia-smi
```

For this deployment path, the host does not need uv. The Docker image uses uv internally during
image build.

## 2. Docker install, model assets and GitHub bootstrap

The command below downloads the engine and central source archives into
`$HOME/volleyball-analysis-engine` and `$HOME/volleyball-monitoring-ai`. It also downloads the
public OSNet/SMP assets using the concrete defaults in the script.

```bash
curl -fsSL https://raw.githubusercontent.com/henry753951/volleyball-analysis-engine/main/scripts/install-ubuntu.sh \
  | bash -s -- \
      --mode docker \
      --server-url https://volleyai.hsulab.net/ \
      --token 'vmai_replace-with-worker-token' \
      --multitask-sdk-url 'https://private.example.com/volleyball_inference_sdk.zip' \
      --gpu-ids 0,1
```

The `--server-url` value may be a base `https://`/`http://` URL or a complete `wss://`/`ws://`
URL. The default is `https://volleyai.hsulab.net/`, normalized to
`wss://volleyai.hsulab.net/api/v2/ai/providers/ws`. The token is written only to the local `.env`
with restrictive permissions and is passed to the outbound worker connection.

The public model links used by the Docker bootstrap are direct URLs:

| Asset | Direct URL | Host destination |
| --- | --- | --- |
| SMP source | [GitHub archive](https://codeload.github.com/holma91/selective-mask-propagation/tar.gz/refs/heads/main) | `.models/selective-mask-propagation` |
| Sports OSNet | [Hugging Face file](https://huggingface.co/datasets/holma91/SAM-Deep-EIoU/resolve/main/checkpoints/osnet_sports.pth.tar?download=true) | `.models/selective-mask-propagation/selective_mask_propagation/osnet/checkpoints/sports_model.pth.tar-60` |

The private multitask SDK must contain `volleyball_sdk/__init__.py` and `best.pth`. Supply either
`--multitask-sdk-root PATH` or a real downloadable zip with `--multitask-sdk-url URL`; the engine
repository does not publish that private artifact. No model manifest or SHA argument is used.

## 3. Multi-GPU behavior

`--gpu-ids 0,1` is an explicit physical GPU selection:

| Mode | Containers | Device value | Instance ID |
| --- | --- | --- | --- |
| Docker | two Compose projects | container-local `cuda:0` | `analysis-worker-gpu0`, `analysis-worker-gpu1` |

Each worker owns its own model replica and receives work through the same central lease queue. Use
`--gpu-id 2 --gpu-id 3` when a comma-separated list is inconvenient. Do not pass GPU IDs with
`--torch-backend cpu`.

The Compose override pins each project to one physical GPU through `device_ids`; inside each
container the assigned card is addressed as `cuda:0`. CPU-only Docker uses
`--torch-backend cpu` and starts one CPU container.

## 4. Docker host layout and operations

The installer downloads the engine and central repositories as sibling archives. The Docker build
uses the central archive as the named `central` build context; no Git metadata is needed. Model
directories are mounted read-only:

```text
host .models/volleyball_inference_sdk       -> /models/volleyball-sdk
host .models/selective-mask-propagation     -> /models/smp
Docker named volume                          -> /workspaces
Docker named volume                          -> /home/worker/.cache
```

Inspect the two GPU projects:

```bash
docker compose -p analysis-worker-gpu0 ps
docker compose -p analysis-worker-gpu1 ps
docker logs -f analysis-worker-gpu0-analysis-worker-1
```

## 5. Uninstall

The default removal stops only the worker containers and preserves source, `.env`, model assets and
named Docker volumes:

```bash
curl -fsSL https://raw.githubusercontent.com/henry753951/volleyball-analysis-engine/main/scripts/uninstall-ubuntu.sh \
  | bash -s -- --mode all
```

Explicit destructive cleanup:

```bash
curl -fsSL https://raw.githubusercontent.com/henry753951/volleyball-analysis-engine/main/scripts/uninstall-ubuntu.sh \
  | bash -s -- --mode all \
      --purge-models \
      --purge-docker-volumes \
      --purge-docker-image \
      --yes
```

Use `--engine-dir` and `--project-prefix` when the install used non-default paths or a different
worker prefix.
