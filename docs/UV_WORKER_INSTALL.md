# Install and run the worker with uv

This is the preferred deployment path while the container release is deferred. The worker makes
one outbound Provider Work v2 WebSocket connection; no inbound port is required. The Python SDK
reconnects with backoff after a network interruption.

## Repository layout

The central SDK is a local package dependency, so clone both repositories as siblings:

```text
work/
├── volleyball-monitoring-ai/
└── volleyball-analysis-engine/
```

```powershell
git clone https://github.com/henry753951/volleyball-monitoring-ai.git
git clone https://github.com/henry753951/volleyball-analysis-engine.git
cd volleyball-analysis-engine
```

Install Git, Python support, and `uv` before continuing. On Windows:

```powershell
winget install --id Git.Git -e
winget install --id astral-sh.uv -e
```

## Model assets

The setup scripts never commit or publish weights. Public model assets are pinned and verified:

| Asset | Source | SHA-256 |
| --- | --- | --- |
| DINOv2 ViT-S/14 registers | Meta DINOv2 | `f433177089a681826f849f194ece3bb48f4d63fb38d32fc837e3dc7a4e5641fb` |
| Sports OSNet | `holma91/SAM-Deep-EIoU` | `8d5b2fd8763db34c2aad69810466adf413f0426d9f8119d322227e0e639c5fbd` |
| Official KPR checkpoint | Official KPR Drive | `9bea1e6dd887fb7af8c2f154912cce846c8c809c4c2357df4b89282889b31a20` |

The project-specific Volleyball multitask SDK is private. Supply one of:

- an existing directory containing `volleyball_sdk/__init__.py` and `best.pth`;
- a zip extracted to `.models/volleyball_inference_sdk`;
- a private downloadable zip URL through `-MultitaskSdkUrl`.

The current trusted `best.pth` digest is
`60ecd86921e13600b7de3f375bdc01ed4cbcd64330e1e2509b77e70f7bcc4ea3`. Override
`-MultitaskSha256` only when intentionally rolling out a newly validated checkpoint.

### Base worker on Windows

Base mode runs unified analysis, ReID association, and identity previews. It downloads the SMP
source and Sports OSNet automatically, while reusing the supplied multitask SDK:

```powershell
.\scripts\setup-uv-worker.ps1 `
  -MultitaskSdkRoot "D:\models\volleyball_inference_sdk" `
  -TorchBackend cu130
```

Stop an already-running worker before rerunning setup; Windows locks the generated worker
executable while the process is active. The script reports the exact worker PID instead of leaving
`uv` in a partial update.

Use `-TorchBackend cpu` on a machine without an NVIDIA GPU. The unified analysis pass produces
player, ball, action, Court60, COCO-17 pose, and group activity; there is no standalone court or pose
model.

### Full ReID feature worker on Windows

Download the named KPR checkpoint from the official KPR pretrained-model folder, then run:

```powershell
.\scripts\setup-uv-worker.ps1 `
  -MultitaskSdkRoot "D:\models\volleyball_inference_sdk" `
  -WithReid `
  -KprCheckpoint "D:\downloads\kpr_occ_pt_IN_82.34_92.33_42323828.pth.tar" `
  -TorchBackend cu130
```

This additionally clones DINOv2 and KPR, downloads DINOv2, creates an isolated `uv` Python 3.10
environment for KPR, and enables `REID_FEATURE_EXTRACTION`. A direct private KPR file URL may be
passed with `-KprCheckpointUrl` instead.

## Configure the Worker Access Token

For a remote or production central server, create a Worker Access Token in the operations console
and run:

```powershell
.\scripts\configure-uv-worker.ps1 `
  -ServerUrl "https://volleyball.example.com" `
  -Token "vmai_replace-with-worker-access-token" `
  -AssetsRoot ".\.models" `
  -WithReid
```

`ServerUrl` accepts either the HTTP base URL or complete WebSocket URL and normalizes it to
`/api/v2/ai/providers/ws`. Omit `-WithReid` if the optional DINO/KPR stack was not installed. The
script detects CUDA, validates every required path, and writes `.env.worker`, which is excluded from
Git. If the token argument is omitted, it prompts without echoing it.

For the local development central server only, the script can create a development token:

```powershell
.\scripts\configure-uv-worker.ps1 `
  -ServerUrl "http://localhost:10000" `
  -CreateLocalToken `
  -CentralHttpUrl "http://localhost:10000" `
  -AssetsRoot ".\.models" `
  -Force
```

The development token endpoint must not be exposed in a production deployment.

## Start and observe

Foreground mode keeps logs in the terminal:

```powershell
.\scripts\start-uv-worker.ps1
```

Background mode returns the process ID and log paths:

```powershell
.\scripts\start-uv-worker.ps1 -Background
```

Rerun `configure-uv-worker.ps1 -Force` to rotate the token or change the central URL. The same
stable `InstanceId` is retained unless explicitly changed.

## Linux

The model downloader is Python and works on Linux. After cloning the sibling repositories:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
cd volleyball-analysis-engine
uv sync --extra cu130 --extra models
uv run --no-sync python scripts/download_worker_models.py \
  --assets-root .models \
  --multitask-sdk-root /srv/vollyai/volleyball_inference_sdk
```

For full ReID, add `--with-reid --kpr-checkpoint /path/to/kpr_occ_pt_IN_82.34_92.33_42323828.pth.tar`,
then create the isolated KPR runtime:

```bash
uv python install 3.10
uv venv --python 3.10 --seed .models/kpr/.venv
uv pip install --python .models/kpr/.venv/bin/python torch torchvision \
  --index-url https://download.pytorch.org/whl/cu130
uv pip install --python .models/kpr/.venv/bin/python -r .models/kpr/requirements.txt
uv pip install --python .models/kpr/.venv/bin/python --editable .models/kpr
```

Set the same `VOLLYAI_*` values shown in `.env.example`, then start with:

```bash
uv run --no-sync volleyball-analysis-worker
```

Do not enable `VOLLYAI_REID_FEATURE_ENABLED` until DINOv2, KPR, KPR Python 3.10, and their
checkpoints all exist. Association and preview jobs may remain enabled independently.
