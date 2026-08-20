# volleyball-analysis-engine

Docker-first worker for running the external AI side of
`volleyball-monitoring-ai`. Online and offline modes use the same model pipeline:

```text
canonical clip
→ Volleyball inference SDK v2 temporal multitask inference on every target frame
  (person/ball/action + Court60 + COCO-17 pose + group activity in one pass)
→ OSNet appearance embeddings
→ upstream DeepEIOU run-local tracking
→ selective SAM3 correction only for low-margin/gap ambiguity windows
→ versioned ReID evidence jobs with co-visibility cannot-links (separate from Local IDs)
→ ball-trajectory contact proposal detection between explicit segment START/END boundaries
→ contact-to-player association
→ AnalysisResult 1.1 JSON + VOV1 overlay + developer artifacts
```

The engine never reads Contract Lab tracking, ball or court JSON as inference output. Contract Lab
and `sdk-analysis-visual-v5` are reference material only. Human `clip_pts`, `clip_time_us` and
`clip_frame_index` values remain immutable anchors from the incoming job. Job 2.0 segment
boundaries define analysis coverage; optional human X contacts are hints, not synthetic serve/end
events. Legacy Job 1.1 remains accepted for historical queued work.

`AnalysisResult.extensions.reid_feature_bank` is a versioned clip-scoped interchange artifact. It
contains exactly one L2-normalized 512-D Sports OSNet prototype per sampled run-local track, split
into left/right/unknown banks from projected court-side observations. Matching follows the
calibrated `volley-reid` eligibility boundary: at least 12 detections at least 28 pixels tall, then
the best-quality observation from each of four temporal bins is averaged into the exported
prototype. Per-frame embeddings remain transient and are never written to the result. The bank
also exports simultaneous-track cannot-links so a downstream match/roster identity service can
consolidate clips without rewriting clip-local track IDs.

## Repository layout

The worker repository contains the inference SDK directly under `src/volleyball_sdk/`; the Docker
image packages that code directly. The multitask checkpoint is downloaded from a Hugging Face file
URL into `.models/volleyball_multitask/`. The central repository remains a build-time dependency
and is cloned inside Docker, so an operator does not need sibling checkouts.

```text
H:\Repos\
├── volleyball-monitoring-ai\
└── volleyball-analysis-engine\
```

The bundled SDK is loaded from `src/volleyball_sdk` during local runs and copied to `/app/src` in
the Docker image. `best.pth` is the only private model asset supplied by URL and is mounted at
`/models/volleyball_multitask/best.pth`.

### Local ID tracking and selective SAM3

`VOLLYAI_LOCAL_TRACKER=deep_eiou` uses the reference DeepEIOU implementation from
`VOLLYAI_SMP_ROOT` and computes OSNet for every person detection. `VOLLYAI_LOCAL_SAM3_ENABLED=true`
adds out-of-process SAM3 correction only when the upstream margin/gap detector opens an ambiguity
window. The bridge uses `VOLLYAI_LOCAL_SAM3_PYTHON`, so its SAM3/PyTorch environment stays isolated
from the multitask worker environment. A missing runtime, timeout, decode mismatch, or invalid
co-visible rename keeps the complete DeepEIOU output and records the fallback in result metadata.

The local launcher enables this path by default. Use `-DisableLocalSam3` or
`volleyball-analysis-worker --disable-local-sam3` to keep DeepEIOU without SAM3. Docker Compose keeps
SAM3 off until the separate runtime and weights are intentionally packaged or mounted.

## First-time setup on this machine

The defaults point at the assets supplied for this project:

```powershell
.\scripts\setup-dev.ps1
```

This validates the bundled `src\volleyball_sdk` and the downloaded `.models\volleyball_multitask\best.pth`, installs CUDA 13.0
PyTorch and model dependencies with `uv`, then strictly loads and warms the multitask model plus
the retained OSNet tracking encoder. Use `-TorchBackend cpu` for CPU-only setup.

The SDK samples five-frame centered clips with its configured temporal spacing and emits one result
for every target frame. Court60 uses ten semantic anchors plus fifty deterministic edge samples;
all sixty points map to the existing 18 x 9 metre court. COCO-17 visibility is preserved as pose
confidence. Group activity is written to `AnalysisData.extensions.group_activity` with its provider
taxonomy, but no server or UI behavior interprets it yet.

Strictly load all checkpoints and inspect a clip:

```powershell
.\scripts\doctor.ps1 `
  -LoadModels `
  -Clip "H:\Repos\volleyball-ai-contract-lab\.data\exports\8469a80e-c0f5-4a57-8859-c8371de7c755\clip.mp4"
```

## Offline mode (no network)

Offline mode does not create an AI worker client, WebSocket, downloader or callback client. It
accepts the same online-shaped job so immutable IDs and media metadata stay contract-valid. An
optional standalone key-point JSON list can replace `job.key_points` for manual experiments.

```powershell
.\scripts\run-offline.ps1 `
  -Job "H:\Repos\volleyball-ai-contract-lab\.data\exports\8469a80e-c0f5-4a57-8859-c8371de7c755\ai-job.json" `
  -Clip "H:\Repos\volleyball-ai-contract-lab\.data\exports\8469a80e-c0f5-4a57-8859-c8371de7c755\clip.mp4" `
  -Output ".\outputs\sample"
```

For a latency benchmark that omits developer-only image/video rendering while preserving the full
typed result and VOV1 overlay, add `-Prewarm -NoDebugArtifacts`. The generated `benchmark.json`
separates model warmup from job wall time and reports effective FPS and real-time factor.

Output:

```text
outputs/sample/
├── analysis-result.json
├── overlay.vov1
├── offline-run.json
├── inference-manifest.json
├── visualization-manifest.json
├── tracks.jsonl
├── ball.jsonl
├── court.jsonl
├── actions.jsonl
├── preview-first-complete.jpg
├── preview-terminal-path.jpg
└── overlay-preview.mp4
```

The preview artifacts retain the Contract Lab `sdk-analysis-visual-v5` layout for direct visual
comparison, but use production-oriented names. Their contents come from the current
multitask-v2, tracking and ReID run.
`overlay-preview.mp4` uses the same 1920x1080 layout: a 1280x720 match
view, 640x720 event panel and 1920x360 canonical-court panel. FFmpeg encodes H.264/yuv420p,
preserves available AAC audio and writes `+faststart` metadata.

For a manually edited marker file, pass either a JSON array of `KeyPointInput` objects or an
object shaped as `{ "key_points": [...] }`:

```powershell
.\scripts\run-offline.ps1 `
  -Job ".\inputs\ai-job.json" `
  -Keypoints ".\inputs\keypoints.json" `
  -Clip ".\inputs\clip.mp4" `
  -Output ".\outputs\manual-run"
```

The key-point file replaces only `job.key_points`; media identity, immutable submission IDs,
clip hash, frame rate and time base still come from the validated job envelope.

## Online worker mode

Create or rotate the Worker Access Token from the central server operations console. The current
protocol has no Integration ID: the bearer token authenticates the worker pool, while each Docker
`VOLLYAI_INSTANCE_ID` identifies one worker replica. The Ubuntu Docker installer persists the
token and connects to `wss://volleyai.hsulab.net/api/v2/ai/providers/ws`.

The worker makes one outbound WebSocket connection, advertises current load, accepts a leased job,
downloads and verifies its canonical clip, runs the shared pipeline, reports progress and sends the
typed result through the authenticated callback.

The supported deployment path is Ubuntu-first. The installer downloads the bootstrap archive and
public model assets, while the Docker build hydrates the bundled SDK/checkpoint and clones the
central repository internally. It persists the WSS endpoint and token in a local protected
environment file and starts one Docker worker per selected GPU. See the deployment runbook:

- [Ubuntu deployment](docs/UBUNTU_DEPLOY.md)

## Ubuntu one-line installation and removal

Docker with two GPUs and an existing worker token:

```bash
curl -fsSL https://raw.githubusercontent.com/henry753951/volleyball-analysis-engine/main/scripts/install-ubuntu.sh \
  | bash -s -- \
      --mode docker \
      --server-url 'wss://volleyai.hsulab.net/api/v2/ai/providers/ws' \
      --token 'vmai_replace-with-worker-token' \
      --multitask-checkpoint-url 'https://huggingface.co/Henry753951/volleyball-analysis-multitask-v2/resolve/main/best.pth?download=true' \
      --osnet-url 'https://huggingface.co/datasets/holma91/SAM-Deep-EIoU/resolve/main/checkpoints/osnet_sports.pth.tar?download=true' \
      --gpu-ids 0,1
```

The recommended `--server-url` value is the complete WebSocket endpoint:
`wss://volleyai.hsulab.net/api/v2/ai/providers/ws`. The installer also accepts the base URL
`https://volleyai.hsulab.net/` and normalizes it to the same endpoint. The command includes the
project's Hugging Face resolve URL for `best.pth` and the concrete public OSNet URL. The SDK code
is bundled in the repository, so no SDK path, manifest URL or SHA argument is required. Use
`--torch-backend cpu` for CPU-only Docker.

Remove worker processes and containers while retaining source, token and model assets:

```bash
curl -fsSL https://raw.githubusercontent.com/henry753951/volleyball-analysis-engine/main/scripts/uninstall-ubuntu.sh \
  | bash -s -- --mode all
```

Only use `--purge-models --purge-docker-volumes --purge-docker-image --yes` when those local
assets are intentionally disposable.

Template links:

- [Ubuntu installer](scripts/install-ubuntu.sh)
- [Ubuntu uninstaller](scripts/uninstall-ubuntu.sh)
- [combined Ubuntu Docker deployment and model guide](docs/UBUNTU_DEPLOY.md)

Models are prewarmed before the worker advertises readiness. Production jobs skip heavy developer
preview rendering by default; download, analysis, and completion durations are logged separately.
The temporal multitask model produces detector, action, court, pose and group evidence for every
target frame. Velocity-aware tracking only bridges an isolated detector miss.

For the local Docker Compose central server, create a managed Worker Token and start the worker in
one command. The token is passed only through the child-process environment and is not written to
the log:

```powershell
.\scripts\start-local-worker.ps1
```

## Quality gates

```powershell
uv run ruff check .
uv run pyright
uv run pytest
```

The model source and checkpoint are trusted local assets supplied by the project owner. The engine
requires Volleyball inference schema `2.0`; the SDK loader enforces at least 98 percent compatible
checkpoint tensor coverage before the worker can advertise readiness.
