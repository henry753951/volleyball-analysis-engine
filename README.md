# volleyball-analysis-engine

Headless `uv` project for running the external AI side of
`volleyball-monitoring-ai`. Online and offline modes use the same model pipeline:

```text
canonical clip
→ RT-DETRv4 + streaming X3D person/ball/action inference
→ OSNet appearance embeddings
→ harmonic-mean EIoU tracking with a retained lost pool
→ YOLO court-keypoint pose + RANSAC homography
→ same-side 2D-court re-entry identity merge (maximum six identities per side)
→ human-keypoint contact association
→ AnalysisResult JSON + VOV1 overlay + developer artifacts
```

The engine never reads Contract Lab tracking, ball or court JSON as inference output. Contract Lab
and `sdk-analysis-visual-v5` are reference material only. Human `clip_pts`, `clip_time_us` and
`clip_frame_index` values remain immutable anchors from the incoming job.

## Repository layout

Keep the central and AI repositories as siblings. The SDK is installed from the sibling path; it is
not vendored or used as a subrepository.

```text
H:\Repos\
├── volleyball-monitoring-ai\
├── volleyball-analysis-engine\
└── volley-ai\
```

Model files and the supplied RT-DETRv4 source are prepared under ignored `.artifacts/`; weights are
never committed.

## First-time setup on this machine

The defaults point at the assets supplied for this project:

```powershell
.\scripts\setup-dev.ps1
```

This extracts `E:\User\Downloads\volleyball_ball_action.zip`, copies `best_stg1.pth` and the
`orderfix` court-keypoint model, installs CUDA 13.0 PyTorch and model dependencies with `uv`, then
runs the environment doctor and strictly loads both checkpoints. Use `-TorchBackend cpu` for CPU-only setup or `-RefreshAssets` to
replace an earlier extraction.

The default X3D temporal backend is the exact rolling-window implementation supplied with the
model. `continual-inference` 1.2.4 cannot convert PyTorchVideo's `ResNetBasicStem`; selecting
`VOLLYAI_RTV4_BACKEND=continual` therefore attempts conversion and records an explicit fallback
to `rolling` instead of silently changing inference semantics.

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
RT-DETRv4/X3D, tracking, court and ReID run.
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

Copy `.env.example` to `.env` for persistent local values, or pass connection secrets to the
PowerShell launcher:

```powershell
.\scripts\run-online-worker.ps1 `
  -CentralUrl "ws://localhost:10000/api/v1/ai/providers/ws" `
  -Token "vmai_replace-with-worker-access-token" `
  -InstanceKey "analysis-worker-local"
```

Create or rotate the Worker Access Token from the central server operations console. The current
protocol has no Integration ID: the bearer token authenticates the worker pool, while
`InstanceKey` identifies this worker process. Port `10000` is the Docker Compose host mapping;
use port `4000` when the central server itself runs directly on the host.

The worker makes one outbound WebSocket connection, advertises current load, accepts a leased job,
downloads and verifies its canonical clip, runs the shared pipeline, reports progress and sends the
typed result through the authenticated callback.

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

The model source and checkpoints are trusted local assets supplied by the project owner. The engine
loads the RT-DETRv4 checkpoint with an exact `strict=True` state-dict match against
`rtv4_x3d_volleyball_v4a_decoupled.yml`. The supplied checkpoint is the 5-frame model
(`120/240/480` encoder channels); the similarly named 7-frame config is intentionally rejected.
