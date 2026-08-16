# volleyball-analysis-engine

Headless `uv` project for running the external AI side of
`volleyball-monitoring-ai`. Online and offline modes use the same model pipeline:

```text
canonical clip
→ Volleyball inference SDK v2 temporal multitask inference on every target frame
  (person/ball/action + Court60 + COCO-17 pose + group activity in one pass)
→ OSNet appearance embeddings
→ harmonic-mean EIoU tracking with a retained lost pool
→ sparse clip-local OSNet prototypes + co-visibility cannot-links (run-local IDs stay unchanged)
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
also exports simultaneous-track cannot-links and the cached OSNet checkpoint SHA256 so a downstream
match/roster identity service can consolidate clips without rewriting clip-local track IDs.

## Repository layout

Keep the central and AI repositories as siblings. The SDK is installed from the sibling path; it is
not vendored or used as a subrepository.

```text
H:\Repos\
├── volleyball-monitoring-ai\
├── volleyball-analysis-engine\
└── volley-ai\
```

The inference SDK and checkpoint remain external trusted assets and are selected with
`VOLLYAI_MULTITASK_SDK_ROOT` and `VOLLYAI_MULTITASK_CHECKPOINT`; weights are never committed.

### Optional ReID VLM

The jersey-number VLM is opt-in. Leave `VOLLYAI_REID_VLM_ENABLED=false` to avoid loading the
model and to omit its artifact kind and model recipe from Worker capability registration. The
worker command can override the environment for one process:

```powershell
volleyball-analysis worker --disable-reid-vlm
volleyball-analysis-worker --enable-reid-vlm
```

The VLM is effective only when `VOLLYAI_REID_FEATURE_ENABLED=true`; analysis and saved every-frame
pose evidence remain independent of it. `scripts/start-local-worker.ps1` and
`scripts/run-online-worker.ps1` keep VLM disabled unless `-EnableReidVlm` is passed explicitly.

## First-time setup on this machine

The defaults point at the assets supplied for this project:

```powershell
.\scripts\setup-dev.ps1
```

This validates `E:\User\Downloads\volleyball_inference_sdk` and its `best.pth`, installs CUDA 13.0
PyTorch and model dependencies with `uv`, then strictly loads and warms the multitask model plus
the retained OSNet tracking encoder. Use `-MultitaskSdkRoot` for another asset location or
`-TorchBackend cpu` for CPU-only setup.

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

### Historical pre-v2 benchmark

The following numbers describe the removed separate-model pipeline and are retained only as a
comparison baseline. The current multitask-v2 path must be benchmarked separately before release.
The old quality profile performed fresh player, ball, action, ReID, and court-line model
inference on every source frame. The v3 model returns a direct 36-point layout for every frame;
the engine consumes that layout once and never sends it through the legacy layout matcher again.
Only `ok` layouts containing all 36 keypoints enter temporal tracking. Ambiguous or abstained
frames never become accepted output and are not replaced by stale geometry.
Model warmup is intentionally outside job wall time because the online worker completes it before
registering. Developer preview rendering is disabled online by default; its final H.264/AAC web
encode uses NVENC when available and falls back to libx264.

Measured on 2026-08-12 with the 884-frame, 59.737 FPS `clip.mp4` (14.798 seconds):

| Path | Job wall time | Effective FPS | Real-time factor |
| --- | ---: | ---: | ---: |
| Every model on every source frame | 73.654 s | 12.00 | 0.201x |
| Every model + 1920x1080 preview package | 103.455 s | 8.54 | 0.143x |

The historical RT-DETRv4/X3D detector cadence was fixed at every source frame.
The tracker may bridge an isolated detector miss for at most two frames using measured velocity,
but its longer ReID recovery pool is never rendered. This prevents stale identities from flashing
back into the overlay while retaining short-gap continuity. The rendered H.264 stream was checked
to contain all 884 source frames; audio ending a few milliseconds earlier does not truncate the
video stream.

Court layout recovery keeps a clip-level Pose36 orientation transform. If a later direct proposal
returns the same geometry with a left/right, near/far, or 180-degree symmetric identity, keypoint
IDs and the output homography are mapped back to the established orientation. Raw model/tracker
geometry stays unchanged, so the lock does not reduce per-frame tracking continuity.

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

Copy `.env.example` to `.env` for persistent local values, or pass connection secrets to the
PowerShell launcher:

```powershell
.\scripts\run-online-worker.ps1 `
  -CentralUrl "ws://localhost:10000/api/v2/ai/providers/ws" `
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
