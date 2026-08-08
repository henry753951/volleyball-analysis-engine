# volleyball-analysis-engine

Outbound AI worker for `volleyball-monitoring-ai`. It uses the central repository's
Python SDK as the wire authority, connects to the central provider WebSocket, accepts
leased jobs, downloads and verifies the canonical clip, emits progress, and returns a
contract-valid analysis JSON plus VOV1 FlatBuffer overlay through the signed callback.

## Current analysis backend

The first vertical slice deliberately uses the recorded outputs in
`volleyball-ai-contract-lab/ai-team-handoff`:

- court keypoints are recomputed into an image-to-court homography with RANSAC;
- player observations come from the saved SAM + Deep-EIoU tracking JSONL;
- track fragments are merged by nearest re-entry position in canonical 2D court space,
  independently on each court side and at most six identities per side;
- hitter association uses the fixed human ball JSON accepted for this phase;
- actions are explicitly marked as an A/B court-position heuristic, not model inference.

The incoming job remains authoritative for clip SHA-256, immutable submission identity,
key-point order, `clip_pts`, `clip_time_us`, and `clip_frame_index`. Fixture frames are
scaled into the canonical clip frame domain; they never replace a key point's anchor.

## Local setup

```powershell
Copy-Item .env.example .env
uv sync --extra dev
uv run ruff check .
uv run pyright
uv run pytest
uv run volleyball-analysis-worker
```

Set `VOLLYAI_FIXTURE_ROOT` to:

```text
H:\Repos\volleyball-ai-contract-lab\ai-team-handoff
```

The project follows uv's explicit PyTorch-index pattern. `uv sync --extra cpu` uses the
CPU wheel index and `uv sync --extra cu130` selects CUDA 13.0; the extras conflict so a
single environment cannot accidentally mix backends.

## Container

The Docker build expects the central SDK submodule at
`vendor/volleyball-monitoring-ai`. Clone with submodules, configure `.env`, then set the
host fixture path and start the worker:

```powershell
$env:VOLLYAI_REFERENCE_ROOT='H:\Repos\volleyball-ai-contract-lab\ai-team-handoff'
docker compose up --build
```

The worker is outbound-only and exposes no inference HTTP port. Horizontal replicas
use distinct `VOLLYAI_INSTANCE_ID` values; the central server selects the globally
least-loaded online instance by active jobs divided by declared concurrency.

## Verified integration

The 2026-08-09 local integration run connected two one-slot containers to the central
WebSocket gateway. A real 398-frame OME canonical clip was assigned to
`analysis-worker-01`, completed through the signed multipart callback, and persisted as
12 court-side tracks (6 left / 6 right), two immutable key-point associations, one ball
path, four overlay chunks and four analysis artifacts. The service/contact anchors
remained exactly aligned at clip frames 180 and 218 from worker input through central
ingest. Identity mapping is intentionally a separate downstream operator step.
