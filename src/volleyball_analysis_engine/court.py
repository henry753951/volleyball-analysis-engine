"""Sparse court-line inference with frame-synchronous layout tracking."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
from numpy.typing import NDArray
from volley_court import CourtLayout, CourtLayoutTracker, CourtLineModel, LayoutTrackingConfig

from .geometry import estimate_homography
from .records import CourtFrame, CourtKeypoint

COURT_WORLD_POINTS = (
    (0.0, 0.0),
    (6.0, 0.0),
    (9.0, 0.0),
    (12.0, 0.0),
    (18.0, 0.0),
    (18.0, 9.0),
    (12.0, 9.0),
    (9.0, 9.0),
    (6.0, 9.0),
    (0.0, 9.0),
)
POSE36_POINT_COUNT = 36


@dataclass(slots=True)
class CourtTiming:
    """Accumulated wall-clock costs for one clip."""

    source_frames: int = 0
    inference_frames: int = 0
    batches: int = 0
    accepted_frames: int = 0
    model_seconds: float = 0.0
    layout_seconds: float = 0.0
    tracking_seconds: float = 0.0

    def to_mapping(self) -> dict[str, int | float]:
        """Return JSON-ready timing counters."""
        total = self.model_seconds + self.layout_seconds + self.tracking_seconds
        return {
            "source_frames": self.source_frames,
            "inference_frames": self.inference_frames,
            "batches": self.batches,
            "accepted_frames": self.accepted_frames,
            "model_seconds": self.model_seconds,
            "layout_seconds": self.layout_seconds,
            "tracking_seconds": self.tracking_seconds,
            "total_seconds": total,
            "source_fps": self.source_frames / max(total, 1e-9),
        }


class CourtLineEstimator:
    """Persistent GPU model that creates bounded per-video processors."""

    def __init__(
        self,
        model: str | Path | None,
        *,
        device: str,
        image_size: int,
        decoder: str,
        batch_size: int,
        layout_every: int,
        refresh_every: int,
        track_every: int,
        max_hold_frames: int,
    ) -> None:
        """Load one verified court model for reuse across clips."""
        self.model_name = str(model or "v1")
        self.batch_size = max(1, batch_size)
        self.layout_every = max(1, layout_every)
        self.refresh_every = max(self.layout_every, refresh_every)
        self.track_every = max(1, track_every)
        self.max_hold_frames = max(0, max_hold_frames)
        self._model = CourtLineModel.from_pretrained(
            model,
            device=device,
            image_size=image_size,
            decoder=decoder,
            half=True,
            include_layout=False,
        )

    def warmup(self) -> None:
        """Materialize CUDA kernels before the worker advertises readiness."""
        self._model.warmup(batch_size=self.batch_size, iterations=2)

    def begin_video(self) -> CourtVideoProcessor:
        """Create isolated temporal state for one clip."""
        return CourtVideoProcessor(
            self._model,
            layout_every=self.layout_every,
            refresh_every=self.refresh_every,
            track_every=self.track_every,
            max_hold_frames=self.max_hold_frames,
        )


class CourtVideoProcessor:
    """Run sparse geometry inference while advancing accepted geometry every frame."""

    def __init__(
        self,
        model: CourtLineModel,
        *,
        layout_every: int,
        refresh_every: int,
        track_every: int,
        max_hold_frames: int,
    ) -> None:
        """Initialize isolated buffering and tracking state."""
        self._model = model
        self._layout_every = layout_every
        self._refresh_every = refresh_every
        self._track_every = track_every
        self._tracker = CourtLayoutTracker(LayoutTrackingConfig(max_hold_frames=max_hold_frames))
        self.timing = CourtTiming()

    def submit(
        self,
        frame_index: int,
        frame: NDArray[np.uint8],
    ) -> dict[int, CourtFrame]:
        """Advance tracking and run fresh geometry only when it is useful."""
        self.timing.source_frames += 1
        has_layout = self._tracker.current is not None
        inference_every = self._refresh_every if has_layout else self._layout_every
        should_infer = frame_index % inference_every == 0
        layout = None
        if should_infer:
            started = perf_counter()
            result = self._model.predict(frame, include_layout=False)
            self.timing.model_seconds += perf_counter() - started
            self.timing.batches += 1
            self.timing.inference_frames += 1
            started = perf_counter()
            layout = self._model.attach_layout(result).layout
            self.timing.layout_seconds += perf_counter() - started

        should_track = layout is not None or frame_index % self._track_every == 0
        if should_track:
            started = perf_counter()
            tracked = self._tracker.update(
                layout,
                width=int(frame.shape[1]),
                height=int(frame.shape[0]),
                frame=frame,
            )
            self.timing.tracking_seconds += perf_counter() - started
        else:
            tracked = self._tracker.current
        court = self._court_frame(frame_index, tracked)
        if court is None:
            return {}
        self.timing.accepted_frames += 1
        return {frame_index: court}

    def finish(self) -> dict[int, CourtFrame]:
        """Return no deferred results; sparse inference is processed online."""
        return {}

    @staticmethod
    def _court_frame(frame_index: int, layout: CourtLayout | None) -> CourtFrame | None:
        if layout is None or layout.status != "ok":
            return None
        raw_points = layout.keypoints
        if len(raw_points) != POSE36_POINT_COUNT:
            return None
        keypoints = tuple(
            CourtKeypoint(
                index=point.id,
                frame_pos_px=(float(point.x), float(point.y)),
                confidence=float(point.score),
                world_pos_m=(
                    COURT_WORLD_POINTS[point.id] if point.id < len(COURT_WORLD_POINTS) else None
                ),
            )
            for point in raw_points
        )
        court = CourtFrame(frame_index=frame_index, available=True, keypoints=keypoints)
        return court if estimate_homography(court) is not None else None
