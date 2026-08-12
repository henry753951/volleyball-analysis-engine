"""Batched court-line inference with frame-synchronous layout tracking."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from itertools import pairwise
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from numpy.typing import NDArray
from volley_court import (
    CourtFrameResult,
    CourtLayout,
    CourtLayoutTracker,
    CourtLineModel,
    LayoutTrackingConfig,
)

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
FULL_SEARCH_RECOVERY_AFTER = 5
MAXIMUM_LAYOUT_JUMP_RATIO = 0.12
ORIENTATION_MARGIN_RATIO = 0.04
MINIMUM_LAYOUT_COMPARISON_POINTS = 4
POSE36_CANONICAL_POINTS = (
    (0.0, 0.0),
    (0.0, 6.0),
    (0.0, 9.0),
    (0.0, 12.0),
    (0.0, 18.0),
    (9.0, 18.0),
    (9.0, 12.0),
    (9.0, 9.0),
    (9.0, 6.0),
    (9.0, 0.0),
    (0.0, 2.0),
    (0.0, 4.0),
    (0.0, 7.0),
    (0.0, 8.0),
    (0.0, 10.0),
    (0.0, 11.0),
    (0.0, 14.0),
    (0.0, 16.0),
    (3.0, 18.0),
    (6.0, 18.0),
    (9.0, 16.0),
    (9.0, 14.0),
    (9.0, 11.0),
    (9.0, 10.0),
    (9.0, 8.0),
    (9.0, 7.0),
    (9.0, 4.0),
    (9.0, 2.0),
    (6.0, 0.0),
    (3.0, 0.0),
    (3.0, 6.0),
    (6.0, 6.0),
    (3.0, 9.0),
    (6.0, 9.0),
    (3.0, 12.0),
    (6.0, 12.0),
)
_POSE36_INDEX = {point: index for index, point in enumerate(POSE36_CANONICAL_POINTS)}
_ORIENTATION_TRANSFORMS = {
    "identity": np.eye(3, dtype=np.float64),
    "left_right": np.asarray(((-1.0, 0.0, 9.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
    "near_far": np.asarray(((1.0, 0.0, 0.0), (0.0, -1.0, 18.0), (0.0, 0.0, 1.0))),
    "both": np.asarray(((-1.0, 0.0, 9.0), (0.0, -1.0, 18.0), (0.0, 0.0, 1.0))),
}
_ORIENTATION_PERMUTATIONS = {
    name: tuple(
        _POSE36_INDEX[
            (
                float(transform[0, 0] * x + transform[0, 2]),
                float(transform[1, 1] * y + transform[1, 2]),
            )
        ]
        for x, y in POSE36_CANONICAL_POINTS
    )
    for name, transform in _ORIENTATION_TRANSFORMS.items()
}


@dataclass(slots=True)
class CourtTiming:
    """Accumulated wall-clock costs for one clip."""

    source_frames: int = 0
    inference_frames: int = 0
    batches: int = 0
    accepted_frames: int = 0
    rejected_layouts: int = 0
    rejected_tracking_updates: int = 0
    recovery_layouts: int = 0
    orientation_corrections: int = 0
    orientation_jump_rejections: int = 0
    interpolated_frames: int = 0
    layout_attempts: int = 0
    prior_refined_matches: int = 0
    full_search_matches: int = 0
    layout_abstentions: int = 0
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
            "rejected_layouts": self.rejected_layouts,
            "rejected_tracking_updates": self.rejected_tracking_updates,
            "recovery_layouts": self.recovery_layouts,
            "orientation_corrections": self.orientation_corrections,
            "orientation_jump_rejections": self.orientation_jump_rejections,
            "interpolated_frames": self.interpolated_frames,
            "layout_attempts": self.layout_attempts,
            "prior_refined_matches": self.prior_refined_matches,
            "full_search_matches": self.full_search_matches,
            "layout_abstentions": self.layout_abstentions,
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
        self.max_hold_frames = max(self.refresh_every, max_hold_frames)
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
            batch_size=self.batch_size,
            layout_every=self.layout_every,
            refresh_every=self.refresh_every,
            track_every=self.track_every,
            max_hold_frames=self.max_hold_frames,
        )


class CourtVideoProcessor:
    """Infer every frame in batches while advancing accepted geometry every frame."""

    def __init__(
        self,
        model: CourtLineModel,
        *,
        batch_size: int,
        layout_every: int,
        refresh_every: int,
        track_every: int,
        max_hold_frames: int,
    ) -> None:
        """Initialize isolated buffering and tracking state."""
        self._model = model
        self._batch_size = max(1, batch_size)
        self._layout_every = layout_every
        self._refresh_every = refresh_every
        self._track_every = track_every
        self.max_hold_frames = max(refresh_every, max_hold_frames)
        self._tracker = CourtLayoutTracker(
            LayoutTrackingConfig(
                smoothing=1.0,
                fast_motion_threshold_px=18.0,
                max_hold_frames=self.max_hold_frames,
            )
        )
        self._pending: list[tuple[int, NDArray[np.uint8]]] = []
        self._last_reliable_layout: CourtLayout | None = None
        self._output_orientation = "identity"
        self._consecutive_match_misses = 0
        self.timing = CourtTiming()

    def submit(
        self,
        frame_index: int,
        frame: NDArray[np.uint8],
    ) -> dict[int, CourtFrame]:
        """Advance tracking and run fresh geometry only when it is useful."""
        self.timing.source_frames += 1
        if self._layout_every == 1:
            self._pending.append((frame_index, frame))
            return self._flush_pending() if len(self._pending) >= self._batch_size else {}

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
            self.timing.layout_attempts += 1
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
        self._last_reliable_layout = tracked
        self.timing.accepted_frames += 1
        return {frame_index: court}

    def finish(self) -> dict[int, CourtFrame]:
        """Flush a final partial full-frame batch."""
        return self._flush_pending()

    def _flush_pending(self) -> dict[int, CourtFrame]:
        if not self._pending:
            return {}
        pending = self._pending
        self._pending = []
        frames = [frame for _, frame in pending]
        started = perf_counter()
        results = self._model.predict_many(frames, include_layout=False)
        self.timing.model_seconds += perf_counter() - started
        self.timing.batches += 1
        self.timing.inference_frames += len(pending)
        output: dict[int, CourtFrame] = {}
        for (frame_index, frame), result in zip(pending, results, strict=True):
            court = self._process_frame_result(frame_index, frame, result)
            if court is not None:
                self.timing.accepted_frames += 1
                output[frame_index] = court
        return output

    def _process_frame_result(
        self,
        frame_index: int,
        frame: NDArray[np.uint8],
        result: CourtFrameResult,
    ) -> CourtFrame | None:
        prior = self._last_reliable_layout or self._tracker.current
        candidate = self._attach_layout(result, prior_layout=prior)
        if candidate is None or candidate.status != "ok":
            self._consecutive_match_misses += 1
            if prior is not None and self._consecutive_match_misses >= FULL_SEARCH_RECOVERY_AFTER:
                recovered = self._attach_layout(result, prior_layout=None)
                if recovered is not None and recovered.status == "ok":
                    candidate = recovered
                    self._tracker.reset()
                    self._consecutive_match_misses = 0
                    self.timing.recovery_layouts += 1
        else:
            self._consecutive_match_misses = 0

        accepted = candidate if candidate is not None and candidate.status == "ok" else None
        if accepted is not None and prior is None:
            self._output_orientation = self._screen_canonical_orientation(accepted)
        if accepted is not None and prior is not None:
            aligned, correction = self._align_orientation(accepted, prior)
            jump = self._layout_jump_px(prior, aligned)
            maximum_jump = MAXIMUM_LAYOUT_JUMP_RATIO * math.hypot(
                float(frame.shape[1]),
                float(frame.shape[0]),
            )
            if jump is None or jump > maximum_jump:
                accepted = None
                self.timing.orientation_jump_rejections += 1
            elif correction != "identity":
                self._output_orientation = self._compose_orientations(
                    correction,
                    self._output_orientation,
                )
                self.timing.orientation_corrections += 1
        tracked = self._track(accepted, frame)
        if accepted is None:
            return None
        output_layout = (
            None
            if tracked is None
            else self._reorient_layout(tracked, self._output_orientation)
        )
        court = self._court_frame(frame_index, output_layout)
        if court is not None:
            self._last_reliable_layout = accepted or tracked
        return court

    def _attach_layout(
        self,
        result: CourtFrameResult,
        *,
        prior_layout: CourtLayout | None,
    ) -> CourtLayout | None:
        started = perf_counter()
        self.timing.layout_attempts += 1
        layout = self._model.attach_layout(result, prior_layout=prior_layout).layout
        self.timing.layout_seconds += perf_counter() - started
        if layout is not None:
            if layout.matcher_mode == "prior_refined":
                self.timing.prior_refined_matches += 1
            elif layout.matcher_mode == "full_search":
                self.timing.full_search_matches += 1
            if layout.status != "ok":
                self.timing.layout_abstentions += 1
        return layout

    def _track(
        self,
        layout: CourtLayout | None,
        frame: NDArray[np.uint8],
    ) -> CourtLayout | None:
        started = perf_counter()
        tracked = self._tracker.update(
            layout,
            width=int(frame.shape[1]),
            height=int(frame.shape[0]),
            frame=frame,
        )
        self.timing.tracking_seconds += perf_counter() - started
        return tracked

    @staticmethod
    def _screen_canonical_orientation(layout: CourtLayout) -> str:
        """Choose one stable world orientation whose axes follow the video screen."""
        if len(layout.keypoints) != POSE36_POINT_COUNT:
            return "identity"

        def orientation_score(name: str) -> float:
            oriented = CourtVideoProcessor._reorient_layout(layout, name)
            points = {point.id: point for point in oriented.keypoints}
            if len(points) != POSE36_POINT_COUNT:
                return float("-inf")
            image_width = max(point.x for point in points.values()) - min(
                point.x for point in points.values()
            )
            image_height = max(point.y for point in points.values()) - min(
                point.y for point in points.values()
            )
            world_left_x = float(np.median([points[index].x for index in (0, 9)]))
            world_right_x = float(np.median([points[index].x for index in (4, 5)]))
            world_far_y = float(np.median([points[index].y for index in range(5)]))
            world_near_y = float(np.median([points[index].y for index in range(5, 10)]))
            return (world_right_x - world_left_x) / max(image_width, 1.0) + (
                world_near_y - world_far_y
            ) / max(image_height, 1.0)

        return max(_ORIENTATION_TRANSFORMS, key=orientation_score)

    @staticmethod
    def _align_orientation(
        candidate: CourtLayout,
        reference: CourtLayout,
    ) -> tuple[CourtLayout, str]:
        """Keep Pose36 left/right and near/far identities stable for one clip."""
        reference_points = {point.id: point for point in reference.keypoints}
        candidate_points = {point.id: point for point in candidate.keypoints}
        if (
            len(reference_points) != POSE36_POINT_COUNT
            or len(candidate_points) != POSE36_POINT_COUNT
        ):
            return candidate, "identity"
        scores = {
            name: float(
                np.median(
                    [
                        math.hypot(
                            candidate_points[candidate_id].x - reference_points[reference_id].x,
                            candidate_points[candidate_id].y - reference_points[reference_id].y,
                        )
                        for reference_id, candidate_id in enumerate(permutation)
                    ]
                )
            )
            for name, permutation in _ORIENTATION_PERMUTATIONS.items()
        }
        best = min(scores, key=scores.__getitem__)
        reference_span = max(
            math.hypot(
                max(point.x for point in reference_points.values())
                - min(point.x for point in reference_points.values()),
                max(point.y for point in reference_points.values())
                - min(point.y for point in reference_points.values()),
            ),
            1.0,
        )
        if (
            best == "identity"
            or scores[best] + ORIENTATION_MARGIN_RATIO * reference_span >= scores["identity"]
        ):
            return candidate, "identity"
        return CourtVideoProcessor._reorient_layout(candidate, best), best

    @staticmethod
    def _reorient_layout(layout: CourtLayout, orientation: str) -> CourtLayout:
        if orientation == "identity":
            return layout
        permutation = _ORIENTATION_PERMUTATIONS[orientation]

        def reindex(points: tuple[Any, ...]) -> tuple[Any, ...]:
            if len(points) != POSE36_POINT_COUNT:
                return points
            by_id = {point.id: point for point in points}
            if len(by_id) != POSE36_POINT_COUNT:
                return points
            return tuple(
                replace(by_id[candidate_id], id=reference_id)
                for reference_id, candidate_id in enumerate(permutation)
            )

        homography = layout.homography
        if homography is not None:
            normalized = (
                np.asarray(homography, dtype=np.float64)
                @ _ORIENTATION_TRANSFORMS[orientation]
            )
            homography = tuple(tuple(float(value) for value in row) for row in normalized)
        return replace(
            layout,
            keypoints=reindex(layout.keypoints),
            candidate_keypoints=reindex(layout.candidate_keypoints),
            homography=homography,
            reason=f"orientation locked ({orientation})",
        )

    @staticmethod
    def _compose_orientations(first: str, second: str) -> str:
        composed = _ORIENTATION_TRANSFORMS[first] @ _ORIENTATION_TRANSFORMS[second]
        return next(
            name
            for name, transform in _ORIENTATION_TRANSFORMS.items()
            if np.allclose(composed, transform)
        )

    @staticmethod
    def _layout_jump_px(current: CourtLayout, candidate: CourtLayout) -> float | None:
        previous = {point.id: point for point in current.keypoints}
        distances = [
            math.hypot(point.x - previous[point.id].x, point.y - previous[point.id].y)
            for point in candidate.keypoints
            if point.id in previous
        ]
        return (
            float(np.median(distances))
            if len(distances) >= MINIMUM_LAYOUT_COMPARISON_POINTS
            else None
        )

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


def interpolate_short_court_gaps(
    courts: dict[int, CourtFrame],
    *,
    max_gap: int = 2,
) -> int:
    """Fill only short gaps bracketed by valid layouts without holding stale geometry."""
    inserted = 0
    valid_frames = sorted(courts)
    for start_frame, end_frame in pairwise(valid_frames):
        gap = end_frame - start_frame - 1
        if gap <= 0 or gap > max_gap:
            continue
        start = courts[start_frame]
        end = courts[end_frame]
        start_points = {point.index: point for point in start.keypoints}
        end_points = {point.index: point for point in end.keypoints}
        if start_points.keys() != end_points.keys():
            continue
        pending: list[CourtFrame] = []
        for offset in range(1, gap + 1):
            alpha = offset / (gap + 1)
            keypoints = tuple(
                _interpolate_keypoint(start_points[index], end_points[index], alpha)
                for index in sorted(start_points)
            )
            court = CourtFrame(
                frame_index=start_frame + offset,
                available=True,
                keypoints=keypoints,
            )
            if estimate_homography(court) is None:
                pending = []
                break
            pending.append(court)
        for court in pending:
            courts[court.frame_index] = court
            inserted += 1
    return inserted


def _interpolate_keypoint(
    start: CourtKeypoint,
    end: CourtKeypoint,
    alpha: float,
) -> CourtKeypoint:
    position = None
    if start.frame_pos_px is not None and end.frame_pos_px is not None:
        position = (
            start.frame_pos_px[0] + alpha * (end.frame_pos_px[0] - start.frame_pos_px[0]),
            start.frame_pos_px[1] + alpha * (end.frame_pos_px[1] - start.frame_pos_px[1]),
        )
    confidence = None
    if start.confidence is not None and end.confidence is not None:
        confidence = start.confidence + alpha * (end.confidence - start.confidence)
    return CourtKeypoint(
        index=start.index,
        frame_pos_px=position,
        confidence=confidence,
        world_pos_m=start.world_pos_m if start.world_pos_m == end.world_pos_m else None,
    )
