"""Shared inference contracts and fallback tracking for the unified model provider."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment  # pyright: ignore[reportUnknownVariableType]
from volleyball_monitoring_ai import AIJobRequest

from .records import (
    ActionObservation,
    BallObservation,
    CourtFrame,
    GroupActivityObservation,
    PersonPoseObservation,
    PlayerObservation,
)

ProgressReporter = Callable[[float, str], None]


def _unit_interval(value: float) -> float:
    """Clamp a model-space frame coordinate to the public video-coordinate domain."""
    return min(1.0, max(0.0, value))


def normalize_frame_bbox(
    bbox: tuple[float, float, float, float],
    *,
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    """Normalize a detector box and absorb harmless post-processing overshoot."""
    x1, y1, x2, y2 = bbox
    left, right = sorted((_unit_interval(x1 / width), _unit_interval(x2 / width)))
    top, bottom = sorted((_unit_interval(y1 / height), _unit_interval(y2 / height)))
    return left, top, right, bottom


def _empty_metadata() -> dict[str, Any]:
    return {}


def _empty_poses() -> dict[int, tuple[PersonPoseObservation, ...]]:
    return {}


def _empty_group_activities() -> dict[int, GroupActivityObservation]:
    return {}


@dataclass(frozen=True, slots=True)
class InferenceResult:
    """All model-owned observations for one canonical clip."""

    players: dict[int, tuple[PlayerObservation, ...]]
    courts: dict[int, CourtFrame]
    balls: dict[int, BallObservation]
    actions: dict[tuple[int, int], ActionObservation]
    frame_count: int
    frame_width: int
    frame_height: int
    fps: float
    poses: dict[int, tuple[PersonPoseObservation, ...]] = field(default_factory=_empty_poses)
    group_activities: dict[int, GroupActivityObservation] = field(
        default_factory=_empty_group_activities
    )
    metadata: dict[str, Any] = field(default_factory=_empty_metadata)


class ObservationProvider(Protocol):
    """Replaceable source of real model observations."""

    def infer(
        self,
        clip_path: Path,
        job: AIJobRequest,
        report: ProgressReporter,
    ) -> InferenceResult:
        """Infer observations from the actual clip bytes."""
        ...


@dataclass(slots=True)
class _Track:
    track_id: int
    bbox: NDArray[np.float32]
    embedding: NDArray[np.float32]
    score: float
    last_frame: int
    velocity: NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class _TrackedDetection:
    track_id: int
    detection_index: int
    bbox: NDArray[np.float32]
    score: float


class HarmonicMeanTracker:
    """Motion-aware appearance/EIoU tracker with separate visible and lost state."""

    def __init__(
        self,
        *,
        max_lost_frames: int = 120,
        max_geometry_lost_frames: int = 120,
        max_prediction_frames: int = 2,
        match_threshold: float = 0.28,
        # volley-reid calibrates raw cosine at 0.9144 with a 0.02 margin. This
        # tracker maps cosine from [-1, 1] to [0, 1], so both values are transformed.
        appearance_recovery_threshold: float = 0.9572026133537292,
        appearance_recovery_margin: float = 0.01,
    ) -> None:
        self.max_lost_frames = max_lost_frames
        self.max_geometry_lost_frames = min(max_lost_frames, max_geometry_lost_frames)
        self.max_prediction_frames = max(0, max_prediction_frames)
        self.match_threshold = match_threshold
        self.appearance_recovery_threshold = appearance_recovery_threshold
        self.appearance_recovery_margin = appearance_recovery_margin
        self._next_id = 1
        self._tracks: dict[int, _Track] = {}
        self._visible_track_ids: set[int] = set()

    def _appearance_recovery_matches(
        self,
        active: list[_Track],
        embeddings: NDArray[np.float32] | None,
        matches: list[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        """Recover only mutual, margin-separated appearance matches after geometry fails."""
        if embeddings is None or not len(embeddings):
            return []
        used_rows = {row for row, _ in matches}
        used_detections = {detection for _, detection in matches}
        rows = [
            row
            for row, track in enumerate(active)
            if row not in used_rows and np.linalg.norm(track.embedding) > 1e-6
        ]
        detections = [
            detection for detection in range(len(embeddings)) if detection not in used_detections
        ]
        if not rows or not detections:
            return []

        similarity = np.stack(
            [_cosine_similarity(active[row].embedding, embeddings[detections]) for row in rows]
        )
        row_best = np.argmax(similarity, axis=1)
        detection_best = np.argmax(similarity, axis=0)
        recovered: list[tuple[int, int]] = []
        for local_row, local_detection in enumerate(row_best):
            if int(detection_best[local_detection]) != local_row:
                continue
            score = float(similarity[local_row, local_detection])
            row_scores = np.sort(similarity[local_row])[::-1]
            detection_scores = np.sort(similarity[:, local_detection])[::-1]
            row_margin = (
                float("inf") if len(row_scores) == 1 else float(row_scores[0] - row_scores[1])
            )
            detection_margin = (
                float("inf")
                if len(detection_scores) == 1
                else float(detection_scores[0] - detection_scores[1])
            )
            if (
                score >= self.appearance_recovery_threshold
                and row_margin >= self.appearance_recovery_margin
                and detection_margin >= self.appearance_recovery_margin
            ):
                recovered.append((rows[local_row], detections[int(local_detection)]))
        return recovered

    def update(
        self,
        frame_index: int,
        boxes: NDArray[np.float32],
        scores: NDArray[np.float32],
        embeddings: NDArray[np.float32] | None,
    ) -> list[_TrackedDetection]:
        """Associate every current person detection without capping raw identities."""
        self._tracks = {
            track_id: track
            for track_id, track in self._tracks.items()
            if frame_index - track.last_frame <= self.max_lost_frames
        }
        active = list(self._tracks.values())
        matches: list[tuple[int, int]] = []
        if active and len(boxes):
            similarity = np.zeros((len(active), len(boxes)), dtype=np.float32)
            for row, track in enumerate(active):
                elapsed = max(0, frame_index - track.last_frame)
                if elapsed > self.max_geometry_lost_frames:
                    continue
                predicted_bbox = track.bbox + track.velocity * elapsed
                geometry = _eiou_similarity(predicted_bbox, boxes)
                if embeddings is None or np.linalg.norm(track.embedding) <= 1e-6:
                    similarity[row] = geometry
                else:
                    appearance = _cosine_similarity(track.embedding, embeddings)
                    denominator = geometry + appearance
                    similarity[row] = np.divide(
                        2.0 * geometry * appearance,
                        denominator,
                        out=np.zeros_like(geometry),
                        where=denominator > 1e-6,
                    )
            assignment = cast(
                "tuple[NDArray[np.int64], NDArray[np.int64]]",
                linear_sum_assignment(1.0 - similarity),
            )
            rows, columns = assignment
            matches = [
                (int(row), int(column))
                for row, column in zip(rows, columns, strict=True)
                if float(similarity[row, column]) >= self.match_threshold
            ]
        matches.extend(self._appearance_recovery_matches(active, embeddings, matches))

        previously_visible = self._visible_track_ids
        used_detections: set[int] = set()
        output: list[_TrackedDetection] = []
        visible_track_ids: set[int] = set()
        for row, detection_index in matches:
            track = active[row]
            elapsed = max(1, frame_index - track.last_frame)
            observed_velocity = (boxes[detection_index] - track.bbox) / elapsed
            track.velocity = 0.65 * track.velocity + 0.35 * observed_velocity
            track.bbox = boxes[detection_index].copy()
            if embeddings is not None:
                track.embedding = (
                    embeddings[detection_index].copy()
                    if np.linalg.norm(track.embedding) <= 1e-6
                    else _normalized_average(
                        track.embedding,
                        embeddings[detection_index],
                    )
                )
            track.score = float(scores[detection_index])
            track.last_frame = frame_index
            used_detections.add(detection_index)
            visible_track_ids.add(track.track_id)
            output.append(
                _TrackedDetection(
                    track.track_id,
                    detection_index,
                    track.bbox.copy(),
                    track.score,
                )
            )

        for detection_index in range(len(boxes)):
            if detection_index in used_detections:
                continue
            track = _Track(
                track_id=self._next_id,
                bbox=boxes[detection_index].copy(),
                embedding=(
                    embeddings[detection_index].copy()
                    if embeddings is not None
                    else np.zeros(512, dtype=np.float32)
                ),
                score=float(scores[detection_index]),
                last_frame=frame_index,
                velocity=np.zeros(4, dtype=np.float32),
            )
            self._tracks[track.track_id] = track
            visible_track_ids.add(track.track_id)
            self._next_id += 1
            output.append(
                _TrackedDetection(
                    track.track_id,
                    detection_index,
                    track.bbox.copy(),
                    track.score,
                )
            )
        for track_id in previously_visible - visible_track_ids:
            track = self._tracks.get(track_id)
            if track is None:
                continue
            elapsed = frame_index - track.last_frame
            if elapsed <= 0 or elapsed > self.max_prediction_frames:
                continue
            visible_track_ids.add(track_id)
            output.append(
                _TrackedDetection(
                    track_id,
                    -1,
                    track.bbox + track.velocity * elapsed,
                    track.score * (0.98**elapsed),
                )
            )
        self._visible_track_ids = visible_track_ids
        return output

    def predict(self, frame_index: int) -> list[_TrackedDetection]:
        """Extrapolate active boxes between detector frames without changing identity state."""
        output: list[_TrackedDetection] = []
        for track_id in self._visible_track_ids:
            track = self._tracks.get(track_id)
            if track is None:
                continue
            elapsed = frame_index - track.last_frame
            if elapsed <= 0 or elapsed > self.max_prediction_frames:
                continue
            output.append(
                _TrackedDetection(
                    track.track_id,
                    -1,
                    track.bbox + track.velocity * elapsed,
                    track.score,
                )
            )
        return output


def _eiou_similarity(
    first: NDArray[np.float32],
    candidates: NDArray[np.float32],
) -> NDArray[np.float32]:
    left = np.maximum(first[0], candidates[:, 0])
    top = np.maximum(first[1], candidates[:, 1])
    right = np.minimum(first[2], candidates[:, 2])
    bottom = np.minimum(first[3], candidates[:, 3])
    intersection = np.maximum(0.0, right - left) * np.maximum(0.0, bottom - top)
    first_area = max(0.0, float(first[2] - first[0])) * max(
        0.0,
        float(first[3] - first[1]),
    )
    candidate_area = np.maximum(0.0, candidates[:, 2] - candidates[:, 0]) * np.maximum(
        0.0,
        candidates[:, 3] - candidates[:, 1],
    )
    union = first_area + candidate_area - intersection
    iou = np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 1e-6,
    )
    first_center = np.asarray(
        [(first[0] + first[2]) / 2.0, (first[1] + first[3]) / 2.0],
        dtype=np.float32,
    )
    candidate_centers = np.column_stack(
        ((candidates[:, 0] + candidates[:, 2]) / 2.0, (candidates[:, 1] + candidates[:, 3]) / 2.0)
    )
    enclosing_left = np.minimum(first[0], candidates[:, 0])
    enclosing_top = np.minimum(first[1], candidates[:, 1])
    enclosing_right = np.maximum(first[2], candidates[:, 2])
    enclosing_bottom = np.maximum(first[3], candidates[:, 3])
    diagonal = np.square(enclosing_right - enclosing_left) + np.square(
        enclosing_bottom - enclosing_top
    )
    center_penalty = np.divide(
        np.square(candidate_centers[:, 0] - first_center[0])
        + np.square(candidate_centers[:, 1] - first_center[1]),
        diagonal,
        out=np.ones_like(diagonal),
        where=diagonal > 1e-6,
    )
    return np.clip(iou - 0.35 * center_penalty, 0.0, 1.0).astype(np.float32)


def _cosine_similarity(
    first: NDArray[np.float32],
    candidates: NDArray[np.float32],
) -> NDArray[np.float32]:
    if not len(candidates):
        return np.empty((0,), dtype=np.float32)
    first_norm = max(float(np.linalg.norm(first)), 1e-6)
    candidate_norms = np.maximum(np.linalg.norm(candidates, axis=1), 1e-6)
    cosine = (candidates @ first) / (candidate_norms * first_norm)
    return np.clip((cosine + 1.0) / 2.0, 0.0, 1.0).astype(np.float32)


def _normalized_average(
    previous: NDArray[np.float32],
    current: NDArray[np.float32],
) -> NDArray[np.float32]:
    combined = 0.85 * previous + 0.15 * current
    norm = max(float(np.linalg.norm(combined)), 1e-6)
    return np.asarray(combined / norm, dtype=np.float32)
