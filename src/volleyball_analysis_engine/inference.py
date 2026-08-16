"""Real clip inference using the supplied RT-DETRv4/X3D and court-pose models."""

from __future__ import annotations

import importlib
import inspect
import logging
import sys
import types
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol, cast

import cv2
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment  # pyright: ignore[reportUnknownVariableType]
from volleyball_monitoring_ai import AIJobRequest

from .ball_tracking import BallTrajectoryTracker
from .court import CourtLineEstimator, interpolate_short_court_gaps
from .nested_reid import NestedPartDescriptorExtractor
from .person_pose import PersonPoseExtractor
from .records import (
    ActionObservation,
    BallObservation,
    CourtFrame,
    PersonPoseObservation,
    PlayerObservation,
    ReIdEmbeddingModel,
    ReIdFeatureSnapshot,
)
from .reid_features import (
    REID_MIN_BBOX_HEIGHT_PX,
    REID_MIN_OBSERVATIONS,
    ReIdFeatureAccumulator,
    sports_osnet_embedding_model,
)

ProgressReporter = Callable[[float, str], None]
LOGGER = logging.getLogger(__name__)
ACTION_NAMES = (
    "waiting",
    "setting",
    "digging",
    "falling",
    "spiking",
    "blocking",
    "jumping",
    "moving",
    "standing",
)


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


def _crop_detection(
    frame: NDArray[np.uint8],
    bbox: NDArray[np.float32],
) -> NDArray[np.uint8] | None:
    """Crop one real detector observation in source pixels for bounded ReID sampling."""
    height, width = frame.shape[:2]
    x1 = max(0, min(width - 1, int(np.floor(float(bbox[0])))))
    y1 = max(0, min(height - 1, int(np.floor(float(bbox[1])))))
    x2 = max(x1 + 1, min(width, int(np.ceil(float(bbox[2])))))
    y2 = max(y1 + 1, min(height, int(np.ceil(float(bbox[3])))))
    crop = frame[y1:y2, x1:x2]
    return np.ascontiguousarray(crop) if crop.size else None


def _empty_metadata() -> dict[str, Any]:
    return {}


def _empty_poses() -> dict[int, tuple[PersonPoseObservation, ...]]:
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
    reid_feature_snapshot: ReIdFeatureSnapshot
    poses: dict[int, tuple[PersonPoseObservation, ...]] = field(default_factory=_empty_poses)
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


@dataclass(frozen=True, slots=True)
class ModelPaths:
    """External model/code assets intentionally kept out of this Git repository."""

    rtv4_root: Path
    rtv4_config: Path
    rtv4_checkpoint: Path
    smp_root: Path
    osnet_checkpoint: Path

    def validate(self) -> None:
        """Fail early with all missing model assets listed together."""
        required = {
            "RT-DETRv4 root": self.rtv4_root / "engine" / "core" / "yaml_config.py",
            "RT-DETRv4 config": self.rtv4_config,
            "RT-DETRv4 checkpoint": self.rtv4_checkpoint,
            "selective-mask-propagation root": self.smp_root / "selective_mask_propagation",
            "OSNet checkpoint": self.osnet_checkpoint,
        }
        missing = [f"{label}: {path}" for label, path in required.items() if not path.exists()]
        if missing:
            raise FileNotFoundError("missing analysis assets:\n" + "\n".join(missing))


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


@dataclass(frozen=True, slots=True)
class _DetectorObservations:
    players: tuple[PlayerObservation, ...]
    ball: BallObservation | None
    actions: tuple[ActionObservation, ...]
    poses: tuple[PersonPoseObservation, ...]
    detector_seconds: float
    postprocess_seconds: float
    embedding_seconds: float
    tracking_seconds: float
    pose_seconds: float


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
        reid_min_observations: int = REID_MIN_OBSERVATIONS,
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
        self._reid_features = ReIdFeatureAccumulator(
            min_observations=reid_min_observations,
        )

    def _observe_reid_frame(
        self,
        *,
        frame_index: int,
        boxes: NDArray[np.float32],
        scores: NDArray[np.float32],
        embeddings: NDArray[np.float32] | None,
        detection_track_ids: dict[int, int],
        frame: NDArray[np.uint8] | None = None,
    ) -> None:
        """Aggregate sufficiently large real detections, never extrapolated boxes."""
        eligible = {
            detection_index: track_id
            for detection_index, track_id in detection_track_ids.items()
            if float(boxes[detection_index][3] - boxes[detection_index][1])
            >= REID_MIN_BBOX_HEIGHT_PX
        }
        self._reid_features.observe_co_visibility(eligible.values())
        if embeddings is None:
            return
        for detection_index, track_id in sorted(eligible.items()):
            bbox_height = float(boxes[detection_index][3] - boxes[detection_index][1])
            score = float(scores[detection_index])
            crop = _crop_detection(frame, boxes[detection_index]) if frame is not None else None
            self._reid_features.observe(
                track_id=track_id,
                frame_index=frame_index,
                embedding=embeddings[detection_index],
                quality=score,
                selection_quality=score * bbox_height,
                crop_bgr=crop,
            )

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
        frame: NDArray[np.uint8] | None = None,
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
        detection_track_ids: dict[int, int] = {}
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
            detection_track_ids[detection_index] = track.track_id
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
            detection_track_ids[detection_index] = track.track_id
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
        self._observe_reid_frame(
            frame_index=frame_index,
            boxes=boxes,
            scores=scores,
            embeddings=embeddings,
            detection_track_ids=detection_track_ids,
            frame=frame,
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

    def reid_feature_snapshot(
        self,
        embedding_model: ReIdEmbeddingModel,
    ) -> ReIdFeatureSnapshot:
        """Return the versioned compact feature state for this tracker run."""
        return ReIdFeatureSnapshot(
            schema_version="1.0.0",
            embedding_model=embedding_model,
            features=self._reid_features.snapshot(),
        )

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


class Rtv4X3DObservationProvider:
    """Headless model pipeline ported from the supplied senior inference program."""

    def __init__(
        self,
        paths: ModelPaths,
        *,
        device: str = "cuda:0",
        backend: str = "continual",
        detector_threshold: float = 0.4,
        detector_input_scale: float = 1.0,
        reid_every: int = 1,
        court_model: str | Path | None = "v3",
        court_imgsz: int = 512,
        court_batch_size: int = 16,
        court_layout_every: int = 1,
        court_refresh_every: int = 120,
        court_track_every: int = 1,
        court_max_hold_frames: int = 180,
        court_decoder: str = "auto",
        disable_amp: bool = False,
        nested_reid: NestedPartDescriptorExtractor | None = None,
        person_pose: PersonPoseExtractor | None = None,
    ) -> None:
        paths.validate()
        self.paths = paths
        self.device_name = device
        self.backend = backend
        self.detector_threshold = detector_threshold
        self.detector_input_scale = min(1.0, max(0.5, detector_input_scale))
        self.reid_every = max(1, reid_every)
        self.court_model = court_model
        self.court_imgsz = court_imgsz
        self.court_batch_size = max(1, court_batch_size)
        self.court_layout_every = max(1, court_layout_every)
        self.court_refresh_every = max(self.court_layout_every, court_refresh_every)
        self.court_track_every = max(1, court_track_every)
        self.court_max_hold_frames = max(0, court_max_hold_frames)
        self.court_decoder = court_decoder
        self.disable_amp = disable_amp
        self.nested_reid = nested_reid
        self.person_pose = person_pose
        self._torch: Any = None
        self._model: Any = None
        self._postprocessor: Any = None
        self._streamer: Any = None
        self._input_size = (576, 1024)
        self._use_amp = False
        self._osnet: Any = None
        self._roi_align: Any = None
        self._detector_frame_tensor: Any = None
        self._osnet_mean: Any = None
        self._osnet_std: Any = None
        self._reid_embedding_model: ReIdEmbeddingModel | None = None
        self._court_estimator: CourtLineEstimator | None = None
        self._effective_backend = backend

    @property
    def effective_backend(self) -> str:
        """Return the actually loaded X3D backend after compatibility fallback."""
        return self._effective_backend

    def _embedding_model_metadata(self) -> ReIdEmbeddingModel:
        if self._reid_embedding_model is None:
            self._reid_embedding_model = sports_osnet_embedding_model(self.paths.osnet_checkpoint)
        return self._reid_embedding_model

    def infer(
        self,
        clip_path: Path,
        job: AIJobRequest,
        report: ProgressReporter,
    ) -> InferenceResult:
        """Decode every canonical frame and run actual models against it."""
        load_started = perf_counter()
        self._load_models(report)
        load_seconds = perf_counter() - load_started
        if self._court_estimator is None:
            raise RuntimeError("court-line estimator was not initialized")
        # Every clip is an independent temporal sequence. A persistent worker or batch
        # offline runner reuses model weights, but must never reuse X3D frame history.
        self._streamer.clean_state()
        self._detector_frame_tensor = None
        capture = cv2.VideoCapture(str(clip_path))
        if not capture.isOpened():
            raise ValueError(f"cannot open canonical clip: {clip_path}")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        expected_frames = int(job.clip.video.total_frames)
        tracker = HarmonicMeanTracker(
            max_lost_frames=max(60, round(fps * 12.0)),
            max_geometry_lost_frames=max(60, round(fps * 2.0)),
        )
        ball_tracker = BallTrajectoryTracker()
        players: dict[int, tuple[PlayerObservation, ...]] = {}
        courts: dict[int, CourtFrame] = {}
        balls: dict[int, BallObservation] = {}
        actions: dict[tuple[int, int], ActionObservation] = {}
        poses: dict[int, tuple[PersonPoseObservation, ...]] = {}
        court_processor = self._court_estimator.begin_video()
        timings = {
            "decode_seconds": 0.0,
            "detector_frames": 0,
            "detector_seconds": 0.0,
            "postprocess_seconds": 0.0,
            "embedding_seconds": 0.0,
            "tracking_seconds": 0.0,
            "pose_seconds": 0.0,
        }
        inference_started = perf_counter()
        frame_index = 0
        try:
            while True:
                started = perf_counter()
                ok, frame = capture.read()
                timings["decode_seconds"] += perf_counter() - started
                if not ok:
                    break
                typed_frame = np.asarray(frame, dtype=np.uint8)
                observed = self._infer_observations(
                    typed_frame,
                    frame_index=frame_index,
                    width=width,
                    height=height,
                    tracker=tracker,
                    ball_tracker=ball_tracker,
                )
                timings["detector_frames"] += 1
                timings["detector_seconds"] += observed.detector_seconds
                timings["postprocess_seconds"] += observed.postprocess_seconds
                timings["embedding_seconds"] += observed.embedding_seconds
                timings["tracking_seconds"] += observed.tracking_seconds
                timings["pose_seconds"] += observed.pose_seconds
                players[frame_index] = observed.players
                poses[frame_index] = observed.poses
                if observed.ball is not None:
                    balls[frame_index] = observed.ball
                for action in observed.actions:
                    actions[(frame_index, action.track_id)] = action

                courts.update(court_processor.submit(frame_index, typed_frame))
                frame_index += 1
                if frame_index % max(1, round(fps)) == 0:
                    report(
                        0.08 + 0.62 * min(1.0, frame_index / max(expected_frames, 1)),
                        "rtv4_x3d_tracking",
                    )
        finally:
            capture.release()
        courts.update(court_processor.finish())
        court_processor.timing.interpolated_frames = interpolate_short_court_gaps(courts)
        inference_seconds = perf_counter() - inference_started
        if frame_index != expected_frames:
            raise ValueError(
                f"decoded frame count mismatch: job={expected_frames}, decoded={frame_index}"
            )
        reid_snapshot = tracker.reid_feature_snapshot(self._embedding_model_metadata())
        if self.nested_reid is not None:
            report(0.70, "nested_part_descriptors")
            reid_snapshot = self.nested_reid.enrich(reid_snapshot)
        return InferenceResult(
            players=players,
            courts=courts,
            balls=balls,
            actions=actions,
            frame_count=frame_index,
            frame_width=width,
            frame_height=height,
            fps=fps,
            reid_feature_snapshot=reid_snapshot,
            poses=poses,
            metadata={
                "detector": "RT-DETRv4+X3D",
                "detector_config": str(self.paths.rtv4_config),
                "detector_stride": 1,
                "detector_input_size": list(self._input_size),
                "detector_input_scale": self.detector_input_scale,
                "reid_every": self.reid_every,
                "tracker": "harmonic-mean-eiou+OSNet-run-local",
                "person_pose_recipe": (
                    self.person_pose.recipe_metadata if self.person_pose is not None else None
                ),
                "court_detector": "court-line-yolo26n-layout-v3+pose36-layout-tracker",
                "court_model": self._court_estimator.model_name,
                "streaming_backend": self._effective_backend,
                "source": "canonical_clip_inference",
                "timing": {
                    "model_load_seconds": load_seconds,
                    "clip_inference_seconds": inference_seconds,
                    "source_fps": fps,
                    "effective_fps": frame_index / max(inference_seconds, 1e-9),
                    **timings,
                    "court": court_processor.timing.to_mapping(),
                },
            },
        )

    def prepare(self, report: ProgressReporter | None = None) -> None:
        """Load and strictly validate all model checkpoints without decoding a clip."""
        self._load_models(report or (lambda _progress, _stage: None))
        if self.nested_reid is not None:
            self.nested_reid.prepare()
        if self.person_pose is not None:
            self.person_pose.prepare()

    def _load_models(self, report: ProgressReporter) -> None:
        if self._model is not None:
            return
        report(0.01, "loading_rtv4_x3d")
        root = str(self.paths.rtv4_root.resolve())
        if root not in sys.path:
            sys.path.insert(0, root)
        torch = importlib.import_module("torch")
        yaml_module = importlib.import_module("engine.core")
        streaming_module = importlib.import_module("engine.backbone.x3d_streaming")
        yaml_config = yaml_module.YAMLConfig(str(self.paths.rtv4_config))
        load_kwargs: dict[str, Any] = {"map_location": "cpu"}
        if "weights_only" in inspect.signature(torch.load).parameters:
            load_kwargs["weights_only"] = False
        checkpoint: Any = torch.load(str(self.paths.rtv4_checkpoint), **load_kwargs)
        state = self._checkpoint_state(checkpoint)
        yaml_config.model.load_state_dict(state, strict=True)
        device = torch.device(self.device_name)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
        model = yaml_config.model.to(device).eval()
        postprocessor = yaml_config.postprocessor.to(device).eval()
        stream_config = yaml_config.yaml_cfg.get("streaming", {})
        frame_num = int(stream_config.get("frame_num", 5))
        configured_input_size = tuple(
            int(value)
            for value in stream_config.get(
                "input_size",
                yaml_config.yaml_cfg.get("eval_spatial_size", [576, 1024]),
            )
        )
        self._input_size = tuple(
            max(160, round(value * self.detector_input_scale / 32.0) * 32)
            for value in configured_input_size
        )
        self._use_amp = (
            bool(stream_config.get("amp", True)) and device.type == "cuda" and not self.disable_amp
        )
        streamer = self._build_streamer(
            streaming_module,
            model.backbone,
            frame_num=frame_num,
        )
        if hasattr(streamer, "to"):
            streamer.to(device)
        streamer.clean_state()
        self._torch = torch
        self._device = device
        self._model = model
        self._postprocessor = postprocessor
        self._streamer = streamer

        report(0.035, "loading_osnet")
        smp_root = str(self.paths.smp_root.resolve())
        if smp_root not in sys.path:
            sys.path.insert(0, smp_root)
        self._install_cython_bbox_fallback()
        osnet_module = importlib.import_module("selective_mask_propagation.osnet.inference")
        self._osnet = osnet_module.build_model(
            device,
            checkpoint=str(self.paths.osnet_checkpoint),
        )
        self._embedding_model_metadata()
        self._roi_align = importlib.import_module("torchvision.ops").roi_align
        self._osnet_mean = torch.tensor(
            [0.485, 0.456, 0.406], device=device, dtype=torch.float32
        ).view(1, 3, 1, 1)
        self._osnet_std = torch.tensor(
            [0.229, 0.224, 0.225], device=device, dtype=torch.float32
        ).view(1, 3, 1, 1)

        report(0.055, "loading_court_lines")
        self._court_estimator = CourtLineEstimator(
            self.court_model,
            device=self.device_name,
            image_size=self.court_imgsz,
            decoder=self.court_decoder,
            batch_size=self.court_batch_size,
            layout_every=self.court_layout_every,
            refresh_every=self.court_refresh_every,
            track_every=self.court_track_every,
            max_hold_frames=self.court_max_hold_frames,
        )
        self._court_estimator.warmup()
        report(0.07, "warming_detector_and_reid")
        self._warmup_detector(frame_num=frame_num)

    def _warmup_detector(self, *, frame_num: int) -> None:
        """Compile detector, ROIAlign and OSNet kernels outside job timing."""
        dummy = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = None
        for _ in range(max(1, frame_num)):
            result = self._infer_detector(dummy, 1920, 1080)
        self._to_numpy(result)
        self._embeddings(
            dummy,
            np.asarray(
                [[160.0, 160.0, 360.0, 720.0], [960.0, 180.0, 1180.0, 760.0]],
                dtype=np.float32,
            ),
        )
        if self._device.type == "cuda":
            self._torch.cuda.synchronize(self._device)
        self._streamer.clean_state()
        self._detector_frame_tensor = None

    @staticmethod
    def _checkpoint_state(checkpoint: Any) -> Any:
        if not isinstance(checkpoint, dict):
            return checkpoint
        checkpoint_dict = cast("dict[str, Any]", checkpoint)
        if "ema" in checkpoint_dict:
            ema: Any = checkpoint_dict["ema"]
            if isinstance(ema, dict):
                ema_dict = cast("dict[str, Any]", ema)
                return ema_dict.get("module", ema_dict)
            return ema
        return checkpoint_dict.get("model", checkpoint_dict)

    def _build_streamer(
        self,
        streaming_module: Any,
        backbone: Any,
        *,
        frame_num: int,
    ) -> Any:
        try:
            streamer = streaming_module.build_x3d_streamer(
                backbone,
                backend=self.backend,
                frame_num=frame_num,
            )
            self._effective_backend = self.backend
            return streamer
        except (ImportError, RuntimeError) as exc:
            if self.backend != "continual":
                raise
            LOGGER.warning(
                "continual X3D conversion is incompatible; using exact rolling backend: %s",
                exc,
            )
            self._effective_backend = "rolling"
            return streaming_module.build_x3d_streamer(
                backbone,
                backend="rolling",
                frame_num=frame_num,
            )

    def _infer_observations(
        self,
        frame: NDArray[np.uint8],
        *,
        frame_index: int,
        width: int,
        height: int,
        tracker: HarmonicMeanTracker,
        ball_tracker: BallTrajectoryTracker,
    ) -> _DetectorObservations:
        """Run the detector on one canonical source frame and normalize observations."""
        started = perf_counter()
        result = self._infer_detector(frame, width, height)
        detector_seconds = perf_counter() - started
        started = perf_counter()
        labels, boxes, scores, action_labels, action_scores = self._to_numpy(result)
        postprocess_seconds = perf_counter() - started
        person_indices = np.flatnonzero((labels == 0) & (scores >= self.detector_threshold))
        person_boxes = boxes[person_indices].astype(np.float32, copy=False)
        person_scores = scores[person_indices].astype(np.float32, copy=False)
        should_embed = frame_index % self.reid_every == 0
        if should_embed:
            started = perf_counter()
            embeddings = self._embeddings(frame, person_boxes)
            embedding_seconds = perf_counter() - started
        else:
            embeddings = None
            embedding_seconds = 0.0
        started = perf_counter()
        tracked = tracker.update(
            frame_index,
            person_boxes,
            person_scores,
            embeddings,
            frame,
        )
        tracking_seconds = perf_counter() - started

        started = perf_counter()
        if self.person_pose is not None:
            poses = self.person_pose.infer_frame(
                frame,
                frame_index=frame_index,
                tracks=[(item.track_id, item.bbox, item.detection_index >= 0) for item in tracked],
            )
        else:
            poses = tuple(
                PersonPoseObservation(
                    frame_index=frame_index,
                    track_id=item.track_id,
                    bbox_source=("DETECTOR" if item.detection_index >= 0 else "TRACKER_PROPAGATED"),
                    frame_bbox=normalize_frame_bbox(
                        (
                            float(item.bbox[0]),
                            float(item.bbox[1]),
                            float(item.bbox[2]),
                            float(item.bbox[3]),
                        ),
                        width=width,
                        height=height,
                    ),
                    crop_transform=(
                        1.0 / width,
                        1.0 / height,
                        max(0.0, float(item.bbox[0]) / width),
                        max(0.0, float(item.bbox[1]) / height),
                    ),
                    status="INFERENCE_FAILED",
                    keypoints=None,
                )
                for item in tracked
            )
        pose_seconds = perf_counter() - started

        players: list[PlayerObservation] = []
        actions: list[ActionObservation] = []
        for item in tracked:
            x1, y1, x2, y2 = (float(value) for value in item.bbox)
            players.append(
                PlayerObservation(
                    frame_index=frame_index,
                    source_track_id=item.track_id,
                    track_id=item.track_id,
                    frame_bbox=normalize_frame_bbox(
                        (x1, y1, x2, y2),
                        width=width,
                        height=height,
                    ),
                    frame_foot_pos=(
                        _unit_interval((x1 + x2) / (2.0 * width)),
                        _unit_interval(y2 / height),
                    ),
                    court_pos=None,
                    confidence=item.score,
                )
            )
            if item.detection_index < 0:
                continue
            model_index = int(person_indices[item.detection_index])
            if action_labels is None or model_index >= len(action_labels):
                continue
            action_index = int(action_labels[model_index])
            if not 0 <= action_index < len(ACTION_NAMES):
                continue
            confidence = (
                float(action_scores[model_index])
                if action_scores is not None and model_index < len(action_scores)
                else None
            )
            actions.append(
                ActionObservation(
                    frame_index,
                    item.track_id,
                    ACTION_NAMES[action_index],
                    confidence,
                )
            )

        ball_indices = np.flatnonzero((labels == 1) & (scores >= self.detector_threshold))
        ball_candidates = [
            (
                _unit_interval(float((boxes[index][0] + boxes[index][2]) / (2.0 * width))),
                _unit_interval(float((boxes[index][1] + boxes[index][3]) / (2.0 * height))),
            )
            for index in ball_indices
        ]
        ball = ball_tracker.update(
            frame_index,
            ball_candidates,
            [float(scores[index]) for index in ball_indices],
        )
        return _DetectorObservations(
            players=tuple(players),
            ball=ball,
            actions=tuple(actions),
            poses=poses,
            detector_seconds=detector_seconds,
            postprocess_seconds=postprocess_seconds,
            embedding_seconds=embedding_seconds,
            tracking_seconds=tracking_seconds,
            pose_seconds=pose_seconds,
        )

    def _infer_detector(self, frame: NDArray[np.uint8], width: int, height: int) -> Any:
        torch = self._torch
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_height, input_width = self._input_size
        if rgb.shape[:2] != self._input_size:
            rgb = cv2.resize(rgb, (input_width, input_height), interpolation=cv2.INTER_LINEAR)
        dtype = torch.float16 if self._use_amp else torch.float32
        tensor = (
            torch.from_numpy(rgb)
            .permute(2, 0, 1)
            .contiguous()
            .unsqueeze(0)
            .to(device=self._device, dtype=dtype, non_blocking=True)
            .div_(255.0)
        )
        self._detector_frame_tensor = tensor
        with (
            torch.inference_mode(),
            torch.autocast(
                device_type=self._device.type,
                dtype=torch.float16,
                enabled=self._use_amp,
            ),
        ):
            features = self._streamer.forward_step(tensor)
            if features is None:
                return None
            encoded: Any = self._model.encoder(features)
            if isinstance(encoded, tuple):
                encoded = cast("Any", encoded[0])
            output = self._model.decoder(encoded, targets=None)
            original_size = torch.tensor(
                [[width, height]],
                dtype=torch.float32,
                device=self._device,
            )
            return self._postprocessor(output, original_size)[0]

    @staticmethod
    def _to_numpy(
        result: Any,
    ) -> tuple[
        NDArray[np.int32],
        NDArray[np.float32],
        NDArray[np.float32],
        NDArray[np.int32] | None,
        NDArray[np.float32] | None,
    ]:
        if result is None:
            return (
                np.empty((0,), dtype=np.int32),
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                None,
                None,
            )
        action_labels = result.get("action_labels")
        action_scores = result.get("action_scores")
        return (
            result["labels"].detach().cpu().numpy().astype(np.int32),
            result["boxes"].detach().cpu().numpy().astype(np.float32),
            result["scores"].detach().cpu().numpy().astype(np.float32),
            (
                action_labels.detach().cpu().numpy().astype(np.int32)
                if action_labels is not None
                else None
            ),
            (
                action_scores.detach().cpu().numpy().astype(np.float32)
                if action_scores is not None
                else None
            ),
        )

    def _embeddings(
        self,
        frame: NDArray[np.uint8],
        boxes: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        if not len(boxes):
            return np.empty((0, 512), dtype=np.float32)
        height = int(frame.shape[0])
        width = int(frame.shape[1])
        input_height, input_width = self._input_size
        scaled = boxes.astype(np.float32, copy=True)
        scaled[:, (0, 2)] = np.clip(scaled[:, (0, 2)], 0.0, float(width)) * (input_width / width)
        scaled[:, (1, 3)] = np.clip(scaled[:, (1, 3)], 0.0, float(height)) * (input_height / height)
        rois = np.concatenate(
            (np.zeros((len(scaled), 1), dtype=np.float32), scaled),
            axis=1,
        )
        roi_tensor = self._torch.from_numpy(rois).to(self._device, non_blocking=True)
        with (
            self._torch.inference_mode(),
            self._torch.autocast(
                device_type=self._device.type,
                dtype=self._torch.float16,
                enabled=self._use_amp,
            ),
        ):
            crops = self._roi_align(
                self._detector_frame_tensor,
                roi_tensor,
                output_size=(256, 128),
                spatial_scale=1.0,
                sampling_ratio=2,
                aligned=True,
            ).float()
            batch = (crops - self._osnet_mean) / self._osnet_std
            features = self._osnet(batch)
        array = np.asarray(features.detach().cpu().numpy(), dtype=np.float32)
        norms = np.maximum(np.linalg.norm(array, axis=1, keepdims=True), 1e-6)
        return np.asarray(array / norms, dtype=np.float32)

    @staticmethod
    def _install_cython_bbox_fallback() -> None:
        try:
            importlib.import_module("cython_bbox")
            return
        except ModuleNotFoundError:
            pass
        module = types.ModuleType("cython_bbox")

        def bbox_overlaps(
            boxes: NDArray[np.float64],
            query_boxes: NDArray[np.float64],
        ) -> NDArray[np.float64]:
            output = np.zeros((len(boxes), len(query_boxes)), dtype=np.float64)
            for index, box in enumerate(boxes):
                left = np.maximum(box[0], query_boxes[:, 0])
                top = np.maximum(box[1], query_boxes[:, 1])
                right = np.minimum(box[2], query_boxes[:, 2])
                bottom = np.minimum(box[3], query_boxes[:, 3])
                intersection = np.maximum(0.0, right - left) * np.maximum(0.0, bottom - top)
                box_area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
                query_area = np.maximum(0.0, query_boxes[:, 2] - query_boxes[:, 0]) * np.maximum(
                    0.0,
                    query_boxes[:, 3] - query_boxes[:, 1],
                )
                union = box_area + query_area - intersection
                output[index] = np.divide(
                    intersection,
                    union,
                    out=np.zeros_like(intersection),
                    where=union > 0,
                )
            return output

        module.bbox_overlaps = bbox_overlaps  # type: ignore[attr-defined]
        sys.modules["cython_bbox"] = module
