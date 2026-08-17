"""Every-frame adapter for the Volleyball Court60 multitask inference SDK."""

from __future__ import annotations

import hashlib
import importlib
import sys
from collections.abc import Iterable, Sequence
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from volleyball_monitoring_ai import AIJobRequest

from .ball_tracking import BallTrajectoryTracker
from .inference import (
    HarmonicMeanTracker,
    InferenceResult,
    ProgressReporter,
    normalize_frame_bbox,
)
from .local_tracking import DeepEiouTracker, SelectiveSam3Augmenter, SelectiveSam3Result
from .records import (
    ActionObservation,
    BallObservation,
    CourtFrame,
    CourtKeypoint,
    GroupActivityObservation,
    PersonPoseObservation,
    PersonPoseStatus,
    PlayerObservation,
)
from .reid_feature_job import SportsOsnetCropEncoder

COURT_BASE_WORLD_XY = (
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
COURT_DENSE_EDGE_BASE_PAIRS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (8, 9),
    (9, 0),
)
COCO17_KEYPOINT_COUNT = 17
COURT60_KEYPOINT_COUNT = 60


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _numpy(value: Any, dtype: Any) -> NDArray[Any]:
    if value is None:
        return np.empty((0,), dtype=dtype)
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _scalar(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if hasattr(value, "detach"):
        value = value.detach().cpu().item()
    return float(value)


def _integer(value: Any, default: int = -1) -> int:
    return int(_scalar(value, float(default)))


def _court_world_points() -> tuple[tuple[float, float], ...]:
    points = list(COURT_BASE_WORLD_XY)
    for first, second in COURT_DENSE_EDGE_BASE_PAIRS:
        start = np.asarray(COURT_BASE_WORLD_XY[first], dtype=np.float64)
        end = np.asarray(COURT_BASE_WORLD_XY[second], dtype=np.float64)
        for fraction in range(1, 6):
            point = start + (end - start) * (fraction / 6.0)
            points.append((float(point[0]), float(point[1])))
    return tuple(points)


COURT60_WORLD_XY = _court_world_points()


class VolleyballMultitaskObservationProvider:
    """Use one temporal model pass for detection, action, court, pose and group state."""

    def __init__(
        self,
        *,
        sdk_root: Path,
        checkpoint: Path,
        smp_root: Path,
        osnet_checkpoint: Path,
        device: str,
        config: Path | None = None,
        detector_threshold: float = 0.4,
        batch_size: int = 4,
        reid_every: int = 1,
        fp16: bool = True,
        warmup: bool = True,
        local_tracker: str = "deep_eiou",
        local_sam3_enabled: bool = True,
        local_sam3_python: Path = Path(
            "../volley-ai/upstream/selective-mask-propagation/.venv/Scripts/python.exe"
        ),
        local_sam3_bridge: Path = Path("scripts/run_selective_sam3.py"),
        local_sam3_timeout_seconds: int = 1800,
        pose_keypoint_confidence: float = 0.3,
        pose_minimum_keypoints: int = 4,
    ) -> None:
        """Configure model assets, temporal batching and run-local tracking."""
        self.sdk_root = sdk_root
        self.checkpoint = checkpoint
        self.config = config
        self.device_name = device
        self.detector_threshold = detector_threshold
        self.batch_size = max(1, batch_size)
        self.reid_every = max(1, reid_every)
        self.fp16 = fp16
        self.warmup = warmup
        if local_tracker not in {"deep_eiou", "harmonic"}:
            raise ValueError(f"unsupported local tracker: {local_tracker}")
        self.local_tracker = local_tracker
        self.smp_root = smp_root
        self.pose_keypoint_confidence = pose_keypoint_confidence
        self.pose_minimum_keypoints = pose_minimum_keypoints
        self._predictor: Any = None
        self._schema_names: tuple[str, ...] = ()
        self._action_names: tuple[str, ...] = ()
        self._group_names: tuple[str, ...] = ()
        self._sdk_version = "unknown"
        self._osnet = SportsOsnetCropEncoder(
            smp_root=smp_root,
            checkpoint=osnet_checkpoint,
            device=device,
        )
        self._sam3 = SelectiveSam3Augmenter(
            enabled=local_sam3_enabled and local_tracker == "deep_eiou",
            python=local_sam3_python,
            bridge=local_sam3_bridge,
            smp_root=smp_root,
            timeout_seconds=local_sam3_timeout_seconds,
        )

    @property
    def effective_backend(self) -> str:
        """Describe the temporal inference backend for doctor output."""
        return f"volleyball-inference-sdk-centered-batch+{self.local_tracker}"

    def validate_assets(self) -> None:
        """Fail before worker registration when model code or weights are absent."""
        if not (self.sdk_root / "volleyball_sdk" / "__init__.py").is_file():
            raise FileNotFoundError(f"missing volleyball inference SDK: {self.sdk_root}")
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"missing volleyball multitask checkpoint: {self.checkpoint}")
        if self.config is not None and not self.config.is_file():
            raise FileNotFoundError(f"missing volleyball multitask config: {self.config}")
        if self.local_tracker == "deep_eiou":
            tracker_source = (
                self.smp_root / "selective_mask_propagation" / "deep_eiou" / "tracker.py"
            )
            if not tracker_source.is_file():
                raise FileNotFoundError(f"missing DeepEIOU tracker source: {tracker_source}")

    def prepare(self, report: ProgressReporter | None = None) -> None:
        """Load and warm the multitask model and run-local tracking encoder once."""
        if self._predictor is not None:
            return

        def noop_report(_progress: float, _stage: str) -> None:
            return

        reporter: ProgressReporter = report or noop_report
        self.validate_assets()
        reporter(0.01, "loading_volleyball_multitask_sdk")
        root = str(self.sdk_root.resolve())
        if root not in sys.path:
            sys.path.insert(0, root)
        sdk = importlib.import_module("volleyball_sdk")
        self._schema_names = tuple(cast("Sequence[str]", sdk.SCHEMA_NAMES))
        self._action_names = tuple(cast("Sequence[str]", sdk.ACTION_NAMES))
        self._group_names = tuple(cast("Sequence[str]", sdk.GROUP_ACTIVITY_NAMES))
        self._sdk_version = str(sdk.SCHEMA_VERSION)
        if self._sdk_version != "2.0":
            raise RuntimeError(f"unsupported volleyball inference schema: {self._sdk_version}")
        predictor_type = sdk.Volleyball
        self._predictor = predictor_type(
            self.checkpoint,
            config=self.config,
            device=self.device_name,
            fp16=self.fp16,
            warmup=self.warmup,
        )
        reporter(0.055, "loading_tracking_osnet")
        self._osnet.prepare()
        reporter(0.07, "volleyball_multitask_ready")

    def infer(
        self,
        clip_path: Path,
        job: AIJobRequest,
        report: ProgressReporter,
    ) -> InferenceResult:
        """Infer all model-owned evidence once for every canonical source frame."""
        load_started = perf_counter()
        self.prepare(report)
        load_seconds = perf_counter() - load_started
        predictor = self._predictor
        video_module = importlib.import_module("volleyball_sdk.video")
        packets = video_module.iter_video_clips(
            clip_path,
            frame_num=int(predictor.frame_num),
            jump_frame=int(predictor.jump_frame),
            sampling_mode=str(predictor.sampling_mode),
            step=1,
        )
        expected_frames = int(job.clip.video.total_frames)
        fps = float(job.clip.video.fps.num) / float(job.clip.video.fps.den)
        tracker: Any = None
        ball_tracker = BallTrajectoryTracker()
        players: dict[int, tuple[PlayerObservation, ...]] = {}
        courts: dict[int, CourtFrame] = {}
        balls: dict[int, BallObservation] = {}
        actions: dict[tuple[int, int], ActionObservation] = {}
        poses: dict[int, tuple[PersonPoseObservation, ...]] = {}
        groups: dict[int, GroupActivityObservation] = {}
        timing = {"model_seconds": 0.0, "tracking_seconds": 0.0, "frames": 0}
        width = 0
        height = 0
        inference_started = perf_counter()
        for batch in self._batches(packets, self.batch_size):
            clips = [packet[2] for packet in batch]
            sizes = [packet[3] for packet in batch]
            started = perf_counter()
            raw_results, model_timing = predictor.predict_batch_raw(clips, image_sizes=sizes)
            timing["model_seconds"] += perf_counter() - started
            for packet, raw in zip(batch, raw_results, strict=True):
                frame_index, _timestamp, clip, (width, height) = packet
                output_slot = (
                    int(predictor.frame_num) // 2
                    if str(predictor.sampling_mode) == "centered"
                    else int(predictor.frame_num) - 1
                )
                frame = np.asarray(clip[output_slot], dtype=np.uint8)
                if tracker is None:
                    tracker = self._build_tracker(fps=fps, width=width, height=height)
                started = perf_counter()
                frame_players, frame_poses, frame_actions = self._people(
                    raw,
                    frame=frame,
                    frame_index=frame_index,
                    width=width,
                    height=height,
                    tracker=tracker,
                )
                timing["tracking_seconds"] += perf_counter() - started
                players[frame_index] = frame_players
                poses[frame_index] = frame_poses
                actions.update({(item.frame_index, item.track_id): item for item in frame_actions})
                court = self._court(raw, frame_index)
                if court is not None:
                    courts[frame_index] = court
                ball = self._ball(raw, frame_index, width, height, ball_tracker)
                if ball is not None:
                    balls[frame_index] = ball
                group = self._group(raw, frame_index)
                if group is not None:
                    groups[frame_index] = group
                timing["frames"] += 1
                if timing["frames"] % max(1, round(fps)) == 0:
                    report(
                        0.08 + 0.62 * min(1.0, timing["frames"] / max(expected_frames, 1)),
                        "volleyball_multitask_tracking",
                    )
            del model_timing
        frame_count = int(timing["frames"])
        if frame_count != expected_frames:
            raise ValueError(
                f"decoded frame count mismatch: job={expected_frames}, decoded={frame_count}"
            )
        sam3_started = perf_counter()
        sam3_result = SelectiveSam3Result("not_applicable", (), 0, 0)
        if isinstance(tracker, DeepEiouTracker):
            report(0.705, "selective_sam3_local_reid")
            sam3_result = self._sam3.augment(
                clip_path=clip_path,
                tracks=tracker.tracks,
                margins=tracker.margins,
            )
            if sam3_result.rename_events:
                try:
                    players, poses, actions = self._remap_local_identities(
                        players,
                        poses,
                        actions,
                        sam3_result,
                    )
                except RuntimeError as exc:
                    sam3_result = SelectiveSam3Result(
                        "invalid_output_fallback_deep_eiou",
                        (),
                        sam3_result.window_count,
                        sam3_result.swap_count,
                        str(exc),
                    )
        timing["sam3_seconds"] = perf_counter() - sam3_started
        return InferenceResult(
            players=players,
            courts=courts,
            balls=balls,
            actions=actions,
            poses=poses,
            group_activities=groups,
            frame_count=frame_count,
            frame_width=width,
            frame_height=height,
            fps=fps,
            metadata={
                "detector": "volleyball-multitask-v2",
                "court_detector": "volleyball-multitask-v2-court60",
                "action_source": "volleyball-multitask-v2",
                "group_activity_source": "volleyball-multitask-v2",
                "tracker": (
                    "deep-eiou+OSNet-run-local"
                    if isinstance(tracker, DeepEiouTracker)
                    else "harmonic-mean-eiou+OSNet-run-local"
                ),
                "selective_sam3": {
                    "status": sam3_result.status,
                    "window_count": sam3_result.window_count,
                    "swap_count": sam3_result.swap_count,
                    "rename_event_count": len(sam3_result.rename_events),
                    "fallback_reason": sam3_result.stderr_tail,
                },
                "sdk_schema_version": self._sdk_version,
                "checkpoint_sha256": _sha256(self.checkpoint),
                "detector_stride": 1,
                "temporal_frame_count": int(predictor.frame_num),
                "temporal_jump_frame": int(predictor.jump_frame),
                "temporal_sampling_mode": str(predictor.sampling_mode),
                "person_pose_recipe": {
                    "namespace": f"volleyball-multitask-v2:{_sha256(self.checkpoint)}:coco17",
                    "model_name": self.checkpoint.stem,
                    "checkpoint_sha256": _sha256(self.checkpoint),
                    "preprocess_version": "sdk-centered-temporal-v2",
                    "keypoint_layout": "COCO_17",
                    "coordinate_space": "NORMALIZED_VIDEO",
                },
                "group_activity_taxonomy": {
                    "namespace": "volleyball-inference-sdk/group-activity",
                    "version": self._sdk_version,
                    "labels": list(self._group_names),
                    "consumer_status": "stored_not_interpreted",
                },
                "source": "canonical_clip_inference",
                "timing": {
                    **timing,
                    "model_load_seconds": load_seconds,
                    "clip_inference_seconds": perf_counter() - inference_started,
                    "source_fps": fps,
                },
            },
        )

    @staticmethod
    def _batches(source: Iterable[Any], size: int) -> Iterable[list[Any]]:
        batch: list[Any] = []
        for item in source:
            batch.append(item)
            if len(batch) == size:
                yield batch
                batch = []
        if batch:
            yield batch

    def _build_tracker(self, *, fps: float, width: int, height: int) -> Any:
        if self.local_tracker == "deep_eiou":
            return DeepEiouTracker(
                smp_root=self.smp_root,
                fps=fps,
                frame_width=width,
                frame_height=height,
            )
        return HarmonicMeanTracker(
            max_lost_frames=max(60, round(fps * 12.0)),
            max_geometry_lost_frames=max(60, round(fps * 2.0)),
        )

    @staticmethod
    def _remap_local_identities(
        players: dict[int, tuple[PlayerObservation, ...]],
        poses: dict[int, tuple[PersonPoseObservation, ...]],
        actions: dict[tuple[int, int], ActionObservation],
        result: SelectiveSam3Result,
    ) -> tuple[
        dict[int, tuple[PlayerObservation, ...]],
        dict[int, tuple[PersonPoseObservation, ...]],
        dict[tuple[int, int], ActionObservation],
    ]:
        remapped_players: dict[int, tuple[PlayerObservation, ...]] = {}
        for frame_index, frame_players in players.items():
            canonical_ids = [
                result.resolve(player.track_id, frame_index) for player in frame_players
            ]
            if len(canonical_ids) != len(set(canonical_ids)):
                raise RuntimeError(
                    f"SAM3 produced co-visible Local ID collision at frame {frame_index}"
                )
            remapped_players[frame_index] = tuple(
                replace(player, track_id=canonical_id)
                for player, canonical_id in zip(frame_players, canonical_ids, strict=True)
            )
        remapped_poses = {
            frame_index: tuple(
                replace(
                    pose,
                    track_id=result.resolve(pose.track_id, frame_index),
                )
                for pose in frame_poses
            )
            for frame_index, frame_poses in poses.items()
        }
        remapped_actions: dict[tuple[int, int], ActionObservation] = {}
        for action in actions.values():
            canonical_id = result.resolve(action.track_id, action.frame_index)
            remapped = replace(action, track_id=canonical_id)
            remapped_actions[(action.frame_index, canonical_id)] = remapped
        return remapped_players, remapped_poses, remapped_actions

    def _people(
        self,
        raw: dict[str, Any],
        *,
        frame: NDArray[np.uint8],
        frame_index: int,
        width: int,
        height: int,
        tracker: Any,
    ) -> tuple[
        tuple[PlayerObservation, ...],
        tuple[PersonPoseObservation, ...],
        tuple[ActionObservation, ...],
    ]:
        labels = _numpy(raw.get("labels"), np.int32)
        boxes = _numpy(raw.get("boxes"), np.float32)
        scores = _numpy(raw.get("scores"), np.float32)
        person_class = self._schema_names.index("person")
        person_indices = np.flatnonzero(
            (labels == person_class) & (scores >= self.detector_threshold)
        )
        person_boxes = boxes[person_indices].astype(np.float32, copy=False)
        person_scores = scores[person_indices].astype(np.float32, copy=False)
        embeddings = self._person_embeddings(frame, person_boxes, frame_index)
        tracked = tracker.update(frame_index, person_boxes, person_scores, embeddings)
        human_points = _numpy(raw.get("human_keypoints"), np.float32)
        human_visibility = _numpy(raw.get("human_visibility"), np.float32)
        action_labels = _numpy(raw.get("action_labels"), np.int32)
        action_scores = _numpy(raw.get("action_scores"), np.float32)
        result_players: list[PlayerObservation] = []
        result_poses: list[PersonPoseObservation] = []
        result_actions: list[ActionObservation] = []
        for item in tracked:
            bbox = (
                float(item.bbox[0]),
                float(item.bbox[1]),
                float(item.bbox[2]),
                float(item.bbox[3]),
            )
            normalized_bbox = normalize_frame_bbox(bbox, width=width, height=height)
            result_players.append(
                PlayerObservation(
                    frame_index=frame_index,
                    source_track_id=item.track_id,
                    track_id=item.track_id,
                    frame_bbox=normalized_bbox,
                    frame_foot_pos=(
                        min(1.0, max(0.0, (bbox[0] + bbox[2]) / (2.0 * width))),
                        min(1.0, max(0.0, bbox[3] / height)),
                    ),
                    court_pos=None,
                    confidence=item.score,
                )
            )
            keypoints: tuple[tuple[float, float, float], ...] | None = None
            status: PersonPoseStatus = "INFERENCE_FAILED"
            if item.detection_index >= 0:
                model_index = int(person_indices[item.detection_index])
                if (
                    human_points.ndim == 3
                    and human_visibility.ndim == 2
                    and model_index < len(human_points)
                    and human_points.shape[1] == COCO17_KEYPOINT_COUNT
                    and human_visibility.shape[1] == COCO17_KEYPOINT_COUNT
                ):
                    keypoints = tuple(
                        (
                            min(1.0, max(0.0, float(point[0]) / width)),
                            min(1.0, max(0.0, float(point[1]) / height)),
                            min(1.0, max(0.0, float(confidence))),
                        )
                        for point, confidence in zip(
                            human_points[model_index], human_visibility[model_index], strict=True
                        )
                    )
                    usable = sum(point[2] >= self.pose_keypoint_confidence for point in keypoints)
                    status = "AVAILABLE" if usable >= self.pose_minimum_keypoints else "LOW_QUALITY"
                if model_index < len(action_labels):
                    action_index = int(action_labels[model_index])
                    if 0 <= action_index < len(self._action_names):
                        result_actions.append(
                            ActionObservation(
                                frame_index=frame_index,
                                track_id=item.track_id,
                                label=self._action_names[action_index],
                                confidence=(
                                    float(action_scores[model_index])
                                    if model_index < len(action_scores)
                                    else None
                                ),
                            )
                        )
            result_poses.append(
                PersonPoseObservation(
                    frame_index=frame_index,
                    track_id=item.track_id,
                    bbox_source="DETECTOR" if item.detection_index >= 0 else "TRACKER_PROPAGATED",
                    frame_bbox=normalized_bbox,
                    crop_transform=(
                        1.0 / width,
                        1.0 / height,
                        normalized_bbox[0],
                        normalized_bbox[1],
                    ),
                    status=status,
                    keypoints=keypoints,
                )
            )
        return tuple(result_players), tuple(result_poses), tuple(result_actions)

    def _person_embeddings(
        self,
        frame: NDArray[np.uint8],
        boxes: NDArray[np.float32],
        frame_index: int,
    ) -> NDArray[np.float32] | None:
        if not len(boxes):
            return None
        if self.local_tracker != "deep_eiou" and frame_index % self.reid_every != 0:
            return None
        height, width = frame.shape[:2]
        crops: list[NDArray[np.uint8]] = []
        for box in boxes:
            x1 = max(0, min(width, int(np.floor(box[0]))))
            y1 = max(0, min(height, int(np.floor(box[1]))))
            x2 = max(0, min(width, int(np.ceil(box[2]))))
            y2 = max(0, min(height, int(np.ceil(box[3]))))
            if x2 <= x1 or y2 <= y1:
                return None
            crops.append(np.ascontiguousarray(frame[y1:y2, x1:x2]))
        return self._osnet.encode(crops, batch_size=max(1, min(self.batch_size, len(crops))))

    def _court(self, raw: dict[str, Any], frame_index: int) -> CourtFrame | None:
        if not bool(raw.get("court_valid", False)):
            return None
        points = _numpy(raw.get("court_keypoints_raw", raw.get("court_keypoints")), np.float32)
        visibility = _numpy(raw.get("court_visibility"), np.float32)
        if points.shape != (COURT60_KEYPOINT_COUNT, 2):
            return None
        if visibility.shape != (COURT60_KEYPOINT_COUNT,):
            visibility = np.full(
                (COURT60_KEYPOINT_COUNT,), _scalar(raw.get("court_combined_score"))
            )
        keypoints = tuple(
            CourtKeypoint(
                index=index,
                frame_pos_px=(float(point[0]), float(point[1])),
                confidence=min(1.0, max(0.0, float(visibility[index]))),
                world_pos_m=COURT60_WORLD_XY[index],
            )
            for index, point in enumerate(points)
        )
        return CourtFrame(frame_index=frame_index, available=True, keypoints=keypoints)

    def _ball(
        self,
        raw: dict[str, Any],
        frame_index: int,
        width: int,
        height: int,
        tracker: BallTrajectoryTracker,
    ) -> BallObservation | None:
        labels = _numpy(raw.get("labels"), np.int32)
        boxes = _numpy(raw.get("boxes"), np.float32)
        scores = _numpy(raw.get("scores"), np.float32)
        ball_class = self._schema_names.index("ball")
        indices = np.flatnonzero((labels == ball_class) & (scores >= self.detector_threshold))
        positions = [
            (
                min(1.0, max(0.0, float((boxes[index][0] + boxes[index][2]) / (2 * width)))),
                min(1.0, max(0.0, float((boxes[index][1] + boxes[index][3]) / (2 * height)))),
            )
            for index in indices
        ]
        return tracker.update(frame_index, positions, [float(scores[index]) for index in indices])

    def _group(self, raw: dict[str, Any], frame_index: int) -> GroupActivityObservation | None:
        label_index = _integer(raw.get("group_label"))
        if not 0 <= label_index < len(self._group_names):
            return None
        return GroupActivityObservation(
            frame_index=frame_index,
            label=self._group_names[label_index],
            confidence=_scalar(raw.get("group_score")),
        )
