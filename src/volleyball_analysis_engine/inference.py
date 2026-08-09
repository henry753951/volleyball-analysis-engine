"""Real clip inference using the supplied RT-DETRv4/X3D and court-pose models."""

from __future__ import annotations

import importlib
import inspect
import logging
import math
import sys
import types
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, cast

import cv2
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment  # pyright: ignore[reportUnknownVariableType]
from volleyball_monitoring_ai import AIJobRequest

from .records import (
    ActionObservation,
    BallObservation,
    CourtFrame,
    CourtKeypoint,
    PlayerObservation,
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


def _empty_metadata() -> dict[str, Any]:
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
    court_checkpoint: Path
    smp_root: Path
    osnet_checkpoint: Path

    def validate(self) -> None:
        """Fail early with all missing model assets listed together."""
        required = {
            "RT-DETRv4 root": self.rtv4_root / "engine" / "core" / "yaml_config.py",
            "RT-DETRv4 config": self.rtv4_config,
            "RT-DETRv4 checkpoint": self.rtv4_checkpoint,
            "court keypose checkpoint": self.court_checkpoint,
            "selective-mask-propagation root": self.smp_root
            / "selective_mask_propagation",
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


@dataclass(frozen=True, slots=True)
class _TrackedDetection:
    track_id: int
    detection_index: int
    bbox: NDArray[np.float32]
    score: float


class HarmonicMeanTracker:
    """Appearance/EIoU tracker with a bounded keep-all lost pool."""

    def __init__(
        self,
        *,
        max_lost_frames: int = 120,
        match_threshold: float = 0.28,
    ) -> None:
        self.max_lost_frames = max_lost_frames
        self.match_threshold = match_threshold
        self._next_id = 1
        self._tracks: dict[int, _Track] = {}

    def update(
        self,
        frame_index: int,
        boxes: NDArray[np.float32],
        scores: NDArray[np.float32],
        embeddings: NDArray[np.float32],
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
                geometry = _eiou_similarity(track.bbox, boxes)
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

        used_detections: set[int] = set()
        output: list[_TrackedDetection] = []
        for row, detection_index in matches:
            track = active[row]
            track.bbox = boxes[detection_index].copy()
            track.embedding = _normalized_average(
                track.embedding,
                embeddings[detection_index],
            )
            track.score = float(scores[detection_index])
            track.last_frame = frame_index
            used_detections.add(detection_index)
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
                embedding=embeddings[detection_index].copy(),
                score=float(scores[detection_index]),
                last_frame=frame_index,
            )
            self._tracks[track.track_id] = track
            self._next_id += 1
            output.append(
                _TrackedDetection(
                    track.track_id,
                    detection_index,
                    track.bbox.copy(),
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
        court_stride: int = 30,
        disable_amp: bool = False,
    ) -> None:
        paths.validate()
        self.paths = paths
        self.device_name = device
        self.backend = backend
        self.detector_threshold = detector_threshold
        self.court_stride = max(1, court_stride)
        self.disable_amp = disable_amp
        self._torch: Any = None
        self._model: Any = None
        self._postprocessor: Any = None
        self._streamer: Any = None
        self._input_size = (576, 1024)
        self._use_amp = False
        self._osnet: Any = None
        self._osnet_transform: Any = None
        self._court_model: Any = None
        self._effective_backend = backend

    @property
    def effective_backend(self) -> str:
        """Return the actually loaded X3D backend after compatibility fallback."""
        return self._effective_backend

    def infer(
        self,
        clip_path: Path,
        job: AIJobRequest,
        report: ProgressReporter,
    ) -> InferenceResult:
        """Decode every canonical frame and run actual models against it."""
        self._load_models(report)
        capture = cv2.VideoCapture(str(clip_path))
        if not capture.isOpened():
            raise ValueError(f"cannot open canonical clip: {clip_path}")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        expected_frames = int(job.clip.video.total_frames)
        tracker = HarmonicMeanTracker(max_lost_frames=max(60, round(fps * 2.0)))
        players: dict[int, tuple[PlayerObservation, ...]] = {}
        courts: dict[int, CourtFrame] = {}
        balls: dict[int, BallObservation] = {}
        actions: dict[tuple[int, int], ActionObservation] = {}
        frame_index = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                typed_frame = np.asarray(frame, dtype=np.uint8)
                result = self._infer_detector(typed_frame, width, height)
                labels, boxes, scores, action_labels, action_scores = self._to_numpy(result)
                person_indices = np.flatnonzero(
                    (labels == 0) & (scores >= self.detector_threshold)
                )
                person_boxes = boxes[person_indices].astype(np.float32, copy=False)
                person_scores = scores[person_indices].astype(np.float32, copy=False)
                embeddings = self._embeddings(typed_frame, person_boxes)
                tracked = tracker.update(
                    frame_index,
                    person_boxes,
                    person_scores,
                    embeddings,
                )
                frame_players: list[PlayerObservation] = []
                for item in tracked:
                    x1, y1, x2, y2 = (float(value) for value in item.bbox)
                    normalized_bbox = normalize_frame_bbox(
                        (x1, y1, x2, y2),
                        width=width,
                        height=height,
                    )
                    player = PlayerObservation(
                        frame_index=frame_index,
                        source_track_id=item.track_id,
                        track_id=item.track_id,
                        frame_bbox=normalized_bbox,
                        frame_foot_pos=(
                            _unit_interval((x1 + x2) / (2.0 * width)),
                            _unit_interval(y2 / height),
                        ),
                        court_pos=None,
                        confidence=item.score,
                    )
                    frame_players.append(player)
                    model_index = int(person_indices[item.detection_index])
                    if action_labels is not None and model_index < len(action_labels):
                        action_index = int(action_labels[model_index])
                        if 0 <= action_index < len(ACTION_NAMES):
                            confidence = (
                                float(action_scores[model_index])
                                if action_scores is not None and model_index < len(action_scores)
                                else None
                            )
                            actions[(frame_index, item.track_id)] = ActionObservation(
                                frame_index,
                                item.track_id,
                                ACTION_NAMES[action_index],
                                confidence,
                            )
                players[frame_index] = tuple(frame_players)

                ball_indices = np.flatnonzero(
                    (labels == 1) & (scores >= self.detector_threshold)
                )
                if len(ball_indices):
                    ball_index = int(ball_indices[np.argmax(scores[ball_indices])])
                    ball_box = boxes[ball_index]
                    balls[frame_index] = BallObservation(
                        frame_index=frame_index,
                        frame_pos=(
                            _unit_interval(
                                float((ball_box[0] + ball_box[2]) / (2.0 * width))
                            ),
                            _unit_interval(
                                float((ball_box[1] + ball_box[3]) / (2.0 * height))
                            ),
                        ),
                        confidence=float(scores[ball_index]),
                    )

                if frame_index % self.court_stride == 0:
                    court = self._infer_court(typed_frame, frame_index)
                    if court is not None:
                        courts[frame_index] = court
                frame_index += 1
                if frame_index % max(1, round(fps)) == 0:
                    report(
                        0.08 + 0.62 * min(1.0, frame_index / max(expected_frames, 1)),
                        "rtv4_x3d_tracking",
                    )
        finally:
            capture.release()
        if frame_index != expected_frames:
            raise ValueError(
                f"decoded frame count mismatch: job={expected_frames}, decoded={frame_index}"
            )
        return InferenceResult(
            players=players,
            courts=courts,
            balls=balls,
            actions=actions,
            frame_count=frame_index,
            frame_width=width,
            frame_height=height,
            fps=fps,
            metadata={
                "detector": "RT-DETRv4+X3D",
                "detector_config": str(self.paths.rtv4_config),
                "tracker": "harmonic-mean-eiou+OSNet+court-reentry",
                "court_detector": "YOLO26-pose",
                "streaming_backend": self._effective_backend,
                "source": "canonical_clip_inference",
            },
        )

    def prepare(self, report: ProgressReporter | None = None) -> None:
        """Load and strictly validate all model checkpoints without decoding a clip."""
        self._load_models(report or (lambda _progress, _stage: None))

    def _load_models(self, report: ProgressReporter) -> None:
        if self._model is not None:
            return
        report(0.01, "loading_rtv4_x3d")
        root = str(self.paths.rtv4_root.resolve())
        if root not in sys.path:
            sys.path.insert(0, root)
        torch = importlib.import_module("torch")
        functional = importlib.import_module("torch.nn.functional")
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
        self._input_size = tuple(
            int(value)
            for value in stream_config.get(
                "input_size",
                yaml_config.yaml_cfg.get("eval_spatial_size", [576, 1024]),
            )
        )
        self._use_amp = (
            bool(stream_config.get("amp", True))
            and device.type == "cuda"
            and not self.disable_amp
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
        self._functional = functional
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
        self._osnet_transform = osnet_module.TRANSFORM

        report(0.055, "loading_court_keypose")
        yolo = importlib.import_module("ultralytics").YOLO
        self._court_model = yolo(str(self.paths.court_checkpoint))

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

    def _infer_detector(self, frame: NDArray[np.uint8], width: int, height: int) -> Any:
        torch = self._torch
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = (
            torch.from_numpy(rgb)
            .permute(2, 0, 1)
            .contiguous()
            .float()
            .div_(255.0)
            .unsqueeze(0)
        )
        tensor = self._functional.interpolate(
            tensor,
            size=self._input_size,
            mode="bilinear",
            align_corners=False,
        ).to(self._device, non_blocking=True)
        with torch.inference_mode(), torch.autocast(
            device_type=self._device.type,
            dtype=torch.float16,
            enabled=self._use_amp,
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
        image_module: Any = importlib.import_module("PIL.Image")
        pil_image: Any = image_module.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        crops: list[Any] = []
        height = int(frame.shape[0])
        width = int(frame.shape[1])
        for box in boxes:
            x1 = max(0, min(width - 1, math.floor(float(box[0]))))
            y1 = max(0, min(height - 1, math.floor(float(box[1]))))
            x2 = max(x1 + 1, min(width, math.ceil(float(box[2]))))
            y2 = max(y1 + 1, min(height, math.ceil(float(box[3]))))
            crops.append(self._osnet_transform(pil_image.crop((x1, y1, x2, y2)).convert("RGB")))
        batch = self._torch.stack(crops).to(self._device)
        with self._torch.inference_mode():
            features = self._osnet(batch)
        array = np.asarray(features.detach().cpu().numpy(), dtype=np.float32)
        norms = np.maximum(np.linalg.norm(array, axis=1, keepdims=True), 1e-6)
        return np.asarray(array / norms, dtype=np.float32)

    def _infer_court(
        self,
        frame: NDArray[np.uint8],
        frame_index: int,
    ) -> CourtFrame | None:
        results = self._court_model.predict(
            source=frame,
            imgsz=640,
            device=self.device_name,
            conf=0.25,
            verbose=False,
            save=False,
        )
        if not results:
            return None
        result = results[0]
        if result.boxes is None or result.keypoints is None or len(result.boxes) == 0:
            return None
        scores = result.boxes.conf.detach().cpu().numpy().reshape(-1)
        best = int(np.argmax(scores))
        positions = result.keypoints.xy[best].detach().cpu().numpy()
        confidences = (
            result.keypoints.conf[best].detach().cpu().numpy()
            if result.keypoints.conf is not None
            else np.ones(len(positions), dtype=np.float32)
        )
        keypoints = tuple(
            CourtKeypoint(
                index=index,
                frame_pos_px=(float(position[0]), float(position[1])),
                confidence=float(confidences[index]),
                world_pos_m=(
                    COURT_WORLD_POINTS[index] if index < len(COURT_WORLD_POINTS) else None
                ),
            )
            for index, position in enumerate(positions)
        )
        return CourtFrame(frame_index=frame_index, available=True, keypoints=keypoints)

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
