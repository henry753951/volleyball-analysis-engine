"""Public inference API for the Volleyball multitask model."""

from __future__ import annotations

import time
import threading
from importlib.resources import files
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

# Registry side effects are required before InferenceConfig creates components.
from . import _model as _model_registry  # noqa: F401
from ._model.core import InferenceConfig
from ._model.volleyball_metadata import (
    ACTION_NAMES,
    BALL_CLASS_ID,
    GROUP_ACTIVITY_NAMES,
    PERSON_CLASS_ID,
)

SCHEMA_VERSION = "2.0"


def _default_config_path() -> Path:
    return Path(str(files("volleyball_sdk").joinpath("config/model.yml")))


def _cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _resolve_device(name: str | torch.device) -> torch.device:
    if isinstance(name, torch.device):
        device = name
    else:
        text = str(name).strip().lower()
        if text == "auto":
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(text)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    return device


def _extract_model_state(payload: Any) -> tuple[dict[str, torch.Tensor], str]:
    if not isinstance(payload, dict):
        raise TypeError("Checkpoint must be a mapping/state-dict")

    ema = payload.get("ema")
    if isinstance(ema, dict):
        candidate = ema.get("module", ema)
        if isinstance(candidate, dict):
            return candidate, "ema"

    for key in ("model", "state_dict"):
        candidate = payload.get(key)
        if isinstance(candidate, dict):
            return candidate, key

    return payload, "root"


def _load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    *,
    minimum_coverage: float = 0.98,
) -> dict[str, Any]:
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    # This deployment package is intended for checkpoints produced by this
    # project. Only load checkpoints from trusted sources.
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    source, source_name = _extract_model_state(payload)
    source = {
        key[7:] if key.startswith("module.") else key: value
        for key, value in source.items()
        if torch.is_tensor(value)
    }

    current = model.state_dict()
    matched = {
        key: value
        for key, value in source.items()
        if key in current and tuple(current[key].shape) == tuple(value.shape)
    }
    if not matched:
        raise RuntimeError(f"No compatible model tensors found in checkpoint: {path}")

    incompatible = model.load_state_dict(matched, strict=False)
    coverage = len(matched) / max(1, len(current))
    if coverage < float(minimum_coverage):
        raise RuntimeError(
            "Checkpoint/model mismatch: "
            f"coverage={coverage:.2%}, required>={minimum_coverage:.2%}. "
            f"missing={len(incompatible.missing_keys)}"
        )

    return {
        "path": str(path),
        "source": source_name,
        "matched_tensors": len(matched),
        "model_tensors": len(current),
        "coverage": coverage,
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }


def _validate_frame(frame: np.ndarray, index: int) -> None:
    if not isinstance(frame, np.ndarray):
        raise TypeError(f"frames_bgr[{index}] must be numpy.ndarray")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"frames_bgr[{index}] must have shape [H,W,3], got {frame.shape}")
    if frame.dtype != np.uint8:
        raise TypeError(f"frames_bgr[{index}] must be uint8 BGR, got dtype={frame.dtype}")


def _as_bgr_image(source: Any) -> np.ndarray:
    """Convert one YOLO-style image source into uint8 BGR OpenCV format."""
    if isinstance(source, np.ndarray):
        _validate_frame(source, 0)
        return source

    if isinstance(source, Image.Image):
        rgb = np.asarray(source.convert("RGB"), dtype=np.uint8)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    if isinstance(source, (str, Path)):
        path = Path(source).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to decode image: {path}")
        return image

    raise TypeError(
        "Unsupported image source. Expected numpy.ndarray (BGR), PIL.Image, or an image path."
    )


def _looks_like_clip(source: Any, frame_num: int) -> bool:
    if isinstance(source, (str, Path, np.ndarray, Image.Image)):
        return False
    if not isinstance(source, Sequence):
        return False
    return len(source) == frame_num


def _preprocess_one_clip(
    frames_bgr: Sequence[np.ndarray],
    *,
    frame_num: int,
    input_size: Sequence[int],
    mean: Sequence[float],
    std: Sequence[float],
) -> torch.Tensor:
    if len(frames_bgr) != frame_num:
        raise ValueError(f"Model expects {frame_num} frames, received {len(frames_bgr)}")

    output_height, output_width = (int(v) for v in input_size)
    tensors = []
    for index, frame_bgr in enumerate(frames_bgr):
        _validate_frame(frame_bgr, index)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        image = TF.resize(image, [output_height, output_width], antialias=True)
        tensor = TF.pil_to_tensor(image).to(torch.float32).div_(255.0)
        tensors.append(tensor)

    video = torch.stack(tensors, dim=1)  # [C,T,H,W]
    mean_t = video.new_tensor(mean).view(3, 1, 1, 1)
    std_t = video.new_tensor(std).view(3, 1, 1, 1)
    return (video - mean_t) / std_t


def _float(value: Any) -> float:
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def _int(value: Any) -> int:
    if torch.is_tensor(value):
        return int(value.detach().cpu().item())
    return int(value)


def _tolist(value: Any) -> list:
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return list(value)


def _confidence(value: Any) -> float:
    """Stable JSON-friendly confidence precision."""
    return round(float(value), 6)


def _coordinate(value: Any) -> float:
    """Keep ample sub-pixel precision without float32 display noise."""
    return round(float(value), 4)


def _xy_points(points: Any) -> list[list[float]]:
    """Serialize Nx2 keypoints as plain pixel-coordinate pairs."""
    if points is None:
        return []
    return [[_coordinate(point[0]), _coordinate(point[1])] for point in _tolist(points)]


def _serialize_result(
    result: dict[str, Any],
    *,
    width: int,
    height: int,
    score_threshold: float,
) -> dict[str, Any]:
    """Convert native model output into the compact public schema.

    Public contract:
      detections    -> [{confidence, label, bbox}]
      human_pose    -> [{bbox, keypoints}]
      court         -> {confidence, keypoints}
      group_activity-> {confidence, label}

    Person detection confidence is the object confidence multiplied by the
    action confidence returned by the postprocessor.  The postprocessor action
    confidence already incorporates actionness when that head is available.
    """
    del width, height  # coordinates are already restored to original pixels

    labels = _tolist(result["labels"])
    scores = _tolist(result["scores"])
    boxes = _tolist(result["boxes"])

    action_labels = _tolist(result["action_labels"]) if "action_labels" in result else None
    action_scores = _tolist(result["action_scores"]) if "action_scores" in result else None
    human_keypoints = result.get("human_keypoints")

    detections: list[dict[str, Any]] = []
    human_pose: list[dict[str, Any]] = []

    for i, (class_id, object_score, box) in enumerate(zip(labels, scores, boxes)):
        class_id = int(class_id)
        object_score = float(object_score)
        bbox = [_coordinate(v) for v in box]

        if class_id == PERSON_CLASS_ID:
            # Preserve the model's original object-detection thresholding.
            # The exported confidence below represents the joint
            # person+action confidence, but filtering remains based on the
            # detector score so simplifying the schema does not drop outputs.
            if object_score < score_threshold:
                continue

            if action_labels is not None and action_scores is not None:
                action_id = int(action_labels[i])
                if 0 <= action_id < len(ACTION_NAMES):
                    label = ACTION_NAMES[action_id]
                    confidence = object_score * float(action_scores[i])
                else:
                    label = "person"
                    confidence = object_score
            else:
                label = "person"
                confidence = object_score

            detections.append(
                {
                    "confidence": _confidence(confidence),
                    "label": label,
                    "bbox": bbox,
                }
            )

            if human_keypoints is not None:
                human_pose.append(
                    {
                        "bbox": bbox.copy(),
                        "keypoints": _xy_points(human_keypoints[i]),
                    }
                )

        elif class_id == BALL_CLASS_ID:
            if object_score < score_threshold:
                continue
            detections.append(
                {
                    "confidence": _confidence(object_score),
                    "label": "ball",
                    "bbox": bbox,
                }
            )

        # Court is intentionally not duplicated in detections.  It has its own
        # scene-level output below with confidence + 60 keypoints.

    group_activity = {"confidence": 0.0, "label": "unknown"}
    if "group_label" in result and "group_score" in result:
        group_id = _int(result["group_label"])
        group_activity = {
            "confidence": _confidence(_float(result["group_score"])),
            "label": GROUP_ACTIVITY_NAMES[group_id]
            if 0 <= group_id < len(GROUP_ACTIVITY_NAMES)
            else str(group_id),
        }

    raw_court_points = result.get("court_keypoints_raw")
    if raw_court_points is None:
        raw_court_points = result.get("court_keypoints")

    if "court_combined_score" in result:
        court_confidence = _float(result["court_combined_score"])
    elif "court_pose_score" in result:
        court_confidence = _float(result["court_pose_score"])
    elif "court_detection_score" in result:
        court_confidence = _float(result["court_detection_score"])
    else:
        court_confidence = 0.0

    court = {
        "confidence": _confidence(court_confidence),
        "keypoints": _xy_points(raw_court_points),
    }

    return {
        "detections": detections,
        "human_pose": human_pose,
        "court": court,
        "group_activity": group_activity,
    }


class VolleyballPredictor:
    """Stable deployment wrapper around the trained multitask checkpoint.

    Input contract:
      - one clip = ``frame_num`` uint8 BGR OpenCV frames;
      - default model uses 5 frames with temporal spacing handled by the caller
        (or by :meth:`predict_video`).

    Output contract:
      - plain Python dicts/lists/numbers;
      - no CUDA tensors are exposed by the normal API;
      - ``predict_batch_raw`` remains available for advanced downstream code.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        config: str | Path | None = None,
        device: str | torch.device = "auto",
        fp16: bool = True,
        minimum_checkpoint_coverage: float = 0.98,
        warmup: bool = True,
    ):
        checkpoint_path = Path(checkpoint).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        self.device = _resolve_device(device)
        self.config_path = Path(config).expanduser().resolve() if config else _default_config_path()
        self.cfg = InferenceConfig(self.config_path)
        runtime = self.cfg.runtime

        self.frame_num = int(runtime["frame_num"])
        self.jump_frame = int(runtime["jump_frame"])
        self.sampling_mode = str(runtime["sampling_mode"]).lower()
        self.input_size = tuple(int(v) for v in runtime["input_size"])
        self.normalize_mean = tuple(float(v) for v in runtime["normalize_mean"])
        self.normalize_std = tuple(float(v) for v in runtime["normalize_std"])
        self.score_threshold = float(runtime["score_threshold"])
        if self.sampling_mode not in {"centered", "causal"}:
            raise ValueError(f"Unsupported sampling_mode: {self.sampling_mode!r}")

        self.model = self.cfg.model
        self.checkpoint_info = _load_checkpoint(
            self.model,
            checkpoint_path,
            minimum_coverage=minimum_checkpoint_coverage,
        )
        self.model = self.model.to(self.device).eval()
        self.postprocessor = self.cfg.postprocessor.to(self.device).eval()
        self.fp16 = bool(fp16 and self.device.type == "cuda")
        self._lock = threading.RLock()

        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        self.warmup_ms = 0.0
        if warmup:
            black = np.zeros((self.input_size[0], self.input_size[1], 3), dtype=np.uint8)
            start = time.perf_counter()
            self.predict_clip([black] * self.frame_num, score_threshold=1.1)
            _cuda_sync(self.device)
            self.warmup_ms = (time.perf_counter() - start) * 1000.0

    def __call__(
        self,
        source: Any,
        *,
        score_threshold: float | None = None,
    ) -> dict[str, Any]:
        """Run inference with a compact YOLO-like ``result = model(source)`` API.

        Accepted sources:
          - one uint8 BGR ``numpy.ndarray``;
          - one ``PIL.Image``;
          - one image path;
          - a sequence of exactly ``frame_num`` images/paths (recommended).

        The network is temporal (default T=5). A single image is replicated T
        times only as a convenience mode. Use a real temporal clip for action
        and group-activity predictions.
        """
        if _looks_like_clip(source, self.frame_num):
            frames = [_as_bgr_image(item) for item in source]
            return self.predict_clip(frames, score_threshold=score_threshold)

        image = _as_bgr_image(source)
        return self.predict_clip(
            [image] * self.frame_num,
            score_threshold=score_threshold,
        )

    def predict(
        self,
        source: Any,
        *,
        score_threshold: float | None = None,
    ) -> dict[str, Any]:
        """Alias of :meth:`__call__`, matching common detector APIs."""
        return self(source, score_threshold=score_threshold)

    def info(self) -> dict[str, Any]:
        return {
            "device": str(self.device),
            "fp16": self.fp16,
            "frame_num": self.frame_num,
            "jump_frame": self.jump_frame,
            "sampling_mode": self.sampling_mode,
            "input_size": list(self.input_size),
            "normalize_mean": list(self.normalize_mean),
            "normalize_std": list(self.normalize_std),
            "score_threshold": self.score_threshold,
            "warmup_ms": self.warmup_ms,
            "checkpoint": dict(self.checkpoint_info),
        }

    def _prepare_batch(self, clips_bgr: Sequence[Sequence[np.ndarray]]) -> torch.Tensor:
        if not clips_bgr:
            raise ValueError("clips_bgr must contain at least one clip")
        clips = [
            _preprocess_one_clip(
                clip,
                frame_num=self.frame_num,
                input_size=self.input_size,
                mean=self.normalize_mean,
                std=self.normalize_std,
            )
            for clip in clips_bgr
        ]
        batch = torch.stack(clips, dim=0)
        return batch.to(self.device, non_blocking=True)

    def predict_batch_raw(
        self,
        clips_bgr: Sequence[Sequence[np.ndarray]],
        *,
        image_sizes: Sequence[tuple[int, int]] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, float]]:
        """Return native postprocessor results and timing.

        ``image_sizes`` uses ``(width, height)`` for each clip. If omitted, the
        center/last output frame size of each clip is used.
        """

        with self._lock:
            start_total = time.perf_counter()
            batch = self._prepare_batch(clips_bgr)
            _cuda_sync(self.device)
            end_pre = time.perf_counter()

            if image_sizes is None:
                output_slot = (
                    self.frame_num // 2 if self.sampling_mode == "centered" else self.frame_num - 1
                )
                image_sizes = [
                    (int(clip[output_slot].shape[1]), int(clip[output_slot].shape[0]))
                    for clip in clips_bgr
                ]
            if len(image_sizes) != len(clips_bgr):
                raise ValueError("image_sizes length must equal batch size")

            with (
                torch.inference_mode(),
                torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.float16,
                    enabled=self.fp16,
                ),
            ):
                outputs = self.model(batch, targets=None)
            _cuda_sync(self.device)
            end_forward = time.perf_counter()

            orig_sizes = torch.tensor(image_sizes, dtype=torch.float32, device=self.device)
            with torch.inference_mode():
                results = self.postprocessor(outputs, orig_sizes)
            _cuda_sync(self.device)
            end_post = time.perf_counter()

        timing = {
            "preprocess_ms": (end_pre - start_total) * 1000.0,
            "forward_ms": (end_forward - end_pre) * 1000.0,
            "postprocess_ms": (end_post - end_forward) * 1000.0,
            "total_ms": (end_post - start_total) * 1000.0,
        }
        return results, timing

    def predict_batch(
        self,
        clips_bgr: Sequence[Sequence[np.ndarray]],
        *,
        image_sizes: Sequence[tuple[int, int]] | None = None,
        score_threshold: float | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, float]]:
        threshold = self.score_threshold if score_threshold is None else float(score_threshold)
        raw_results, timing = self.predict_batch_raw(clips_bgr, image_sizes=image_sizes)

        if image_sizes is None:
            output_slot = (
                self.frame_num // 2 if self.sampling_mode == "centered" else self.frame_num - 1
            )
            image_sizes = [
                (int(clip[output_slot].shape[1]), int(clip[output_slot].shape[0]))
                for clip in clips_bgr
            ]

        results = [
            _serialize_result(
                raw,
                width=int(size[0]),
                height=int(size[1]),
                score_threshold=threshold,
            )
            for raw, size in zip(raw_results, image_sizes)
        ]
        return results, timing

    def predict_clip(
        self,
        frames_bgr: Sequence[np.ndarray],
        *,
        score_threshold: float | None = None,
    ) -> dict[str, Any]:
        output_slot = (
            self.frame_num // 2 if self.sampling_mode == "centered" else self.frame_num - 1
        )
        if len(frames_bgr) != self.frame_num:
            raise ValueError(f"Model expects {self.frame_num} frames, received {len(frames_bgr)}")
        output_frame = frames_bgr[output_slot]
        results, _timing = self.predict_batch(
            [frames_bgr],
            image_sizes=[(int(output_frame.shape[1]), int(output_frame.shape[0]))],
            score_threshold=score_threshold,
        )
        return results[0]

    def predict_video(
        self,
        video_path: str | Path,
        *,
        step: int = 1,
        batch_size: int = 1,
        score_threshold: float | None = None,
    ) -> Iterable[dict[str, Any]]:
        """Yield one result per selected target frame from a video.

        ``jump_frame`` controls temporal spacing inside each 5-frame model clip.
        ``step`` independently controls how often an output is produced.
        """
        from .video import iter_video_clips

        if step <= 0:
            raise ValueError("step must be > 0")
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        pending: list[tuple[int, float, list[np.ndarray], tuple[int, int]]] = []
        for packet in iter_video_clips(
            video_path,
            frame_num=self.frame_num,
            jump_frame=self.jump_frame,
            sampling_mode=self.sampling_mode,
            step=step,
        ):
            pending.append(packet)
            if len(pending) < batch_size:
                continue
            yield from self._predict_video_packets(pending, score_threshold)
            pending.clear()

        if pending:
            yield from self._predict_video_packets(pending, score_threshold)

    def _predict_video_packets(
        self,
        packets: Sequence[tuple[int, float, list[np.ndarray], tuple[int, int]]],
        score_threshold: float | None,
    ) -> Iterable[dict[str, Any]]:
        clips = [packet[2] for packet in packets]
        sizes = [packet[3] for packet in packets]
        results, _timing = self.predict_batch(
            clips,
            image_sizes=sizes,
            score_threshold=score_threshold,
        )
        for packet, result in zip(packets, results):
            frame_index, timestamp_sec, _clip, _size = packet
            result["video"] = {
                "frame_index": int(frame_index),
                "timestamp_sec": float(timestamp_sec),
            }
            yield result
