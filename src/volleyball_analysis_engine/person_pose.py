"""Every-frame person COCO-17 evidence extracted once during base analysis."""

from __future__ import annotations

import hashlib
import importlib
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from .records import BboxObservationSource, PersonPoseObservation, PersonPoseStatus

LOGGER = logging.getLogger(__name__)
COCO17_KEYPOINT_COUNT = 17
KEYPOINT_ARRAY_NDIM = 3
KEYPOINT_CONFIDENCE_NDIM = 2


@dataclass(frozen=True, slots=True)
class PoseCrop:
    """One tracked-player crop and its lossless coordinate transform."""

    track_id: int
    bbox_source: BboxObservationSource
    frame_bbox: tuple[float, float, float, float]
    crop_transform: tuple[float, float, float, float]
    image: NDArray[np.uint8] | None


@lru_cache(maxsize=8)
def _checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class PersonPoseExtractor:
    """Batch a frame's tracked player crops without sampling away frame coverage."""

    def __init__(
        self,
        checkpoint: Path,
        *,
        device: str,
        batch_size: int = 32,
        imgsz: int = 640,
        confidence: float = 0.15,
        keypoint_confidence: float = 0.3,
        minimum_keypoints: int = 4,
    ) -> None:
        """Configure a reusable Ultralytics COCO-17 pose model."""
        self.checkpoint = checkpoint
        self.device_name = device
        self.batch_size = max(1, batch_size)
        self.imgsz = imgsz
        self.confidence = confidence
        self.keypoint_confidence = keypoint_confidence
        self.minimum_keypoints = minimum_keypoints
        self._model: Any = None

    @property
    def recipe_namespace(self) -> str:
        """Return the content-addressed recipe used by every observation."""
        return (
            f"ultralytics-coco17:{_checkpoint_sha256(self.checkpoint)}:"
            f"imgsz{self.imgsz}:conf{self.confidence:g}:crop-v1"
        )

    @property
    def recipe_metadata(self) -> dict[str, str]:
        """Describe the immutable model and normalized crop preprocessing contract."""
        return {
            "namespace": self.recipe_namespace,
            "model_name": self.checkpoint.stem,
            "checkpoint_sha256": _checkpoint_sha256(self.checkpoint),
            "preprocess_version": "tracked-player-crop-normalized-v1",
            "keypoint_layout": "COCO_17",
            "coordinate_space": "NORMALIZED_VIDEO",
        }

    def prepare(self) -> None:
        """Load the model once before frame inference begins."""
        if self._model is not None:
            return
        if not self.checkpoint.exists():
            message = f"missing COCO-17 pose checkpoint: {self.checkpoint}"
            raise FileNotFoundError(message)
        ultralytics = importlib.import_module("ultralytics")

        self._model = ultralytics.YOLO(str(self.checkpoint.resolve()))

    @staticmethod
    def _crop(
        frame: NDArray[np.uint8],
        *,
        track_id: int,
        bbox: NDArray[np.float32],
        detector_observed: bool,
    ) -> PoseCrop:
        height, width = frame.shape[:2]
        raw_x1, raw_y1, raw_x2, raw_y2 = (float(value) for value in bbox)
        source: BboxObservationSource = "DETECTOR" if detector_observed else "TRACKER_PROPAGATED"
        finite = all(np.isfinite(value) for value in (raw_x1, raw_y1, raw_x2, raw_y2))
        ordered_x1, ordered_x2 = sorted((raw_x1, raw_x2)) if finite else (0.0, 0.0)
        ordered_y1, ordered_y2 = sorted((raw_y1, raw_y2)) if finite else (0.0, 0.0)
        x1 = max(0, min(width, int(np.floor(ordered_x1))))
        y1 = max(0, min(height, int(np.floor(ordered_y1))))
        x2 = max(0, min(width, int(np.ceil(ordered_x2))))
        y2 = max(0, min(height, int(np.ceil(ordered_y2))))
        frame_bbox = (x1 / width, y1 / height, x2 / width, y2 / height)
        crop_transform = (1.0 / width, 1.0 / height, x1 / width, y1 / height)
        usable = finite and x2 > x1 and y2 > y1
        crop = frame[y1:y2, x1:x2] if usable else None
        return PoseCrop(
            track_id=track_id,
            bbox_source=source,
            frame_bbox=frame_bbox,
            crop_transform=crop_transform,
            image=(np.ascontiguousarray(crop) if crop is not None and crop.size else None),
        )

    @staticmethod
    def _selected_keypoints(
        result: object, crop: PoseCrop
    ) -> tuple[tuple[float, float, float], ...] | None:
        if crop.image is None:
            return None
        dynamic_result = cast("Any", result)
        try:
            boxes = dynamic_result.boxes.xyxy.detach().cpu().numpy()
            xy = dynamic_result.keypoints.xy.detach().cpu().numpy()
            confidence_tensor = dynamic_result.keypoints.conf
            if confidence_tensor is None:
                return None
            confidence = confidence_tensor.detach().cpu().numpy()
        except (AttributeError, TypeError, ValueError):
            return None
        if not len(boxes):
            return None
        if (
            xy.ndim != KEYPOINT_ARRAY_NDIM
            or xy.shape[1] != COCO17_KEYPOINT_COUNT
            or confidence.ndim != KEYPOINT_CONFIDENCE_NDIM
            or confidence.shape[:2] != xy.shape[:2]
        ):
            return None
        height, width = crop.image.shape[:2]
        center = np.asarray((width / 2, height / 2), dtype=np.float32)
        centers = np.stack(
            ((boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2), axis=1
        )
        areas = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
        scores = areas / max(width * height, 1) - 0.2 * np.linalg.norm(
            (centers - center) / np.asarray((width, height)), axis=1
        )
        selected = int(np.argmax(scores))
        scale_x, scale_y, offset_x, offset_y = crop.crop_transform
        points: list[tuple[float, float, float]] = []
        for index in range(COCO17_KEYPOINT_COUNT):
            score = float(confidence[selected, index])
            if not np.isfinite(score) or score < 0:
                points.append((-1.0, -1.0, -1.0))
                continue
            x = min(1.0, max(0.0, float(xy[selected, index, 0]) * scale_x + offset_x))
            y = min(1.0, max(0.0, float(xy[selected, index, 1]) * scale_y + offset_y))
            points.append((x, y, min(1.0, score)))
        return tuple(points)

    def infer_frame(
        self,
        frame: NDArray[np.uint8],
        *,
        frame_index: int,
        tracks: list[tuple[int, NDArray[np.float32], bool]],
    ) -> tuple[PersonPoseObservation, ...]:
        """Return one explicit pose state for every supplied player observation."""
        if not tracks:
            return ()
        self.prepare()
        crops = [
            self._crop(
                frame,
                track_id=track_id,
                bbox=bbox,
                detector_observed=detector_observed,
            )
            for track_id, bbox, detector_observed in tracks
        ]
        usable = [(index, crop) for index, crop in enumerate(crops) if crop.image is not None]
        results: dict[int, Any] = {}
        try:
            for start in range(0, len(usable), self.batch_size):
                batch = usable[start : start + self.batch_size]
                predicted = self._model.predict(
                    source=[crop.image for _, crop in batch],
                    device=self.device_name,
                    imgsz=self.imgsz,
                    conf=self.confidence,
                    batch=self.batch_size,
                    half=self.device_name.startswith("cuda"),
                    verbose=False,
                )
                for (index, _), result in zip(batch, predicted, strict=True):
                    results[index] = result
        except Exception:
            LOGGER.exception("person pose inference failed for canonical frame %s", frame_index)
            results.clear()

        observations: list[PersonPoseObservation] = []
        for index, crop in enumerate(crops):
            try:
                keypoints = (
                    None if index not in results else self._selected_keypoints(results[index], crop)
                )
            except (IndexError, TypeError, ValueError):
                LOGGER.exception(
                    "invalid pose result for canonical frame %s track %s",
                    frame_index,
                    crop.track_id,
                )
                keypoints = None
            if crop.image is None:
                status: PersonPoseStatus = "NO_USABLE_BBOX"
            elif index not in results or keypoints is None:
                status = "INFERENCE_FAILED"
            elif (
                sum(point[2] >= self.keypoint_confidence for point in keypoints)
                < self.minimum_keypoints
            ):
                status = "LOW_QUALITY"
            else:
                status = "AVAILABLE"
            observations.append(
                PersonPoseObservation(
                    frame_index=frame_index,
                    track_id=crop.track_id,
                    bbox_source=crop.bbox_source,
                    frame_bbox=crop.frame_bbox,
                    crop_transform=crop.crop_transform,
                    status=status,
                    keypoints=keypoints,
                )
            )
        return tuple(observations)
