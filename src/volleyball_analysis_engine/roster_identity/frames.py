"""Pick the frames of a tracklet in which the jersey number is most likely legible.

The dominant failure mode is not blur or size, it is **body orientation**.  A broadcast
camera sits at the side of the court, so most frames show a player in profile, where the
number is compressed onto a near-vertical strip and is unreadable at any resolution.  A
frontality score derived from the shoulder/hip keypoints is therefore weighted highest.

Numbers are printed on both the front and the back of the jersey, so no front/back
classification is needed; the useful distinction is frontal-or-rear versus profile.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
from numpy.typing import NDArray

from .records import CanonicalTracklet, SelectedFrame

L_SHOULDER, R_SHOULDER = 5, 6
L_HIP, R_HIP = 11, 12
TORSO_KEYPOINTS = (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP)

# Below this many pixels the shoulder/hip geometry is noise, so frontality is not derived.
MIN_TORSO_HEIGHT_PX = 4
# Distance from the frame edge at which a box counts as touching it.
EDGE_MARGIN_PX = 3
# Smallest crop worth keeping; anything smaller has no readable jersey region.
MIN_CROP_WIDTH_PX = 8
MIN_CROP_HEIGHT_PX = 16

# A jersey number spans roughly 45% of torso height.  Below this many pixels of torso the
# number is smaller than a couple of ViT patches and carries almost no information.
TORSO_PIXEL_SATURATION = 60.0

# A failed pose usually means the crop does not contain a whole player: the tracker has
# drifted onto empty court or half a body.  Penalise hard rather than exclude, so a
# tracklet whose pose never fires still yields something to look at.
NO_POSE_PENALTY = 0.30

DEFAULT_WEIGHTS = {
    "frontality": 0.35,
    "torso_px": 0.25,
    "sharpness": 0.15,
    "confidence": 0.10,
    "completeness": 0.15,
}


@dataclass(frozen=True, slots=True)
class FrameSelectionSettings:
    """How many frames to keep and how widely to spread them over the tracklet."""

    top_k: int = 6
    candidate_count: int = 60
    min_gap: int = 8
    padding: float = 0.06
    pose_confidence: float = 0.35
    torso_expand: float = 0.15


class PoseEstimator:
    """Thin wrapper over YOLO pose, kept out of import time for the base package."""

    def __init__(self, weights: Path, device: int | str = 0, imgsz: int = 320) -> None:
        """Load the pose checkpoint onto the given device."""
        from ultralytics import YOLO

        # Ultralytics ships no type information, so the handle is explicitly untyped.
        self._model: Any = cast("Any", YOLO(str(weights)))
        self._device = device
        self._imgsz = imgsz

    def infer(
        self, crops: list[NDArray[np.uint8]], batch: int = 32
    ) -> list[NDArray[np.float32] | None]:
        """Crops are BGR.  Returns COCO-17 keypoints per crop, largest person only."""
        results: list[NDArray[np.float32] | None] = []
        for start in range(0, len(crops), batch):
            chunk = crops[start : start + batch]
            predictions: Any = self._model.predict(
                chunk, imgsz=self._imgsz, conf=0.15, device=self._device, verbose=False
            )
            for prediction in cast("list[Any]", predictions):
                keypoints: Any = getattr(prediction.keypoints, "data", None)
                if keypoints is None or len(keypoints) == 0:
                    results.append(None)
                    continue
                array = cast("NDArray[np.float32]", keypoints.cpu().numpy())
                if prediction.boxes is not None and len(array) > 1:
                    boxes = cast("NDArray[np.float32]", prediction.boxes.xyxy.cpu().numpy())
                    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                    array = array[[int(np.argmax(areas))]]
                results.append(array[0])
        return results


def torso_geometry(
    keypoints: NDArray[np.float32] | None, confidence: float
) -> dict[str, Any] | None:
    """Frontality and torso box from the four torso keypoints.

    `frontality` is shoulder span over torso height: about 0.85-1.1 when the chest or back
    faces the camera, about 0.1-0.3 in profile.
    """
    if keypoints is None:
        return None
    if float(np.min(keypoints[list(TORSO_KEYPOINTS), 2])) < confidence:
        return None
    left_shoulder, right_shoulder = keypoints[L_SHOULDER, :2], keypoints[R_SHOULDER, :2]
    left_hip, right_hip = keypoints[L_HIP, :2], keypoints[R_HIP, :2]
    torso_height = abs(
        (left_hip[1] + right_hip[1]) / 2.0 - (left_shoulder[1] + right_shoulder[1]) / 2.0
    )
    if torso_height < MIN_TORSO_HEIGHT_PX:
        return None
    shoulder_span = abs(left_shoulder[0] - right_shoulder[0])
    hip_span = abs(left_hip[0] - right_hip[0])
    # Shoulders dominate; hips are more often occluded by arms or shorts.
    ratio = (0.75 * shoulder_span + 0.25 * hip_span) / torso_height
    xs = [left_shoulder[0], right_shoulder[0], left_hip[0], right_hip[0]]
    ys = [left_shoulder[1], right_shoulder[1], left_hip[1], right_hip[1]]
    return {
        "frontality": float(min(1.0, max(0.0, ratio / 0.85))),
        "torso_height": float(torso_height),
        "torso_box": (float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))),
    }


def torso_roi(
    geometry: dict[str, Any] | None, width: int, height: int, expand: float
) -> tuple[int, int, int, int]:
    """Jersey region, falling back to the top 55% of the box when pose is unavailable."""
    if geometry is None:
        return (0, 0, width, max(8, int(height * 0.55)))
    x1, y1, x2, y2 = geometry["torso_box"]
    pad_x, pad_y = (x2 - x1) * expand, (y2 - y1) * expand
    return (
        int(max(0, x1 - pad_x)),
        int(max(0, y1 - pad_y)),
        int(min(width, x2 + pad_x)),
        int(min(height, y2 + pad_y)),
    )


def _sharpness(image: NDArray[np.uint8]) -> float:
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = float(cast("Any", cv2.Laplacian(grey, cv2.CV_64F)).var())
    return float(min(1.0, max(0.0, math.log10(variance + 1.0) / 3.0)))


def _completeness(box: tuple[int, int, int, int], width: int, height: int) -> float:
    """Score how fully the player fits inside the frame.

    A box touching the frame edge means the player is cut off.
    """
    x1, y1, x2, y2 = box
    touching = sum(
        [
            x1 <= EDGE_MARGIN_PX,
            y1 <= EDGE_MARGIN_PX,
            x2 >= width - EDGE_MARGIN_PX,
            y2 >= height - EDGE_MARGIN_PX,
        ]
    )
    return max(0.0, 1.0 - 0.35 * touching)


def _score(
    geometry: dict[str, Any] | None,
    box: tuple[int, int, int, int],
    torso: NDArray[np.uint8],
    *,
    confidence: float,
    frame_width: int,
    frame_height: int,
    weights: dict[str, float],
) -> tuple[float, dict[str, float]]:
    frontality = geometry["frontality"] if geometry else 0.25
    torso_height = (
        geometry["torso_height"] if geometry else (box[3] - box[1]) * 0.3
    )
    parts = {
        "frontality": frontality,
        "torso_px": min(1.0, torso_height / TORSO_PIXEL_SATURATION),
        "sharpness": _sharpness(torso) if torso.size else 0.0,
        "confidence": min(1.0, confidence),
        "completeness": _completeness(box, frame_width, frame_height),
    }
    quality = sum(weights[name] * value for name, value in parts.items())
    if geometry is None:
        quality *= NO_POSE_PENALTY
    return quality, parts


def select_frames(
    tracklet: CanonicalTracklet,
    video_path: Path,
    output_dir: Path,
    *,
    pose: PoseEstimator,
    settings: FrameSelectionSettings,
    weights: dict[str, float] | None = None,
) -> list[SelectedFrame]:
    """Decode candidate frames of one tracklet, score them and keep the best few."""
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    observations = list(tracklet.observations)
    step = max(1, math.ceil(len(observations) / settings.candidate_count))
    planned = {row.frame_index: row for row in observations[::step][: settings.candidate_count]}
    if not planned:
        return []

    capture = cv2.VideoCapture(str(video_path))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    crops: list[NDArray[np.uint8]] = []
    boxes: list[tuple[int, int, int, int]] = []
    frame_indices: list[int] = []
    confidences: list[float] = []
    index, last = 0, max(planned)
    while index <= last:
        ok, frame = capture.read()
        if not ok:
            break
        row = planned.get(index)
        if row is not None:
            box = _padded_box(row.frame_bbox, settings.padding, frame_width, frame_height)
            if box is not None:
                crop = cast("NDArray[np.uint8]", frame[box[1] : box[3], box[0] : box[2]])
                crops.append(crop)
                boxes.append(box)
                frame_indices.append(index)
                confidences.append(row.confidence or 0.0)
        index += 1
    capture.release()
    if not crops:
        return []

    keypoints = pose.infer(crops)
    output_dir.mkdir(parents=True, exist_ok=True)
    scored: list[SelectedFrame] = []
    for crop, box, frame_index, confidence, points in zip(
        crops, boxes, frame_indices, confidences, keypoints, strict=True
    ):
        geometry = torso_geometry(points, settings.pose_confidence)
        height, width = crop.shape[:2]
        x1, y1, x2, y2 = torso_roi(geometry, width, height, settings.torso_expand)
        torso = crop[y1:y2, x1:x2]
        quality, parts = _score(
            geometry,
            box,
            torso,
            confidence=confidence,
            frame_width=frame_width,
            frame_height=frame_height,
            weights=weights,
        )
        full_path = output_dir / f"f{frame_index:06d}_full.jpg"
        torso_path = output_dir / f"f{frame_index:06d}_torso.jpg"
        cv2.imwrite(str(full_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if torso.size:
            cv2.imwrite(str(torso_path), torso, [cv2.IMWRITE_JPEG_QUALITY, 95])
        scored.append(
            SelectedFrame(
                frame_index=frame_index,
                bbox_px=box,
                quality=quality,
                frontality=parts["frontality"],
                torso_px=parts["torso_px"],
                sharpness=parts["sharpness"],
                completeness=parts["completeness"],
                pose_ok=geometry is not None,
                full_path=str(full_path),
                torso_path=str(torso_path) if torso.size else "",
            )
        )
    return _pick(scored, settings)


def _padded_box(
    bbox: tuple[float, float, float, float],
    padding: float,
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = (
        bbox[0] * frame_width,
        bbox[1] * frame_height,
        bbox[2] * frame_width,
        bbox[3] * frame_height,
    )
    pad_x, pad_y = (x2 - x1) * padding, (y2 - y1) * padding
    box = (
        int(max(0, x1 - pad_x)),
        int(max(0, y1 - pad_y)),
        int(min(frame_width, x2 + pad_x)),
        int(min(frame_height, y2 + pad_y)),
    )
    if box[2] - box[0] < MIN_CROP_WIDTH_PX or box[3] - box[1] < MIN_CROP_HEIGHT_PX:
        return None
    return box


def _pick(scored: list[SelectedFrame], settings: FrameSelectionSettings) -> list[SelectedFrame]:
    """Take the best frames, spread over time.

    A player turns, and adjacent frames show the same angle, so a second nearby frame adds
    no new evidence.
    """
    if not scored:
        return []
    span = scored[-1].frame_index - scored[0].frame_index
    min_gap = min(settings.min_gap, max(1, int(span / (settings.top_k * 1.5))))
    picked: list[SelectedFrame] = []
    for frame in sorted(scored, key=lambda item: -item.quality):
        if all(abs(frame.frame_index - other.frame_index) >= min_gap for other in picked):
            picked.append(frame)
        if len(picked) >= settings.top_k:
            break
    for frame in sorted(scored, key=lambda item: -item.quality):
        if len(picked) >= settings.top_k:
            break
        if frame not in picked:
            picked.append(frame)
    return sorted(picked, key=lambda item: item.frame_index)
