"""Run-local DeepEIOU tracking and optional selective SAM3 correction."""

from __future__ import annotations

import importlib
import json
import logging
import subprocess
import sys
import tempfile
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

LOGGER = logging.getLogger(__name__)
MIN_TARGET_BOX_IOU = 0.5


@dataclass(frozen=True, slots=True)
class TrackedDetection:
    """One DeepEIOU track materialized against this frame's detections."""

    track_id: int
    detection_index: int
    bbox: NDArray[np.float32]
    score: float


@dataclass(frozen=True, slots=True)
class SelectiveSam3Result:
    """Identity-only SAM3 correction result with an explicit fallback status."""

    status: str
    rename_events: tuple[tuple[int, int, int], ...]
    window_count: int
    swap_count: int
    stderr_tail: str | None = None

    def resolve(self, raw_track_id: int, frame_index: int) -> int:
        """Resolve one raw DeepEIOU ID using the upstream frame-effective ledger."""
        resolved = raw_track_id
        effective_frame = -1
        for source_id, canonical_id, event_frame in self.rename_events:
            if source_id != raw_track_id:
                continue
            if event_frame <= frame_index and event_frame >= effective_frame:
                resolved = canonical_id
                effective_frame = event_frame
        return resolved


class DeepEiouTracker:
    """Adapter around the reference Deep-EIoU tracker from the SMP checkout."""

    def __init__(
        self,
        *,
        smp_root: Path,
        fps: float,
        frame_width: int,
        frame_height: int,
    ) -> None:
        """Load the upstream tracker and initialize one run-local state machine."""
        self.smp_root = smp_root.expanduser().resolve()
        tracker_source = self.smp_root / "selective_mask_propagation" / "deep_eiou" / "tracker.py"
        if not tracker_source.is_file():
            message = f"missing DeepEIOU tracker source: {tracker_source}"
            raise FileNotFoundError(message)
        root = str(self.smp_root)
        if root not in sys.path:
            sys.path.insert(0, root)
        _install_cython_bbox_fallback()
        tracker_module = importlib.import_module("selective_mask_propagation.deep_eiou.tracker")
        self._tracker = tracker_module.Deep_EIoU(
            frame_rate=max(1, round(fps)),
            frame_width=frame_width,
            frame_height=frame_height,
            with_reid=True,
        )
        self._next_frame_index = 0
        self.tracks: dict[int, dict[int, NDArray[np.float32]]] = {}
        self.margins: dict[int, dict[int, float]] = {}

    def update(
        self,
        frame_index: int,
        boxes: NDArray[np.float32],
        scores: NDArray[np.float32],
        embeddings: NDArray[np.float32] | None,
    ) -> tuple[TrackedDetection, ...]:
        """Advance one canonical frame and preserve detection-index attribution."""
        if frame_index != self._next_frame_index:
            message = (
                "DeepEIOU requires contiguous canonical frames: "
                f"expected={self._next_frame_index}, actual={frame_index}"
            )
            raise ValueError(message)
        self._next_frame_index += 1
        if embeddings is None and len(boxes):
            message = "DeepEIOU requires an OSNet embedding for every person detection"
            raise ValueError(message)
        detection_rows = (
            np.concatenate((boxes, scores.reshape(-1, 1)), axis=1).astype(np.float32, copy=False)
            if len(boxes)
            else np.empty((0, 5), dtype=np.float32)
        )
        feature_rows = (
            embeddings.astype(np.float32, copy=False)
            if embeddings is not None
            else np.empty((0, 512), dtype=np.float32)
        )
        targets = self._tracker.update(detection_rows, feature_rows)
        target_boxes = np.asarray([target.last_tlbr for target in targets], dtype=np.float32)
        detection_indexes = match_target_boxes(target_boxes, boxes)
        tracked = tuple(
            TrackedDetection(
                track_id=int(target.track_id),
                detection_index=detection_indexes[index],
                bbox=target_boxes[index],
                score=float(target.score),
            )
            for index, target in enumerate(targets)
        )
        self.tracks[frame_index] = {
            item.track_id: item.bbox.astype(np.float32, copy=True) for item in tracked
        }
        self.margins[frame_index] = {
            int(target.track_id): float(target.margin) for target in targets
        }
        return tracked


class SelectiveSam3Augmenter:
    """Run selective SAM3 out-of-process and fall back safely to DeepEIOU."""

    def __init__(
        self,
        *,
        enabled: bool,
        python: Path,
        bridge: Path,
        smp_root: Path,
        timeout_seconds: int,
    ) -> None:
        """Configure the isolated SAM3 runtime and its bounded execution time."""
        self.enabled = enabled
        self.python = python.expanduser().resolve()
        self.bridge = bridge.expanduser().resolve()
        self.smp_root = smp_root.expanduser().resolve()
        self.timeout_seconds = timeout_seconds

    def augment(
        self,
        *,
        clip_path: Path,
        tracks: dict[int, dict[int, NDArray[np.float32]]],
        margins: dict[int, dict[int, float]],
    ) -> SelectiveSam3Result:
        """Return upstream rename events; any operational failure keeps base IDs."""
        if not self.enabled:
            return SelectiveSam3Result("disabled", (), 0, 0)
        missing = [path for path in (self.python, self.bridge) if not path.is_file()]
        if missing:
            message = "missing selective SAM3 runtime: " + ", ".join(str(path) for path in missing)
            LOGGER.warning(message)
            return SelectiveSam3Result("unavailable_fallback_deep_eiou", (), 0, 0, message)
        payload = {
            "tracks": {
                str(frame): {
                    str(track_id): [float(value) for value in bbox]
                    for track_id, bbox in frame_tracks.items()
                }
                for frame, frame_tracks in tracks.items()
            },
            "margins": {
                str(frame): {
                    str(track_id): (value if np.isfinite(value) else None)
                    for track_id, value in frame_margins.items()
                }
                for frame, frame_margins in margins.items()
            },
        }
        try:
            with tempfile.TemporaryDirectory(prefix="vollyai-sam3-") as temp_dir_name:
                temp_dir = Path(temp_dir_name)
                input_path = temp_dir / "input.json"
                output_path = temp_dir / "output.json"
                input_path.write_text(json.dumps(payload), encoding="utf-8")
                completed = subprocess.run(  # noqa: S603
                    [
                        str(self.python),
                        str(self.bridge),
                        "--smp-root",
                        str(self.smp_root),
                        "--clip",
                        str(clip_path.expanduser().resolve()),
                        "--input",
                        str(input_path),
                        "--output",
                        str(output_path),
                        "--workspace",
                        str(temp_dir / "sequence"),
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                )
                stderr_tail = completed.stderr[-2000:] or None
                if completed.returncode != 0 or not output_path.is_file():
                    message = stderr_tail or completed.stdout[-2000:] or "SAM3 bridge failed"
                    LOGGER.warning("selective SAM3 failed; keeping DeepEIOU IDs: %s", message)
                    return SelectiveSam3Result("failed_fallback_deep_eiou", (), 0, 0, message)
                result = json.loads(output_path.read_text(encoding="utf-8"))
                events = tuple(
                    (int(event[0]), int(event[1]), int(event[2]))
                    for event in result.get("rename_events", [])
                )
                return SelectiveSam3Result(
                    status=str(result.get("status", "completed")),
                    rename_events=events,
                    window_count=int(result.get("window_count", 0)),
                    swap_count=int(result.get("swap_count", 0)),
                    stderr_tail=stderr_tail,
                )
        except (OSError, subprocess.TimeoutExpired, ValueError, json.JSONDecodeError) as exc:
            message = f"{type(exc).__name__}: {exc}"
            LOGGER.warning("selective SAM3 failed; keeping DeepEIOU IDs: %s", message)
            return SelectiveSam3Result("failed_fallback_deep_eiou", (), 0, 0, message)


def match_target_boxes(
    target_boxes: NDArray[np.float32],
    detection_boxes: NDArray[np.float32],
) -> tuple[int, ...]:
    """Recover target-to-detection indexes without depending on tracker internals."""
    if not len(target_boxes):
        return ()
    result = [-1] * len(target_boxes)
    if not len(detection_boxes):
        return tuple(result)
    overlaps = _bbox_overlaps(target_boxes.astype(np.float64), detection_boxes.astype(np.float64))
    pairs = sorted(
        (
            (float(overlaps[row, column]), row, column)
            for row in range(overlaps.shape[0])
            for column in range(overlaps.shape[1])
        ),
        reverse=True,
    )
    used_rows: set[int] = set()
    used_columns: set[int] = set()
    for overlap, row, column in pairs:
        if overlap < MIN_TARGET_BOX_IOU:
            break
        if row in used_rows or column in used_columns:
            continue
        result[row] = column
        used_rows.add(row)
        used_columns.add(column)
    return tuple(result)


def _bbox_overlaps(
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
            0.0, query_boxes[:, 3] - query_boxes[:, 1]
        )
        union = box_area + query_area - intersection
        output[index] = np.divide(
            intersection,
            union,
            out=np.zeros_like(intersection),
            where=union > 0,
        )
    return output


def _install_cython_bbox_fallback() -> None:
    try:
        importlib.import_module("cython_bbox")
    except ModuleNotFoundError:
        pass
    else:
        return
    module = types.ModuleType("cython_bbox")
    module.bbox_overlaps = _bbox_overlaps  # type: ignore[attr-defined]
    sys.modules["cython_bbox"] = module
