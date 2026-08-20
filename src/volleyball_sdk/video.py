"""Sequential OpenCV video sampling with exact model temporal spacing."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


def _sample_offsets(frame_num: int, jump_frame: int, sampling_mode: str) -> list[int]:
    if frame_num <= 0 or jump_frame <= 0:
        raise ValueError("frame_num and jump_frame must be > 0")
    mode = str(sampling_mode).lower()
    if mode == "centered":
        half = frame_num // 2
        return [(i - half) * jump_frame for i in range(frame_num)]
    if mode == "causal":
        return [-(frame_num - 1 - i) * jump_frame for i in range(frame_num)]
    raise ValueError(f"Unsupported sampling_mode: {sampling_mode!r}")


def iter_video_clips(
    video_path: str | Path,
    *,
    frame_num: int,
    jump_frame: int,
    sampling_mode: str,
    step: int = 1,
) -> Iterable[tuple[int, float, list[np.ndarray], tuple[int, int]]]:
    """Yield ``(frame_index, timestamp_sec, clip_bgr, (width,height))``.

    Frames are decoded once sequentially. Border samples are clamped to the
    first/last frame, matching the existing GUI seek behavior.
    """

    if step <= 0:
        raise ValueError("step must be > 0")
    path = Path(video_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Video not found: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    fps = fps if fps > 0 else 1.0
    offsets = _sample_offsets(frame_num, jump_frame, sampling_mode)
    min_offset, max_offset = min(offsets), max(offsets)
    retention = max_offset - min_offset + 1
    frames: deque[tuple[int, np.ndarray]] = deque(maxlen=max(1, retention))
    first_frame: np.ndarray | None = None
    last_frame: np.ndarray | None = None
    next_target = 0
    last_index = -1

    def get_frame(index: int, final_index: int | None = None) -> np.ndarray:
        nonlocal first_frame, last_frame
        if index <= 0 and first_frame is not None:
            return first_frame
        if final_index is not None and index >= final_index and last_frame is not None:
            return last_frame
        for frame_index, frame in frames:
            if frame_index == index:
                return frame
        raise RuntimeError(f"Internal video buffer miss for frame {index}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            last_index += 1
            if first_frame is None:
                first_frame = frame.copy()
            last_frame = frame.copy()
            frames.append((last_index, frame.copy()))

            # A target is ready when all non-clamped future context has arrived.
            while next_target <= last_index and next_target + max_offset <= last_index:
                indices = [max(0, next_target + offset) for offset in offsets]
                clip = [get_frame(index).copy() for index in indices]
                h, w = clip[frame_num // 2 if sampling_mode == "centered" else -1].shape[:2]
                yield next_target, next_target / fps, clip, (w, h)
                next_target += step

        if last_index < 0:
            raise RuntimeError(f"Video contains no decodable frames: {path}")

        # Flush final targets using last-frame clamping for future context.
        while next_target <= last_index:
            indices = [min(last_index, max(0, next_target + offset)) for offset in offsets]
            clip = [get_frame(index, final_index=last_index).copy() for index in indices]
            output_slot = frame_num // 2 if sampling_mode == "centered" else frame_num - 1
            h, w = clip[output_slot].shape[:2]
            yield next_target, next_target / fps, clip, (w, h)
            next_target += step
    finally:
        cap.release()
