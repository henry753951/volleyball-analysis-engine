"""Out-of-process selective SAM3 bridge for the analysis worker."""

from __future__ import annotations

import argparse
import configparser
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smp-root", type=Path, required=True)
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    return parser


def _decode_sequence(clip: Path, workspace: Path, expected_frames: int) -> None:
    if workspace.exists():
        shutil.rmtree(workspace)
    image_dir = workspace / "img1"
    image_dir.mkdir(parents=True)
    capture = cv2.VideoCapture(str(clip.resolve(strict=True)))
    if not capture.isOpened():
        message = f"cannot open canonical clip: {clip}"
        raise RuntimeError(message)
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            output = image_dir / f"{frame_index:08d}.jpg"
            if not cv2.imwrite(str(output), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                message = f"failed to write SAM3 source frame: {output}"
                raise RuntimeError(message)
            frame_index += 1
    finally:
        capture.release()
    if frame_index != expected_frames:
        message = f"SAM3 decode frame mismatch: expected={expected_frames}, actual={frame_index}"
        raise RuntimeError(message)
    config = configparser.ConfigParser()
    config["Sequence"] = {
        "name": clip.stem,
        "imDir": "img1",
        "frameRate": str(max(1, round(fps))),
        "seqLength": str(frame_index),
        "imWidth": str(width),
        "imHeight": str(height),
        "imExt": ".jpg",
    }
    with (workspace / "seqinfo.ini").open("w", encoding="utf-8") as handle:
        config.write(handle)


def _arrays(
    payload: dict[str, Any],
) -> tuple[dict[int, dict[int, np.ndarray]], dict[int, dict[int, float]]]:
    tracks = {
        int(frame): {
            int(track_id): np.asarray(bbox, dtype=np.float64)
            for track_id, bbox in frame_tracks.items()
        }
        for frame, frame_tracks in payload["tracks"].items()
    }
    margins = {
        int(frame): {
            int(track_id): (float(value) if value is not None else float("inf"))
            for track_id, value in frame_margins.items()
        }
        for frame, frame_margins in payload["margins"].items()
    }
    return tracks, margins


def main() -> None:
    """Run selective SAM3 only when the upstream window detector finds ambiguity."""
    arguments = _parser().parse_args()
    smp_root = str(arguments.smp_root.expanduser().resolve(strict=True))
    if smp_root not in sys.path:
        sys.path.insert(0, smp_root)
    payload = json.loads(arguments.input.read_text(encoding="utf-8"))
    tracks, margins = _arrays(payload)
    from selective_mask_propagation.core.sam2 import WindowOutcome, find_windows

    specs = find_windows(tracks, margins)
    if not specs:
        result = {
            "status": "no_ambiguous_windows",
            "window_count": 0,
            "swap_count": 0,
            "rename_events": [],
        }
    else:
        expected_frames = max(tracks, default=-1) + 1
        _decode_sequence(arguments.clip, arguments.workspace, expected_frames)
        from selective_mask_propagation.core.sam3 import build_predictor, run_sam

        predictor = build_predictor()
        _masks, windows, rename_events, _history = run_sam(
            predictor,
            specs,
            tracks,
            margins,
            str(arguments.workspace),
        )
        result = {
            "status": "completed",
            "window_count": len(windows),
            "swap_count": sum(window.outcome == WindowOutcome.SWAP for window in windows),
            "rename_events": [list(event) for event in rename_events],
        }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result), encoding="utf-8")


if __name__ == "__main__":
    main()
