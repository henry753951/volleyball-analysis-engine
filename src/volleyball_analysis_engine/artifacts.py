"""Developer-facing data and video artifacts produced from real inference."""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import cv2

from .records import ActionObservation, BallObservation, CourtFrame, FrameObservation


def write_inference_artifacts(
    *,
    output_dir: Path,
    clip_path: Path,
    frames: list[FrameObservation],
    balls: dict[int, BallObservation],
    courts: dict[int, CourtFrame],
    actions: dict[tuple[int, int], ActionObservation],
    fps: float,
) -> dict[str, str]:
    """Write inspectable JSONL and an H.264/AAC-compatible visualization."""
    output_dir.mkdir(parents=True, exist_ok=True)
    tracks_path = output_dir / "tracks.jsonl"
    ball_path = output_dir / "ball.jsonl"
    court_path = output_dir / "court.jsonl"
    action_path = output_dir / "actions.jsonl"
    overlay_path = output_dir / "overlay.mp4"
    _write_jsonl(
        tracks_path,
        (
            {
                "frame_index": frame.frame_index,
                "players": [
                    {
                        "track_id": player.track_id,
                        "source_track_id": player.source_track_id,
                        "frame_bbox": player.frame_bbox,
                        "frame_foot_pos": player.frame_foot_pos,
                        "court_pos": player.court_pos,
                        "confidence": player.confidence,
                    }
                    for player in frame.players
                ],
            }
            for frame in frames
        ),
    )
    _write_jsonl(
        ball_path,
        (
            {
                "frame_index": ball.frame_index,
                "frame_pos": ball.frame_pos,
                "confidence": ball.confidence,
            }
            for ball in balls.values()
        ),
    )
    _write_jsonl(
        court_path,
        (
            {
                "frame_index": court.frame_index,
                "available": court.available,
                "keypoints": [
                    {
                        "index": point.index,
                        "frame_pos_px": point.frame_pos_px,
                        "confidence": point.confidence,
                        "world_pos_m": point.world_pos_m,
                    }
                    for point in court.keypoints
                ],
            }
            for court in courts.values()
        ),
    )
    _write_jsonl(
        action_path,
        (
            {
                "frame_index": action.frame_index,
                "track_id": action.track_id,
                "label": action.label,
                "confidence": action.confidence,
            }
            for action in actions.values()
        ),
    )
    _render_overlay(
        clip_path=clip_path,
        output_path=overlay_path,
        frames={frame.frame_index: frame for frame in frames},
        balls=balls,
        actions=actions,
        fps=fps,
    )
    manifest_path = output_dir / "inference-manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "source": "real_clip_inference",
                "clip": str(clip_path.resolve()),
                "frame_count": len(frames),
                "fps": fps,
                "files": {
                    "tracks": tracks_path.name,
                    "ball": ball_path.name,
                    "court": court_path.name,
                    "actions": action_path.name,
                    "video": overlay_path.name,
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "tracks": str(tracks_path),
        "ball": str(ball_path),
        "court": str(court_path),
        "actions": str(action_path),
        "video": str(overlay_path),
        "manifest": str(manifest_path),
    }


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            stream.write("\n")
    temporary.replace(path)


def _render_overlay(
    *,
    clip_path: Path,
    output_path: Path,
    frames: dict[int, FrameObservation],
    balls: dict[int, BallObservation],
    actions: dict[tuple[int, int], ActionObservation],
    fps: float,
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to create overlay.mp4")
    capture = cv2.VideoCapture(str(clip_path))
    if not capture.isOpened():
        raise ValueError(f"cannot open clip for visualization: {clip_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps:.6f}",
        "-i",
        "pipe:0",
        "-i",
        str(clip_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "160k",
        "-shortest",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)  # noqa: S603
    if process.stdin is None:
        capture.release()
        process.kill()
        raise RuntimeError("failed to open ffmpeg input pipe")
    frame_index = 0
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            frame = frames.get(frame_index)
            if frame is not None:
                for player in frame.players:
                    x1 = round(player.frame_bbox[0] * width)
                    y1 = round(player.frame_bbox[1] * height)
                    x2 = round(player.frame_bbox[2] * width)
                    y2 = round(player.frame_bbox[3] * height)
                    color = _track_color(player.track_id)
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
                    action = actions.get((frame_index, player.source_track_id))
                    label = f"P{player.track_id}"
                    if action is not None:
                        label = f"{label} · {action.label}"
                    cv2.putText(
                        image,
                        label,
                        (x1, max(22, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.62,
                        color,
                        2,
                        cv2.LINE_AA,
                    )
            ball = balls.get(frame_index)
            if ball is not None:
                center = (
                    round(ball.frame_pos[0] * width),
                    round(ball.frame_pos[1] * height),
                )
                cv2.circle(image, center, 10, (0, 215, 255), 3, cv2.LINE_AA)
            process.stdin.write(image.tobytes())
            frame_index += 1
    finally:
        capture.release()
        process.stdin.close()
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg overlay render failed with exit code {return_code}")


def _track_color(track_id: int) -> tuple[int, int, int]:
    return (
        80 + (track_id * 67) % 176,
        80 + (track_id * 43) % 176,
        80 + (track_id * 29) % 176,
    )
