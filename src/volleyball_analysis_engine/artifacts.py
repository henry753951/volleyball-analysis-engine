"""Developer-facing data and video artifacts produced from real inference."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from volleyball_monitoring_ai import AIJobRequest, AnalysisResult

from .records import ActionObservation, BallObservation, CourtFrame, FrameObservation
from .visual_v5 import write_visual_v5_package


def write_inference_artifacts(
    *,
    output_dir: Path,
    clip_path: Path,
    job: AIJobRequest,
    result: AnalysisResult,
    frames: list[FrameObservation],
    balls: dict[int, BallObservation],
    courts: dict[int, CourtFrame],
    actions: dict[tuple[int, int], ActionObservation],
    fps: float,
    frame_width: int,
    frame_height: int,
) -> dict[str, str]:
    """Write raw observations plus the Contract Lab visual-v5 package."""
    output_dir.mkdir(parents=True, exist_ok=True)
    tracks_path = output_dir / "tracks.jsonl"
    ball_path = output_dir / "ball.jsonl"
    court_path = output_dir / "court.jsonl"
    action_path = output_dir / "actions.jsonl"
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
    visual_paths = write_visual_v5_package(
        output_dir=output_dir,
        clip_path=clip_path,
        job=job,
        result=result,
        frames=frames,
        balls=balls,
        courts=courts,
        fps=fps,
        frame_width=frame_width,
        frame_height=frame_height,
        action_frame_count=len({action.frame_index for action in actions.values()}),
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
                    "video": Path(visual_paths["video"]).name,
                    "result": Path(visual_paths["result"]).name,
                    "visual_manifest": Path(visual_paths["manifest"]).name,
                    "preview_first_complete": Path(visual_paths["preview_first_complete"]).name,
                    "preview_terminal_path": Path(visual_paths["preview_terminal_path"]).name,
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
        "video": visual_paths["video"],
        "result": visual_paths["result"],
        "visual_manifest": visual_paths["manifest"],
        "preview_first_complete": visual_paths["preview_first_complete"],
        "preview_terminal_path": visual_paths["preview_terminal_path"],
        "manifest": str(manifest_path),
    }


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            stream.write("\n")
    temporary.replace(path)
