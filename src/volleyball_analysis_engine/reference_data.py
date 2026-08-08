"""Streaming readers for recorded detector and tracker outputs."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import orjson

from .records import BallObservation, CourtFrame, CourtKeypoint, PlayerObservation


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield non-empty JSONL records without loading the file into memory."""
    with path.open("rb") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = orjson.loads(line)
            except orjson.JSONDecodeError as exc:
                msg = f"invalid JSONL at {path}:{line_number}"
                raise ValueError(msg) from exc
            if not isinstance(value, dict):
                msg = f"JSONL record must be an object at {path}:{line_number}"
                raise TypeError(msg)
            yield cast("dict[str, Any]", value)


def load_court_frames(path: Path) -> dict[int, CourtFrame]:
    """Load recorded court detections keyed by source frame."""
    frames: dict[int, CourtFrame] = {}
    for record in read_jsonl(path):
        frame_index = int(record["frame_index"])
        keypoints = tuple(
            CourtKeypoint(
                index=int(point["index"]),
                frame_pos_px=(
                    None
                    if point.get("x_px") is None or point.get("y_px") is None
                    else (float(point["x_px"]), float(point["y_px"]))
                ),
                confidence=(
                    None if point.get("confidence") is None else float(point["confidence"])
                ),
                world_pos_m=(
                    None
                    if point.get("world_pos_m") is None
                    else (
                        float(point["world_pos_m"][0]),
                        float(point["world_pos_m"][1]),
                    )
                ),
            )
            for point in cast("list[dict[str, Any]]", record["keypoints"])
        )
        frames[frame_index] = CourtFrame(
            frame_index=frame_index,
            available=bool(record.get("available", False)),
            keypoints=keypoints,
        )
    return frames


def load_player_frames(path: Path) -> dict[int, tuple[PlayerObservation, ...]]:
    """Load normalized player tracker output keyed by source frame."""
    frames: dict[int, tuple[PlayerObservation, ...]] = {}
    for record in read_jsonl(path):
        frame_index = int(record["frame_index"])
        observations: list[PlayerObservation] = []
        for player in cast("list[dict[str, Any]]", record["players"]):
            bbox = cast("dict[str, Any]", player["frame_bbox"])
            foot = cast("dict[str, Any]", player["frame_foot_pos"])
            court = cast("dict[str, Any] | None", player.get("court_pos"))
            track_id = int(player["track_id"])
            observations.append(
                PlayerObservation(
                    frame_index=frame_index,
                    source_track_id=track_id,
                    track_id=track_id,
                    frame_bbox=(
                        float(bbox["x1"]),
                        float(bbox["y1"]),
                        float(bbox["x2"]),
                        float(bbox["y2"]),
                    ),
                    frame_foot_pos=(float(foot["x"]), float(foot["y"])),
                    court_pos=(None if court is None else (float(court["x"]), float(court["y"]))),
                    confidence=(
                        None if player.get("confidence") is None else float(player["confidence"])
                    ),
                )
            )
        frames[frame_index] = tuple(observations)
    return frames


def load_ball_positions(path: Path) -> dict[int, BallObservation]:
    """Load the temporary manual ball JSON accepted for the first integration."""
    raw_payload = orjson.loads(path.read_bytes())
    if not isinstance(raw_payload, dict):
        msg = "ball fixture must be an object"
        raise TypeError(msg)
    payload = cast("dict[str, Any]", raw_payload)
    if payload.get("coordinate_space") != "normalized-video-frame":
        msg = "ball fixture must use normalized-video-frame"
        raise ValueError(msg)
    result: dict[int, BallObservation] = {}
    for point in cast("list[dict[str, Any]]", payload["points"]):
        frame_pos = cast("dict[str, Any]", point["frame_pos"])
        frame_index = int(point["clip_frame_index"])
        result[frame_index] = BallObservation(
            frame_index=frame_index,
            frame_pos=(float(frame_pos["x"]), float(frame_pos["y"])),
        )
    return result
