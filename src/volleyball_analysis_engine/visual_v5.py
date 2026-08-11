"""Contract Lab visual-v5 compatible rendering backed by real inference data."""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
from volleyball_monitoring_ai import AIJobRequest, AnalysisResult

from .records import BallObservation, CourtFrame, FrameObservation

OUTPUT_WIDTH = 1920
OUTPUT_HEIGHT = 1080
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720
INFO_LEFT = VIDEO_WIDTH
COURT_TOP = VIDEO_HEIGHT
PATH_SCROLL_TRANSITION_FRAMES = 12


def _frame_point(
    value: Mapping[str, float] | tuple[float, float],
    width: int,
    height: int,
) -> tuple[int, int]:
    x, y = value if isinstance(value, tuple) else (float(value["x"]), float(value["y"]))
    x = max(0.0, min(1.0, float(x)))
    y = max(0.0, min(1.0, float(y)))
    return round(x * (width - 1)), round(y * (height - 1))


def _court_point(
    value: Mapping[str, float] | tuple[float, float],
    rect: tuple[int, int, int, int],
) -> tuple[int, int]:
    left, top, width, height = rect
    x, y = value if isinstance(value, tuple) else (float(value["x"]), float(value["y"]))
    # AI-owned court coordinates intentionally remain unclamped.
    return round(left + float(x) * width), round(top + float(y) * height)


def _inside(point: tuple[int, int], rect: tuple[int, int, int, int]) -> bool:
    left, top, width, height = rect
    return left <= point[0] <= left + width and top <= point[1] <= top + height


def _clip_line(
    start: tuple[int, int],
    end: tuple[int, int],
    rect: tuple[int, int, int, int],
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    clipped, clipped_start, clipped_end = cv2.clipLine(rect, start, end)
    if not clipped:
        return None
    return (
        (int(clipped_start[0]), int(clipped_start[1])),
        (int(clipped_end[0]), int(clipped_end[1])),
    )


def _text(
    frame: Any,
    value: str,
    at: tuple[int, int],
    scale: float = 0.55,
    color: tuple[int, int, int] = (255, 255, 255),
    thickness: int = 1,
) -> None:
    cv2.putText(
        frame,
        value,
        at,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness + 3,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        value,
        at,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _boxed_text(
    frame: Any,
    value: str,
    at: tuple[int, int],
    *,
    foreground: tuple[int, int, int],
    background: tuple[int, int, int],
    scale: float = 0.34,
    background_alpha: float = 0.78,
) -> None:
    (text_width, text_height), baseline = cv2.getTextSize(
        value,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        1,
    )
    x = max(2, min(VIDEO_WIDTH - text_width - 10, at[0]))
    y = max(text_height + 6, min(VIDEO_HEIGHT - baseline - 4, at[1]))
    layer = frame.copy()
    cv2.rectangle(
        layer,
        (x - 4, y - text_height - 4),
        (x + text_width + 5, y + baseline + 3),
        background,
        -1,
    )
    cv2.addWeighted(layer, background_alpha, frame, 1.0 - background_alpha, 0, frame)
    cv2.putText(
        frame,
        value,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        foreground,
        1,
        cv2.LINE_AA,
    )


def _track_color(track_id: int) -> tuple[int, int, int]:
    palette = (
        (80, 255, 120),
        (255, 150, 80),
        (120, 220, 255),
        (220, 120, 255),
        (120, 255, 220),
        (255, 220, 120),
    )
    return palette[track_id % len(palette)]


def _action_color(label: str) -> tuple[int, int, int]:
    return {
        "serving": (80, 240, 145),
        "spiking": (50, 165, 255),
        "passing": (255, 210, 75),
        "digging": (255, 210, 75),
        "setting": (220, 145, 255),
        "blocking": (100, 205, 255),
        "falling": (100, 145, 255),
        "moving": (160, 215, 255),
        "standing": (175, 185, 195),
        "ball_control": (220, 145, 255),
        "ready_position": (175, 185, 195),
    }.get(label, (175, 185, 195))


def _path_color(index: int, total: int) -> tuple[int, int, int]:
    progress = (index + 1) / max(total, 1)
    return (
        round(28 + progress * 42),
        round(105 + progress * 105),
        round(185 + progress * 70),
    )


def _side_letter(side: str) -> str:
    return {"left": "L", "right": "R"}.get(side.lower(), "?")


def _draw_side_badge(
    frame: Any,
    side: str,
    at: tuple[int, int],
    *,
    width: int = 28,
    height: int = 20,
) -> None:
    """Draw a compact court-side badge with a net-like hatch."""
    x = max(0, min(frame.shape[1] - width, at[0]))
    y = max(0, min(frame.shape[0] - height, at[1]))
    accent = {
        "left": (70, 190, 245),
        "right": (235, 170, 90),
    }.get(side.lower(), (145, 155, 165))
    badge = np.full((height, width, 3), (18, 24, 29), dtype=np.uint8)
    hatch = tuple(round(channel * 0.58) for channel in accent)
    for offset in range(-height, width + height, 6):
        cv2.line(
            badge,
            (offset, height - 1),
            (offset + height, 0),
            hatch,
            1,
            cv2.LINE_AA,
        )
    cv2.rectangle(badge, (0, 0), (width - 1, height - 1), accent, 1, cv2.LINE_AA)
    label = _side_letter(side)
    (text_width, text_height), _ = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        1,
    )
    cv2.putText(
        badge,
        label,
        ((width - text_width) // 2, (height + text_height) // 2 - 1),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (248, 250, 252),
        1,
        cv2.LINE_AA,
    )
    frame[y : y + height, x : x + width] = badge


def path_scroll_offset(
    segments: list[dict[str, Any]],
    frame_index: int,
    *,
    visible_rows: int = 7,
    transition_frames: int = PATH_SCROLL_TRANSITION_FRAMES,
) -> float:
    """Return a smooth row offset that keeps the latest completed path visible."""
    if len(segments) <= visible_rows:
        return 0.0
    completed = sum(int(segment.get("end_frame_index", 0)) <= frame_index for segment in segments)
    target = min(max(0, completed - visible_rows), len(segments) - visible_rows)
    if target <= 0:
        return 0.0
    trigger_index = target + visible_rows - 1
    trigger_frame = int(segments[trigger_index].get("end_frame_index", frame_index))
    progress = max(0.0, min(1.0, (frame_index - trigger_frame) / max(transition_frames, 1)))
    eased = progress * progress * (3.0 - 2.0 * progress)
    return (target - 1) + eased


def _path_position_side(
    position: dict[str, Any] | None,
    track_sides: dict[int, str],
) -> str:
    if position is None:
        return "unknown"
    track_id = position.get("track_id")
    if track_id is not None:
        return track_sides.get(int(track_id), "unknown")
    court_pos = position.get("court_pos")
    if court_pos is None:
        return "unknown"
    return "left" if float(court_pos["x"]) < 0.5 else "right"


def _draw_court_keypoints(
    frame: Any,
    court: CourtFrame | None,
    source_width: int,
    source_height: int,
    confidence_threshold: float = 0.25,
) -> bool:
    if court is None or not court.available:
        return False
    layer = frame.copy()
    drawn = False
    label_boxes: list[tuple[int, int, int, int]] = []
    for keypoint in court.keypoints:
        if keypoint.frame_pos_px is None:
            continue
        if keypoint.confidence is not None and keypoint.confidence < confidence_threshold:
            continue
        source_x, source_y = keypoint.frame_pos_px
        x = round(source_x / max(source_width, 1) * (VIDEO_WIDTH - 1))
        y = round(source_y / max(source_height, 1) * (VIDEO_HEIGHT - 1))
        if not (0 <= x < VIDEO_WIDTH and 0 <= y < VIDEO_HEIGHT):
            continue
        color = (210, 195, 90)
        diamond = np.asarray(
            [(x, y - 5), (x + 5, y), (x, y + 5), (x - 5, y)],
            dtype=np.int32,
        )
        cv2.fillConvexPoly(layer, diamond, (18, 28, 34), cv2.LINE_AA)
        cv2.polylines(layer, [diamond], True, color, 1, cv2.LINE_AA)
        cv2.line(layer, (x - 2, y), (x + 2, y), color, 1, cv2.LINE_AA)
        cv2.line(layer, (x, y - 2), (x, y + 2), color, 1, cv2.LINE_AA)
        label = f"K{keypoint.index + 1:02d}"
        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.30,
            1,
        )
        candidates = (
            (9, -7),
            (9, 19),
            (-text_width - 13, -7),
            (-text_width - 13, 19),
            (9, 35),
            (-text_width - 13, 35),
        )
        chosen: tuple[int, int, int, int, int, int] | None = None
        for offset_x, offset_y in candidates:
            label_x = max(3, min(VIDEO_WIDTH - text_width - 6, x + offset_x))
            label_y = max(
                text_height + 4,
                min(VIDEO_HEIGHT - baseline - 3, y + offset_y),
            )
            box = (
                label_x - 3,
                label_y - text_height - 3,
                label_x + text_width + 4,
                label_y + baseline + 2,
            )
            overlaps = any(
                box[0] < other[2] + 3
                and box[2] + 3 > other[0]
                and box[1] < other[3] + 3
                and box[3] + 3 > other[1]
                for other in label_boxes
            )
            if not overlaps:
                chosen = (*box, label_x, label_y)
                break
        if chosen is None:
            offset_x, offset_y = candidates[keypoint.index % len(candidates)]
            label_x = max(3, min(VIDEO_WIDTH - text_width - 6, x + offset_x))
            label_y = max(
                text_height + 4,
                min(VIDEO_HEIGHT - baseline - 3, y + offset_y),
            )
            chosen = (
                label_x - 3,
                label_y - text_height - 3,
                label_x + text_width + 4,
                label_y + baseline + 2,
                label_x,
                label_y,
            )
        box_left, box_top, box_right, box_bottom, label_x, label_y = chosen
        if abs(label_x - x) > 12 or abs(label_y - y) > 18:
            cv2.line(
                layer,
                (x, y),
                (label_x, label_y - text_height // 2),
                color,
                1,
                cv2.LINE_AA,
            )
        cv2.rectangle(
            layer,
            (box_left, box_top),
            (box_right, box_bottom),
            (10, 18, 23),
            -1,
        )
        cv2.putText(
            layer,
            label,
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.30,
            color,
            1,
            cv2.LINE_AA,
        )
        label_boxes.append((box_left, box_top, box_right, box_bottom))
        drawn = True
    if drawn:
        cv2.addWeighted(layer, 0.42, frame, 0.58, 0, frame)
    return drawn


def _action_states_for_frame(
    events: list[dict[str, Any]],
    frame_index: int,
    fps: float,
) -> dict[int, tuple[str, bool]]:
    states: dict[int, tuple[str, bool]] = {}
    duration_frames = max(1, round(fps))
    highlight_frames = max(1, round(fps * 0.10))
    for event in events:
        event_frame = int(event.get("resolved_frame_index", event["anchor_frame_index"]))
        if not event_frame <= frame_index < event_frame + duration_frames:
            continue
        for actor in event.get("actors", []):
            action = actor.get("action")
            if action:
                states[int(actor["track_id"])] = (
                    str(action["label"]),
                    frame_index - event_frame <= highlight_frames,
                )
    return states


def _draw_tracking(
    frame: Any,
    observation: FrameObservation | None,
    action_states: dict[int, tuple[str, bool]],
) -> None:
    if observation is None or not observation.players:
        return
    layer = frame.copy()
    for player in observation.players:
        top_left = _frame_point(player.frame_bbox[:2], VIDEO_WIDTH, VIDEO_HEIGHT)
        bottom_right = _frame_point(player.frame_bbox[2:], VIDEO_WIDTH, VIDEO_HEIGHT)
        color = _track_color(player.track_id)
        cv2.rectangle(layer, top_left, bottom_right, color, 2, cv2.LINE_AA)
        foot = _frame_point(player.frame_foot_pos, VIDEO_WIDTH, VIDEO_HEIGHT)
        cv2.circle(layer, foot, 4, color, -1, cv2.LINE_AA)
    cv2.addWeighted(layer, 0.46, frame, 0.54, 0, frame)
    for player in observation.players:
        top_left = _frame_point(player.frame_bbox[:2], VIDEO_WIDTH, VIDEO_HEIGHT)
        bottom_right = _frame_point(player.frame_bbox[2:], VIDEO_WIDTH, VIDEO_HEIGHT)
        label, highlight = action_states.get(
            player.track_id,
            ("ready_position", False),
        )
        display_label = {
            "ready_position": "READY",
            "ball_control": "CONTROL",
        }.get(label, label.upper())
        color = _action_color(label)
        label_y = max(21, top_left[1] - 5)
        if label != "ready_position":
            cv2.rectangle(
                frame,
                top_left,
                bottom_right,
                (0, 255, 255) if highlight else color,
                4 if highlight else 2,
                cv2.LINE_AA,
            )
        _draw_side_badge(
            frame,
            player.court_side,
            (top_left[0] + 2, label_y - 18),
            width=26,
            height=18,
        )
        _boxed_text(
            frame,
            f"T{player.track_id} | {display_label}",
            (top_left[0] + 32, label_y),
            foreground=(15, 20, 24) if highlight else (245, 248, 250),
            background=(
                (0, 255, 255)
                if highlight
                else ((34, 74, 88) if label == "ready_position" else color)
            ),
            scale=0.27 if label == "ready_position" else 0.34,
            background_alpha=1.0 if highlight else (0.48 if label == "ready_position" else 0.76),
        )


def _draw_dashed_circle(
    frame: Any,
    center: tuple[int, int],
    radius: int,
    color: tuple[int, int, int],
    phase_degrees: int,
) -> None:
    for start in range(phase_degrees, phase_degrees + 360, 32):
        cv2.ellipse(
            frame,
            center,
            (radius, radius),
            0,
            start,
            start + 17,
            color,
            2,
            cv2.LINE_AA,
        )


def _draw_ball(
    frame: Any,
    balls: dict[int, BallObservation],
    events: list[dict[str, Any]],
    frame_index: int,
    fps: float,
) -> bool:
    frame_ball = balls.get(frame_index)
    if frame_ball is not None:
        trail_layer = frame.copy()
        previous_point: tuple[int, int] | None = None
        for offset in range(8, -1, -1):
            trail = balls.get(frame_index - offset)
            if trail is None:
                continue
            point = _frame_point(trail.frame_pos, VIDEO_WIDTH, VIDEO_HEIGHT)
            if previous_point is not None:
                cv2.line(
                    trail_layer,
                    previous_point,
                    point,
                    (0, 175, 255),
                    2,
                    cv2.LINE_AA,
                )
            previous_point = point
        cv2.addWeighted(trail_layer, 0.28, frame, 0.72, 0, frame)
        center = _frame_point(frame_ball.frame_pos, VIDEO_WIDTH, VIDEO_HEIGHT)
        cv2.circle(frame, center, 11, (18, 36, 48), -1, cv2.LINE_AA)
        cv2.circle(frame, center, 8, (0, 205, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, center, 4, (245, 252, 255), -1, cv2.LINE_AA)

    preview_frames = max(1, round(fps * 0.70))
    upcoming = next(
        (
            event
            for event in events
            if 0
            <= int(event.get("resolved_frame_index", event["anchor_frame_index"])) - frame_index
            <= preview_frames
            and event.get("ball", {}).get("frame_pos")
        ),
        None,
    )
    if upcoming is not None:
        event_frame = int(upcoming.get("resolved_frame_index", upcoming["anchor_frame_index"]))
        frames_until = event_frame - frame_index
        center = _frame_point(
            upcoming["ball"]["frame_pos"],
            VIDEO_WIDTH,
            VIDEO_HEIGHT,
        )
        progress = 1.0 - frames_until / preview_frames
        radius = round(21 - progress * 7)
        phase = round(frame_index * 7) % 32
        if frames_until <= max(1, round(fps * 0.10)):
            cv2.circle(frame, center, radius + 5, (0, 255, 255), 3, cv2.LINE_AA)
            _boxed_text(
                frame,
                "HIT NOW",
                (center[0] + radius + 7, center[1] - radius),
                foreground=(15, 20, 24),
                background=(0, 255, 255),
                scale=0.36,
            )
        else:
            _draw_dashed_circle(frame, center, radius, (255, 225, 90), phase)
            _boxed_text(
                frame,
                "NEXT HIT",
                (center[0] + radius + 6, center[1] - radius),
                foreground=(240, 248, 255),
                background=(62, 72, 78),
                scale=0.32,
            )
    return frame_ball is not None


def _draw_info_panel(
    frame: Any,
    event: dict[str, Any],
    point: dict[str, Any],
    frame_index: int,
    fps: float,
    tracking_count: int,
) -> None:
    cv2.rectangle(
        frame,
        (INFO_LEFT, 0),
        (OUTPUT_WIDTH - 1, VIDEO_HEIGHT - 1),
        (15, 20, 25),
        -1,
    )
    cv2.line(frame, (INFO_LEFT, 0), (INFO_LEFT, VIDEO_HEIGHT), (54, 65, 75), 1)
    _text(
        frame,
        "VOLLEYBALL AI ANALYSIS",
        (INFO_LEFT + 24, 38),
        0.72,
        (95, 220, 255),
        2,
    )
    _text(
        frame,
        "tracking: REAL MODEL + REID",
        (INFO_LEFT + 24, 68),
        0.42,
        (110, 240, 160),
    )
    _text(
        frame,
        "action: RT-DETRV4-X3D",
        (INFO_LEFT + 322, 68),
        0.38,
        (120, 170, 255),
    )
    _text(
        frame,
        f"frame {frame_index}  {frame_index / fps:.3f}s",
        (INFO_LEFT + 24, 104),
        0.48,
        (190, 200, 210),
    )
    cv2.line(
        frame,
        (INFO_LEFT + 24, 122),
        (OUTPUT_WIDTH - 24, 122),
        (55, 64, 73),
        1,
    )

    actors = event.get("actors", [])
    actor = actors[0] if actors else None
    action = actor.get("action", {}).get("label", "n/a") if actor else "n/a"
    actor_id = str(actor["track_id"]) if actor else "NO PLAYER"
    ball = event.get("ball", {}).get("frame_pos")
    representative = event.get("representative_court_positions", [])
    court = representative[0]["court_pos"] if representative else None
    lines = [
        (
            f"EVENT #{event.get('sequence_index', 'n/a')}  {event.get('marker_kind', 'n/a')}",
            (255, 255, 255),
        ),
        (f"anchor frame  {event.get('anchor_frame_index', 'n/a')}", (180, 190, 200)),
        (f"resolved frame  {event.get('resolved_frame_index', 'n/a')}", (180, 190, 200)),
        (f"who hit  {actor_id}", (95, 240, 150) if actor else (100, 170, 255)),
        (f"association  {event.get('association_state', 'n/a')}", (180, 190, 200)),
        (f"action  {action}", (120, 170, 255)),
        (
            f"ball  ({ball['x']:.4f}, {ball['y']:.4f})" if ball else "ball  missing",
            (80, 220, 255),
        ),
        (
            f"court  ({court['x']:.4f}, {court['y']:.4f})" if court else "court  unavailable",
            (255, 200, 100),
        ),
        (f"players on frame  {tracking_count}", (180, 190, 200)),
        (f"keypoint  {str(point.get('key_point_id', 'n/a'))[:18]}...", (150, 160, 170)),
    ]
    for index, (line, color) in enumerate(lines):
        _text(frame, line, (INFO_LEFT + 24, 162 + index * 39), 0.52, color)

    _text(frame, "QUALITY", (INFO_LEFT + 24, 548), 0.48, (170, 180, 190))
    for index, flag in enumerate(event.get("quality_flags", [])[:4]):
        _text(
            frame,
            f"- {flag}",
            (INFO_LEFT + 24, 578 + index * 24),
            0.39,
            (150, 195, 225),
        )


def _draw_path_badge(
    frame: Any,
    value: str,
    point: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    (text_width, text_height), baseline = cv2.getTextSize(
        value,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.28,
        1,
    )
    x = max(2, min(648 - text_width - 8, point[0] + 6))
    y = max(
        COURT_TOP + text_height + 4,
        min(OUTPUT_HEIGHT - baseline - 3, point[1] - 7),
    )
    cv2.rectangle(
        frame,
        (x - 3, y - text_height - 3),
        (x + text_width + 4, y + baseline + 2),
        (8, 13, 17),
        -1,
    )
    cv2.rectangle(
        frame,
        (x - 3, y - text_height - 3),
        (x + text_width + 4, y + baseline + 2),
        color,
        1,
    )
    cv2.putText(
        frame,
        value,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.28,
        color,
        1,
        cv2.LINE_AA,
    )


def _draw_court_panel(
    frame: Any,
    result: dict[str, Any],
    frame_index: int,
    observation: FrameObservation | None,
    ball_court: tuple[float, float] | None,
) -> None:
    cv2.rectangle(
        frame,
        (0, COURT_TOP),
        (OUTPUT_WIDTH - 1, OUTPUT_HEIGHT - 1),
        (10, 14, 18),
        -1,
    )
    cv2.line(frame, (0, COURT_TOP), (OUTPUT_WIDTH, COURT_TOP), (59, 72, 82), 1)
    path_clip_rect = (12, COURT_TOP + 8, 658, OUTPUT_HEIGHT - COURT_TOP - 18)
    stage_left, stage_top, stage_width, stage_height = path_clip_rect
    cv2.rectangle(
        frame,
        (stage_left, stage_top),
        (stage_left + stage_width, stage_top + stage_height),
        (15, 24, 29),
        -1,
    )
    cv2.rectangle(
        frame,
        (stage_left, stage_top),
        (stage_left + stage_width, stage_top + stage_height),
        (45, 58, 66),
        1,
    )
    court_rect = (28, COURT_TOP + 60, 560, 280)
    left, top, width, height = court_rect
    right, bottom = left + width, top + height
    cv2.rectangle(frame, (left, top), (right, bottom), (34, 76, 89), -1)
    cv2.rectangle(frame, (left, top), (right, bottom), (220, 235, 242), 2)
    for ratio in (1 / 3, 2 / 3):
        x = left + round(width * ratio)
        cv2.line(frame, (x, top), (x, bottom), (155, 180, 190), 1, cv2.LINE_AA)
    net_x = left + width // 2
    cv2.line(frame, (net_x, top), (net_x, bottom), (80, 210, 255), 3, cv2.LINE_AA)

    players = observation.players if observation is not None else ()
    for player in players:
        if player.court_pos is None:
            continue
        point = _court_point(player.court_pos, court_rect)
        if not _inside(point, court_rect):
            continue
        color = _track_color(player.track_id)
        cv2.circle(frame, point, 6, (9, 15, 19), -1, cv2.LINE_AA)
        cv2.circle(frame, point, 4, color, -1, cv2.LINE_AA)
        _text(
            frame,
            str(player.track_id),
            (point[0] + 6, point[1] + 4),
            0.30,
            (205, 218, 225),
        )
    if ball_court is not None:
        ball_point = _court_point(ball_court, court_rect)
        if _inside(ball_point, path_clip_rect):
            cv2.circle(frame, ball_point, 8, (0, 220, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, ball_point, 11, (255, 255, 255), 1, cv2.LINE_AA)
            _text(
                frame,
                "BALL",
                (ball_point[0] + 10, ball_point[1] + 4),
                0.34,
                (0, 220, 255),
            )

    visible_segments = [
        segment
        for segment in result.get("path_segments", [])
        if int(segment.get("end_frame_index", 0)) <= frame_index
    ]
    path_outline = frame.copy()
    path_color_layer = frame.copy()
    path_badges: list[tuple[str, tuple[int, int], tuple[int, int, int]]] = []
    for path_index, segment in enumerate(visible_segments):
        start_positions = segment.get("start_court_positions", [])
        end_positions = segment.get("end_court_positions", [])
        if not start_positions or not end_positions:
            continue
        start = _court_point(start_positions[0]["court_pos"], court_rect)
        end = _court_point(end_positions[0]["court_pos"], court_rect)
        clipped_line = _clip_line(start, end, path_clip_rect)
        if clipped_line is None:
            continue
        clipped_start, clipped_end = clipped_line
        path_color = _path_color(path_index, len(visible_segments))
        cv2.arrowedLine(
            path_outline,
            clipped_start,
            clipped_end,
            (4, 8, 12),
            7,
            cv2.LINE_AA,
            tipLength=0.055,
        )
        cv2.arrowedLine(
            path_color_layer,
            clipped_start,
            clipped_end,
            path_color,
            3,
            cv2.LINE_AA,
            tipLength=0.055,
        )
        cv2.circle(path_color_layer, clipped_start, 5, (7, 12, 16), -1, cv2.LINE_AA)
        cv2.circle(path_color_layer, clipped_start, 4, path_color, 2, cv2.LINE_AA)
        sequence = segment["sequence_index"]
        if _inside(start, path_clip_rect):
            path_badges.append((f"A{sequence}", start, path_color))
        end_label = "LAND" if end_positions[0].get("track_id") is None else f"B{sequence}"
        if _inside(end, path_clip_rect):
            badge_color = (255, 120, 255) if end_label == "LAND" else path_color
            path_badges.append((end_label, end, badge_color))
    cv2.addWeighted(path_outline, 0.72, frame, 0.28, 0, frame)
    cv2.addWeighted(path_color_layer, 0.88, frame, 0.12, 0, frame)
    for label, position, color in path_badges:
        _draw_path_badge(frame, label, position, color)

    _text(
        frame,
        "CANONICAL COURT 18m x 9m",
        (24, COURT_TOP + 28),
        0.52,
        (185, 220, 245),
    )
    panel_left = 690
    _text(
        frame,
        "A / B PATH RESOLUTION  |  COURT SIDE",
        (panel_left, COURT_TOP + 30),
        0.52,
        (185, 220, 245),
    )
    _text(
        frame,
        "A = hitter footprint | B = next hitter footprint or projected landing",
        (panel_left, COURT_TOP + 58),
        0.42,
        (145, 160, 175),
    )
    segments = cast("list[dict[str, Any]]", result.get("path_segments", []))
    track_sides = {
        int(track["track_id"]): str(track.get("court_side", "unknown"))
        for track in result.get("tracks", [])
    }
    row_spacing = 32
    visible_rows = 7
    scroll_offset = path_scroll_offset(
        segments,
        frame_index,
        visible_rows=visible_rows,
    )
    for index, segment in enumerate(segments):
        row_position = index - scroll_offset
        if not -0.75 <= row_position <= visible_rows - 0.15:
            continue
        row_y = COURT_TOP + 96 + round(row_position * row_spacing)
        start = segment.get("start_court_positions", [])
        end = segment.get("end_court_positions", [])
        start_position = start[0] if start else None
        end_position = end[0] if end else None
        start_id = start[0].get("track_id") if start else None
        end_id = end[0].get("track_id") if end else None
        start_label = f"T{start_id}" if start_id is not None else "n/a"
        end_label = f"T{end_id}" if end_id is not None else ("LAND" if end else "n/a")
        start_side = _path_position_side(start_position, track_sides)
        end_side = _path_position_side(end_position, track_sides)
        state = "DONE" if int(segment.get("end_frame_index", 0)) <= frame_index else "WAIT"
        color = (90, 230, 145) if state == "DONE" else (115, 125, 135)
        _text(
            frame,
            f"#{index + 1:02d}",
            (panel_left, row_y),
            0.44,
            color,
        )
        _draw_side_badge(frame, start_side, (panel_left + 50, row_y - 16))
        _text(frame, f"A {start_label}", (panel_left + 86, row_y), 0.44, color)
        _text(frame, "->", (panel_left + 176, row_y), 0.44, (155, 170, 182))
        _draw_side_badge(frame, end_side, (panel_left + 218, row_y - 16))
        _text(frame, f"B {end_label}", (panel_left + 254, row_y), 0.44, color)
        _text(
            frame,
            state,
            (OUTPUT_WIDTH - 84, row_y),
            0.39,
            color,
        )
    _text(
        frame,
        "PLAYER COURT | TERMINAL BALL LANDING | A->B path",
        (panel_left, OUTPUT_HEIGHT - 22),
        0.42,
        (170, 185, 198),
    )


def _draw_timeline(
    frame: Any,
    job: dict[str, Any],
    frame_index: int,
) -> None:
    total = max(int(job["clip"]["video"]["total_frames"]) - 1, 1)
    left, right, y = INFO_LEFT + 24, OUTPUT_WIDTH - 24, VIDEO_HEIGHT - 20
    _text(frame, "CLIP TIMELINE", (left, y - 18), 0.36, (140, 155, 168))
    cv2.line(frame, (left, y), (right, y), (100, 110, 120), 2)
    for point in job["key_points"]:
        x = left + round(int(point["clip_frame_index"]) / total * (right - left))
        color = (80, 220, 80) if point["marker_kind"] == "service" else (20, 190, 255)
        cv2.circle(
            frame,
            (x, y),
            6 if point["is_terminal"] else 4,
            color,
            -1,
            cv2.LINE_AA,
        )
    cursor_x = left + round(frame_index / total * (right - left))
    cv2.line(frame, (cursor_x, y - 9), (cursor_x, y + 9), (255, 255, 255), 1)


def preview_frame_indices(
    result: dict[str, Any],
    *,
    fps: float,
    total_frames: int,
) -> tuple[int, int]:
    """Select the same two visual milestones as Contract Lab visual-v5."""
    paths = result.get("path_segments", [])
    events = result.get("contact_events", [])
    first_complete = int(paths[0]["end_frame_index"]) if paths else 0
    terminal = next((event for event in reversed(events) if event.get("is_terminal")), None)
    terminal_frame = int(terminal["anchor_frame_index"] if terminal is not None else first_complete)
    upper = max(total_frames - 1, 0)
    return (
        min(upper, max(0, first_complete + round(fps * 0.15))),
        min(
            upper,
            max(
                0,
                terminal_frame + max(PATH_SCROLL_TRANSITION_FRAMES, round(fps / 30.0)),
            ),
        ),
    )


def _terminal_ball_court(
    event: dict[str, Any],
    frame_index: int,
) -> tuple[float, float] | None:
    if not event.get("is_terminal") or frame_index < int(event["anchor_frame_index"]):
        return None
    landing = next(
        (
            position
            for position in event.get("representative_court_positions", [])
            if position.get("track_id") is None
        ),
        None,
    )
    if landing is None:
        return None
    court = landing.get("court_pos")
    return (float(court["x"]), float(court["y"])) if court else None


def _encode_web_video(video_only: Path, original: Path, output: Path) -> str:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to create the visual-v5 package")
    common = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_only),
        "-i",
        str(original),
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
    ]
    web_options = [
        "-pix_fmt",
        "yuv420p",
        "-profile:v",
        "high",
        "-level:v",
        "4.2",
        "-tag:v",
        "avc1",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        "-shortest",
    ]
    encoders = (
        (
            "h264_nvenc",
            ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", "20", "-b:v", "0"],
        ),
        ("libx264", ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"]),
    )
    failures: list[str] = []
    for name, encoder_options in encoders:
        output.unlink(missing_ok=True)
        completed = subprocess.run(
            [*common, *encoder_options, *web_options, str(output)],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode == 0:
            video_only.unlink(missing_ok=True)
            return name
        failures.append(f"{name}: {completed.stderr.strip()}")
    output.unlink(missing_ok=True)
    raise RuntimeError("ffmpeg visual-v5 encode failed: " + " | ".join(failures))


def _render_video(
    *,
    clip_path: Path,
    output_path: Path,
    job: dict[str, Any],
    result: dict[str, Any],
    frames: list[FrameObservation],
    balls: dict[int, BallObservation],
    courts: dict[int, CourtFrame],
    fps: float,
    source_width: int,
    source_height: int,
    preview_paths: tuple[Path, Path],
) -> dict[str, int | float | bool | str]:
    capture = cv2.VideoCapture(str(clip_path))
    if not capture.isOpened():
        raise ValueError(f"cannot open clip for visualization: {clip_path}")
    video_only = output_path.with_name(f"{output_path.stem}.opencv-video-only.mp4")
    writer = cv2.VideoWriter(
        str(video_only),
        cv2.VideoWriter.fourcc(*"mp4v"),
        fps,
        (OUTPUT_WIDTH, OUTPUT_HEIGHT),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("OpenCV could not create the visual-v5 video")

    frame_map = {observation.frame_index: observation for observation in frames}
    events = cast("list[dict[str, Any]]", result["contact_events"])
    event_frames = [int(event["anchor_frame_index"]) for event in events]
    key_points = cast("list[dict[str, Any]]", job["key_points"])
    points_by_id = {point["key_point_id"]: point for point in key_points}
    total_frames = int(job["clip"]["video"]["total_frames"])
    preview_indices = preview_frame_indices(result, fps=fps, total_frames=total_frames)
    preview_by_index: dict[int, list[Path]] = {}
    for preview_index, preview_path in zip(preview_indices, preview_paths, strict=True):
        preview_by_index.setdefault(preview_index, []).append(preview_path)
    sorted_courts = sorted(courts.items())
    court_cursor = 0
    active_court: CourtFrame | None = None
    frame_index = 0
    tracking_frames = 0
    tracking_rows = 0
    ball_frames = 0
    court_frames = 0
    try:
        while True:
            ok, source_frame = capture.read()
            if not ok:
                break
            while (
                court_cursor < len(sorted_courts) and sorted_courts[court_cursor][0] <= frame_index
            ):
                active_court = sorted_courts[court_cursor][1]
                court_cursor += 1
            image = np.zeros((OUTPUT_HEIGHT, OUTPUT_WIDTH, 3), dtype=np.uint8)
            image[:VIDEO_HEIGHT, :VIDEO_WIDTH] = cv2.resize(
                source_frame,
                (VIDEO_WIDTH, VIDEO_HEIGHT),
                interpolation=cv2.INTER_LINEAR,
            )
            observation = frame_map.get(frame_index)
            action_states = _action_states_for_frame(events, frame_index, fps)
            if _draw_court_keypoints(
                image,
                active_court,
                source_width,
                source_height,
            ):
                court_frames += 1
            if observation is not None and observation.players:
                tracking_frames += 1
                tracking_rows += len(observation.players)
                _draw_tracking(image, observation, action_states)
            if _draw_ball(image, balls, events, frame_index, fps):
                ball_frames += 1

            if events:
                nearest_index = min(
                    range(len(events)),
                    key=lambda index: abs(event_frames[index] - frame_index),
                )
                event = events[nearest_index]
            else:
                event = {}
            point = points_by_id.get(
                event.get("key_point_id"),
                {"key_point_id": "unavailable"},
            )
            _draw_info_panel(
                image,
                event,
                point,
                frame_index,
                fps,
                len(observation.players) if observation is not None else 0,
            )
            _draw_timeline(image, job, frame_index)
            _draw_court_panel(
                image,
                result,
                frame_index,
                observation,
                _terminal_ball_court(event, frame_index) if event else None,
            )
            for preview_path in preview_by_index.get(frame_index, []):
                cv2.imwrite(str(preview_path), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            writer.write(image)
            frame_index += 1
    finally:
        capture.release()
        writer.release()

    if frame_index != total_frames:
        video_only.unlink(missing_ok=True)
        raise ValueError(f"decoded {frame_index} frames but AI job declares {total_frames}")
    video_encoder = _encode_web_video(video_only, clip_path, output_path)
    return {
        "width": OUTPUT_WIDTH,
        "height": OUTPUT_HEIGHT,
        "fps": fps,
        "frames_written": frame_index,
        "audio_preserved": bool(job["clip"]["video"]["has_audio"]),
        "video_codec": "h264",
        "video_encoder": video_encoder,
        "pixel_format": "yuv420p",
        "faststart": True,
        "layout": "match_1280x720_plus_info_and_bottom_canonical_court",
        "tracking_overlay": tracking_frames > 0,
        "tracking_frames_rendered": tracking_frames,
        "tracking_rows_rendered": tracking_rows,
        "ball_frames_rendered": ball_frames,
        "court_keypoint_frames_rendered": court_frames,
    }


def write_visual_v5_package(
    *,
    output_dir: Path,
    clip_path: Path,
    job: AIJobRequest,
    result: AnalysisResult,
    frames: list[FrameObservation],
    balls: dict[int, BallObservation],
    courts: dict[int, CourtFrame],
    fps: float,
    frame_width: int,
    frame_height: int,
    action_frame_count: int,
) -> dict[str, str]:
    """Write the visual-v5 filenames and layout using only current inference data."""
    output_dir.mkdir(parents=True, exist_ok=True)
    job_document = job.model_dump(mode="json", exclude_none=True)
    result_document = result.model_dump(mode="json", exclude_none=True)
    result_path = output_dir / "analysis-result.json"
    video_path = output_dir / "overlay-preview.mp4"
    first_preview = output_dir / "preview-first-complete.jpg"
    terminal_preview = output_dir / "preview-terminal-path.jpg"
    manifest_path = output_dir / "visualization-manifest.json"
    result_path.write_text(
        json.dumps(result_document, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    rendering = _render_video(
        clip_path=clip_path,
        output_path=video_path,
        job=job_document,
        result=result_document,
        frames=frames,
        balls=balls,
        courts=courts,
        fps=fps,
        source_width=frame_width,
        source_height=frame_height,
        preview_paths=(first_preview, terminal_preview),
    )
    manifest = {
        "schema_version": "1.0.0",
        "input_job_schema_version": str(job_document["schema_version"]),
        "output_result_schema_version": str(result_document["schema_version"]),
        "network_calls": 0,
        "synthetic_ai_fields": False,
        "action_source": "rt_detrv4_x3d_model",
        "action_model_calls": action_frame_count,
        "default_non_hitter_action": "ready_position",
        "real_keypoint_passthrough": True,
        "tracking_output": "in_memory_model_inference",
        "tracking_available": any(frame.players for frame in frames),
        "ball_annotations": bool(balls),
        "ball_annotation_points": len(balls),
        "video": video_path.name,
        "result": result_path.name,
        "rendering": rendering,
        "notes": [
            "who_hit is contact_events[].actors[].track_id",
            "hit time/frame comes from the real input keypoint anchor",
            "actions come from the RT-DETRv4/X3D checkpoint",
            "tracking and 12-player identity consolidation are produced by the engine",
            "court_pos is AI-owned, unclamped canonical court space",
            "no precomputed Contract Lab result is read during inference",
            "out-of-court A/B geometry is clipped by a padded court stage",
        ],
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "result": str(result_path),
        "video": str(video_path),
        "manifest": str(manifest_path),
        "preview_first_complete": str(first_preview),
        "preview_terminal_path": str(terminal_preview),
    }
