"""Ball-to-hitter association and temporary A/B action rules."""

from __future__ import annotations

from dataclasses import dataclass

from .records import BallObservation, PlayerObservation

COURT_MIDLINE_X = 0.5


@dataclass(frozen=True, slots=True)
class HitAssociation:
    """Resolved evidence for one human key point."""

    ball: BallObservation | None
    player: PlayerObservation | None
    observation_frame: int | None
    mode: str
    confidence: float | None


def _iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if intersection > 0.0 and union > 0.0 else 0.0


def _best_player(
    ball: BallObservation,
    players: tuple[PlayerObservation, ...],
    *,
    frame_width: int,
    frame_height: int,
) -> tuple[PlayerObservation | None, float]:
    radius_px = max(4.0, min(frame_width, frame_height) * 0.018)
    radius_x, radius_y = radius_px / frame_width, radius_px / frame_height
    ball_box = (
        ball.frame_pos[0] - radius_x,
        ball.frame_pos[1] - radius_y,
        ball.frame_pos[0] + radius_x,
        ball.frame_pos[1] + radius_y,
    )
    best: PlayerObservation | None = None
    best_score = 0.0
    for player in players:
        x1, y1, x2, y2 = player.frame_bbox
        width, height = max(1 / frame_width, x2 - x1), max(1 / frame_height, y2 - y1)
        expanded = (
            x1 - max(radius_x, width * 0.35),
            y1 - max(radius_y, height * 0.35),
            x2 + max(radius_x, width * 0.35),
            y2 + max(radius_y, height * 0.35),
        )
        score = _iou(ball_box, expanded)
        if score > best_score:
            best, best_score = player, score
    return (best, best_score) if best_score > 0 else (None, 0.0)


def associate_hit(
    *,
    anchor_frame: int,
    next_anchor_frame: int,
    is_terminal: bool,
    balls: dict[int, BallObservation],
    players: dict[int, tuple[PlayerObservation, ...]],
    frame_width: int,
    frame_height: int,
) -> HitAssociation:
    """Associate a marker with ball evidence and the nearest plausible player."""
    if not balls:
        return HitAssociation(None, None, None, "ball_missing", None)
    ball_frame = min(balls, key=lambda frame: abs(frame - anchor_frame))
    ball = balls[ball_frame]
    player, score = _best_player(
        ball,
        players.get(ball_frame, ()),
        frame_width=frame_width,
        frame_height=frame_height,
    )
    if player is not None:
        return HitAssociation(ball, player, ball_frame, "direct_hit_ball_iou", score)
    if not is_terminal:
        for frame_index in range(ball_frame + 1, next_anchor_frame):
            player, score = _best_player(
                ball,
                players.get(frame_index, ()),
                frame_width=frame_width,
                frame_height=frame_height,
            )
            if player is not None:
                return HitAssociation(
                    ball,
                    player,
                    frame_index,
                    "forward_player_frames_at_fixed_hit_ball",
                    score,
                )
    mode = "terminal_ground_candidate" if is_terminal else "no_player"
    return HitAssociation(ball, None, ball_frame, mode, None)


def classify_action(
    start_position: tuple[float, float] | None,
    end_position: tuple[float, float] | None,
    *,
    is_service: bool,
) -> tuple[str, bool] | None:
    """Temporary action label derived only from adjacent A/B court positions."""
    if start_position is None or end_position is None:
        return None
    crosses_court = (start_position[0] < COURT_MIDLINE_X) != (end_position[0] < COURT_MIDLINE_X)
    if crosses_court and is_service:
        return "serving", True
    if crosses_court:
        return "spiking", True
    return "passing", False
