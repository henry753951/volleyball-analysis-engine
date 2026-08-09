"""Action-aware ball-to-hitter association and temporary A/B action rules."""

from __future__ import annotations

from dataclasses import dataclass

from .records import ActionObservation, BallObservation, PlayerObservation

COURT_MIDLINE_X = 0.5
CONTACT_ACTION_PRIORITY = {"spiking": 0, "passing": 1, "setting": 1, "digging": 2}
BALL_ACTION_FRAME_TOLERANCE = 3


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


def _nearby_frames(
    anchor_frame: int,
    *,
    lower_frame: int,
    upper_frame: int,
    radius: int,
) -> list[int]:
    start = max(lower_frame, anchor_frame - radius)
    end = min(upper_frame, anchor_frame + radius)
    return sorted(range(start, end + 1), key=lambda frame: (abs(frame - anchor_frame), frame))


def _nearest_ball(
    balls: dict[int, BallObservation],
    frame_index: int,
    *,
    lower_frame: int,
    upper_frame: int,
    max_distance: int | None = None,
) -> BallObservation | None:
    candidates = [
        ball
        for candidate_frame, ball in balls.items()
        if lower_frame <= candidate_frame <= upper_frame
        and (max_distance is None or abs(candidate_frame - frame_index) <= max_distance)
    ]
    return min(candidates, key=lambda ball: abs(ball.frame_index - frame_index), default=None)


def _action_candidate(
    *,
    anchor_frame: int,
    lower_frame: int,
    upper_frame: int,
    search_radius: int,
    balls: dict[int, BallObservation],
    players: dict[int, tuple[PlayerObservation, ...]],
    actions: dict[tuple[int, int], ActionObservation],
    frame_width: int,
    frame_height: int,
) -> HitAssociation | None:
    """Prefer the closest spike/set/dig evidence before spatial-only fallback."""
    for frame_index in _nearby_frames(
        anchor_frame,
        lower_frame=lower_frame,
        upper_frame=upper_frame,
        radius=search_radius,
    ):
        action_players: list[tuple[PlayerObservation, ActionObservation]] = []
        for player in players.get(frame_index, ()):
            action = actions.get((frame_index, player.source_track_id))
            if action is not None and action.label.lower() in CONTACT_ACTION_PRIORITY:
                action_players.append((player, action))
        if not action_players:
            continue

        ball = _nearest_ball(
            balls,
            frame_index,
            lower_frame=lower_frame,
            upper_frame=upper_frame,
            max_distance=BALL_ACTION_FRAME_TOLERANCE,
        )
        if ball is not None:
            player, score = _best_player(
                ball,
                tuple(candidate[0] for candidate in action_players),
                frame_width=frame_width,
                frame_height=frame_height,
            )
            if player is not None:
                action = next(
                    candidate_action
                    for candidate_player, candidate_action in action_players
                    if candidate_player.source_track_id == player.source_track_id
                )
                return HitAssociation(
                    ball,
                    player,
                    frame_index,
                    f"action_{action.label.lower()}_ball_iou",
                    score,
                )

        player, action = min(
            action_players,
            key=lambda candidate: (
                CONTACT_ACTION_PRIORITY[candidate[1].label.lower()],
                -(candidate[1].confidence or 0.0),
                candidate[0].source_track_id,
            ),
        )
        event_ball = ball or _nearest_ball(
            balls,
            anchor_frame,
            lower_frame=lower_frame,
            upper_frame=upper_frame,
        )
        return HitAssociation(
            event_ball,
            player,
            frame_index,
            f"action_{action.label.lower()}_near_anchor",
            action.confidence,
        )
    return None


def associate_hit(
    *,
    anchor_frame: int,
    previous_anchor_frame: int,
    next_anchor_frame: int,
    is_terminal: bool,
    balls: dict[int, BallObservation],
    players: dict[int, tuple[PlayerObservation, ...]],
    actions: dict[tuple[int, int], ActionObservation],
    frame_width: int,
    frame_height: int,
    action_search_radius: int,
) -> HitAssociation:
    """Resolve the nearest contact action, then fall back to spatial ball IoU."""
    lower_frame = max(0, previous_anchor_frame + 1)
    upper_frame = max(lower_frame, next_anchor_frame - 1)
    action_match = _action_candidate(
        anchor_frame=anchor_frame,
        lower_frame=lower_frame,
        upper_frame=upper_frame,
        search_radius=action_search_radius,
        balls=balls,
        players=players,
        actions=actions,
        frame_width=frame_width,
        frame_height=frame_height,
    )
    if action_match is not None:
        return action_match
    if not balls:
        return HitAssociation(None, None, None, "ball_missing", None)
    ball = _nearest_ball(
        balls,
        anchor_frame,
        lower_frame=lower_frame,
        upper_frame=upper_frame,
    )
    if ball is None:
        return HitAssociation(None, None, None, "ball_missing_in_event_window", None)
    for frame_index in _nearby_frames(
        anchor_frame,
        lower_frame=lower_frame,
        upper_frame=upper_frame,
        radius=action_search_radius,
    ):
        player, score = _best_player(
            ball,
            players.get(frame_index, ()),
            frame_width=frame_width,
            frame_height=frame_height,
        )
        if player is not None:
            mode = (
                "direct_hit_ball_iou"
                if frame_index == ball.frame_index
                else ("nearby_player_frames_at_fixed_hit_ball")
            )
            return HitAssociation(ball, player, frame_index, mode, score)
    mode = "terminal_ground_candidate" if is_terminal else "no_player"
    return HitAssociation(ball, None, ball.frame_index, mode, None)


def classify_action(
    start_position: tuple[float, float] | None,
    end_position: tuple[float, float] | None,
) -> tuple[str, bool] | None:
    """Temporary action label derived only from adjacent A/B court positions."""
    if start_position is None or end_position is None:
        return None
    crosses_court = (start_position[0] < COURT_MIDLINE_X) != (end_position[0] < COURT_MIDLINE_X)
    if crosses_court:
        return "Spiking", True
    return "Waiting", False
