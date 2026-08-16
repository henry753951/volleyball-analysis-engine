"""Action-aware ball-to-hitter association and temporary A/B action rules."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import hypot
from typing import Any

from .records import ActionObservation, BallObservation, PersonPoseObservation, PlayerObservation

COURT_MIDLINE_X = 0.5
CONTACT_ACTION_PRIORITY = {"spiking": 0, "passing": 1, "setting": 1, "digging": 2}
BALL_ACTION_FRAME_TOLERANCE = 3
POSE_KEYPOINT_CONFIDENCE = 0.3
POSE_MAX_NORMALIZED_DISTANCE = 0.45
POSE_MIN_RUNNER_UP_MARGIN = 0.08
POSE_TEMPORAL_PENALTY = 0.04
SEGMENT_EPSILON = 1e-12


def _empty_evidence() -> dict[str, Any]:
    return {}


@dataclass(frozen=True, slots=True)
class HitAssociation:
    """Resolved evidence for one human key point."""

    ball: BallObservation | None
    player: PlayerObservation | None
    observation_frame: int | None
    mode: str
    confidence: float | None
    evidence: dict[str, Any] = field(default_factory=_empty_evidence)


def _point_segment_distance(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    dx, dy = end[0] - start[0], end[1] - start[1]
    denominator = dx * dx + dy * dy
    if denominator <= SEGMENT_EPSILON:
        return hypot(point[0] - start[0], point[1] - start[1])
    projection = min(
        1.0,
        max(0.0, ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / denominator),
    )
    closest = (start[0] + projection * dx, start[1] + projection * dy)
    return hypot(point[0] - closest[0], point[1] - closest[1])


def _arm_distance(
    ball: BallObservation,
    pose: PersonPoseObservation,
) -> tuple[float, dict[str, Any]] | None:
    if pose.status != "AVAILABLE" or pose.keypoints is None:
        return None
    x1, y1, x2, y2 = pose.frame_bbox
    scale = max(hypot(x2 - x1, y2 - y1), 1e-6)
    candidates: list[tuple[float, str, float, float]] = []
    for side, elbow_index, wrist_index in (("left", 7, 9), ("right", 8, 10)):
        elbow = pose.keypoints[elbow_index]
        wrist = pose.keypoints[wrist_index]
        if wrist[2] < POSE_KEYPOINT_CONFIDENCE:
            continue
        wrist_distance = hypot(ball.frame_pos[0] - wrist[0], ball.frame_pos[1] - wrist[1])
        candidates.append((wrist_distance, f"{side}_wrist", wrist[2], elbow[2]))
        if elbow[2] >= POSE_KEYPOINT_CONFIDENCE:
            candidates.append(
                (
                    _point_segment_distance(
                        ball.frame_pos,
                        (elbow[0], elbow[1]),
                        (wrist[0], wrist[1]),
                    ),
                    f"{side}_forearm",
                    wrist[2],
                    elbow[2],
                )
            )
    if not candidates:
        return None
    distance, geometry, wrist_confidence, elbow_confidence = min(candidates)
    return distance / scale, {
        "geometry": geometry,
        "raw_video_distance": distance,
        "bbox_diagonal": scale,
        "wrist_confidence": wrist_confidence,
        "elbow_confidence": elbow_confidence,
        "bbox_source": pose.bbox_source,
    }


def _pose_candidate(
    *,
    anchor_frame: int,
    lower_frame: int,
    upper_frame: int,
    search_radius: int,
    balls: dict[int, BallObservation],
    players: dict[int, tuple[PlayerObservation, ...]],
    poses: dict[int, tuple[PersonPoseObservation, ...]],
    actions: dict[tuple[int, int], ActionObservation],
) -> tuple[HitAssociation | None, dict[str, Any]]:
    ball = _nearest_ball(
        balls,
        anchor_frame,
        lower_frame=lower_frame,
        upper_frame=upper_frame,
        max_distance=search_radius,
    )
    if ball is None:
        return None, {"pose_fallback_reason": "ball_missing_in_pose_window"}
    ranked: list[tuple[float, float, PlayerObservation, int, dict[str, Any]]] = []
    for frame_index in _nearby_frames(
        anchor_frame,
        lower_frame=lower_frame,
        upper_frame=upper_frame,
        radius=search_radius,
    ):
        player_by_track = {player.track_id: player for player in players.get(frame_index, ())}
        for pose in poses.get(frame_index, ()):
            player = player_by_track.get(pose.track_id)
            if player is None:
                continue
            measured = _arm_distance(ball, pose)
            if measured is None:
                continue
            spatial_distance, evidence = measured
            temporal_offset = abs(frame_index - ball.frame_index)
            action = actions.get((frame_index, player.source_track_id))
            action_bonus = (
                0.03
                if action is not None and action.label.lower() in CONTACT_ACTION_PRIORITY
                else 0.0
            )
            rank_score = spatial_distance + temporal_offset * POSE_TEMPORAL_PENALTY - action_bonus
            ranked.append(
                (
                    rank_score,
                    spatial_distance,
                    player,
                    frame_index,
                    {
                        **evidence,
                        "track_id": player.track_id,
                        "pose_frame_index": frame_index,
                        "ball_frame_index": ball.frame_index,
                        "temporal_offset": temporal_offset,
                        "action_label": None if action is None else action.label,
                        "action_bonus": action_bonus,
                        "normalized_distance": spatial_distance,
                        "rank_score": rank_score,
                    },
                )
            )
    if not ranked:
        return None, {"pose_fallback_reason": "no_reliable_arm_keypoints"}
    ranked.sort(key=lambda candidate: (candidate[0], candidate[2].track_id, candidate[3]))
    best = ranked[0]
    runner_up = next(
        (candidate for candidate in ranked[1:] if candidate[2].track_id != best[2].track_id), None
    )
    margin = None if runner_up is None else runner_up[0] - best[0]
    audit = {
        "pose_recipe": "coco17-hand-forearm-association-v1",
        "absolute_distance_gate": POSE_MAX_NORMALIZED_DISTANCE,
        "runner_up_margin_gate": POSE_MIN_RUNNER_UP_MARGIN,
        "best": best[4],
        "runner_up": None if runner_up is None else runner_up[4],
        "runner_up_margin": margin,
    }
    if best[1] > POSE_MAX_NORMALIZED_DISTANCE:
        return None, {**audit, "pose_fallback_reason": "outside_distance_gate"}
    if margin is not None and margin < POSE_MIN_RUNNER_UP_MARGIN:
        return None, {**audit, "pose_fallback_reason": "ambiguous_runner_up"}
    confidence = max(0.0, min(1.0, 1.0 - best[1] / POSE_MAX_NORMALIZED_DISTANCE))
    return HitAssociation(ball, best[2], best[3], "pose_hand_nearest", confidence, audit), audit


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
    action_label: str | None = None,
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
        horizontal_margin = max(radius_x, width * 0.35)
        if action_label in {"spiking", "setting"}:
            # Overhead contacts must be close to the head/arms, not merely the feet.
            expanded = (
                x1 - horizontal_margin,
                y1 - max(radius_y, height * 0.50),
                x2 + horizontal_margin,
                y1 + height * 0.65,
            )
        elif action_label in {"passing", "digging"}:
            # Forearm contacts happen around the torso and below, but not far overhead.
            expanded = (
                x1 - horizontal_margin,
                y1 + height * 0.10,
                x2 + horizontal_margin,
                y2 + max(radius_y, height * 0.20),
            )
        else:
            expanded = (
                x1 - horizontal_margin,
                y1 - max(radius_y, height * 0.35),
                x2 + horizontal_margin,
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
        if ball is None:
            continue
        ranked: list[tuple[float, PlayerObservation, ActionObservation]] = []
        for candidate_player, candidate_action in action_players:
            matched_player, score = _best_player(
                ball,
                (candidate_player,),
                frame_width=frame_width,
                frame_height=frame_height,
                action_label=candidate_action.label.lower(),
            )
            if matched_player is not None:
                ranked.append((score, matched_player, candidate_action))
        if ranked:
            score, player, action = max(ranked, key=lambda candidate: candidate[0])
            return HitAssociation(
                ball,
                player,
                frame_index,
                f"action_{action.label.lower()}_ball_iou",
                score,
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
    poses: dict[int, tuple[PersonPoseObservation, ...]] | None = None,
    frame_width: int,
    frame_height: int,
    action_search_radius: int,
) -> HitAssociation:
    """Resolve a hitter from reusable pose evidence, then degrade deterministically."""
    lower_frame = max(0, previous_anchor_frame + 1)
    upper_frame = max(lower_frame, next_anchor_frame - 1)
    pose_audit: dict[str, Any]
    if poses:
        pose_match, pose_audit = _pose_candidate(
            anchor_frame=anchor_frame,
            lower_frame=lower_frame,
            upper_frame=upper_frame,
            search_radius=action_search_radius,
            balls=balls,
            players=players,
            poses=poses,
            actions=actions,
        )
        if pose_match is not None:
            return pose_match
    else:
        pose_audit = {"pose_fallback_reason": "pose_evidence_unavailable"}
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
        return replace(action_match, evidence=pose_audit)
    if not balls:
        return HitAssociation(None, None, None, "ball_missing", None, pose_audit)
    ball = _nearest_ball(
        balls,
        anchor_frame,
        lower_frame=lower_frame,
        upper_frame=upper_frame,
    )
    if ball is None:
        return HitAssociation(
            None,
            None,
            None,
            "ball_missing_in_event_window",
            None,
            pose_audit,
        )
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
            return HitAssociation(ball, player, frame_index, mode, score, pose_audit)
    mode = "terminal_ground_candidate" if is_terminal else "no_player"
    return HitAssociation(ball, None, ball.frame_index, mode, None, pose_audit)


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
