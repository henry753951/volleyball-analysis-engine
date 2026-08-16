"""Action-aware hitter association regression tests."""

from volleyball_analysis_engine.association import HitAssociation, associate_hit
from volleyball_analysis_engine.records import (
    ActionObservation,
    BallObservation,
    PersonPoseObservation,
    PlayerObservation,
)


def player(frame: int, source_track_id: int) -> PlayerObservation:
    return PlayerObservation(
        frame_index=frame,
        source_track_id=source_track_id,
        track_id=source_track_id,
        frame_bbox=(0.40, 0.25, 0.60, 0.80),
        frame_foot_pos=(0.50, 0.80),
        court_pos=(0.25 if source_track_id % 2 else 0.75, 0.50),
        confidence=0.90,
    )


def associate(
    *,
    anchor: int,
    players: dict[int, tuple[PlayerObservation, ...]],
    actions: dict[tuple[int, int], ActionObservation],
    poses: dict[int, tuple[PersonPoseObservation, ...]] | None = None,
    ball_position: tuple[float, float] = (0.50, 0.45),
) -> HitAssociation:
    return associate_hit(
        anchor_frame=anchor,
        previous_anchor_frame=0,
        next_anchor_frame=30,
        is_terminal=False,
        balls={frame: BallObservation(frame, ball_position, 0.95) for frame in range(1, 30)},
        players=players,
        actions=actions,
        poses=poses,
        frame_width=1920,
        frame_height=1080,
        action_search_radius=6,
    )


def pose(
    frame: int,
    track_id: int,
    *,
    wrist: tuple[float, float],
) -> PersonPoseObservation:
    keypoints = [(-1.0, -1.0, -1.0)] * 17
    keypoints[7] = (wrist[0], wrist[1] + 0.06, 0.90)
    keypoints[9] = (wrist[0], wrist[1], 0.95)
    return PersonPoseObservation(
        frame_index=frame,
        track_id=track_id,
        bbox_source="DETECTOR",
        frame_bbox=(0.40, 0.25, 0.60, 0.80),
        crop_transform=(1 / 1920, 1 / 1080, 0.40, 0.25),
        status="AVAILABLE",
        keypoints=tuple(keypoints),
    )


def test_nearby_spiking_actor_wins_before_spatial_only_fallback() -> None:
    result = associate(
        anchor=10,
        players={10: (player(10, 1),), 12: (player(12, 2),)},
        actions={(12, 2): ActionObservation(12, 2, "spiking", 0.92)},
    )

    assert result.player is not None
    assert result.player.source_track_id == 2
    assert result.observation_frame == 12
    assert result.mode == "action_spiking_ball_iou"


def test_setting_actor_is_found_before_the_manual_anchor() -> None:
    result = associate(
        anchor=10,
        players={8: (player(8, 3),)},
        actions={(8, 3): ActionObservation(8, 3, "setting", 0.87)},
    )

    assert result.player is not None
    assert result.player.source_track_id == 3
    assert result.observation_frame == 8
    assert result.mode == "action_setting_ball_iou"


def test_spatial_fallback_searches_symmetrically_before_anchor() -> None:
    result = associate(
        anchor=10,
        players={7: (player(7, 4),)},
        actions={},
    )

    assert result.player is not None
    assert result.player.source_track_id == 4
    assert result.observation_frame == 7
    assert result.mode == "nearby_player_frames_at_fixed_hit_ball"


def test_setting_action_does_not_validate_a_ball_near_the_players_feet() -> None:
    result = associate(
        anchor=10,
        players={10: (player(10, 5),)},
        actions={(10, 5): ActionObservation(10, 5, "setting", 0.95)},
        ball_position=(0.50, 0.82),
    )

    assert result.player is not None
    assert result.mode == "direct_hit_ball_iou"


def test_reliable_hand_pose_wins_before_action_and_bbox_fallbacks() -> None:
    result = associate(
        anchor=10,
        players={10: (player(10, 1), player(10, 2))},
        actions={(10, 2): ActionObservation(10, 2, "spiking", 0.99)},
        poses={
            10: (
                pose(10, 1, wrist=(0.505, 0.45)),
                pose(10, 2, wrist=(0.70, 0.45)),
            )
        },
    )

    assert result.player is not None
    assert result.player.track_id == 1
    assert result.mode == "pose_hand_nearest"
    assert result.evidence["best"]["geometry"] in {"left_wrist", "left_forearm"}


def test_ambiguous_pose_abstains_and_records_the_bbox_fallback_reason() -> None:
    result = associate(
        anchor=10,
        players={10: (player(10, 1), player(10, 2))},
        actions={(10, 2): ActionObservation(10, 2, "spiking", 0.99)},
        poses={
            10: (
                pose(10, 1, wrist=(0.49, 0.45)),
                pose(10, 2, wrist=(0.51, 0.45)),
            )
        },
    )

    assert result.player is not None
    assert result.player.track_id == 2
    assert result.mode == "action_spiking_ball_iou"
    assert result.evidence["pose_fallback_reason"] == "ambiguous_runner_up"
