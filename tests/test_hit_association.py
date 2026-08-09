"""Action-aware hitter association regression tests."""

from volleyball_analysis_engine.association import HitAssociation, associate_hit
from volleyball_analysis_engine.records import (
    ActionObservation,
    BallObservation,
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
) -> HitAssociation:
    return associate_hit(
        anchor_frame=anchor,
        previous_anchor_frame=0,
        next_anchor_frame=30,
        is_terminal=False,
        balls={frame: BallObservation(frame, (0.50, 0.45), 0.95) for frame in range(1, 30)},
        players=players,
        actions=actions,
        frame_width=1920,
        frame_height=1080,
        action_search_radius=6,
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
