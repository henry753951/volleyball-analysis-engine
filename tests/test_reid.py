"""Court-position ReID tests."""

from volleyball_analysis_engine.records import FrameObservation, PlayerObservation
from volleyball_analysis_engine.reid import CourtPositionReidentifier


def player(frame: int, source: int, x: float, y: float) -> PlayerObservation:
    return PlayerObservation(
        frame_index=frame,
        source_track_id=source,
        track_id=source,
        frame_bbox=(x - 0.02, y - 0.1, x + 0.02, y),
        frame_foot_pos=(x, y),
        court_pos=(x, y),
    )


def test_reentry_uses_nearest_missing_identity_on_same_court_side() -> None:
    frames = [
        FrameObservation(0, (player(0, 10, 0.2, 0.4), player(0, 20, 0.8, 0.5)), True),
        FrameObservation(1, (player(1, 20, 0.79, 0.5),), True),
        FrameObservation(2, (player(2, 30, 0.21, 0.41), player(2, 20, 0.78, 0.5)), True),
    ]
    output = CourtPositionReidentifier().apply(frames)
    first_left = output[0].players[0].track_id
    reentered_left = next(item for item in output[2].players if item.source_track_id == 30)
    assert reentered_left.track_id == first_left


def test_reentry_never_crosses_team_side() -> None:
    frames = [
        FrameObservation(0, (player(0, 1, 0.49, 0.5),), True),
        FrameObservation(1, (), True),
        FrameObservation(2, (player(2, 2, 0.51, 0.5),), True),
    ]
    output = CourtPositionReidentifier().apply(frames)
    assert output[0].players[0].track_id != output[2].players[0].track_id


def test_each_court_side_is_bounded_to_six_player_identities() -> None:
    players = tuple(player(0, source, 0.6 + source * 0.01, 0.5) for source in range(7))
    output = CourtPositionReidentifier(max_per_side=6).apply([FrameObservation(0, players, True)])
    assert len(output[0].players) == 6
    assert len({observation.track_id for observation in output[0].players}) == 6


def test_reappearing_source_collision_never_duplicates_identity_in_one_frame() -> None:
    frames = [
        FrameObservation(0, (player(0, 10, 0.2, 0.4),), True),
        FrameObservation(1, (), True),
        FrameObservation(2, (player(2, 20, 0.21, 0.4),), True),
        FrameObservation(
            3,
            (player(3, 10, 0.2, 0.4), player(3, 20, 0.21, 0.4)),
            True,
        ),
    ]

    output = CourtPositionReidentifier().apply(frames)
    identities = [observation.track_id for observation in output[3].players]

    assert len(identities) == len(set(identities))


def test_full_side_never_forces_far_source_onto_existing_identity() -> None:
    first = tuple(player(0, source, 0.51 + source * 0.01, 0.5) for source in range(6))
    frames = [
        FrameObservation(0, first, True),
        FrameObservation(1, first[1:], True),
        FrameObservation(2, (player(2, 20, 0.95, 0.95), *first[1:]), True),
    ]

    output = CourtPositionReidentifier(max_per_side=6).apply(frames)
    original = {observation.track_id for observation in output[0].players}
    far = next(item for item in output[2].players if item.source_track_id == 20)

    assert far.track_id not in original
