"""Focused tests for clip-local ReID feature-bank projection."""

from __future__ import annotations

import pytest

from volleyball_analysis_engine.records import (
    CourtSide,
    FrameObservation,
    PlayerObservation,
)
from volleyball_analysis_engine.reid_features import (
    _descriptor_bytes,  # pyright: ignore[reportPrivateUsage]
    _enforce_fixed_roster_capacity,  # pyright: ignore[reportPrivateUsage]
    _select_on_court_roster,  # pyright: ignore[reportPrivateUsage]
    resolve_track_court_sides,
)

_TRACK_ID = 7


def _frames_for_side_counts(*, left: int, right: int, unknown: int) -> list[FrameObservation]:
    court_positions = [(0.25, 0.5)] * left + [(0.75, 0.5)] * right + [None] * unknown
    return [
        FrameObservation(
            frame_index=frame_index,
            players=(
                PlayerObservation(
                    frame_index=frame_index,
                    source_track_id=_TRACK_ID,
                    track_id=_TRACK_ID,
                    frame_bbox=(0.1, 0.2, 0.3, 0.8),
                    frame_foot_pos=(0.2, 0.8),
                    court_pos=court_pos,
                    confidence=0.9,
                ),
            ),
            homography_available=court_pos is not None,
        )
        for frame_index, court_pos in enumerate(court_positions)
    ]


def _feature_side(*, left: int, right: int, unknown: int) -> CourtSide:
    frames = _frames_for_side_counts(left=left, right=right, unknown=unknown)
    return resolve_track_court_sides(frames)[_TRACK_ID]


@pytest.mark.parametrize(
    ("left", "right", "unknown", "expected"),
    [
        pytest.param(110, 0, 131, "left", id="consistent-known-outvotes-more-unknown"),
        pytest.param(4, 0, 96, "left", id="known-evidence-outvotes-unknown"),
        pytest.param(9, 2, 0, "left", id="dominant-left-outvotes-right"),
        pytest.param(2, 9, 0, "right", id="dominant-right-outvotes-left"),
        pytest.param(5, 5, 0, "unknown", id="known-sides-tie"),
        pytest.param(0, 0, 10, "unknown", id="all-observations-unknown"),
    ],
)
def test_feature_side_uses_volley_reid_known_side_majority(
    left: int,
    right: int,
    unknown: int,
    expected: CourtSide,
) -> None:
    assert _feature_side(left=left, right=right, unknown=unknown) == expected


def test_zero_descriptor_is_downgraded_to_missing_instead_of_fabricated() -> None:
    assert _descriptor_bytes([(0.0,) * 384]) is None


def _player(
    *,
    frame_index: int,
    track_id: int,
    court_pos: tuple[float, float],
    confidence: float = 0.9,
) -> PlayerObservation:
    return PlayerObservation(
        frame_index=frame_index,
        source_track_id=track_id,
        track_id=track_id,
        frame_bbox=(0.1, 0.2, 0.3, 0.8),
        frame_foot_pos=(0.2, 0.8),
        court_pos=court_pos,
        confidence=confidence,
    )


def test_final_side_capacity_drops_transient_seventh_track_instead_of_failing() -> None:
    right_players = tuple(
        _player(frame_index=0, track_id=track_id, court_pos=(0.75, 0.5)) for track_id in range(1, 7)
    )
    frames = [
        FrameObservation(
            frame_index=0,
            players=(
                *right_players,
                _player(
                    frame_index=0,
                    track_id=7,
                    court_pos=(0.75, 0.5),
                    confidence=0.1,
                ),
            ),
            homography_available=True,
        ),
        *[
            FrameObservation(
                frame_index=frame_index,
                players=(
                    *(
                        _player(
                            frame_index=frame_index,
                            track_id=track_id,
                            court_pos=(0.25, 0.5),
                        )
                        for track_id in range(8, 14)
                    ),
                    _player(
                        frame_index=frame_index,
                        track_id=7,
                        court_pos=(0.25, 0.5),
                        confidence=0.1,
                    ),
                ),
                homography_available=True,
            )
            for frame_index in (1, 2)
        ],
    ]

    selected = _select_on_court_roster(frames)
    sides = resolve_track_court_sides(frames, selected_by_frame=selected)
    assert sides[7] == "right"
    assert len(selected[0]) == 7

    constrained = _enforce_fixed_roster_capacity(
        frames,
        selected_by_frame=selected,
        sides=sides,
    )

    assert constrained[0] == set(range(1, 7))
