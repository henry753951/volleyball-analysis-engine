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
