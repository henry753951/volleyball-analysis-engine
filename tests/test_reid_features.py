"""Focused tests for clip-local ReID feature-bank projection."""

from __future__ import annotations

import pytest

from volleyball_analysis_engine.records import (
    CourtSide,
    FrameObservation,
    PlayerObservation,
    ReIdEmbeddingModel,
    ReIdFeatureSnapshot,
    ReIdTrackFeature,
)
from volleyball_analysis_engine.reid_features import build_reid_feature_bank

_TRACK_ID = 7


def _frames_for_side_counts(*, left: int, right: int, unknown: int) -> list[FrameObservation]:
    court_positions = (
        [(0.25, 0.5)] * left
        + [(0.75, 0.5)] * right
        + [None] * unknown
    )
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
    snapshot = ReIdFeatureSnapshot(
        schema_version="1.0.0",
        embedding_model=ReIdEmbeddingModel(
            name="sports-osnet",
            checkpoint_sha256="a" * 64,
            preprocess_version="roi-align-rgb-imagenet-v1",
            dimension=1,
            distance="cosine",
        ),
        features=(
            ReIdTrackFeature(
                track_id=_TRACK_ID,
                prototype=(1.0,),
                sample_count=len(frames),
                first_frame_index=0,
                last_frame_index=len(frames) - 1,
                mean_quality=0.9,
                cannot_link_track_ids=(),
            ),
        ),
    )
    feature_bank = build_reid_feature_bank(snapshot, frames, map_frame=lambda value: value)
    populated_banks = [
        bank for bank in feature_bank["side_feature_banks"] if bank["features"]
    ]
    assert len(populated_banks) == 1
    return populated_banks[0]["court_side"]


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
