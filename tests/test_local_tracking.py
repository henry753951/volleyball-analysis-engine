"""Run-local DeepEIOU/SAM3 adapter invariants."""

import numpy as np

from volleyball_analysis_engine.local_tracking import (
    SelectiveSam3Result,
    match_target_boxes,
)


def test_target_box_matching_preserves_detection_indexes_after_reordering() -> None:
    detections = np.asarray(
        [[10.0, 10.0, 30.0, 60.0], [80.0, 10.0, 100.0, 60.0]],
        dtype=np.float32,
    )

    assert match_target_boxes(detections[::-1].copy(), detections) == (1, 0)


def test_sam3_rename_events_are_frame_effective_and_latest_wins() -> None:
    result = SelectiveSam3Result(
        status="completed",
        rename_events=((7, 3, 20), (7, 11, 40)),
        window_count=2,
        swap_count=2,
    )

    assert result.resolve(7, 19) == 7
    assert result.resolve(7, 20) == 3
    assert result.resolve(7, 39) == 3
    assert result.resolve(7, 40) == 11


def test_sam3_never_changes_unrelated_raw_track_ids() -> None:
    result = SelectiveSam3Result(
        status="completed",
        rename_events=((7, 3, 20),),
        window_count=1,
        swap_count=1,
    )

    assert result.resolve(8, 100) == 8
