"""Person-pose crop invariants for detector and tracker observations."""

# pyright: reportPrivateUsage=false

import numpy as np

from volleyball_analysis_engine.person_pose import PersonPoseExtractor


def test_crop_orders_inverted_tracker_bbox_before_normalizing() -> None:
    frame = np.zeros((100, 200, 3), dtype=np.uint8)

    crop = PersonPoseExtractor._crop(  # noqa: SLF001
        frame,
        track_id=7,
        bbox=np.asarray([180.0, 90.0, 20.0, 10.0], dtype=np.float32),
        detector_observed=False,
    )

    assert crop.bbox_source == "TRACKER_PROPAGATED"
    assert crop.frame_bbox == (0.1, 0.1, 0.9, 0.9)
    assert crop.crop_transform == (0.005, 0.01, 0.1, 0.1)
    assert crop.image is not None
    assert crop.image.shape == (80, 160, 3)


def test_crop_marks_non_finite_bbox_unusable() -> None:
    frame = np.zeros((100, 200, 3), dtype=np.uint8)

    crop = PersonPoseExtractor._crop(  # noqa: SLF001
        frame,
        track_id=8,
        bbox=np.asarray([0.0, np.nan, 20.0, 30.0], dtype=np.float32),
        detector_observed=True,
    )

    assert crop.frame_bbox == (0.0, 0.0, 0.0, 0.0)
    assert crop.image is None
