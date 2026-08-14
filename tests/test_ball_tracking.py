"""Temporal ball candidate identity tests."""

from volleyball_analysis_engine.ball_tracking import (
    BallTrackingConfig,
    BallTrajectoryTracker,
)


def test_tracker_rejects_a_higher_confidence_distant_false_positive() -> None:
    tracker = BallTrajectoryTracker()
    assert tracker.update(0, [(0.10, 0.20)], [0.8]) is not None
    assert tracker.update(1, [(0.12, 0.20)], [0.8]) is not None

    selected = tracker.update(2, [(0.14, 0.20), (0.82, 0.08)], [0.72, 0.99])

    assert selected is not None
    assert selected.frame_pos == (0.14, 0.20)


def test_tracker_does_not_emit_predicted_points_during_detection_loss() -> None:
    tracker = BallTrajectoryTracker()
    tracker.update(0, [(0.10, 0.20)], [0.9])

    assert tracker.update(1, [], []) is None


def test_tracker_reacquires_only_after_sustained_loss_resets_identity() -> None:
    tracker = BallTrajectoryTracker(BallTrackingConfig(maximum_missed_frames=1))
    tracker.update(0, [(0.10, 0.20)], [0.9])
    assert tracker.update(1, [], []) is None
    assert tracker.update(2, [], []) is None

    reacquired = tracker.update(3, [(0.80, 0.10)], [0.85])

    assert reacquired is not None
    assert reacquired.frame_pos == (0.80, 0.10)
