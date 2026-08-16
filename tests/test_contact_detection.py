"""Ball-trajectory contact proposal tests."""

from volleyball_analysis_engine.contact_detection import detect_contact_proposals
from volleyball_analysis_engine.records import BallObservation


def ball(frame: int, x: float, y: float, confidence: float = 0.95) -> BallObservation:
    return BallObservation(frame_index=frame, frame_pos=(x, y), confidence=confidence)


def test_detects_abrupt_direction_change_and_preserves_canonical_frame() -> None:
    balls = {frame: ball(frame, frame / 100, (frame / 100) ** 2) for frame in range(20)}
    for frame in range(20, 41):
        balls[frame] = ball(frame, 0.4 - (frame - 20) / 80, 0.04 + (frame - 20) / 120)
    proposals = detect_contact_proposals(
        balls,
        start_frame=0,
        end_frame=40,
        fps=60,
        protected_frames={0, 40},
    )
    assert proposals
    assert abs(proposals[0].frame_index - 20) <= 2
    assert proposals[0].confidence >= 0.5
    assert proposals[0].direction_change > 0.12


def test_rejects_smooth_parabola_and_sparse_gaps() -> None:
    smooth = {
        frame: ball(frame, frame / 100, 0.2 + (frame - 20) ** 2 / 10_000) for frame in range(41)
    }
    sparse = {frame: smooth[frame] for frame in range(0, 41, 8)}
    assert detect_contact_proposals(smooth, start_frame=0, end_frame=40, fps=60) == []
    assert detect_contact_proposals(sparse, start_frame=0, end_frame=40, fps=60) == []


def test_rejects_noisy_apex_that_only_changes_vertical_speed() -> None:
    smooth = {
        frame: ball(
            frame,
            0.25 + frame / 200,
            0.18 + (frame - 24) ** 2 / 9000 + (0.0007 if frame % 2 else -0.0007),
        )
        for frame in range(49)
    }

    assert detect_contact_proposals(smooth, start_frame=0, end_frame=48, fps=60) == []


def test_nms_excludes_existing_human_anchor_neighborhood() -> None:
    balls = {frame: ball(frame, frame / 100, frame / 200) for frame in range(41)}
    for frame in range(20, 41):
        balls[frame] = ball(frame, 0.2 - (frame - 20) / 100, 0.1)
    assert (
        detect_contact_proposals(
            balls,
            start_frame=0,
            end_frame=40,
            fps=60,
            protected_frames={20},
        )
        == []
    )


def test_nms_suppresses_delayed_breakpoint_near_human_anchor() -> None:
    balls = {frame: ball(frame, frame / 100, (frame / 100) ** 2) for frame in range(41)}
    for frame in range(20, 41):
        balls[frame] = ball(frame, 0.4 - (frame - 20) / 80, 0.04 + (frame - 20) / 120)

    assert (
        detect_contact_proposals(
            balls,
            start_frame=0,
            end_frame=40,
            fps=60,
            protected_frames={35},
        )
        == []
    )
