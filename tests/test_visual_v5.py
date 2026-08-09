"""Visual-v5 compatibility milestone tests."""

from volleyball_analysis_engine.visual_v5 import path_scroll_offset, preview_frame_indices


def test_preview_frames_match_contract_lab_visual_milestones() -> None:
    result = {
        "path_segments": [{"end_frame_index": 234}],
        "contact_events": [
            {"anchor_frame_index": 120, "is_terminal": False},
            {
                "anchor_frame_index": 850,
                "resolved_frame_index": 842,
                "is_terminal": True,
            },
        ],
    }

    assert preview_frame_indices(result, fps=60.0, total_frames=1033) == (243, 862)


def test_preview_frames_are_clamped_to_decoded_clip() -> None:
    result = {
        "path_segments": [{"end_frame_index": 99}],
        "contact_events": [{"anchor_frame_index": 99, "is_terminal": True}],
    }

    assert preview_frame_indices(result, fps=60.0, total_frames=100) == (99, 99)


def test_path_rows_scroll_smoothly_when_completed_events_overflow() -> None:
    segments = [{"end_frame_index": frame} for frame in range(10, 130, 10)]

    assert path_scroll_offset(segments, 79, transition_frames=10) == 0.0
    assert 0.0 < path_scroll_offset(segments, 85, transition_frames=10) < 1.0
    assert path_scroll_offset(segments, 90, transition_frames=10) == 1.0
    assert path_scroll_offset(segments, 130, transition_frames=10) == 5.0
