"""Visual-v5 compatibility milestone tests."""

from volleyball_analysis_engine.visual_v5 import preview_frame_indices


def test_preview_frames_match_contract_lab_visual_milestones() -> None:
    result = {
        "path_segments": [{"end_frame_index": 234}],
        "contact_events": [
            {"anchor_frame_index": 120, "is_terminal": False},
            {
                "anchor_frame_index": 850,
                "resolved_frame_index": 850,
                "is_terminal": True,
            },
        ],
    }

    assert preview_frame_indices(result, fps=60.0, total_frames=1033) == (243, 852)


def test_preview_frames_are_clamped_to_decoded_clip() -> None:
    result = {
        "path_segments": [{"end_frame_index": 99}],
        "contact_events": [{"anchor_frame_index": 99, "is_terminal": True}],
    }

    assert preview_frame_indices(result, fps=60.0, total_frames=100) == (99, 99)
