"""Visual-v5 compatibility milestone tests."""

import numpy as np

from volleyball_analysis_engine.records import CourtFrame, CourtKeypoint
from volleyball_analysis_engine.visual_v5 import (
    _action_states_for_frame,  # pyright: ignore[reportPrivateUsage]
    _draw_court_keypoints,  # pyright: ignore[reportPrivateUsage]
    path_scroll_offset,
    preview_frame_indices,
)


def test_action_highlight_uses_contact_anchor_not_actor_observation_frame() -> None:
    events = [
        {
            "anchor_frame_index": "20",
            "resolved_frame_index": "16",
            "actors": [{"track_id": "7", "action": {"label": "setting"}}],
        }
    ]

    assert _action_states_for_frame(events, frame_index=16, fps=30.0) == {}
    assert _action_states_for_frame(events, frame_index=20, fps=30.0) == {7: ("setting", True)}


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


def test_accepted_court_draws_complete_virtual_lines() -> None:
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    positions = (
        (100.0, 600.0),
        (160.0, 440.0),
        (210.0, 330.0),
        (260.0, 220.0),
        (320.0, 100.0),
        (960.0, 100.0),
        (1020.0, 220.0),
        (1070.0, 330.0),
        (1120.0, 440.0),
        (1180.0, 600.0),
    )
    court = CourtFrame(
        frame_index=0,
        available=True,
        keypoints=tuple(
            CourtKeypoint(index, position, 1.0, None) for index, position in enumerate(positions)
        ),
    )

    assert _draw_court_keypoints(frame, court, 1280, 720)
    assert np.any(frame[598:603, 630:650])
