"""Tests for the isolated predictions importer."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from volleyball_analysis_engine.contact_detection import ContactProposal
from volleyball_analysis_engine.predictions_import import (
    ContactPhaseCandidate,
    PredictionIndex,
    SourceMetadata,
    _build_contact_events,  # pyright: ignore[reportPrivateUsage]
    _candidate_semantic,  # pyright: ignore[reportPrivateUsage]
    _contact_phase_candidates,  # pyright: ignore[reportPrivateUsage]
    build_prediction_index,
    create_plan,
    detect_segments,
    load_prediction_index,
)


def test_contact_events_preserve_human_key_points_instead_of_adding_ai_events() -> None:
    point = SimpleNamespace(
        key_point_id="manual-point-1",
        clip_frame_index="12",
        clip_pts="3072",
        clip_time_us="200000",
        marker_kind="contact",
        is_terminal=False,
    )
    job = SimpleNamespace(
        key_points=[point],
        clip=SimpleNamespace(video=SimpleNamespace(total_frames="60")),
    )

    events = _build_contact_events(
        job=job,
        phase_candidates=[],
        proposals=[],
        balls={},
        players={},
        actions={},
        poses={},
        homographies={},
        fps=60.0,
        frame_width=1920,
        frame_height=1080,
    )

    assert [(event["key_point_id"], event["anchor_origin"]) for event in events] == [
        ("manual-point-1", "human_anchor")
    ]
    assert events[0]["source_key_point_id"] == "manual-point-1"
    assert events[0]["anchor_frame_index"] == "12"


def test_contact_events_deduplicate_candidates_aligned_to_the_same_frame() -> None:
    job = SimpleNamespace(
        key_points=[],
        rally_submission_id="submission-1",
        clip=SimpleNamespace(video=SimpleNamespace(total_frames="30")),
    )
    proposal = ContactProposal(
        frame_index=10,
        confidence=0.9,
        direction_change=0.5,
        acceleration=0.5,
        speed_ratio=0.5,
        model_improvement=0.5,
        prediction_error=0.5,
    )

    events = _build_contact_events(
        job=job,
        phase_candidates=[
            ContactPhaseCandidate(8, 9, "l_set", 0.8),
            ContactPhaseCandidate(10, 12, "l_spike", 0.7),
        ],
        proposals=[proposal],
        balls={},
        players={},
        actions={},
        poses={},
        homographies={},
        fps=10.0,
        frame_width=1920,
        frame_height=1080,
    )

    assert len(events) == 1
    assert events[0]["sequence_index"] == 0
    assert events[0]["anchor_frame_index"] == "10"


def test_candidate_semantic_maps_legacy_side_and_ball_type() -> None:
    assert _candidate_semantic(ContactPhaseCandidate(1, 2, "l_pass", 0.9)) == {
        "court_side": "left",
        "phase": "pass",
        "ball_event_kind": "receive",
    }
    assert _candidate_semantic(ContactPhaseCandidate(3, 4, "r_spike", 0.8)) == {
        "court_side": "right",
        "phase": "spike",
        "ball_event_kind": "spike",
    }


def _index(activity_ids: list[int], *, fps: float = 10.0) -> PredictionIndex:
    return PredictionIndex(
        source_path=Path("source.json"),
        source_size=1,
        source_mtime_ns=1,
        metadata=SourceMetadata(
            width=1920,
            height=1080,
            fps=fps,
            frame_count=len(activity_ids),
            duration_sec=len(activity_ids) / fps,
            checkpoint=None,
        ),
        offsets=np.arange(len(activity_ids), dtype=np.uint64),
        activity_ids=np.asarray(activity_ids, dtype=np.uint8),
        activity_confidences=np.ones(len(activity_ids), dtype=np.float32),
        activity_names={0: "non_rally", 1: "l_pass"},
    )


def test_detect_segments_closes_short_gaps_and_removes_noise() -> None:
    source = _index([0] * 20 + [1] * 30 + [0] * 5 + [1] * 30 + [0] * 20)
    segments = detect_segments(
        source,
        smooth_seconds=0,
        max_gap_seconds=1,
        min_active_seconds=2,
        padding_before_seconds=0,
        padding_after_seconds=0,
    )
    assert [(item.source_start_frame, item.source_end_frame_exclusive) for item in segments] == [
        (20, 85)
    ]


def test_contact_phase_candidates_reject_pass_only_noise_and_merge_short_blips() -> None:
    labels = (
        ["l_pass"] * 10
        + ["l_winpoint"] * 15
        + ["l_pass"] * 10
        + ["l_set"]
        + ["l_pass"] * 9
        + ["l_set"] * 10
        + ["l_spike"] * 10
    )
    candidates = _contact_phase_candidates(
        [(frame, label, 0.9) for frame, label in enumerate(labels)],
        fps=10.0,
    )

    assert [(item.frame_index, item.end_frame_index, item.label) for item in candidates] == [
        (25, 44, "l_pass"),
        (45, 54, "l_set"),
        (55, 64, "l_spike"),
    ]


def test_build_and_reload_index_without_loading_whole_export(tmp_path: Path) -> None:
    source = tmp_path / "predictions.json"
    rows: list[dict[str, Any]] = [
        {
            "frame_index": index,
            "time_sec": index / 10,
            "detections": [],
            "court": {"valid": False, "keypoints": []},
            "group_activity": {
                "id": int(index >= 2),
                "name": "l_pass" if index >= 2 else "non_rally",
                "confidence": 0.9,
            },
            "ball": {"visible": False, "center_xy": [None, None], "confidence": 0},
        }
        for index in range(5)
    ]
    with source.open("w", encoding="utf-8") as handle:
        handle.write("{\n")
        handle.write(
            '  "video": {"width":1920,"height":1080,"fps":10.0,'
            '"source_frame_count":5,"source_duration_sec":0.5},\n'
        )
        handle.write('  "checkpoint": "/models/best.pth",\n')
        handle.write('  "frames": [\n')
        for index, row in enumerate(rows):
            handle.write("    " + json.dumps(row, separators=(",", ":")))
            handle.write(",\n" if index < len(rows) - 1 else "\n")
        handle.write("  ]\n}\n")

    index_path = tmp_path / "predictions.index.npz"
    built = build_prediction_index(source, index_path)
    loaded = load_prediction_index(index_path)

    assert built.metadata.frame_count == loaded.metadata.frame_count == 5
    assert loaded.activity_names == {0: "non_rally", 1: "l_pass"}
    assert loaded.offsets.tolist() == sorted(loaded.offsets.tolist())

    plan_path = tmp_path / "plan.json"
    plan = create_plan(
        index_path,
        plan_path,
        smooth_seconds=0,
        max_gap_seconds=0,
        min_active_seconds=0.1,
        padding_before_seconds=0,
        padding_after_seconds=0,
    )
    assert len(plan["segments"]) == 1
    assert plan_path.exists()
