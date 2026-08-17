"""Canonical frame and contract integration tests for the pipeline."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from volleyball_monitoring_ai import AIJobRequest, validate_analysis_data_bytes

from volleyball_analysis_engine.inference import InferenceResult
from volleyball_analysis_engine.pipeline import AnalysisPipeline, resolve_track_court_sides
from volleyball_analysis_engine.records import (
    ActionObservation,
    BallObservation,
    CourtFrame,
    CourtKeypoint,
    FrameObservation,
    GroupActivityObservation,
    PlayerObservation,
)


class FakeProvider:
    """In-memory model boundary used only by unit tests."""

    def infer(
        self,
        clip_path: Path,
        job: AIJobRequest,
        report: Callable[[float, str], None],
    ) -> InferenceResult:
        """Return deterministic observations without reading a fixture file."""
        del clip_path, job, report
        court_points = tuple(
            CourtKeypoint(index, (x, y), 0.99, (wx, wy))
            for index, (x, y, wx, wy) in enumerate(
                [
                    (0, 0, 0, 0),
                    (50, 0, 6, 0),
                    (100, 0, 18, 0),
                    (0, 100, 0, 9),
                    (50, 100, 6, 9),
                    (100, 100, 18, 9),
                ]
            )
        )
        players = {
            frame: (
                PlayerObservation(
                    frame_index=frame,
                    source_track_id=1 if frame < 2 else 7,
                    track_id=1 if frame < 2 else 7,
                    frame_bbox=(0.08, 0.2, 0.28, 0.7),
                    frame_foot_pos=(0.18, 0.7),
                    court_pos=None,
                    confidence=0.9,
                ),
            )
            for frame in range(3)
        }
        return InferenceResult(
            players=players,
            courts={frame: CourtFrame(frame, True, court_points) for frame in range(3)},
            balls={frame: BallObservation(frame, (0.18, 0.45), 0.95) for frame in range(3)},
            actions={(0, 1): ActionObservation(0, 1, "setting", 0.88)},
            frame_count=3,
            frame_width=100,
            frame_height=100,
            fps=60.0,
            metadata={
                "provider": "fake",
                "person_pose_recipe": {
                    "namespace": "pose/test/every-frame-v1",
                    "model_name": "pose-test",
                    "checkpoint_sha256": "b" * 64,
                    "preprocess_version": "crop-v1",
                    "keypoint_layout": "COCO_17",
                    "coordinate_space": "NORMALIZED_VIDEO",
                },
            },
        )


class NoPlayersProvider(FakeProvider):
    """Return observed balls and court geometry without any nearby player."""

    def infer(
        self,
        clip_path: Path,
        job: AIJobRequest,
        report: Callable[[float, str], None],
    ) -> InferenceResult:
        """Remove players so representative positions must use ball projection."""
        base = super().infer(clip_path, job, report)
        court_points = tuple(
            CourtKeypoint(index, (x, y), 0.99, (wx, wy))
            for index, (x, y, wx, wy) in enumerate(
                [
                    (0, 0, 0, 0),
                    (50, 0, 9, 0),
                    (100, 0, 18, 0),
                    (100, 100, 18, 9),
                    (50, 100, 9, 9),
                    (0, 100, 0, 9),
                ]
            )
        )
        return replace(
            base,
            players=dict.fromkeys(range(base.frame_count), ()),
            courts={frame: CourtFrame(frame, True, court_points) for frame in range(3)},
            actions={},
        )


class ManyPlayersProvider(FakeProvider):
    """Return seven simultaneous run-local tracks on the same projected side."""

    def infer(
        self,
        clip_path: Path,
        job: AIJobRequest,
        report: Callable[[float, str], None],
    ) -> InferenceResult:
        """Replace the base observations with seven stable simultaneous tracks."""
        base = super().infer(clip_path, job, report)
        track_ids = tuple(range(1, 8))
        players = {
            frame: tuple(
                PlayerObservation(
                    frame_index=frame,
                    source_track_id=track_id,
                    track_id=track_id,
                    frame_bbox=(0.04 * track_id, 0.2, 0.04 * track_id + 0.03, 0.7),
                    frame_foot_pos=(0.04 * track_id + 0.015, 0.7),
                    court_pos=(
                        (0.75, 0.5) if frame == 0 and track_id == 7 else (0.04 * track_id, 0.5)
                    ),
                    confidence=0.9,
                )
                for track_id in track_ids
            )
            for frame in range(3)
        }
        return InferenceResult(
            players=players,
            courts={},
            balls=base.balls,
            actions={(0, 1): ActionObservation(0, 1, "setting", 0.88)},
            frame_count=base.frame_count,
            frame_width=base.frame_width,
            frame_height=base.frame_height,
            fps=base.fps,
            metadata=base.metadata,
        )


class CourtSideEvidenceProvider(FakeProvider):
    """Return known-side tracks with different amounts of conflicting evidence."""

    def infer(
        self,
        clip_path: Path,
        job: AIJobRequest,
        report: Callable[[float, str], None],
    ) -> InferenceResult:
        """Expose conflicting court-side evidence for majority resolution."""
        base = super().infer(clip_path, job, report)
        strong_left = [(0.25, 0.5)] * 110 + [None] * 131
        uncertain = [(0.25, 0.5)] * 107 + [(0.75, 0.5)] * 13 + [None] * 121
        players = {
            frame: tuple(
                PlayerObservation(
                    frame_index=frame,
                    source_track_id=track_id,
                    track_id=track_id,
                    frame_bbox=(0.08 * track_id, 0.2, 0.08 * track_id + 0.05, 0.7),
                    frame_foot_pos=(0.08 * track_id + 0.025, 0.7),
                    court_pos=court_pos,
                    confidence=0.9,
                )
                for track_id, court_pos in ((1, strong_left[frame]), (2, uncertain[frame]))
            )
            for frame in range(len(strong_left))
        }
        return InferenceResult(
            players=players,
            courts={},
            balls=base.balls,
            actions=base.actions,
            frame_count=len(strong_left),
            frame_width=base.frame_width,
            frame_height=base.frame_height,
            fps=base.fps,
            metadata=base.metadata,
        )


def job() -> AIJobRequest:
    expiry = datetime.now(UTC) + timedelta(hours=1)
    return AIJobRequest.model_validate(
        {
            "schema_version": "3.0.0",
            "ai_job_id": "job-1",
            "rally_submission_id": "submission-1",
            "rally_id": "rally-1",
            "match_id": "match-1",
            "annotation_revision": "9",
            "clip": {
                "clip_asset_id": "clip-1",
                "download_url": "https://example.test/clip.mp4",
                "download_url_expires_at": expiry.isoformat(),
                "sha256": "a" * 64,
                "byte_length": "123",
                "content_type": "video/mp4",
                "video": {
                    "width": 100,
                    "height": 100,
                    "fps": {"num": 60, "den": 1},
                    "time_base": {"num": 1, "den": 15360},
                    "total_frames": "3",
                    "duration_us": "50000",
                    "has_audio": True,
                },
            },
            "key_points": [
                {
                    "key_point_id": "point-1",
                    "sequence_index": 0,
                    "marker_kind": "contact",
                    "is_terminal": False,
                    "clip_pts": "0",
                    "clip_time_us": "0",
                    "clip_frame_index": "0",
                },
                {
                    "key_point_id": "point-2",
                    "sequence_index": 1,
                    "marker_kind": "contact",
                    "is_terminal": False,
                    "clip_pts": "512",
                    "clip_time_us": "33333",
                    "clip_frame_index": "2",
                },
            ],
            "boundaries": [
                {
                    "kind": "start",
                    "clip_pts": "0",
                    "clip_time_us": "0",
                    "clip_frame_index": "0",
                },
                {
                    "kind": "end",
                    "clip_pts": "512",
                    "clip_time_us": "33333",
                    "clip_frame_index": "2",
                },
            ],
            "analysis_plan": {
                "mode": "full",
                "modules": {
                    "court": "run",
                    "tracking": "run",
                    "reid": "run",
                    "contacts": "run",
                },
                "source_analysis_data": None,
                "preserve_manual_corrections": True,
            },
            "outcome": {"score_resolution": "resolved", "scoring_court_side": "left"},
            "callback": {
                "url": "https://example.test/callback",
                "token": "x" * 32,
                "expires_at": expiry.isoformat(),
            },
        }
    )


def test_pipeline_preserves_authoritative_keypoint_frames_and_builds_analysis_data(
    tmp_path: Path,
) -> None:
    incoming = job()
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"unit-test")
    bundle = AnalysisPipeline(FakeProvider()).analyze(incoming, clip)
    assert bundle.domain.schema_version == "1.0.0"
    assert bundle.domain.producer.sdk_version == "0.5.0"
    assert [event.anchor_frame_index for event in bundle.domain.contact_events] == ["0", "2"]
    assert bundle.domain.path_segments[0].start_frame_index == "0"
    assert bundle.domain.path_segments[0].end_frame_index == "2"
    assert bundle.domain.extensions["canonical_frame_count"] == 3
    assert bundle.domain.extensions["contact_suggestions"] == []
    assert bundle.domain.extensions["provider_work_boundary"] == "base-analysis-without-identity"
    assert "fixed_roster_reid" not in bundle.domain.extensions
    action = bundle.domain.contact_events[0].actors[0].action
    assert action is not None
    assert action.label == "setting"
    validate_analysis_data_bytes(bundle.analysis_data_bytes)


def test_no_player_events_use_ball_projection_without_terminal_marker(tmp_path: Path) -> None:
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"unit-test")

    bundle = AnalysisPipeline(NoPlayersProvider()).analyze(job(), clip)

    assert [event.association_state for event in bundle.domain.contact_events] == [
        "no_player",
        "no_player",
    ]
    for event in bundle.domain.contact_events:
        assert len(event.representative_court_positions) == 1
        representative = event.representative_court_positions[0]
        assert representative.track_id is None
        assert representative.basis == "terminal_projection"

    path = bundle.domain.path_segments[0]
    assert path.render_state == "complete"
    assert len(path.start_court_positions) == 1
    assert len(path.end_court_positions) == 1


def test_provider_work_base_analysis_does_not_build_or_publish_reid_bank(
    tmp_path: Path,
) -> None:
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"unit-test")

    result = AnalysisPipeline(FakeProvider()).analyze_provider_work(job(), clip)
    assert result.bundle.domain.extensions["provider_work_boundary"] == (
        "base-analysis-without-identity"
    )
    assert "reid" not in result.bundle.domain.extensions
    assert "fixed_roster_reid" not in result.bundle.domain.extensions
    assert result.evidence.manifest.analysis_run_id == result.bundle.domain.analysis_id


def test_pipeline_stores_group_activity_without_interpreting_it(tmp_path: Path) -> None:
    class GroupProvider(FakeProvider):
        def infer(
            self,
            clip_path: Path,
            job: AIJobRequest,
            report: Callable[[float, str], None],
        ) -> InferenceResult:
            base = super().infer(clip_path, job, report)
            return replace(
                base,
                group_activities={
                    0: GroupActivityObservation(0, "r_pass", 0.91),
                    2: GroupActivityObservation(2, "l_spike", 0.87),
                },
            )

    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"unit-test")
    bundle = AnalysisPipeline(GroupProvider()).analyze(job(), clip)

    assert bundle.domain.extensions["group_activity"] == {
        "status": "stored_not_interpreted",
        "taxonomy": None,
        "frames": [
            {"frame_index": 0, "label": "r_pass", "confidence": 0.91},
            {"frame_index": 2, "label": "l_spike", "confidence": 0.87},
        ],
    }


def test_pipeline_does_not_generate_contact_proposals_over_manual_markers(
    tmp_path: Path,
) -> None:
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"unit-test")

    with patch("volleyball_analysis_engine.pipeline.detect_contact_proposals") as detect_proposals:
        bundle = AnalysisPipeline(FakeProvider()).analyze(job(), clip)

    detect_proposals.assert_not_called()
    assert [event.anchor_origin for event in bundle.domain.contact_events] == [
        "human_anchor",
        "human_anchor",
    ]
    assert bundle.domain.extensions["contact_suggestions"] == []


def test_pipeline_uses_v2_boundaries_when_no_human_contacts_are_supplied(
    tmp_path: Path,
) -> None:
    payload = job().model_dump(mode="json")
    payload.update(
        {
            "schema_version": "3.0.0",
            "boundaries": [
                {
                    "kind": "start",
                    "clip_pts": "0",
                    "clip_time_us": "0",
                    "clip_frame_index": "0",
                },
                {
                    "kind": "end",
                    "clip_pts": "512",
                    "clip_time_us": "33333",
                    "clip_frame_index": "2",
                },
            ],
            "key_points": [],
            "outcome": {"score_resolution": "pending", "scoring_court_side": None},
        }
    )
    incoming = AIJobRequest.model_validate(payload)
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"unit-test")

    bundle = AnalysisPipeline(FakeProvider()).analyze(incoming, clip)

    assert bundle.domain.contact_events == []
    assert bundle.domain.path_segments == []
    assert bundle.domain.extensions["contact_suggestions"] == []


def test_pipeline_keeps_all_run_local_tracks_without_fixed_roster_filtering(
    tmp_path: Path,
) -> None:
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"unit-test")

    bundle = AnalysisPipeline(ManyPlayersProvider()).analyze(job(), clip)

    assert [track.track_id for track in bundle.domain.tracks] == [1, 2, 3, 4, 5, 6, 7]
    assert "fixed_roster_reid" not in bundle.domain.extensions


def test_pipeline_uses_majority_track_side_resolution(
    tmp_path: Path,
) -> None:
    payload = job().model_dump(mode="json")
    payload["clip"]["video"].update(
        {
            "total_frames": "241",
            "duration_us": "4016667",
        }
    )
    incoming = AIJobRequest.model_validate(payload)
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"unit-test")

    bundle = AnalysisPipeline(CourtSideEvidenceProvider()).analyze(incoming, clip)

    track_sides = {track.track_id: track.court_side for track in bundle.domain.tracks}
    assert track_sides == {1: "left", 2: "left"}


def test_track_side_abstains_for_midline_jitter_and_short_evidence() -> None:
    frames = [
        FrameObservation(
            frame_index=frame_index,
            players=(
                PlayerObservation(
                    frame_index=frame_index,
                    source_track_id=1,
                    track_id=1,
                    frame_bbox=(0.4, 0.2, 0.6, 0.8),
                    frame_foot_pos=(0.5, 0.8),
                    court_pos=(court_x, 0.5),
                    confidence=0.9,
                ),
            ),
            homography_available=True,
        )
        for frame_index, court_x in enumerate((0.48, 0.51, 0.47, 0.52, 0.49, 0.5))
    ]

    assert resolve_track_court_sides(frames) == {1: "unknown"}
