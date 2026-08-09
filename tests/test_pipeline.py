"""Canonical frame and contract integration tests for the pipeline."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

from volleyball_monitoring_ai import AIJobRequest, validate_overlay_bytes

from volleyball_analysis_engine.inference import InferenceResult
from volleyball_analysis_engine.pipeline import AnalysisPipeline
from volleyball_analysis_engine.records import (
    ActionObservation,
    BallObservation,
    CourtFrame,
    CourtKeypoint,
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
            balls={
                frame: BallObservation(frame, (0.18, 0.45), 0.95) for frame in range(3)
            },
            actions={(0, 1): ActionObservation(0, 1, "setting", 0.88)},
            frame_count=3,
            frame_width=100,
            frame_height=100,
            fps=60.0,
            metadata={"provider": "fake"},
        )


def job() -> AIJobRequest:
    expiry = datetime.now(UTC) + timedelta(hours=1)
    return AIJobRequest.model_validate(
        {
            "schema_version": "1.1.0",
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
                    "marker_kind": "service",
                    "is_terminal": False,
                    "clip_pts": "0",
                    "clip_time_us": "0",
                    "clip_frame_index": "0",
                },
                {
                    "key_point_id": "point-2",
                    "sequence_index": 1,
                    "marker_kind": "contact",
                    "is_terminal": True,
                    "clip_pts": "512",
                    "clip_time_us": "33333",
                    "clip_frame_index": "2",
                },
            ],
            "outcome": {"score_resolution": "resolved", "scoring_court_side": "left"},
            "callback": {
                "url": "https://example.test/callback",
                "token": "x" * 32,
                "expires_at": expiry.isoformat(),
            },
        }
    )


def test_pipeline_preserves_authoritative_keypoint_frames_and_builds_overlay(
    tmp_path: Path,
) -> None:
    incoming = job()
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"unit-test")
    bundle = AnalysisPipeline(FakeProvider()).analyze(incoming, clip)
    assert [event.anchor_frame_index for event in bundle.result.contact_events] == ["0", "2"]
    assert bundle.result.path_segments[0].start_frame_index == "0"
    assert bundle.result.path_segments[0].end_frame_index == "2"
    assert bundle.result.extensions["canonical_frame_count"] == 3
    action = bundle.result.contact_events[0].actors[0].action
    assert action is not None
    assert action.label == "setting"
    validate_overlay_bytes(bundle.overlay_bytes)
