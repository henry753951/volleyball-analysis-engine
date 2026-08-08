"""Canonical frame and contract integration tests for the pipeline."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import orjson
from volleyball_monitoring_ai import AIJobRequest, validate_overlay_bytes

from volleyball_analysis_engine.pipeline import AnalysisPipeline, PipelineConfig


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(orjson.dumps(record) + b"\n" for record in records))


def fixture_root(tmp_path: Path) -> Path:
    root = tmp_path / "handoff"
    keypoints = [
        {"index": index, "x_px": x, "y_px": y, "confidence": 0.99, "world_pos_m": [wx, wy]}
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
    ]
    write_jsonl(
        root / "tracking-data" / "court-keypoints.jsonl",
        [{"frame_index": frame, "available": True, "keypoints": keypoints} for frame in range(3)],
    )
    write_jsonl(
        root / "tracking-data" / "tracks-sam-deep-eiou.jsonl",
        [
            {
                "frame_index": frame,
                "players": [
                    {
                        "track_id": 1 if frame < 2 else 7,
                        "frame_bbox": {"x1": 0.08, "y1": 0.2, "x2": 0.28, "y2": 0.7},
                        "frame_foot_pos": {"x": 0.18, "y": 0.7},
                        "court_pos": {"x": 0.2, "y": 0.5},
                        "confidence": 0.9,
                    }
                ],
            }
            for frame in range(3)
        ],
    )
    ball = {
        "coordinate_space": "normalized-video-frame",
        "points": [
            {"clip_frame_index": str(frame), "frame_pos": {"x": 0.18, "y": 0.45}}
            for frame in range(3)
        ],
    }
    (root / "input").mkdir(parents=True)
    (root / "input" / "ball-annotations.manual.json").write_bytes(orjson.dumps(ball))
    return root


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
    bundle = AnalysisPipeline(PipelineConfig(fixture_root(tmp_path))).analyze(incoming)
    assert [event.anchor_frame_index for event in bundle.result.contact_events] == ["0", "2"]
    assert bundle.result.path_segments[0].start_frame_index == "0"
    assert bundle.result.path_segments[0].end_frame_index == "2"
    assert bundle.result.extensions["canonical_frame_count"] == 3
    validate_overlay_bytes(bundle.overlay_bytes)
