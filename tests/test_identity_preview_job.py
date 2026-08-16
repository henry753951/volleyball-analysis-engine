"""Identity preview tests proving saved pose is reused without model inference."""

from __future__ import annotations

import hashlib
import io
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import cv2
import numpy as np
from PIL import Image
from volleyball_monitoring_ai import AIJobRequest, IdentityPreviewJobRequest

from volleyball_analysis_engine.config import Settings
from volleyball_analysis_engine.evidence_artifacts import build_analysis_evidence_artifacts
from volleyball_analysis_engine.identity_preview_job import (
    IdentityPreviewInputs,
    build_identity_preview_artifacts,
)
from volleyball_analysis_engine.records import PersonPoseObservation
from volleyball_analysis_engine.worker import provider_work_capabilities


def _job() -> AIJobRequest:
    expiry = datetime.now(UTC) + timedelta(hours=1)
    return AIJobRequest.model_validate(
        {
            "schema_version": "3.0.0",
            "ai_job_id": "preview-source-job",
            "rally_submission_id": "submission-1",
            "rally_id": "rally-1",
            "match_id": "match-1",
            "annotation_revision": "1",
            "clip": {
                "clip_asset_id": "clip-1",
                "download_url": "https://example.test/clip.avi",
                "download_url_expires_at": expiry.isoformat(),
                "sha256": "a" * 64,
                "byte_length": "123",
                "content_type": "video/mp4",
                "video": {
                    "width": 80,
                    "height": 80,
                    "fps": {"num": 10, "den": 1},
                    "time_base": {"num": 1, "den": 10},
                    "total_frames": "3",
                    "duration_us": "300000",
                    "has_audio": False,
                },
            },
            "key_points": [],
            "boundaries": [
                {
                    "kind": "start",
                    "clip_pts": "0",
                    "clip_time_us": "0",
                    "clip_frame_index": "0",
                },
                {
                    "kind": "end",
                    "clip_pts": "2",
                    "clip_time_us": "200000",
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
            "outcome": {"score_resolution": "pending", "scoring_court_side": None},
            "callback": {
                "url": "https://example.test/callback",
                "token": "x" * 32,
                "expires_at": expiry.isoformat(),
            },
        }
    )


def _write_video(path: Path) -> None:
    fourcc = cast("Any", cv2).VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, 10, (80, 80))
    assert writer.isOpened()
    for index in range(3):
        frame = np.full((80, 80, 3), 20 + index * 20, dtype=np.uint8)
        cv2.rectangle(frame, (22 + index, 8), (48 + index, 75), (30, 180, 240), -1)
        writer.write(frame)
    writer.release()


def _pose(frame_index: int) -> PersonPoseObservation:
    bbox = (0.25 + frame_index * 0.01, 0.10, 0.65 + frame_index * 0.01, 0.95)
    return PersonPoseObservation(
        frame_index=frame_index,
        track_id=7,
        bbox_source="DETECTOR",
        frame_bbox=bbox,
        crop_transform=(0.01, 0.01, bbox[0], bbox[1]),
        status="AVAILABLE",
        keypoints=tuple((0.45, 0.5, 0.9) for _ in range(17)),
    )


def test_preview_uses_exact_saved_pose_frames_and_emits_animated_webp(tmp_path: Path) -> None:
    clip_path = tmp_path / "clip.avi"
    _write_video(clip_path)
    analysis_run_id = str(uuid4())
    evidence = build_analysis_evidence_artifacts(
        job=_job(),
        analysis_run_id=analysis_run_id,
        analysis_data_bytes=b"test-analysis-data",
        poses={index: (_pose(index),) for index in range(3)},
        pose_recipe={
            "namespace": "pose/test/every-frame-v1",
            "model_name": "test-pose",
            "checkpoint_sha256": "b" * 64,
            "preprocess_version": "tracked-player-crop-normalized-v1",
            "keypoint_layout": "COCO_17",
            "coordinate_space": "NORMALIZED_VIDEO",
        },
        chunk_frame_count=2,
    )
    by_kind: dict[str, list[bytes]] = {}
    for artifact in evidence.artifacts:
        by_kind.setdefault(artifact.kind, []).append(artifact.read_bytes())
    pose_manifest = json.loads(by_kind["PERSON_POSE_EVIDENCE_MANIFEST"][0])
    crop_manifest = json.loads(by_kind["PLAYER_CROP_SOURCE_MANIFEST"][0])
    request = IdentityPreviewJobRequest.model_validate(
        {
            "schema_version": "1.1.0",
            "provider_job_id": str(uuid4()),
            "preview_id": str(uuid4()),
            "analysis_run_id": analysis_run_id,
            "tracklet_id": str(uuid4()),
            "canonical_track_id": 7,
            "crop_source_manifest_artifact_id": "crop-source-input",
            "pose_manifest_artifact_id": "pose-manifest-input",
            "selected_frame_indices": ["0", "2"],
            "recipe": {
                "namespace": "identity-preview/animated-webp/v1",
                "output_format": "ANIMATED_WEBP",
                "target_width": 128,
                "crop_padding_ratio": 0.15,
                "frame_duration_ms": 160,
            },
        }
    )
    result = build_identity_preview_artifacts(
        request=request,
        inputs=IdentityPreviewInputs(
            clip_path=clip_path,
            crop_manifest=crop_manifest,
            pose_manifest=pose_manifest,
            pose_chunks=tuple(by_kind["PERSON_POSE_EVIDENCE_CHUNK"]),
        ),
    )

    assert result.result.source_frame_indices == ["0", "2"]
    assert result.result.frame_count == 2
    assert (result.result.width, result.result.height) == (128, 192)
    media = next(artifact for artifact in result.artifacts if artifact.kind == "IDENTITY_PREVIEW")
    media_bytes = media.read_bytes()
    assert hashlib.sha256(media_bytes).hexdigest() == result.result.media_artifact.sha256
    image = Image.open(io.BytesIO(media_bytes))
    image.seek(1)
    assert image.tell() == 1


def test_preview_capability_is_only_advertised_after_explicit_enablement() -> None:
    disabled = provider_work_capabilities(Settings())
    enabled = provider_work_capabilities(Settings(identity_preview_enabled=True))

    assert all(
        item.work_kind != "IDENTITY_PREVIEW_GENERATION" for item in disabled.work_capabilities
    )
    preview = next(
        item
        for item in enabled.work_capabilities
        if item.work_kind == "IDENTITY_PREVIEW_GENERATION"
    )
    assert preview.request_schema_versions == ["1.1.0"]
    assert "PERSON_POSE_EVIDENCE_CHUNK" in preview.accepted_input_artifact_kinds
