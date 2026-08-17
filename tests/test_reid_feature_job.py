"""Independent ReID feature job tests over immutable base-analysis evidence."""

# pyright: reportUnknownArgumentType=false, reportUnknownMemberType=false

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import cv2
import numpy as np
from numpy.typing import NDArray
from volleyball_monitoring_ai import (
    AIJobRequest,
    AnalysisDomainData,
    ReidFeatureJobRequest,
    build_analysis_data,
)

from volleyball_analysis_engine.config import Settings
from volleyball_analysis_engine.evidence_artifacts import build_analysis_evidence_artifacts
from volleyball_analysis_engine.records import PersonPoseObservation
from volleyball_analysis_engine.reid_feature_job import (
    ReidFeatureInputs,
    build_reid_feature_artifacts,
)
from volleyball_analysis_engine.worker import provider_work_capabilities


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def _semantic(payload: dict[str, Any]) -> dict[str, Any]:
    return {**payload, "content_sha256": hashlib.sha256(_json_bytes(payload)).hexdigest()}


def _job() -> AIJobRequest:
    expiry = datetime.now(UTC) + timedelta(hours=1)
    return AIJobRequest.model_validate(
        {
            "schema_version": "3.0.0",
            "ai_job_id": "job-1",
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
                    "width": 100,
                    "height": 100,
                    "fps": {"num": 10, "den": 1},
                    "time_base": {"num": 1, "den": 10},
                    "total_frames": "4",
                    "duration_us": "400000",
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
                    "clip_pts": "3",
                    "clip_time_us": "300000",
                    "clip_frame_index": "3",
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


def _keypoints(x1: float, x2: float) -> tuple[tuple[float, float, float], ...]:
    points = [[(x1 + x2) / 2, 0.5, 0.8] for _ in range(17)]
    points[5] = [x1 + 0.04, 0.32, 0.9]
    points[6] = [x2 - 0.04, 0.32, 0.9]
    points[11] = [x1 + 0.05, 0.62, 0.9]
    points[12] = [x2 - 0.05, 0.62, 0.9]
    return tuple(tuple(value) for value in points)  # type: ignore[return-value]


def _write_video(path: Path) -> None:
    fourcc = cast("Any", cv2).VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, 10, (100, 100))
    assert writer.isOpened()
    for index in range(4):
        frame = np.full((100, 100, 3), 20 + index * 10, dtype=np.uint8)
        cv2.rectangle(frame, (20, 10), (45, 90), (255, 100, 20), -1)
        cv2.rectangle(frame, (55, 10), (80, 90), (20, 100, 255), -1)
        writer.write(frame)
    writer.release()


class FakeNestedWithoutPose:
    """Descriptor fake deliberately has no pose model or pose inference method."""

    def encode_dino_crops(self, crops: list[NDArray[np.uint8]]) -> NDArray[np.float32]:
        """Return deterministic normalized DINO-like vectors."""
        values = np.ones((len(crops), 384), dtype=np.float32)
        return values / np.linalg.norm(values, axis=1, keepdims=True)

    def encode_kpr_crops(
        self,
        *,
        keys: list[tuple[int, int]],
        crops: list[NDArray[np.uint8]],
        prompts: list[list[list[float]] | None],
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.bool_]]:
        """Return KPR-like values while requiring supplied saved-pose prompts."""
        assert len(keys) == len(crops) == len(prompts)
        assert all(prompt is not None for prompt in prompts)
        values = np.ones((len(crops), 9, 512), dtype=np.float32)
        return values, values * 2, np.ones(len(crops), dtype=np.bool_)


class FakeOsnet:
    """Small deterministic OSNet replacement."""

    def encode(self, crops: list[NDArray[np.uint8]], *, batch_size: int) -> NDArray[np.float32]:
        """Return normalized OSNet-like vectors."""
        assert batch_size > 0
        values = np.ones((len(crops), 512), dtype=np.float32)
        return values / np.linalg.norm(values, axis=1, keepdims=True)


def test_feature_job_reuses_saved_pose_and_emits_traceable_multimodal_evidence(
    tmp_path: Path,
) -> None:
    job = _job()
    clip_path = tmp_path / "clip.avi"
    _write_video(clip_path)
    analysis_run_id = str(uuid4())
    evidence_set_id = str(uuid4())
    frame_records: list[dict[str, Any]] = []
    poses: dict[int, tuple[PersonPoseObservation, ...]] = {}
    for frame_index in range(4):
        players: list[dict[str, Any]] = []
        observations: list[PersonPoseObservation] = []
        for track_id, side, bbox in (
            (1, "left", (0.20, 0.10, 0.45, 0.90)),
            (2, "right", (0.55, 0.10, 0.80, 0.90)),
        ):
            players.append(
                {
                    "track_id": track_id,
                    "frame_bbox": dict(zip(("x1", "y1", "x2", "y2"), bbox, strict=True)),
                    "frame_foot_pos": {"x": (bbox[0] + bbox[2]) / 2, "y": bbox[3]},
                    "court_pos": {"x": 0.25 if side == "left" else 0.75, "y": 0.5},
                    "confidence": 0.9,
                }
            )
            observations.append(
                PersonPoseObservation(
                    frame_index=frame_index,
                    track_id=track_id,
                    bbox_source="DETECTOR",
                    frame_bbox=bbox,
                    crop_transform=(0.01, 0.01, bbox[0], bbox[1]),
                    status="AVAILABLE",
                    keypoints=_keypoints(bbox[0], bbox[2]),
                )
            )
        frame_records.append({"frame_index": frame_index, "players": players})
        poses[frame_index] = tuple(observations)
    domain = AnalysisDomainData.model_validate(
        {
            "schema_version": "1.0.0",
            "analysis_id": analysis_run_id,
            "analysis_version": "test",
            "ai_job_id": job.ai_job_id,
            "rally_submission_id": job.rally_submission_id,
            "rally_id": job.rally_id,
            "match_id": job.match_id,
            "annotation_revision": job.annotation_revision,
            "clip_asset_id": job.clip.clip_asset_id,
            "input_clip_sha256": job.clip.sha256,
            "producer": {"name": "test", "build_id": "test"},
            "tracks": [
                {
                    "track_id": 1,
                    "court_side": "left",
                    "first_frame_index": "0",
                    "last_frame_index": "3",
                },
                {
                    "track_id": 2,
                    "court_side": "right",
                    "first_frame_index": "0",
                    "last_frame_index": "3",
                },
            ],
            "contact_events": [],
            "path_segments": [],
            "summary": {
                "track_count": 2,
                "contact_event_count": 0,
                "path_segment_count": 0,
                "unresolved_event_count": 0,
            },
        }
    )
    analysis_data = build_analysis_data(
        job,
        domain=domain,
        frame_records=frame_records,
        ball_positions={},
    )
    recipe = {
        "namespace": "pose/test/every-frame-v1",
        "model_name": "test-pose",
        "checkpoint_sha256": "b" * 64,
        "preprocess_version": "tracked-player-crop-normalized-v1",
        "keypoint_layout": "COCO_17",
        "coordinate_space": "NORMALIZED_VIDEO",
    }
    evidence = build_analysis_evidence_artifacts(
        job=job,
        analysis_run_id=analysis_run_id,
        analysis_data_bytes=analysis_data,
        poses=poses,
        pose_recipe=recipe,
        chunk_frame_count=2,
    )
    artifacts_by_kind: dict[str, list[bytes]] = {}
    for artifact in evidence.artifacts:
        data = artifact.read_bytes()
        artifacts_by_kind.setdefault(artifact.kind, []).append(data)
    roster = _semantic(
        {
            "schema_version": "1.0.0",
            "roster_snapshot_id": str(uuid4()),
            "match_id": job.match_id,
            "rally_submission_id": job.rally_submission_id,
            "as_of_position": {"set_number": 1, "rally_ordinal": 1},
            "teams": [
                {
                    "team_id": "left-team",
                    "court_side": "LEFT",
                    "entries": [
                        {
                            "roster_entry_id": "left-11",
                            "player_id": "player-11",
                            "jersey_number": "11",
                            "display_name": "Left 11",
                            "position": "OH",
                            "active": True,
                        }
                    ],
                },
                {
                    "team_id": "right-team",
                    "court_side": "RIGHT",
                    "entries": [
                        {
                            "roster_entry_id": "right-2",
                            "player_id": "player-2",
                            "jersey_number": "2",
                            "display_name": "Right 2",
                            "position": "S",
                            "active": True,
                        }
                    ],
                },
            ],
        }
    )
    request = ReidFeatureJobRequest.model_validate(
        {
            "schema_version": "2.0.0",
            "provider_job_id": str(uuid4()),
            "evidence_set_id": evidence_set_id,
            "analysis_run_id": analysis_run_id,
            "match_id": job.match_id,
            "analysis_evidence_artifact_id": "analysis-evidence",
            "roster_snapshot_artifact_id": "roster-snapshot",
            "pose_recipe_namespace": recipe["namespace"],
            "frame_selection_recipe_version": "reid-frame-selection/pose-frontality-v1",
            "requested_recipes": [
                {"modality": "DINO", "model_namespace": "dinov2/vits14-reg/v1"},
                {"modality": "OSNET", "model_namespace": "sports-osnet/x1/v1"},
                {
                    "modality": "KPR_PROMPT",
                    "model_namespace": "kpr/coco17-prompt/v1",
                },
            ],
        }
    )
    result = build_reid_feature_artifacts(
        request=request,
        inputs=ReidFeatureInputs(
            clip_path=clip_path,
            analysis_data=analysis_data,
            analysis_manifest=json.loads(artifacts_by_kind["ANALYSIS_EVIDENCE_MANIFEST"][0]),
            pose_manifest=json.loads(artifacts_by_kind["PERSON_POSE_EVIDENCE_MANIFEST"][0]),
            crop_manifest=json.loads(artifacts_by_kind["PLAYER_CROP_SOURCE_MANIFEST"][0]),
            roster_snapshot=roster,
            pose_chunks=tuple(artifacts_by_kind["PERSON_POSE_EVIDENCE_CHUNK"]),
        ),
        nested=FakeNestedWithoutPose(),  # type: ignore[arg-type]
        osnet=FakeOsnet(),  # type: ignore[arg-type]
        candidate_count=4,
        top_k=2,
        min_gap=1,
    )

    assert result.result.status == "READY"
    assert len(result.result.tracklets) == 2
    assert all(
        {vector.modality for vector in tracklet.vectors} == {"DINO", "OSNET", "KPR_PROMPT"}
        for tracklet in result.result.tracklets
    )
    assert all(len(tracklet.cannot_link_tracklet_ids) == 1 for tracklet in result.result.tracklets)


def test_reid_feature_capability_is_only_advertised_after_explicit_enablement() -> None:
    disabled = provider_work_capabilities(Settings())
    enabled = provider_work_capabilities(Settings(reid_feature_enabled=True))

    assert [item.work_kind for item in disabled.work_capabilities] == ["ANALYSIS"]
    assert [item.work_kind for item in enabled.work_capabilities] == [
        "ANALYSIS",
        "REID_FEATURE_EXTRACTION",
    ]
