"""Provider Work base-analysis evidence artifact regression tests."""

from datetime import UTC, datetime, timedelta

from volleyball_monitoring_ai import (
    AIJobRequest,
    PlayerCropSourceManifest,
    decode_person_pose_evidence_chunk,
)

from volleyball_analysis_engine.evidence_artifacts import (
    AnalysisEvidenceArtifacts,
    build_analysis_evidence_artifacts,
)
from volleyball_analysis_engine.records import PersonPoseObservation, PersonPoseStatus


def job() -> AIJobRequest:
    expiry = datetime.now(UTC) + timedelta(hours=1)
    return AIJobRequest.model_validate(
        {
            "schema_version": "3.0.0",
            "ai_job_id": "job-evidence-1",
            "rally_submission_id": "submission-1",
            "rally_id": "rally-1",
            "match_id": "match-1",
            "annotation_revision": "1",
            "clip": {
                "clip_asset_id": "clip-1",
                "download_url": "https://example.test/clip.mp4",
                "download_url_expires_at": expiry.isoformat(),
                "sha256": "a" * 64,
                "byte_length": "123",
                "content_type": "video/mp4",
                "video": {
                    "width": 1920,
                    "height": 1080,
                    "fps": {"num": 60, "den": 1},
                    "time_base": {"num": 1, "den": 60},
                    "total_frames": "3",
                    "duration_us": "50000",
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
            "outcome": {
                "score_resolution": "pending",
                "scoring_court_side": None,
            },
            "callback": {
                "url": "https://example.test/callback",
                "token": "x" * 32,
                "expires_at": expiry.isoformat(),
            },
        }
    )


def observation(
    frame_index: int,
    *,
    status: PersonPoseStatus = "AVAILABLE",
) -> PersonPoseObservation:
    keypoints = tuple((0.5, 0.5, 0.9) for _ in range(17))
    return PersonPoseObservation(
        frame_index=frame_index,
        track_id=7,
        bbox_source="DETECTOR",
        frame_bbox=(0.4, 0.2, 0.6, 0.8),
        crop_transform=(1 / 1920, 1 / 1080, 0.4, 0.2),
        status=status,
        keypoints=keypoints,
    )


def artifact_bytes(result: AnalysisEvidenceArtifacts, part_name: str) -> bytes:
    artifact = next(item for item in result.artifacts if item.part_name == part_name)
    assert isinstance(artifact.data, bytes)
    return artifact.data


def test_analysis_evidence_covers_empty_frames_without_fabricating_pose() -> None:
    result = build_analysis_evidence_artifacts(
        job=job(),
        analysis_run_id="analysis-1",
        analysis_data_bytes=b"\x00\x00\x00\x00VAD1test",
        poses={
            0: (observation(0),),
            1: (),
            2: (observation(2, status="LOW_QUALITY"),),
        },
        pose_recipe={
            "namespace": "pose/test/every-frame-v1",
            "model_name": "pose-test",
            "checkpoint_sha256": "b" * 64,
            "preprocess_version": "crop-v1",
            "keypoint_layout": "COCO_17",
            "coordinate_space": "NORMALIZED_VIDEO",
        },
        chunk_frame_count=2,
    )

    assert result.manifest.analysis_run_id == "analysis-1"
    assert sum(artifact.kind == "PERSON_POSE_EVIDENCE_CHUNK" for artifact in result.artifacts) == 2
    first = decode_person_pose_evidence_chunk(artifact_bytes(result, "pose_chunk_0000"))
    second = decode_person_pose_evidence_chunk(artifact_bytes(result, "pose_chunk_0001"))
    assert [len(frame) for frame in first.frames] == [1, 0]
    assert second.start_frame_index == 2
    assert second.frames[0][0].status == "LOW_QUALITY"


def test_crop_source_points_to_the_same_clip_and_pose_manifest() -> None:
    result = build_analysis_evidence_artifacts(
        job=job(),
        analysis_run_id="analysis-2",
        analysis_data_bytes=b"\x00\x00\x00\x00VAD1test",
        poses={0: (), 1: (), 2: ()},
        pose_recipe={
            "namespace": "pose/test/every-frame-v1",
            "model_name": "pose-test",
            "checkpoint_sha256": "b" * 64,
            "preprocess_version": "crop-v1",
            "keypoint_layout": "COCO_17",
            "coordinate_space": "NORMALIZED_VIDEO",
        },
    )

    crop = PlayerCropSourceManifest.model_validate_json(
        artifact_bytes(result, "crop_source_manifest")
    )
    assert crop.clip_artifact.artifact_id == "clip-1"
    assert crop.pose_manifest_artifact.sha256 == result.manifest.pose_manifest_artifact.sha256
    assert crop.crop_recipe.decode_alignment == "CANONICAL_FRAME_INDEX"
