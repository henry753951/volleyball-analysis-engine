"""Build immutable base-analysis evidence artifacts for Provider Work v2."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel
from volleyball_monitoring_ai import (
    AIJobRequest,
    AnalysisEvidenceManifest,
    PersonPoseEvidenceManifest,
    PersonPoseRecord,
    PlayerCropSourceManifest,
    ProviderResultArtifact,
    build_person_pose_evidence_chunk,
)
from volleyball_monitoring_ai.provider_work import ImmutableArtifactReference

from .records import PersonPoseObservation

POSE_CHUNK_FRAME_COUNT = 120
JSON_CONTENT_TYPE = "application/json"
ANALYSIS_DATA_CONTENT_TYPE = "application/vnd.volleyball.analysis-data+flatbuffers;version=1"
POSE_CHUNK_CONTENT_TYPE = "application/vnd.volleyball.person-pose-evidence+flatbuffers;version=1"


@dataclass(frozen=True, slots=True)
class AnalysisEvidenceArtifacts:
    """Callback-ready artifacts plus their validated top-level manifest."""

    artifacts: tuple[ProviderResultArtifact, ...]
    manifest: AnalysisEvidenceManifest


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _semantic_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_json_bytes(payload)).hexdigest()


def _content_addressed_model[ModelT: BaseModel](
    model_type: type[ModelT], payload: dict[str, Any]
) -> ModelT:
    body = {**payload, "content_sha256": _semantic_hash(payload)}
    return model_type.model_validate(body)


def _artifact_reference(
    *,
    artifact_id: str,
    kind: str,
    data: bytes,
    content_type: str,
) -> ImmutableArtifactReference:
    return ImmutableArtifactReference(
        artifact_id=artifact_id,
        kind=kind,
        schema_version="1.0.0",
        sha256=hashlib.sha256(data).hexdigest(),
        byte_length=str(len(data)),
        content_type=content_type,
    )


def _record(observation: PersonPoseObservation) -> PersonPoseRecord:
    return PersonPoseRecord(
        track_id=observation.track_id,
        bbox_source=observation.bbox_source,
        frame_bbox=observation.frame_bbox,
        crop_transform=observation.crop_transform,
        status=observation.status,
        keypoints=observation.keypoints,
    )


def build_analysis_evidence_artifacts(
    *,
    job: AIJobRequest,
    analysis_run_id: str,
    analysis_data_bytes: bytes,
    poses: dict[int, tuple[PersonPoseObservation, ...]],
    pose_recipe: dict[str, str],
    chunk_frame_count: int = POSE_CHUNK_FRAME_COUNT,
) -> AnalysisEvidenceArtifacts:
    """Create full canonical-frame pose coverage and downstream crop sources."""
    canonical_frame_count = int(job.clip.video.total_frames)
    if canonical_frame_count < 1:
        message = "pose evidence requires at least one canonical frame"
        raise ValueError(message)
    if chunk_frame_count < 1:
        message = "pose evidence chunk size must be positive"
        raise ValueError(message)

    analysis_ref = _artifact_reference(
        artifact_id=f"{analysis_run_id}:analysis-data",
        kind="ANALYSIS_DATA",
        data=analysis_data_bytes,
        content_type=ANALYSIS_DATA_CONTENT_TYPE,
    )
    callback_artifacts: list[ProviderResultArtifact] = [
        ProviderResultArtifact(
            part_name="analysis_data",
            kind=analysis_ref.kind,
            schema_version=analysis_ref.schema_version,
            content_type=analysis_ref.content_type,
            data=analysis_data_bytes,
            filename="analysis-data.vad1",
        )
    ]
    chunk_entries: list[dict[str, Any]] = []
    total_players = 0
    total_available = 0
    for chunk_index, start in enumerate(range(0, canonical_frame_count, chunk_frame_count)):
        end = min(canonical_frame_count, start + chunk_frame_count)
        frame_records = [
            [_record(observation) for observation in poses.get(frame_index, ())]
            for frame_index in range(start, end)
        ]
        player_count = sum(len(frame) for frame in frame_records)
        available_count = sum(
            record.status == "AVAILABLE" for frame in frame_records for record in frame
        )
        chunk_bytes = build_person_pose_evidence_chunk(
            analysis_run_id=analysis_run_id,
            pose_recipe_namespace=pose_recipe["namespace"],
            start_frame_index=start,
            frames=frame_records,
        )
        chunk_ref = _artifact_reference(
            artifact_id=f"{analysis_run_id}:pose-chunk:{chunk_index:04d}",
            kind="PERSON_POSE_EVIDENCE_CHUNK",
            data=chunk_bytes,
            content_type=POSE_CHUNK_CONTENT_TYPE,
        )
        chunk_entries.append(
            {
                "index": chunk_index,
                "start_frame_index": str(start),
                "end_frame_index": str(end - 1),
                "player_observation_count": str(player_count),
                "pose_observation_count": str(available_count),
                "missing_observation_count": str(player_count - available_count),
                "artifact": chunk_ref.model_dump(mode="json"),
            }
        )
        callback_artifacts.append(
            ProviderResultArtifact(
                part_name=f"pose_chunk_{chunk_index:04d}",
                kind=chunk_ref.kind,
                schema_version=chunk_ref.schema_version,
                content_type=chunk_ref.content_type,
                data=chunk_bytes,
                filename=f"person-pose-{chunk_index:04d}.vpe1",
            )
        )
        total_players += player_count
        total_available += available_count

    pose_manifest = _content_addressed_model(
        PersonPoseEvidenceManifest,
        {
            "schema_version": "1.0.0",
            "analysis_run_id": analysis_run_id,
            "clip_asset_id": job.clip.clip_asset_id,
            "canonical_frame_count": str(canonical_frame_count),
            "player_observation_count": str(total_players),
            "pose_observation_count": str(total_available),
            "missing_observation_count": str(total_players - total_available),
            "pose_recipe": pose_recipe,
            "chunks": chunk_entries,
        },
    )
    pose_manifest_bytes = _json_bytes(pose_manifest.model_dump(mode="json"))
    pose_manifest_ref = _artifact_reference(
        artifact_id=f"{analysis_run_id}:pose-manifest",
        kind="PERSON_POSE_EVIDENCE_MANIFEST",
        data=pose_manifest_bytes,
        content_type=JSON_CONTENT_TYPE,
    )
    callback_artifacts.append(
        ProviderResultArtifact(
            part_name="pose_manifest",
            kind=pose_manifest_ref.kind,
            schema_version=pose_manifest_ref.schema_version,
            content_type=pose_manifest_ref.content_type,
            data=pose_manifest_bytes,
            filename="person-pose-manifest.json",
        )
    )

    clip_ref = ImmutableArtifactReference(
        artifact_id=job.clip.clip_asset_id,
        kind="CANONICAL_CLIP",
        schema_version="1.0.0",
        sha256=job.clip.sha256,
        byte_length=job.clip.byte_length,
        content_type=job.clip.content_type,
    )
    crop_manifest = _content_addressed_model(
        PlayerCropSourceManifest,
        {
            "schema_version": "1.0.0",
            "analysis_run_id": analysis_run_id,
            "clip_asset_id": job.clip.clip_asset_id,
            "canonical_frame_count": str(canonical_frame_count),
            "coordinate_space": "NORMALIZED_VIDEO",
            "clip_artifact": clip_ref.model_dump(mode="json"),
            "pose_manifest_artifact": pose_manifest_ref.model_dump(mode="json"),
            "crop_recipe": {
                "namespace": "player-crop/pose-bbox/pad-0.10-v1",
                "bbox_source": "PERSON_POSE_EVIDENCE",
                "decode_alignment": "CANONICAL_FRAME_INDEX",
                "padding_ratio": 0.1,
                "clamp_to_frame": True,
            },
        },
    )
    crop_manifest_bytes = _json_bytes(crop_manifest.model_dump(mode="json"))
    crop_manifest_ref = _artifact_reference(
        artifact_id=f"{analysis_run_id}:crop-source",
        kind="PLAYER_CROP_SOURCE_MANIFEST",
        data=crop_manifest_bytes,
        content_type=JSON_CONTENT_TYPE,
    )
    callback_artifacts.append(
        ProviderResultArtifact(
            part_name="crop_source_manifest",
            kind=crop_manifest_ref.kind,
            schema_version=crop_manifest_ref.schema_version,
            content_type=crop_manifest_ref.content_type,
            data=crop_manifest_bytes,
            filename="player-crop-source-manifest.json",
        )
    )

    evidence_manifest = _content_addressed_model(
        AnalysisEvidenceManifest,
        {
            "schema_version": "1.0.0",
            "analysis_run_id": analysis_run_id,
            "match_id": job.match_id,
            "rally_submission_id": job.rally_submission_id,
            "clip_asset_id": job.clip.clip_asset_id,
            "analysis_data_artifact": analysis_ref.model_dump(mode="json"),
            "pose_manifest_artifact": pose_manifest_ref.model_dump(mode="json"),
            "crop_source_manifest_artifact": crop_manifest_ref.model_dump(mode="json"),
            "unavailable_evidence": [],
        },
    )
    evidence_manifest_bytes = _json_bytes(evidence_manifest.model_dump(mode="json"))
    callback_artifacts.append(
        ProviderResultArtifact(
            part_name="analysis_evidence_manifest",
            kind="ANALYSIS_EVIDENCE_MANIFEST",
            schema_version="1.0.0",
            content_type=JSON_CONTENT_TYPE,
            data=evidence_manifest_bytes,
            filename="analysis-evidence-manifest.json",
        )
    )
    return AnalysisEvidenceArtifacts(
        artifacts=tuple(callback_artifacts),
        manifest=evidence_manifest,
    )
