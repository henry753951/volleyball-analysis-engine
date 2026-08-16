"""Versioned ReID association tests with explicit immutable vector locations."""

from __future__ import annotations

import hashlib
import json
from typing import Any
from uuid import uuid4

import numpy as np
from volleyball_monitoring_ai.provider_work import ReidAssociationJobRequest

from volleyball_analysis_engine.reid_association_job import (
    ReidAssociationInputs,
    build_reid_association_artifacts,
)


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def _semantic(payload: dict[str, Any]) -> dict[str, Any]:
    return {**payload, "content_sha256": hashlib.sha256(_json_bytes(payload)).hexdigest()}


def _normalized(dimension: int, index: int = 0) -> bytes:
    vector = np.zeros(dimension, dtype="<f4")
    vector[index] = 1
    return vector.tobytes()


def test_association_resolves_only_explicitly_eligible_tracklets() -> None:
    match_id = str(uuid4())
    evidence_set_id = str(uuid4())
    feature_job_id = str(uuid4())
    association_job_id = str(uuid4())
    association_run_id = str(uuid4())
    bank_id = str(uuid4())
    team_id = str(uuid4())
    roster_entry_id = str(uuid4())
    cluster_id = str(uuid4())
    current_tracklet_id = str(uuid4())
    excluded_tracklet_id = str(uuid4())
    current_vector_id = str(uuid4())
    history_tracklet_id = str(uuid4())
    history_vector_id = str(uuid4())
    history_artifact_id = str(uuid4())
    current = _normalized(384)
    history = _normalized(384)
    current_ref = {
        "artifact_id": str(uuid4()),
        "kind": "REID_DESCRIPTOR_BUNDLE",
        "schema_version": "1.0.0",
        "sha256": hashlib.sha256(current).hexdigest(),
        "byte_length": str(len(current)),
        "content_type": "application/vnd.volleyball.reid-descriptors+octet-stream;version=1",
    }
    vector = {
        "vector_id": current_vector_id,
        "modality": "DINO",
        "model_namespace": "dinov2/vits14-reg/v1",
        "dimension": 384,
        "normalization": "L2",
        "distance": "COSINE",
        "byte_offset": "0",
        "byte_length": str(len(current)),
        "sha256": hashlib.sha256(current).hexdigest(),
        "source_frame_indices": ["1"],
    }
    jersey_artifact = {
        "artifact_id": str(uuid4()),
        "kind": "JERSEY_VLM_RESPONSE",
        "schema_version": "1.0.0",
        "sha256": "a" * 64,
        "byte_length": "1",
        "content_type": "application/vnd.volleyball.reid-jersey-vlm-responses+json;version=1",
    }
    feature = _semantic(
        {
            "schema_version": "1.0.0",
            "provider_job_id": feature_job_id,
            "evidence_set_id": evidence_set_id,
            "analysis_run_id": str(uuid4()),
            "match_id": match_id,
            "status": "READY",
            "descriptor_artifact": current_ref,
            "jersey_vlm_response_artifact": jersey_artifact,
            "tracklets": [
                {
                    "tracklet_id": current_tracklet_id,
                    "canonical_track_id": 1,
                    "track_id_aliases": [1],
                    "court_side": "LEFT",
                    "first_frame_index": "0",
                    "last_frame_index": "3",
                    "cannot_link_tracklet_ids": [excluded_tracklet_id],
                    "vectors": [vector],
                    "jersey_vlm": {
                        "model_namespace": "jersey-vlm/qwen-v1",
                        "raw_response_key": "response-current",
                        "raw_response_sha256": "b" * 64,
                        "candidate_numbers": [11],
                        "selected_frame_indices": ["1"],
                    },
                },
                {
                    "tracklet_id": excluded_tracklet_id,
                    "canonical_track_id": 2,
                    "track_id_aliases": [2],
                    "court_side": "RIGHT",
                    "first_frame_index": "0",
                    "last_frame_index": "3",
                    "cannot_link_tracklet_ids": [current_tracklet_id],
                    "vectors": [],
                    "jersey_vlm": None,
                },
            ],
            "unavailable_evidence": [],
        }
    )
    bank = _semantic(
        {
            "schema_version": "1.1.0",
            "bank_snapshot_id": bank_id,
            "match_id": match_id,
            "team_id": team_id,
            "revision": "1",
            "as_of_position": {"set_number": 1, "rally_ordinal": 2},
            "clusters": [{"person_cluster_id": cluster_id, "roster_entry_id": roster_entry_id}],
            "evidence_artifacts": [
                {
                    "artifact_id": history_artifact_id,
                    "sha256": hashlib.sha256(history).hexdigest(),
                    "byte_length": str(len(history)),
                }
            ],
            "vectors": [
                {
                    "vector_id": history_vector_id,
                    "artifact_id": history_artifact_id,
                    "modality": "DINO",
                    "model_namespace": "dinov2/vits14-reg/v1",
                    "dimension": 384,
                    "normalization": "L2",
                    "distance": "COSINE",
                    "byte_offset": "0",
                    "byte_length": str(len(history)),
                    "sha256": hashlib.sha256(history).hexdigest(),
                }
            ],
            "memberships": [
                {
                    "membership_id": str(uuid4()),
                    "person_cluster_id": cluster_id,
                    "tracklet_id": history_tracklet_id,
                    "vector_ids": [history_vector_id],
                    "evidence_state": "CONFIRMED",
                    "evidence_role": "POSITIVE",
                    "weight": 1,
                    "source_revision": "1",
                    "roster_entry_id": roster_entry_id,
                }
            ],
            "cannot_links": [],
        }
    )
    roster = _semantic(
        {
            "schema_version": "1.0.0",
            "roster_snapshot_id": str(uuid4()),
            "match_id": match_id,
            "rally_submission_id": str(uuid4()),
            "as_of_position": {"set_number": 1, "rally_ordinal": 2},
            "teams": [
                {
                    "team_id": team_id,
                    "court_side": "LEFT",
                    "entries": [
                        {
                            "roster_entry_id": roster_entry_id,
                            "player_id": None,
                            "jersey_number": "11",
                            "display_name": None,
                            "position": "OH",
                            "active": True,
                        }
                    ],
                },
                {
                    "team_id": str(uuid4()),
                    "court_side": "RIGHT",
                    "entries": [],
                },
            ],
        }
    )
    request = ReidAssociationJobRequest.model_validate(
        {
            "schema_version": "1.1.0",
            "provider_job_id": association_job_id,
            "association_run_id": association_run_id,
            "match_id": match_id,
            "evidence_set_id": evidence_set_id,
            "eligible_tracklet_ids": [current_tracklet_id],
            "evidence_result_artifact_id": current_ref["artifact_id"],
            "bank_snapshot_id": bank_id,
            "bank_snapshot_artifact_id": str(uuid4()),
            "roster_snapshot_artifact_id": str(uuid4()),
            "recipe": {
                "namespace": "reid/nested-part-v2",
                "candidate_modalities": ["DINO", "JERSEY_VLM"],
                "same_clip_grouping": True,
                "allow_abstention": True,
                "manual_assignment_precedence": True,
            },
        }
    )

    output = build_reid_association_artifacts(
        request=request,
        inputs=ReidAssociationInputs(
            feature_result=feature,
            current_descriptors=current,
            bank_snapshot=bank,
            roster_snapshot=roster,
            bank_descriptor_artifacts={history_artifact_id: history},
        ),
    )

    assert output.result.status == "COMPLETED"
    assert len(output.result.decisions) == 1
    decision = output.result.decisions[0]
    assert decision.tracklet_id == current_tracklet_id
    assert decision.association_state == "RESOLVED"
    assert decision.selected_person_cluster_id == cluster_id
    assert decision.selected_roster_entry_id == roster_entry_id
    assert {score.component for score in decision.candidates[0].scores} == {
        "DINO",
        "JERSEY_VLM",
    }
