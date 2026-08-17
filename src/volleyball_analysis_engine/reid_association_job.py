"""Versioned ReID association over current evidence and an immutable eligible bank."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from itertools import pairwise
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from volleyball_monitoring_ai import ProviderResultArtifact
from volleyball_monitoring_ai.provider_work import (
    ReidAssociationJobRequest,
    ReidAssociationResult,
    ReidBankSnapshot,
    ReidFeatureResult,
    ReidRosterSnapshot,
)

JSON_CONTENT_TYPE = "application/json"
COMPONENT_WEIGHTS = {
    "DINO": 0.25,
    "OSNET": 0.20,
    "KPR": 0.35,
    "KPR_PROMPT": 0.35,
}
RESOLVE_THRESHOLD = 0.75
MARGIN_THRESHOLD = 0.12
SAME_CLIP_NEW_GID_THRESHOLD = 0.86


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _javascript_json_numbers(value: object) -> object:
    """Match JSON.stringify for the finite integral floats used by wire artifacts."""
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("ReID JSON artifacts cannot contain non-finite numbers")
        if value == 0.0 or (value.is_integer() and abs(value) <= 9_007_199_254_740_991):
            return int(value)
        return value
    if isinstance(value, list):
        return [_javascript_json_numbers(item) for item in cast("list[object]", value)]
    if isinstance(value, dict):
        items = cast("dict[str, object]", value)
        return {key: _javascript_json_numbers(item) for key, item in items.items()}
    return value


def _semantic_hash(payload: dict[str, Any]) -> str:
    normalized = cast("dict[str, Any]", _javascript_json_numbers(payload))
    return hashlib.sha256(_json_bytes(normalized)).hexdigest()


def _validate_semantic_hash(payload: dict[str, Any], *, label: str) -> None:
    body = {key: value for key, value in payload.items() if key != "content_sha256"}
    if payload.get("content_sha256") != _semantic_hash(body):
        message = f"{label} semantic content hash mismatch"
        raise ValueError(message)


def _vector(
    data: bytes, *, offset: int, length: int, digest: str, dimension: int
) -> NDArray[np.float32]:
    if offset < 0 or length != dimension * 4 or offset + length > len(data):
        raise ValueError("descriptor vector byte range is invalid")
    raw = data[offset : offset + length]
    if hashlib.sha256(raw).hexdigest() != digest.lower():
        raise ValueError("descriptor vector checksum mismatch")
    return np.frombuffer(raw, dtype="<f4").astype(np.float32, copy=True)


def _cosine(left: NDArray[np.float32], right: NDArray[np.float32]) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-12:
        return 0.0
    return min(1.0, max(-1.0, float(np.dot(left, right) / denominator)))


@dataclass(frozen=True, slots=True)
class ReidAssociationInputs:
    """Exact current, bank, roster, and descriptor artifacts leased to one job."""

    feature_result: dict[str, Any]
    current_descriptors: bytes
    bank_snapshot: dict[str, Any]
    roster_snapshot: dict[str, Any]
    bank_descriptor_artifacts: dict[str, bytes]


@dataclass(frozen=True, slots=True)
class ReidAssociationArtifacts:
    """Callback-ready association result and its validated model."""

    artifacts: tuple[ProviderResultArtifact, ...]
    result: ReidAssociationResult


@dataclass(frozen=True, slots=True)
class _CurrentVector:
    modality: str
    model_namespace: str
    value: NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class _BankVector:
    modality: str
    model_namespace: str
    value: NDArray[np.float32]


def soft_lineup_candidate_rows(
    rows: list[dict[str, Any]],
    *,
    reserved: dict[str, list[str]],
    cannot_link_tracklet_ids: set[str],
    expected_count: int,
) -> list[dict[str, Any]]:
    """Reject impossible co-visibility only; apply lineup size as a score prior."""
    occupied_covisible_clusters = {
        cluster_id
        for cluster_id, owners in reserved.items()
        if any(owner in cannot_link_tracklet_ids for owner in owners)
    }
    available: list[dict[str, Any]] = []
    for row in rows:
        cluster_id = str(row["person_cluster_id"])
        if any(owner in cannot_link_tracklet_ids for owner in reserved.get(cluster_id, [])):
            continue
        candidate = dict(row)
        candidate["scores"] = list(cast("list[dict[str, Any]]", row["scores"]))
        if (
            len(occupied_covisible_clusters) >= expected_count
            and cluster_id not in occupied_covisible_clusters
        ):
            candidate["confidence"] = float(candidate["confidence"]) * 0.9
            candidate["scores"].append(
                {
                    "component": "CONSTRAINT",
                    "value": -0.1,
                    "model_namespace": "soft-team-occupancy/v1",
                }
            )
        available.append(candidate)
    available.sort(key=lambda row: (-float(row["confidence"]), str(row["person_cluster_id"])))
    for rank, row in enumerate(available, 1):
        row["rank"] = rank
    return available


def _same_clip_similarity(
    left: dict[str, _CurrentVector], right: dict[str, _CurrentVector]
) -> float:
    weighted_sum = 0.0
    total_weight = 0.0
    for modality, left_value in left.items():
        right_value = right.get(modality)
        if right_value is None or right_value.model_namespace != left_value.model_namespace:
            continue
        weight = COMPONENT_WEIGHTS.get(modality, 0.0)
        weighted_sum += ((_cosine(left_value.value, right_value.value) + 1.0) / 2.0) * weight
        total_weight += weight
    return weighted_sum / total_weight if total_weight else 0.0


def _decode_current(
    result: ReidFeatureResult, descriptors: bytes
) -> dict[str, dict[str, _CurrentVector]]:
    decoded: dict[str, dict[str, _CurrentVector]] = {}
    for tracklet in result.tracklets:
        values: dict[str, _CurrentVector] = {}
        ranges: list[tuple[int, int]] = []
        for reference in tracklet.vectors:
            offset, length = int(reference.byte_offset), int(reference.byte_length)
            ranges.append((offset, offset + length))
            values[reference.modality] = _CurrentVector(
                reference.modality,
                reference.model_namespace,
                _vector(
                    descriptors,
                    offset=offset,
                    length=length,
                    digest=reference.sha256,
                    dimension=reference.dimension,
                ),
            )
        for (_, left_end), (right_start, _) in pairwise(sorted(ranges)):
            if right_start < left_end:
                raise ValueError("current descriptor vector byte ranges overlap")
        decoded[tracklet.tracklet_id] = values
    return decoded


def _decode_bank(bank: ReidBankSnapshot, artifacts: dict[str, bytes]) -> dict[str, _BankVector]:
    expected = {artifact.artifact_id: artifact for artifact in bank.evidence_artifacts}
    if set(artifacts) != set(expected):
        raise ValueError("bank descriptor artifact set does not match the snapshot")
    for artifact_id, reference in expected.items():
        data = artifacts[artifact_id]
        if (
            len(data) != int(reference.byte_length)
            or hashlib.sha256(data).hexdigest() != reference.sha256.lower()
        ):
            raise ValueError("bank descriptor artifact integrity mismatch")
    decoded: dict[str, _BankVector] = {}
    ranges: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for reference in bank.vectors:
        data = artifacts[reference.artifact_id]
        offset, length = int(reference.byte_offset), int(reference.byte_length)
        ranges[reference.artifact_id].append((offset, offset + length))
        decoded[reference.vector_id] = _BankVector(
            reference.modality,
            reference.model_namespace,
            _vector(
                data,
                offset=offset,
                length=length,
                digest=reference.sha256,
                dimension=reference.dimension,
            ),
        )
    for artifact_ranges in ranges.values():
        ordered = sorted(artifact_ranges)
        if any(right_start < left_end for (_, left_end), (right_start, _) in pairwise(ordered)):
            raise ValueError("bank descriptor vector byte ranges overlap")
    return decoded


def build_reid_association_artifacts(
    *, request: ReidAssociationJobRequest, inputs: ReidAssociationInputs
) -> ReidAssociationArtifacts:
    """Score only explicitly eligible current tracklets and abstain on weak evidence."""
    for label, payload in (
        ("ReID feature result", inputs.feature_result),
        ("ReID bank snapshot", inputs.bank_snapshot),
        ("ReID roster snapshot", inputs.roster_snapshot),
    ):
        _validate_semantic_hash(payload, label=label)
    feature = ReidFeatureResult.model_validate(inputs.feature_result)
    bank = ReidBankSnapshot.model_validate(inputs.bank_snapshot)
    roster = ReidRosterSnapshot.model_validate(inputs.roster_snapshot)
    if (
        feature.provider_job_id == request.provider_job_id
        or feature.evidence_set_id != request.evidence_set_id
        or bank.bank_snapshot_id != request.bank_snapshot_id
        or bank.match_id != request.match_id
        or roster.match_id != request.match_id
    ):
        raise ValueError("association input passthrough identity mismatch")
    feature_by_id = {tracklet.tracklet_id: tracklet for tracklet in feature.tracklets}
    if not set(request.eligible_tracklet_ids).issubset(feature_by_id):
        raise ValueError("eligible association tracklet is absent from feature evidence")
    current = _decode_current(feature, inputs.current_descriptors)
    history = _decode_bank(bank, inputs.bank_descriptor_artifacts)
    if not any(team.team_id == bank.team_id for team in roster.teams):
        raise ValueError("bank team is absent from the immutable roster snapshot")
    clusters = {cluster.person_cluster_id: cluster for cluster in bank.clusters}
    memberships: dict[str, list[Any]] = defaultdict(list)
    for membership in bank.memberships:
        memberships[membership.person_cluster_id].append(membership)

    candidate_rows: dict[str, list[dict[str, Any]]] = {}
    for tracklet_id in request.eligible_tracklet_ids:
        tracklet = feature_by_id[tracklet_id]
        rows: list[dict[str, Any]] = []
        for cluster_id, cluster in clusters.items():
            component_scores: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
            for membership in memberships.get(cluster_id, []):
                direction = 1.0 if membership.evidence_role == "POSITIVE" else -0.65
                for vector_id in membership.vector_ids:
                    historical = history[vector_id]
                    if historical.modality not in request.recipe.candidate_modalities:
                        continue
                    present = current[tracklet_id].get(historical.modality)
                    if present is None or present.model_namespace != historical.model_namespace:
                        continue
                    similarity = (_cosine(present.value, historical.value) + 1.0) / 2.0
                    component_scores[(historical.modality, historical.model_namespace)].append(
                        (direction * similarity, membership.weight)
                    )
            scores: list[dict[str, Any]] = []
            weighted_sum = 0.0
            total_weight = 0.0
            for (component, namespace), values in sorted(component_scores.items()):
                denominator = sum(weight for _, weight in values)
                value = sum(score * weight for score, weight in values) / max(denominator, 1e-12)
                value = min(1.0, max(0.0, value))
                scores.append(
                    {"component": component, "value": value, "model_namespace": namespace}
                )
                component_weight = COMPONENT_WEIGHTS.get(component, 0.0)
                weighted_sum += value * component_weight
                total_weight += component_weight
            confidence = weighted_sum / total_weight if total_weight else 0.0
            rows.append(
                {
                    "candidate_key": f"{tracklet_id}:{cluster_id}",
                    "person_cluster_id": cluster_id,
                    "roster_entry_id": cluster.roster_entry_id,
                    "confidence": confidence,
                    "scores": scores,
                }
            )
        rows.sort(key=lambda row: (-float(row["confidence"]), str(row["person_cluster_id"])))
        for rank, row in enumerate(rows, 1):
            row["rank"] = rank
        candidate_rows[tracklet_id] = rows

    reserved: dict[str, list[str]] = defaultdict(list)
    new_gid_groups: dict[str, list[str]] = {}
    decisions: list[dict[str, Any]] = []
    ordered_tracklets = sorted(
        request.eligible_tracklet_ids,
        key=lambda tracklet_id: (
            -float(candidate_rows[tracklet_id][0]["confidence"])
            if candidate_rows[tracklet_id]
            else 0.0
        ),
    )
    for tracklet_id in ordered_tracklets:
        tracklet = feature_by_id[tracklet_id]
        rows = candidate_rows[tracklet_id]
        cannot_link = set(tracklet.cannot_link_tracklet_ids)
        available = soft_lineup_candidate_rows(
            rows,
            reserved=reserved,
            cannot_link_tracklet_ids=cannot_link,
            expected_count=request.recipe.team_occupancy_prior.expected_count,
        )
        top = available[0] if available else None
        second = available[1] if len(available) > 1 else None
        confidence = float(top["confidence"]) if top else 0.0
        margin = confidence - (float(second["confidence"]) if second else 0.0)
        if top is not None and confidence >= RESOLVE_THRESHOLD and margin >= MARGIN_THRESHOLD:
            action = "MATCH_EXISTING_GID"
            selected_cluster = str(top["person_cluster_id"])
            selected_roster = top["roster_entry_id"]
            new_gid_group_key = None
            decision_confidence = confidence
            rationale = "existing GID passed confidence, margin, and co-visibility checks"
            reserved[selected_cluster].append(tracklet_id)
        else:
            action = "CREATE_NEW_GID"
            selected_cluster = selected_roster = None
            reusable_group: str | None = None
            reusable_similarity = 0.0
            for group_key, member_tracklet_ids in new_gid_groups.items():
                if any(
                    member_id in cannot_link
                    or tracklet_id in set(feature_by_id[member_id].cannot_link_tracklet_ids)
                    for member_id in member_tracklet_ids
                ):
                    continue
                similarity = max(
                    _same_clip_similarity(current[tracklet_id], current[member_id])
                    for member_id in member_tracklet_ids
                )
                if similarity > reusable_similarity:
                    reusable_group = group_key
                    reusable_similarity = similarity
            if reusable_group is not None and reusable_similarity >= SAME_CLIP_NEW_GID_THRESHOLD:
                new_gid_group_key = reusable_group
                new_gid_groups[reusable_group].append(tracklet_id)
                decision_confidence = reusable_similarity
                rationale = "non-co-visible Local matched a new same-clip GID seed"
            else:
                new_gid_group_key = f"new:{request.evidence_set_id}:{bank.team_id}:{tracklet_id}"
                new_gid_groups[new_gid_group_key] = [tracklet_id]
                decision_confidence = max(0.6, 1.0 - confidence)
                rationale = "no existing GID was reliable; activate a new unbound GID"
        decisions.append(
            {
                "tracklet_id": tracklet_id,
                "group_key": f"{request.evidence_set_id}:{bank.team_id}",
                "action": action,
                "selected_person_cluster_id": selected_cluster,
                "new_gid_group_key": new_gid_group_key,
                "selected_roster_entry_id": selected_roster,
                "confidence": decision_confidence,
                "candidates": available,
                "rationale": rationale,
            }
        )
    decisions.sort(
        key=lambda decision: request.eligible_tracklet_ids.index(decision["tracklet_id"])
    )
    body: dict[str, Any] = {
        "schema_version": "2.0.0",
        "provider_job_id": request.provider_job_id,
        "association_run_id": request.association_run_id,
        "evidence_set_id": request.evidence_set_id,
        "bank_snapshot_id": request.bank_snapshot_id,
        "status": "COMPLETED",
        "decisions": decisions,
    }
    result = ReidAssociationResult.model_validate({**body, "content_sha256": _semantic_hash(body)})
    result_bytes = _json_bytes(result.model_dump(mode="json"))
    return ReidAssociationArtifacts(
        artifacts=(
            ProviderResultArtifact(
                part_name="reid_association_result",
                kind="REID_ASSOCIATION_RESULT",
                schema_version="2.0.0",
                content_type=JSON_CONTENT_TYPE,
                data=result_bytes,
                filename="reid-association-result.json",
            ),
        ),
        result=result,
    )
