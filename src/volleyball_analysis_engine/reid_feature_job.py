"""Independent Provider Work v2 ReID feature extraction from immutable evidence.

The base analysis owns person pose. This module consumes its every-frame VPE1
artifacts and never loads or invokes a pose model.
"""

# pyright: reportPrivateUsage=false, reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false

from __future__ import annotations

import hashlib
import importlib
import json
import math
import sys
import types
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from uuid import NAMESPACE_URL, UUID, uuid5

import cv2
import numpy as np
from flatbuffers.table import Table
from numpy.typing import NDArray
from volleyball_monitoring_ai import (
    AnalysisDomainData,
    AnalysisEvidenceManifest,
    PersonPoseEvidenceManifest,
    PlayerCropSourceManifest,
    ProviderResultArtifact,
    ReidFeatureJobRequest,
    ReidFeatureResult,
    ReidRosterSnapshot,
    decode_person_pose_evidence_chunk,
    validate_analysis_data_bytes,
)
from volleyball_monitoring_ai.provider_work import ImmutableArtifactReference

from .nested_reid import NestedPartDescriptorExtractor

JSON_CONTENT_TYPE = "application/json"
DESCRIPTOR_CONTENT_TYPE = "application/vnd.volleyball.reid-descriptors+octet-stream;version=1"
L_SHOULDER, R_SHOULDER, L_HIP, R_HIP = 5, 6, 11, 12
TORSO_KEYPOINTS = (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP)
VECTOR_DIMENSIONS = {"DINO": 384, "OSNET": 512, "KPR": 4096, "KPR_PROMPT": 4096}
SUPPORTED_RECIPES = {
    "DINO": "dinov2/vits14-reg/v1",
    "OSNET": "sports-osnet/x1/v1",
    "KPR": "kpr/plain/v1",
    "KPR_PROMPT": "kpr/coco17-prompt/v1",
}


def _install_cython_bbox_fallback() -> None:
    """Install the tiny cython_bbox API required by the upstream OSNet package."""
    try:
        importlib.import_module("cython_bbox")
    except ModuleNotFoundError:
        pass
    else:
        return

    module = types.ModuleType("cython_bbox")

    def bbox_overlaps(
        boxes: NDArray[np.float64],
        query_boxes: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        output = np.zeros((len(boxes), len(query_boxes)), dtype=np.float64)
        for index, box in enumerate(boxes):
            left = np.maximum(box[0], query_boxes[:, 0])
            top = np.maximum(box[1], query_boxes[:, 1])
            right = np.minimum(box[2], query_boxes[:, 2])
            bottom = np.minimum(box[3], query_boxes[:, 3])
            intersection = np.maximum(0.0, right - left) * np.maximum(0.0, bottom - top)
            box_area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
            query_area = np.maximum(0.0, query_boxes[:, 2] - query_boxes[:, 0]) * np.maximum(
                0.0,
                query_boxes[:, 3] - query_boxes[:, 1],
            )
            union = box_area + query_area - intersection
            output[index] = np.divide(
                intersection,
                union,
                out=np.zeros_like(intersection),
                where=union > 0,
            )
        return output

    module.bbox_overlaps = bbox_overlaps  # type: ignore[attr-defined]
    sys.modules["cython_bbox"] = module


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _semantic_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_json_bytes(payload)).hexdigest()


def _normalize(value: NDArray[np.floating[Any]]) -> NDArray[np.float32]:
    vector = np.asarray(value, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    return np.asarray(vector / norm, dtype=np.float32) if norm > 1e-12 else vector


def _validate_semantic_hash(payload: dict[str, Any], *, label: str) -> None:
    claimed = payload.get("content_sha256")
    semantic = {key: value for key, value in payload.items() if key != "content_sha256"}
    if claimed != _semantic_hash(semantic):
        raise ValueError(f"{label} semantic content hash mismatch")


def _artifact_reference(
    *, artifact_id: str, kind: str, data: bytes, content_type: str
) -> ImmutableArtifactReference:
    return ImmutableArtifactReference(
        artifact_id=artifact_id,
        kind=kind,
        schema_version="1.0.0",
        sha256=hashlib.sha256(data).hexdigest(),
        byte_length=str(len(data)),
        content_type=content_type,
    )


def _stable_uuid(namespace_value: str, name: str) -> str:
    try:
        namespace = UUID(namespace_value)
    except ValueError:
        namespace = uuid5(NAMESPACE_URL, namespace_value)
    return str(uuid5(namespace, name))


def _vad_domain(data: bytes) -> AnalysisDomainData:
    validate_analysis_data_bytes(data)
    root = int.from_bytes(data[0:4], "little")
    if root <= 0 or root >= len(data):
        raise ValueError("AnalysisData root offset is invalid")
    table = Table(bytearray(data), root)
    offset = table.Offset(4 + 34 * 2)
    if offset == 0:
        raise ValueError("AnalysisData does not contain domain JSON")
    return AnalysisDomainData.model_validate_json(bytes(table.String(offset + table.Pos)))


@dataclass(frozen=True, slots=True)
class PoseSample:
    frame_index: int
    track_id: int
    frame_bbox: tuple[float, float, float, float]
    status: str
    keypoints: tuple[tuple[float, float, float], ...] | None


@dataclass(frozen=True, slots=True)
class SelectedCrop:
    frame_index: int
    track_id: int
    crop: NDArray[np.uint8]
    torso: NDArray[np.uint8]
    prompt: list[list[float]] | None
    quality: float


@dataclass(frozen=True, slots=True)
class ReidFeatureInputs:
    clip_path: Path
    analysis_data: bytes
    analysis_manifest: dict[str, Any]
    pose_manifest: dict[str, Any]
    crop_manifest: dict[str, Any]
    roster_snapshot: dict[str, Any]
    pose_chunks: tuple[bytes, ...]


@dataclass(frozen=True, slots=True)
class ReidFeatureArtifacts:
    artifacts: tuple[ProviderResultArtifact, ...]
    result: ReidFeatureResult


class SportsOsnetCropEncoder:
    """Standalone Sports OSNet crop encoder for the independent feature job."""

    def __init__(self, *, smp_root: Path, checkpoint: Path, device: str) -> None:
        self.smp_root = smp_root
        self.checkpoint = checkpoint
        self.device_name = device
        self._torch: Any = None
        self._device: Any = None
        self._model: Any = None

    def prepare(self) -> None:
        if self._model is not None:
            return
        if not self.checkpoint.exists():
            raise FileNotFoundError(f"missing Sports OSNet checkpoint: {self.checkpoint}")
        root = str(self.smp_root.resolve())
        if root not in sys.path:
            sys.path.insert(0, root)
        _install_cython_bbox_fallback()
        import torch

        self._torch = torch
        self._device = torch.device(self.device_name)
        module = importlib.import_module("selective_mask_propagation.osnet.inference")
        self._model = module.build_model(self._device, checkpoint=str(self.checkpoint.resolve()))

    def encode(self, crops: list[NDArray[np.uint8]], *, batch_size: int) -> NDArray[np.float32]:
        self.prepare()
        torch = self._torch
        mean = torch.tensor((0.485, 0.456, 0.406), device=self._device).view(1, 3, 1, 1)
        std = torch.tensor((0.229, 0.224, 0.225), device=self._device).view(1, 3, 1, 1)
        output: list[NDArray[np.float32]] = []
        with torch.inference_mode():
            for start in range(0, len(crops), batch_size):
                tensors = []
                for crop in crops[start : start + batch_size]:
                    rgb = cv2.cvtColor(cv2.resize(crop, (128, 256)), cv2.COLOR_BGR2RGB)
                    tensors.append(torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().div_(255))
                values = self._model((torch.stack(tensors).to(self._device) - mean) / std)
                array = np.asarray(values.detach().float().cpu().numpy(), dtype=np.float32)
                output.extend(_normalize(row) for row in array)
        result = np.stack(output)
        if result.shape != (len(crops), VECTOR_DIMENSIONS["OSNET"]):
            raise ValueError(f"unexpected Sports OSNet descriptor shape: {result.shape}")
        return result


def _padded_box(
    bbox: tuple[float, float, float, float], padding: float, width: int, height: int
) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = bbox
    pad_x = (x2 - x1) * padding
    pad_y = (y2 - y1) * padding
    result = (
        max(0, math.floor((x1 - pad_x) * width)),
        max(0, math.floor((y1 - pad_y) * height)),
        min(width, math.ceil((x2 + pad_x) * width)),
        min(height, math.ceil((y2 + pad_y) * height)),
    )
    return result if result[2] - result[0] >= 8 and result[3] - result[1] >= 16 else None


def _crop_prompt(
    sample: PoseSample, box: tuple[int, int, int, int], width: int, height: int
) -> list[list[float]] | None:
    if sample.keypoints is None:
        return None
    points = []
    usable = 0
    for x, y, confidence in sample.keypoints:
        if confidence < 0:
            points.append([-1.0, -1.0, -1.0])
            continue
        crop_x = min(max(x * width - box[0], 0.0), max(box[2] - box[0] - 1e-3, 0.0))
        crop_y = min(max(y * height - box[1], 0.0), max(box[3] - box[1] - 1e-3, 0.0))
        points.append([crop_x, crop_y, confidence])
        usable += confidence >= 0.3
    return points if usable >= 4 else None


def _torso_geometry(
    prompt: list[list[float]] | None,
) -> tuple[float, float, tuple[int, int, int, int]] | None:
    if prompt is None or min(prompt[index][2] for index in TORSO_KEYPOINTS) < 0.35:
        return None
    left_shoulder, right_shoulder = np.asarray(prompt[L_SHOULDER]), np.asarray(prompt[R_SHOULDER])
    left_hip, right_hip = np.asarray(prompt[L_HIP]), np.asarray(prompt[R_HIP])
    torso_height = abs(
        float((left_hip[1] + right_hip[1] - left_shoulder[1] - right_shoulder[1]) / 2)
    )
    if torso_height < 4:
        return None
    span = 0.75 * abs(float(left_shoulder[0] - right_shoulder[0])) + 0.25 * abs(
        float(left_hip[0] - right_hip[0])
    )
    frontality = min(1.0, max(0.0, span / torso_height / 0.85))
    xs = [prompt[index][0] for index in TORSO_KEYPOINTS]
    ys = [prompt[index][1] for index in TORSO_KEYPOINTS]
    return frontality, torso_height, (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))


def _score_crop(
    crop: NDArray[np.uint8], prompt: list[list[float]] | None
) -> tuple[float, NDArray[np.uint8]]:
    geometry = _torso_geometry(prompt)
    if geometry is None:
        torso = crop[: max(8, int(crop.shape[0] * 0.55))]
        frontality, torso_height, pose_penalty = 0.25, crop.shape[0] * 0.3, 0.3
    else:
        frontality, torso_height, (x1, y1, x2, y2) = geometry
        pad_x, pad_y = int((x2 - x1) * 0.15), int((y2 - y1) * 0.15)
        x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        x2, y2 = min(crop.shape[1], x2 + pad_x), min(crop.shape[0], y2 + pad_y)
        torso = crop[y1:y2, x1:x2]
        pose_penalty = 1.0
    grey = cv2.cvtColor(torso, cv2.COLOR_BGR2GRAY) if torso.size else None
    sharpness = (
        min(1.0, max(0.0, math.log10(float(cv2.Laplacian(grey, cv2.CV_64F).var()) + 1) / 3))
        if grey is not None
        else 0.0
    )
    confidence = float(np.mean([point[2] for point in prompt if point[2] >= 0])) if prompt else 0.0
    quality = pose_penalty * (
        0.35 * frontality
        + 0.25 * min(1.0, torso_height / 60.0)
        + 0.15 * sharpness
        + 0.10 * confidence
        + 0.15
    )
    return quality, torso


def _candidate_samples(samples: list[PoseSample], count: int) -> list[PoseSample]:
    if len(samples) <= count:
        return samples
    step = math.ceil(len(samples) / count)
    return samples[::step][:count]


def _pick_selected(crops: list[SelectedCrop], *, top_k: int, min_gap: int) -> list[SelectedCrop]:
    if not crops:
        return []
    span = max(item.frame_index for item in crops) - min(item.frame_index for item in crops)
    gap = min(min_gap, max(1, int(span / (top_k * 1.5))))
    picked: list[SelectedCrop] = []
    for item in sorted(crops, key=lambda value: -value.quality):
        if all(abs(item.frame_index - other.frame_index) >= gap for other in picked):
            picked.append(item)
        if len(picked) >= top_k:
            break
    for item in sorted(crops, key=lambda value: -value.quality):
        if len(picked) >= top_k:
            break
        if item not in picked:
            picked.append(item)
    return sorted(picked, key=lambda value: value.frame_index)


def _decode_selected(
    clip_path: Path,
    samples_by_frame: dict[int, list[PoseSample]],
    *,
    padding: float,
) -> dict[int, list[SelectedCrop]]:
    capture = cv2.VideoCapture(str(clip_path))
    if not capture.isOpened():
        raise ValueError(f"cannot open canonical clip: {clip_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    maximum = max(samples_by_frame, default=-1)
    selected: dict[int, list[SelectedCrop]] = defaultdict(list)
    frame_index = 0
    while frame_index <= maximum:
        ok, frame = capture.read()
        if not ok:
            break
        for sample in samples_by_frame.get(frame_index, []):
            box = _padded_box(sample.frame_bbox, padding, width, height)
            if box is None:
                continue
            crop = cast(
                "NDArray[np.uint8]",
                np.ascontiguousarray(frame[box[1] : box[3], box[0] : box[2]]),
            )
            prompt = _crop_prompt(sample, box, width, height)
            quality, torso = _score_crop(crop, prompt)
            selected[sample.track_id].append(
                SelectedCrop(frame_index, sample.track_id, crop, torso, prompt, quality)
            )
        frame_index += 1
    capture.release()
    if maximum >= 0 and frame_index <= maximum:
        raise ValueError("canonical clip ended before a selected evidence frame")
    return selected


def _aggregate_kpr(values: NDArray[np.float32], indices: list[int]) -> NDArray[np.float32]:
    embeddings = values[indices, 1:]
    parts = []
    for part_index in range(embeddings.shape[1]):
        part = embeddings[:, part_index]
        visible = np.linalg.norm(part, axis=1) > 1e-12
        parts.append(
            _normalize(part[visible].mean(axis=0))
            if np.any(visible)
            else np.zeros(512, dtype=np.float32)
        )
    return np.concatenate(parts)


def build_reid_feature_artifacts(
    *,
    request: ReidFeatureJobRequest,
    inputs: ReidFeatureInputs,
    nested: NestedPartDescriptorExtractor,
    osnet: SportsOsnetCropEncoder,
    batch_size: int = 32,
    candidate_count: int = 60,
    top_k: int = 6,
    min_gap: int = 8,
) -> ReidFeatureArtifacts:
    """Build versioned appearance descriptors without recomputing person pose."""
    for label, payload in (
        ("analysis evidence manifest", inputs.analysis_manifest),
        ("pose evidence manifest", inputs.pose_manifest),
        ("crop source manifest", inputs.crop_manifest),
        ("roster snapshot", inputs.roster_snapshot),
    ):
        _validate_semantic_hash(payload, label=label)
    analysis_manifest = AnalysisEvidenceManifest.model_validate(inputs.analysis_manifest)
    pose_manifest = PersonPoseEvidenceManifest.model_validate(inputs.pose_manifest)
    crop_manifest = PlayerCropSourceManifest.model_validate(inputs.crop_manifest)
    roster = ReidRosterSnapshot.model_validate(inputs.roster_snapshot)
    domain = _vad_domain(inputs.analysis_data)
    if {
        analysis_manifest.analysis_run_id,
        pose_manifest.analysis_run_id,
        crop_manifest.analysis_run_id,
        request.analysis_run_id,
    } != {request.analysis_run_id}:
        raise ValueError("ReID feature inputs reference different analysis runs")
    if domain.match_id != request.match_id or roster.match_id != request.match_id:
        raise ValueError("ReID feature inputs reference a different match")
    if pose_manifest.pose_recipe.namespace != request.pose_recipe_namespace:
        raise ValueError("ReID feature request pose recipe does not match persisted evidence")

    by_track: dict[int, list[PoseSample]] = defaultdict(list)
    expected_chunks = sorted(pose_manifest.chunks, key=lambda item: item.index)
    if len(expected_chunks) != len(inputs.pose_chunks):
        raise ValueError("pose chunk count does not match its manifest")
    for entry, chunk_bytes in zip(expected_chunks, inputs.pose_chunks, strict=True):
        if hashlib.sha256(chunk_bytes).hexdigest() != entry.artifact.sha256.lower():
            raise ValueError(f"pose chunk {entry.index} hash mismatch")
        decoded = decode_person_pose_evidence_chunk(chunk_bytes)
        if (
            decoded.analysis_run_id != request.analysis_run_id
            or decoded.pose_recipe_namespace != request.pose_recipe_namespace
            or decoded.start_frame_index != int(entry.start_frame_index)
        ):
            raise ValueError(f"pose chunk {entry.index} identity mismatch")
        for relative_index, frame in enumerate(decoded.frames):
            frame_index = decoded.start_frame_index + relative_index
            for record in frame:
                by_track[record.track_id].append(
                    PoseSample(
                        frame_index=frame_index,
                        track_id=record.track_id,
                        frame_bbox=record.frame_bbox,
                        status=record.status,
                        keypoints=record.keypoints,
                    )
                )
    track_map = {track.track_id: track for track in domain.tracks}
    if not set(by_track).issubset(track_map):
        raise ValueError("pose evidence references a track absent from AnalysisData")

    planned: dict[int, list[PoseSample]] = defaultdict(list)
    for samples in by_track.values():
        for sample in _candidate_samples(
            sorted(samples, key=lambda item: item.frame_index), candidate_count
        ):
            planned[sample.frame_index].append(sample)
    decoded = _decode_selected(
        inputs.clip_path,
        planned,
        padding=crop_manifest.crop_recipe.padding_ratio,
    )
    selected_by_track = {
        track_id: _pick_selected(crops, top_k=top_k, min_gap=min_gap)
        for track_id, crops in decoded.items()
    }

    ordered_tracks = sorted(by_track)
    tracklet_ids = {
        track_id: _stable_uuid(request.evidence_set_id, f"tracklet:{track_id}")
        for track_id in ordered_tracks
    }
    visible_frames = {
        track_id: {sample.frame_index for sample in samples}
        for track_id, samples in by_track.items()
    }
    cannot_links = {
        track_id: [
            other
            for other in ordered_tracks
            if other != track_id and visible_frames[track_id] & visible_frames[other]
        ]
        for track_id in ordered_tracks
    }
    flattened = [
        crop for track_id in ordered_tracks for crop in selected_by_track.get(track_id, [])
    ]
    indices_by_track: dict[int, list[int]] = defaultdict(list)
    for index, crop in enumerate(flattened):
        indices_by_track[crop.track_id].append(index)

    recipes = {recipe.modality: recipe.model_namespace for recipe in request.requested_recipes}
    unsupported = {
        modality: namespace
        for modality, namespace in recipes.items()
        if SUPPORTED_RECIPES.get(modality) != namespace
    }
    if unsupported:
        message = f"unsupported ReID feature recipes: {unsupported}"
        raise ValueError(message)
    vectors_by_track: dict[int, dict[str, NDArray[np.float32]]] = defaultdict(dict)
    unavailable: list[dict[str, Any]] = []

    def unavailable_for(modality: str, reason: str, message: str, track_ids: list[int]) -> None:
        unavailable.extend(
            {
                "tracklet_id": tracklet_ids[track_id],
                "modality": modality,
                "reason_code": reason,
                "message": message[:1000],
            }
            for track_id in track_ids
        )

    missing_crops = [track_id for track_id in ordered_tracks if not indices_by_track[track_id]]
    for modality in recipes:
        if missing_crops:
            unavailable_for(
                modality, "NO_USABLE_CROPS", "no selected crop was usable", missing_crops
            )

    if flattened and "DINO" in recipes:
        try:
            values = nested.encode_dino_crops([crop.crop for crop in flattened])
            for track_id, indices in indices_by_track.items():
                vectors_by_track[track_id]["DINO"] = _normalize(values[indices].mean(axis=0))
        except Exception as error:
            unavailable_for("DINO", "MODEL_UNAVAILABLE", str(error), list(indices_by_track))
    if flattened and "OSNET" in recipes:
        try:
            values = osnet.encode([crop.crop for crop in flattened], batch_size=batch_size)
            for track_id, indices in indices_by_track.items():
                vectors_by_track[track_id]["OSNET"] = _normalize(values[indices].mean(axis=0))
        except Exception as error:
            unavailable_for("OSNET", "MODEL_UNAVAILABLE", str(error), list(indices_by_track))
    kpr_modalities = {value for value in ("KPR", "KPR_PROMPT") if value in recipes}
    if flattened and kpr_modalities:
        try:
            plain, prompted, _ = nested.encode_kpr_crops(
                keys=[(crop.track_id, crop.frame_index) for crop in flattened],
                crops=[crop.crop for crop in flattened],
                prompts=[crop.prompt for crop in flattened],
            )
            for track_id, indices in indices_by_track.items():
                if "KPR" in kpr_modalities:
                    vectors_by_track[track_id]["KPR"] = _aggregate_kpr(plain, indices)
                if "KPR_PROMPT" in kpr_modalities:
                    vectors_by_track[track_id]["KPR_PROMPT"] = _aggregate_kpr(prompted, indices)
        except Exception as error:
            for modality in sorted(kpr_modalities):
                unavailable_for(modality, "MODEL_UNAVAILABLE", str(error), list(indices_by_track))

    descriptor = bytearray()
    tracklet_payloads = []
    for track_id in ordered_tracks:
        vector_payloads = []
        for modality in ("DINO", "OSNET", "KPR", "KPR_PROMPT"):
            vector = vectors_by_track[track_id].get(modality)
            if vector is None:
                continue
            vector = _normalize(vector).astype("<f4", copy=False)
            expected = VECTOR_DIMENSIONS[modality]
            if vector.shape != (expected,):
                raise ValueError(f"{modality} descriptor has invalid shape {vector.shape}")
            raw = vector.tobytes()
            offset = len(descriptor)
            descriptor.extend(raw)
            vector_payloads.append(
                {
                    "vector_id": _stable_uuid(
                        request.evidence_set_id, f"vector:{track_id}:{modality}:{recipes[modality]}"
                    ),
                    "modality": modality,
                    "model_namespace": recipes[modality],
                    "dimension": expected,
                    "normalization": "L2",
                    "distance": "COSINE",
                    "byte_offset": str(offset),
                    "byte_length": str(len(raw)),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "source_frame_indices": [
                        str(crop.frame_index) for crop in selected_by_track.get(track_id, [])
                    ],
                }
            )
        track = track_map[track_id]
        tracklet_payloads.append(
            {
                "tracklet_id": tracklet_ids[track_id],
                "canonical_track_id": track_id,
                "track_id_aliases": [track_id],
                "court_side": track.court_side.upper(),
                "first_frame_index": track.first_frame_index,
                "last_frame_index": track.last_frame_index,
                "cannot_link_tracklet_ids": [
                    tracklet_ids[value] for value in cannot_links[track_id]
                ],
                "vectors": vector_payloads,
            }
        )

    descriptor_bytes = bytes(descriptor)
    descriptor_ref = _artifact_reference(
        artifact_id=f"{request.evidence_set_id}:descriptors",
        kind="REID_DESCRIPTOR_BUNDLE",
        data=descriptor_bytes,
        content_type=DESCRIPTOR_CONTENT_TYPE,
    )
    callback_artifacts: list[ProviderResultArtifact] = []
    result_body = {
        "schema_version": "2.0.0",
        "provider_job_id": request.provider_job_id,
        "evidence_set_id": request.evidence_set_id,
        "analysis_run_id": request.analysis_run_id,
        "match_id": request.match_id,
        "status": "PARTIAL" if unavailable else "READY",
        "descriptor_artifact": descriptor_ref.model_dump(mode="json"),
        "tracklets": tracklet_payloads,
        "unavailable_evidence": unavailable,
    }
    result = ReidFeatureResult.model_validate(
        {**result_body, "content_sha256": _semantic_hash(result_body)}
    )
    result_bytes = _json_bytes(result.model_dump(mode="json"))
    callback_artifacts.insert(
        0,
        ProviderResultArtifact(
            part_name="reid_feature_result",
            kind="REID_FEATURE_RESULT",
            schema_version="2.0.0",
            content_type=JSON_CONTENT_TYPE,
            data=result_bytes,
            filename="reid-feature-result.json",
        ),
    )
    callback_artifacts.insert(
        1,
        ProviderResultArtifact(
            part_name="reid_descriptor_bundle",
            kind="REID_DESCRIPTOR_BUNDLE",
            schema_version="1.0.0",
            content_type=DESCRIPTOR_CONTENT_TYPE,
            data=descriptor_bytes,
            filename="reid-descriptors.f32le",
        ),
    )
    return ReidFeatureArtifacts(tuple(callback_artifacts), result)
