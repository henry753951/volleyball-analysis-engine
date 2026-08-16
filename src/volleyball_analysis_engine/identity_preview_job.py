"""Deterministic animated identity previews from saved every-frame pose evidence."""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
from PIL import Image
from volleyball_monitoring_ai import (
    IdentityPreviewJobRequest,
    IdentityPreviewResult,
    PersonPoseEvidenceManifest,
    PlayerCropSourceManifest,
    ProviderResultArtifact,
    decode_person_pose_evidence_chunk,
)

JSON_CONTENT_TYPE = "application/json"
WEBP_CONTENT_TYPE = "image/webp"


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _semantic_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_json_bytes(payload)).hexdigest()


def _validate_semantic_hash(payload: dict[str, Any], *, label: str) -> None:
    body = {key: value for key, value in payload.items() if key != "content_sha256"}
    if payload.get("content_sha256") != _semantic_hash(body):
        message = f"{label} semantic content hash mismatch"
        raise ValueError(message)


@dataclass(frozen=True, slots=True)
class IdentityPreviewInputs:
    """Exact canonical media and saved pose artifacts leased to one preview job."""

    clip_path: Path
    crop_manifest: dict[str, Any]
    pose_manifest: dict[str, Any]
    pose_chunks: tuple[bytes, ...]


@dataclass(frozen=True, slots=True)
class IdentityPreviewArtifacts:
    """Preview media plus its content-addressed result manifest."""

    artifacts: tuple[ProviderResultArtifact, ...]
    result: IdentityPreviewResult


def _selected_boxes(
    *,
    request: IdentityPreviewJobRequest,
    manifest: PersonPoseEvidenceManifest,
    chunks: tuple[bytes, ...],
) -> dict[int, tuple[float, float, float, float]]:
    expected_frames = {int(value) for value in request.selected_frame_indices}
    boxes: dict[int, tuple[float, float, float, float]] = {}
    entries = sorted(manifest.chunks, key=lambda item: item.index)
    if len(entries) != len(chunks):
        message = "preview pose chunk count does not match its manifest"
        raise ValueError(message)
    for entry, chunk_bytes in zip(entries, chunks, strict=True):
        if hashlib.sha256(chunk_bytes).hexdigest() != entry.artifact.sha256.lower():
            message = f"preview pose chunk {entry.index} hash mismatch"
            raise ValueError(message)
        decoded = decode_person_pose_evidence_chunk(chunk_bytes)
        if (
            decoded.analysis_run_id != request.analysis_run_id
            or decoded.pose_recipe_namespace != manifest.pose_recipe.namespace
            or decoded.start_frame_index != int(entry.start_frame_index)
        ):
            message = f"preview pose chunk {entry.index} identity mismatch"
            raise ValueError(message)
        for relative_index, observations in enumerate(decoded.frames):
            frame_index = decoded.start_frame_index + relative_index
            if frame_index not in expected_frames:
                continue
            matches = [
                observation
                for observation in observations
                if observation.track_id == request.canonical_track_id
            ]
            if len(matches) != 1:
                message = (
                    f"preview frame {frame_index} does not contain exactly one requested track"
                )
                raise ValueError(message)
            boxes[frame_index] = matches[0].frame_bbox
    if set(boxes) != expected_frames:
        message = "preview pose evidence does not cover every selected frame"
        raise ValueError(message)
    return boxes


def _pixel_box(
    bbox: tuple[float, float, float, float],
    *,
    padding: float,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    box_width = x2 - x1
    box_height = y2 - y1
    if box_width <= 0 or box_height <= 0:
        message = "preview source bbox is empty"
        raise ValueError(message)
    x1 -= box_width * padding
    x2 += box_width * padding
    y1 -= box_height * padding
    y2 += box_height * padding
    left = max(0, min(width - 1, int(np.floor(x1 * width))))
    top = max(0, min(height - 1, int(np.floor(y1 * height))))
    right = max(left + 1, min(width, int(np.ceil(x2 * width))))
    bottom = max(top + 1, min(height, int(np.ceil(y2 * height))))
    return left, top, right, bottom


def _letterbox(crop_bgr: np.ndarray[Any, np.dtype[np.uint8]], *, width: int) -> Image.Image:
    canvas_height = max(1, round(width * 1.5))
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(crop_rgb)
    scale = min(width / image.width, canvas_height / image.height)
    resized = image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new("RGB", (width, canvas_height), (12, 16, 20))
    canvas.paste(resized, ((width - resized.width) // 2, (canvas_height - resized.height) // 2))
    return canvas


def _decode_preview_frames(
    clip_path: Path,
    *,
    boxes: dict[int, tuple[float, float, float, float]],
    padding: float,
    target_width: int,
) -> list[Image.Image]:
    capture = cv2.VideoCapture(str(clip_path))
    if not capture.isOpened():
        message = f"cannot open canonical clip: {clip_path}"
        raise ValueError(message)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        capture.release()
        message = "canonical clip has invalid dimensions"
        raise ValueError(message)
    selected: dict[int, Image.Image] = {}
    maximum = max(boxes)
    frame_index = 0
    try:
        while frame_index <= maximum:
            ok, frame = capture.read()
            if not ok:
                break
            bbox = boxes.get(frame_index)
            if bbox is not None:
                left, top, right, bottom = _pixel_box(
                    bbox,
                    padding=padding,
                    width=width,
                    height=height,
                )
                crop = cast("np.ndarray[Any, np.dtype[np.uint8]]", frame[top:bottom, left:right])
                selected[frame_index] = _letterbox(crop, width=target_width)
            frame_index += 1
    finally:
        capture.release()
    if set(selected) != set(boxes):
        message = "canonical clip ended before all preview frames were decoded"
        raise ValueError(message)
    return [selected[index] for index in sorted(selected)]


def build_identity_preview_artifacts(
    *,
    request: IdentityPreviewJobRequest,
    inputs: IdentityPreviewInputs,
) -> IdentityPreviewArtifacts:
    """Build an animated WebP without loading or rerunning any pose model."""
    if request.recipe.output_format != "ANIMATED_WEBP":
        message = "this worker only advertises animated WebP identity previews"
        raise ValueError(message)
    crop_payload = inputs.crop_manifest
    pose_payload = inputs.pose_manifest
    crop_manifest = PlayerCropSourceManifest.model_validate(crop_payload)
    pose_manifest = PersonPoseEvidenceManifest.model_validate(pose_payload)
    _validate_semantic_hash(crop_payload, label="preview crop source manifest")
    _validate_semantic_hash(pose_payload, label="preview pose manifest")
    if (
        crop_manifest.analysis_run_id != request.analysis_run_id
        or pose_manifest.analysis_run_id != request.analysis_run_id
        or crop_manifest.pose_manifest_artifact.sha256
        != hashlib.sha256(_json_bytes(pose_payload)).hexdigest()
    ):
        message = "preview request and immutable evidence references do not match"
        raise ValueError(message)
    boxes = _selected_boxes(request=request, manifest=pose_manifest, chunks=inputs.pose_chunks)
    frames = _decode_preview_frames(
        inputs.clip_path,
        boxes=boxes,
        padding=request.recipe.crop_padding_ratio,
        target_width=request.recipe.target_width,
    )
    media = io.BytesIO()
    frames[0].save(
        media,
        format="WEBP",
        save_all=True,
        append_images=frames[1:],
        duration=request.recipe.frame_duration_ms,
        loop=0,
        lossless=False,
        quality=82,
        method=4,
    )
    media_bytes = media.getvalue()
    media_sha = hashlib.sha256(media_bytes).hexdigest()
    source_frames = list(request.selected_frame_indices)
    body: dict[str, Any] = {
        "schema_version": "1.0.0",
        "provider_job_id": request.provider_job_id,
        "preview_id": request.preview_id,
        "tracklet_id": request.tracklet_id,
        "recipe_namespace": request.recipe.namespace,
        "media_artifact": {
            "artifact_id": f"{request.preview_id}:media",
            "kind": "IDENTITY_PREVIEW",
            "sha256": media_sha,
            "byte_length": str(len(media_bytes)),
            "content_type": WEBP_CONTENT_TYPE,
        },
        "source_frame_indices": source_frames,
        "start_frame_index": source_frames[0],
        "end_frame_index": source_frames[-1],
        "width": frames[0].width,
        "height": frames[0].height,
        "frame_count": len(frames),
        "duration_ms": len(frames) * request.recipe.frame_duration_ms,
    }
    result = IdentityPreviewResult.model_validate({**body, "content_sha256": _semantic_hash(body)})
    result_bytes = _json_bytes(result.model_dump(mode="json"))
    return IdentityPreviewArtifacts(
        artifacts=(
            ProviderResultArtifact(
                part_name="identity_preview",
                kind="IDENTITY_PREVIEW",
                schema_version="1.0.0",
                content_type=WEBP_CONTENT_TYPE,
                data=media_bytes,
                filename="identity-preview.webp",
            ),
            ProviderResultArtifact(
                part_name="identity_preview_result",
                kind="IDENTITY_PREVIEW_RESULT",
                schema_version="1.0.0",
                content_type=JSON_CONTENT_TYPE,
                data=result_bytes,
                filename="identity-preview-result.json",
            ),
        ),
        result=result,
    )
