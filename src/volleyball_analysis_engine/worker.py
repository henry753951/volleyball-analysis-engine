"""Durable outbound worker entrypoint using the central SDK."""

from __future__ import annotations

import asyncio
import json
import logging

from volleyball_monitoring_ai import (
    AIJobRequest,
    IdentityPreviewJobRequest,
    ProviderAnalysisJobRequest,
    ProviderWorkCapabilities,
    ProviderWorkContext,
    ProviderWorkerClient,
    ProviderWorkerConfig,
    ProviderWorkHandler,
    ReidAssociationJobRequest,
    ReidFeatureJobRequest,
)
from volleyball_monitoring_ai.provider_work import ProviderWorkKind

from .config import Settings
from .identity_preview_job import IdentityPreviewInputs, build_identity_preview_artifacts
from .inference import ModelPaths, Rtv4X3DObservationProvider
from .nested_reid import NestedPartDescriptorExtractor, NestedReidPaths
from .person_pose import PersonPoseExtractor
from .pipeline import AnalysisPipeline, PipelineConfig
from .reid_association_job import (
    ReidAssociationInputs,
    build_reid_association_artifacts,
)
from .reid_feature_job import (
    CandidateConstrainedJerseyVlm,
    ReidFeatureInputs,
    SportsOsnetCropEncoder,
    build_reid_feature_artifacts,
)

LOGGER = logging.getLogger(__name__)


def provider_work_capabilities(settings: Settings) -> ProviderWorkCapabilities:
    """Advertise only handlers that are executable through Provider Work v2."""
    work_capabilities = [
        {
            "work_kind": "ANALYSIS",
            "request_schema_versions": ["1.0.0"],
            "result_schema_versions": ["1.0.0"],
            "accepted_input_artifact_kinds": ["CANONICAL_CLIP"],
            "produced_artifact_kinds": [
                "ANALYSIS_DATA",
                "ANALYSIS_EVIDENCE_MANIFEST",
                "PERSON_POSE_EVIDENCE_MANIFEST",
                "PERSON_POSE_EVIDENCE_CHUNK",
                "PLAYER_CROP_SOURCE_MANIFEST",
            ],
            "model_recipe_namespaces": ["analysis/base-every-frame-pose-v1"],
            "hardware": {"accelerator": "CUDA"},
            "max_concurrency": settings.max_concurrency,
        }
    ]
    if settings.reid_feature_enabled:
        produced_artifact_kinds = [
            "REID_FEATURE_RESULT",
            "REID_DESCRIPTOR_BUNDLE",
        ]
        model_recipe_namespaces = [
            "dinov2/vits14-reg/v1",
            "sports-osnet/x1/v1",
            "kpr/coco17-prompt/v1",
            "kpr/plain/v1",
        ]
        if settings.reid_vlm_enabled:
            produced_artifact_kinds.append("JERSEY_VLM_RESPONSE")
            model_recipe_namespaces.append("jersey-vlm/qwen-v1")
        work_capabilities.append(
            {
                "work_kind": "REID_FEATURE_EXTRACTION",
                "request_schema_versions": ["1.0.0"],
                "result_schema_versions": ["1.0.0"],
                "accepted_input_artifact_kinds": [
                    "CANONICAL_CLIP",
                    "ANALYSIS_DATA",
                    "ANALYSIS_EVIDENCE_MANIFEST",
                    "PERSON_POSE_EVIDENCE_MANIFEST",
                    "PERSON_POSE_EVIDENCE_CHUNK",
                    "PLAYER_CROP_SOURCE_MANIFEST",
                    "REID_ROSTER_SNAPSHOT",
                ],
                "produced_artifact_kinds": produced_artifact_kinds,
                "model_recipe_namespaces": model_recipe_namespaces,
                "hardware": {"accelerator": "CUDA"},
                "max_concurrency": settings.max_concurrency,
            }
        )
    if settings.reid_association_enabled:
        work_capabilities.append(
            {
                "work_kind": "REID_ASSOCIATION",
                "request_schema_versions": ["1.1.0"],
                "result_schema_versions": ["1.0.0"],
                "accepted_input_artifact_kinds": [
                    "REID_FEATURE_RESULT",
                    "REID_DESCRIPTOR_BUNDLE",
                    "REID_BANK_SNAPSHOT",
                    "REID_ROSTER_SNAPSHOT",
                ],
                "produced_artifact_kinds": ["REID_ASSOCIATION_RESULT"],
                "model_recipe_namespaces": ["reid/nested-part-v2"],
                "hardware": {"accelerator": "ANY"},
                "max_concurrency": settings.max_concurrency,
            }
        )
    if settings.identity_preview_enabled:
        work_capabilities.append(
            {
                "work_kind": "IDENTITY_PREVIEW_GENERATION",
                "request_schema_versions": ["1.1.0"],
                "result_schema_versions": ["1.0.0"],
                "accepted_input_artifact_kinds": [
                    "CANONICAL_CLIP",
                    "PERSON_POSE_EVIDENCE_MANIFEST",
                    "PERSON_POSE_EVIDENCE_CHUNK",
                    "PLAYER_CROP_SOURCE_MANIFEST",
                ],
                "produced_artifact_kinds": [
                    "IDENTITY_PREVIEW",
                    "IDENTITY_PREVIEW_RESULT",
                ],
                "model_recipe_namespaces": ["identity-preview/animated-webp/v1"],
                "hardware": {"accelerator": "ANY"},
                "max_concurrency": settings.max_concurrency,
            }
        )
    return ProviderWorkCapabilities.model_validate(
        {
            "schema_version": "3.0.0",
            "provider_name": "volleyball-analysis-engine",
            "provider_build_id": settings.provider_build_id,
            "work_capabilities": work_capabilities,
        }
    )


def _analysis_job(
    context: ProviderWorkContext,
    request: ProviderAnalysisJobRequest,
) -> AIJobRequest:
    clips = [
        artifact for artifact in context.work.input_artifacts if artifact.kind == "CANONICAL_CLIP"
    ]
    if len(clips) != 1:
        message = "Provider Work analysis requires exactly one CANONICAL_CLIP artifact"
        raise ValueError(message)
    artifact = clips[0]
    return AIJobRequest.model_validate(
        {
            "schema_version": "3.0.0",
            "ai_job_id": request.ai_job_id,
            "rally_submission_id": request.rally_submission_id,
            "rally_id": request.rally_id,
            "match_id": request.match_id,
            "annotation_revision": request.annotation_revision,
            "clip": {
                "clip_asset_id": request.clip.clip_asset_id,
                "download_url": artifact.download_url,
                "download_url_expires_at": artifact.download_url_expires_at,
                "sha256": artifact.sha256,
                "byte_length": artifact.byte_length,
                "content_type": artifact.content_type,
                "video": request.clip.video.model_dump(mode="json"),
            },
            "key_points": [point.model_dump(mode="json") for point in request.key_points],
            "boundaries": [boundary.model_dump(mode="json") for boundary in request.boundaries],
            # Internal compatibility only. ProviderAnalysisJobRequest has no ReID module.
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
            "outcome": request.outcome.model_dump(mode="json"),
            "callback": {
                "url": context.work.callback.url,
                "token": context.work.callback.token,
                "expires_at": context.work.callback.expires_at,
            },
        }
    )


async def run_provider_work_worker(settings: Settings) -> None:
    """Run the v2 base-analysis handler with multi-artifact completion."""
    pipeline = build_pipeline(settings)
    if settings.prewarm_models:
        await asyncio.to_thread(
            pipeline.prepare,
            lambda progress, stage: LOGGER.info("model prewarm %.1f%% %s", progress * 100, stage),
        )
    config_options = {
        "server_ws_url": settings.server_ws_url,
        "token": settings.token,
        "workspace": settings.workspace,
        "provider_build_id": settings.provider_build_id,
        "capabilities": provider_work_capabilities(settings),
    }
    if settings.instance_id is not None:
        config_options["instance_id"] = settings.instance_id
    client = ProviderWorkerClient(ProviderWorkerConfig(**config_options))  # type: ignore[arg-type]

    feature_nested = (
        NestedPartDescriptorExtractor(
            NestedReidPaths(
                dinov2_root=settings.dinov2_root,
                dinov2_checkpoint=settings.dinov2_checkpoint,
                kpr_python=settings.kpr_python,
                kpr_root=settings.kpr_root,
                kpr_checkpoint=settings.kpr_checkpoint,
                kpr_bridge=settings.kpr_bridge,
            ),
            device=settings.device,
            batch_size=settings.reid_feature_batch_size,
        )
        if settings.reid_feature_enabled
        else None
    )
    feature_osnet = (
        SportsOsnetCropEncoder(
            smp_root=settings.smp_root,
            checkpoint=settings.osnet_checkpoint,
            device=settings.device,
        )
        if settings.reid_feature_enabled
        else None
    )
    feature_vlm = (
        CandidateConstrainedJerseyVlm(
            model_id=settings.reid_vlm_model_id,
            device=settings.device,
            dtype=settings.reid_vlm_dtype,
            max_new_tokens=settings.reid_vlm_max_new_tokens,
        )
        if settings.reid_feature_enabled and settings.reid_vlm_enabled
        else None
    )
    # Model instances and the X3D streamer retain mutable CUDA/runtime state.
    # Provider work may download or upload concurrently, but GPU inference must
    # be serialized unless each lease owns an isolated model instance.
    gpu_work_lock = asyncio.Lock()

    async def handle_analysis(context: ProviderWorkContext) -> None:
        request = ProviderAnalysisJobRequest.model_validate(context.work.request)
        if request.provider_job_id != context.work.provider_job_id:
            message = "provider analysis request/job ID mismatch"
            raise ValueError(message)
        clip_artifacts = [
            artifact
            for artifact in context.work.input_artifacts
            if artifact.kind == "CANONICAL_CLIP"
        ]
        if len(clip_artifacts) != 1:
            message = "Provider Work analysis requires exactly one CANONICAL_CLIP artifact"
            raise ValueError(message)
        clip_path = await context.download_artifact(
            clip_artifacts[0].artifact_id,
            filename="canonical.mp4",
        )
        incoming = _analysis_job(context, request)
        loop = asyncio.get_running_loop()

        def report(progress: float, stage: str) -> None:
            future = asyncio.run_coroutine_threadsafe(
                context.report_progress(progress, stage),
                loop,
            )
            future.result(timeout=15)

        async with gpu_work_lock:
            result = await asyncio.to_thread(
                pipeline.analyze_provider_work,
                incoming,
                clip_path,
                report,
                context.workspace / "artifacts" if settings.write_debug_artifacts else None,
            )
        await context.complete(
            result_schema_version="1.0.0",
            artifacts=list(result.evidence.artifacts),
        )

    async def handle_reid_feature(context: ProviderWorkContext) -> None:
        if feature_nested is None or feature_osnet is None:
            message = "ReID feature handler was invoked while the capability is disabled"
            raise RuntimeError(message)
        request = ReidFeatureJobRequest.model_validate(context.work.request)
        if request.provider_job_id != context.work.provider_job_id:
            message = "provider ReID feature request/job ID mismatch"
            raise ValueError(message)

        def exactly_one(kind: str):  # noqa: ANN202
            artifacts = [item for item in context.work.input_artifacts if item.kind == kind]
            if len(artifacts) != 1:
                message = f"ReID feature work requires exactly one {kind} artifact"
                raise ValueError(message)
            return artifacts[0]

        clip_artifact = exactly_one("CANONICAL_CLIP")
        analysis_data_artifact = exactly_one("ANALYSIS_DATA")
        pose_manifest_artifact = exactly_one("PERSON_POSE_EVIDENCE_MANIFEST")
        crop_manifest_artifact = exactly_one("PLAYER_CROP_SOURCE_MANIFEST")
        analysis_manifest_artifact = context.input_artifact(request.analysis_evidence_artifact_id)
        roster_artifact = context.input_artifact(request.roster_snapshot_artifact_id)
        if analysis_manifest_artifact.kind != "ANALYSIS_EVIDENCE_MANIFEST":
            message = "analysis_evidence_artifact_id is not an analysis evidence manifest"
            raise ValueError(message)
        if roster_artifact.kind != "REID_ROSTER_SNAPSHOT":
            message = "roster_snapshot_artifact_id is not a roster snapshot"
            raise ValueError(message)
        chunk_artifacts = [
            item
            for item in context.work.input_artifacts
            if item.kind == "PERSON_POSE_EVIDENCE_CHUNK"
        ]
        await context.report_progress(0.02, "downloading_reid_evidence")
        downloads = await asyncio.gather(
            context.download_artifact(clip_artifact.artifact_id, filename="canonical.mp4"),
            context.download_artifact(
                analysis_data_artifact.artifact_id, filename="analysis-data.vad1"
            ),
            context.download_artifact(
                analysis_manifest_artifact.artifact_id, filename="analysis-evidence.json"
            ),
            context.download_artifact(
                pose_manifest_artifact.artifact_id, filename="person-pose-manifest.json"
            ),
            context.download_artifact(
                crop_manifest_artifact.artifact_id, filename="crop-source-manifest.json"
            ),
            context.download_artifact(roster_artifact.artifact_id, filename="roster-snapshot.json"),
            *[
                context.download_artifact(item.artifact_id, filename=f"pose-{index:04d}.vpe1")
                for index, item in enumerate(chunk_artifacts)
            ],
        )
        clip_path, analysis_path, analysis_manifest_path, pose_manifest_path = downloads[:4]
        crop_manifest_path, roster_path = downloads[4:6]
        chunk_paths = downloads[6:]
        pose_manifest_payload = json.loads(pose_manifest_path.read_text(encoding="utf-8"))
        chunk_bytes_by_sha = {
            artifact.sha256.lower(): path.read_bytes()
            for artifact, path in zip(chunk_artifacts, chunk_paths, strict=True)
        }
        ordered_chunks = tuple(
            chunk_bytes_by_sha[entry["artifact"]["sha256"].lower()]
            for entry in sorted(pose_manifest_payload["chunks"], key=lambda value: value["index"])
        )
        feature_inputs = ReidFeatureInputs(
            clip_path=clip_path,
            analysis_data=analysis_path.read_bytes(),
            analysis_manifest=json.loads(analysis_manifest_path.read_text(encoding="utf-8")),
            pose_manifest=pose_manifest_payload,
            crop_manifest=json.loads(crop_manifest_path.read_text(encoding="utf-8")),
            roster_snapshot=json.loads(roster_path.read_text(encoding="utf-8")),
            pose_chunks=ordered_chunks,
        )
        await context.report_progress(0.15, "extracting_reid_features_from_saved_pose")
        async with gpu_work_lock:
            result = await asyncio.to_thread(
                build_reid_feature_artifacts,
                request=request,
                inputs=feature_inputs,
                nested=feature_nested,
                osnet=feature_osnet,
                vlm=feature_vlm,
                batch_size=settings.reid_feature_batch_size,
                candidate_count=settings.reid_feature_candidate_frames,
                top_k=settings.reid_feature_selected_frames,
                min_gap=settings.reid_feature_min_frame_gap,
            )
        await context.report_progress(0.92, "uploading_reid_feature_evidence")
        await context.complete(result_schema_version="1.0.0", artifacts=list(result.artifacts))

    async def handle_reid_association(context: ProviderWorkContext) -> None:
        request = ReidAssociationJobRequest.model_validate(context.work.request)
        if request.provider_job_id != context.work.provider_job_id:
            message = "provider ReID association request/job ID mismatch"
            raise ValueError(message)
        feature_artifact = context.input_artifact(request.evidence_result_artifact_id)
        bank_artifact = context.input_artifact(request.bank_snapshot_artifact_id)
        roster_artifact = context.input_artifact(request.roster_snapshot_artifact_id)
        if feature_artifact.kind != "REID_FEATURE_RESULT":
            message = "evidence_result_artifact_id is not a feature result"
            raise ValueError(message)
        if bank_artifact.kind != "REID_BANK_SNAPSHOT":
            message = "bank_snapshot_artifact_id is not a bank snapshot"
            raise ValueError(message)
        if roster_artifact.kind != "REID_ROSTER_SNAPSHOT":
            message = "roster_snapshot_artifact_id is not a roster snapshot"
            raise ValueError(message)
        descriptor_artifacts = [
            item for item in context.work.input_artifacts if item.kind == "REID_DESCRIPTOR_BUNDLE"
        ]
        await context.report_progress(0.05, "downloading_association_evidence")
        downloads = await asyncio.gather(
            context.download_artifact(feature_artifact.artifact_id, filename="feature-result.json"),
            context.download_artifact(bank_artifact.artifact_id, filename="bank-snapshot.json"),
            context.download_artifact(roster_artifact.artifact_id, filename="roster-snapshot.json"),
            *[
                context.download_artifact(
                    item.artifact_id, filename=f"descriptor-{index:04d}.f32le"
                )
                for index, item in enumerate(descriptor_artifacts)
            ],
        )
        feature_payload = json.loads(downloads[0].read_text(encoding="utf-8"))
        bank_payload = json.loads(downloads[1].read_text(encoding="utf-8"))
        descriptor_bytes = {
            artifact.artifact_id: path.read_bytes()
            for artifact, path in zip(descriptor_artifacts, downloads[3:], strict=True)
        }
        current_reference = feature_payload.get("descriptor_artifact", {})
        current_sha = str(current_reference.get("sha256", "")).lower()
        current_artifact = next(
            (item for item in descriptor_artifacts if item.sha256.lower() == current_sha),
            None,
        )
        if current_artifact is None:
            message = "current feature descriptor bundle is absent from association inputs"
            raise ValueError(message)
        current_descriptors = descriptor_bytes.pop(current_artifact.artifact_id)
        bank_artifact_ids = {
            str(item["artifact_id"]) for item in bank_payload.get("evidence_artifacts", [])
        }
        if set(descriptor_bytes) != bank_artifact_ids:
            message = "historical descriptor inputs do not match the bank snapshot"
            raise ValueError(message)
        await context.report_progress(0.35, "scoring_versioned_reid_association")
        result = await asyncio.to_thread(
            build_reid_association_artifacts,
            request=request,
            inputs=ReidAssociationInputs(
                feature_result=feature_payload,
                current_descriptors=current_descriptors,
                bank_snapshot=bank_payload,
                roster_snapshot=json.loads(downloads[2].read_text(encoding="utf-8")),
                bank_descriptor_artifacts=descriptor_bytes,
            ),
        )
        await context.complete(result_schema_version="1.0.0", artifacts=list(result.artifacts))

    async def handle_identity_preview(context: ProviderWorkContext) -> None:
        request = IdentityPreviewJobRequest.model_validate(context.work.request)
        if request.provider_job_id != context.work.provider_job_id:
            message = "provider identity preview request/job ID mismatch"
            raise ValueError(message)
        clip_artifacts = [
            item for item in context.work.input_artifacts if item.kind == "CANONICAL_CLIP"
        ]
        if len(clip_artifacts) != 1:
            message = "identity preview work requires exactly one CANONICAL_CLIP artifact"
            raise ValueError(message)
        crop_artifact = context.input_artifact(request.crop_source_manifest_artifact_id)
        pose_artifact = context.input_artifact(request.pose_manifest_artifact_id)
        if crop_artifact.kind != "PLAYER_CROP_SOURCE_MANIFEST":
            message = "crop_source_manifest_artifact_id is not a crop source manifest"
            raise ValueError(message)
        if pose_artifact.kind != "PERSON_POSE_EVIDENCE_MANIFEST":
            message = "pose_manifest_artifact_id is not a person pose manifest"
            raise ValueError(message)
        chunk_artifacts = [
            item
            for item in context.work.input_artifacts
            if item.kind == "PERSON_POSE_EVIDENCE_CHUNK"
        ]
        await context.report_progress(0.05, "downloading_saved_pose_for_preview")
        downloads = await asyncio.gather(
            context.download_artifact(clip_artifacts[0].artifact_id, filename="canonical.mp4"),
            context.download_artifact(crop_artifact.artifact_id, filename="crop-source.json"),
            context.download_artifact(pose_artifact.artifact_id, filename="pose-manifest.json"),
            *[
                context.download_artifact(item.artifact_id, filename=f"pose-{index:04d}.vpe1")
                for index, item in enumerate(chunk_artifacts)
            ],
        )
        pose_payload = json.loads(downloads[2].read_text(encoding="utf-8"))
        chunk_bytes_by_sha = {
            artifact.sha256.lower(): path.read_bytes()
            for artifact, path in zip(chunk_artifacts, downloads[3:], strict=True)
        }
        ordered_chunks = tuple(
            chunk_bytes_by_sha[entry["artifact"]["sha256"].lower()]
            for entry in sorted(pose_payload["chunks"], key=lambda value: value["index"])
        )
        await context.report_progress(0.35, "rendering_identity_preview_from_saved_pose")
        result = await asyncio.to_thread(
            build_identity_preview_artifacts,
            request=request,
            inputs=IdentityPreviewInputs(
                clip_path=downloads[0],
                crop_manifest=json.loads(downloads[1].read_text(encoding="utf-8")),
                pose_manifest=pose_payload,
                pose_chunks=ordered_chunks,
            ),
        )
        await context.complete(result_schema_version="1.0.0", artifacts=list(result.artifacts))

    handlers: dict[ProviderWorkKind, ProviderWorkHandler] = {"ANALYSIS": handle_analysis}
    if settings.reid_feature_enabled:
        handlers["REID_FEATURE_EXTRACTION"] = handle_reid_feature
    if settings.reid_association_enabled:
        handlers["REID_ASSOCIATION"] = handle_reid_association
    if settings.identity_preview_enabled:
        handlers["IDENTITY_PREVIEW_GENERATION"] = handle_identity_preview
    await client.run_forever(handlers)


async def run_worker(settings: Settings) -> None:
    """Connect forever and accept only versioned Provider Work leases."""
    settings.validate_online()
    await run_provider_work_worker(settings)


def build_pipeline(settings: Settings) -> AnalysisPipeline:
    """Build the single model pipeline shared by online and offline entrypoints."""
    person_pose = (
        PersonPoseExtractor(
            settings.pose_checkpoint,
            device=settings.device,
            batch_size=settings.person_pose_batch_size,
            imgsz=settings.person_pose_imgsz,
            confidence=settings.person_pose_confidence,
            keypoint_confidence=settings.person_pose_keypoint_confidence,
            minimum_keypoints=settings.person_pose_minimum_keypoints,
        )
        if settings.person_pose_enabled
        else None
    )
    provider = Rtv4X3DObservationProvider(
        ModelPaths(
            rtv4_root=settings.rtv4_root,
            rtv4_config=settings.rtv4_config,
            rtv4_checkpoint=settings.rtv4_checkpoint,
            smp_root=settings.smp_root,
            osnet_checkpoint=settings.osnet_checkpoint,
        ),
        device=settings.device,
        backend=settings.rtv4_backend,
        detector_threshold=settings.detector_threshold,
        detector_input_scale=settings.detector_input_scale,
        reid_every=settings.reid_every,
        court_model=settings.court_model,
        court_imgsz=settings.court_imgsz,
        court_batch_size=settings.court_batch_size,
        court_layout_every=settings.court_layout_every,
        court_refresh_every=settings.court_refresh_every,
        court_track_every=settings.court_track_every,
        court_max_hold_frames=settings.court_max_hold_frames,
        court_decoder=settings.court_decoder,
        disable_amp=settings.disable_amp,
        person_pose=person_pose,
    )
    return AnalysisPipeline(provider, PipelineConfig())
