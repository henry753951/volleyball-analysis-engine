"""Durable outbound worker entrypoint using the central SDK."""

from __future__ import annotations

import asyncio
import logging
from time import perf_counter

from volleyball_monitoring_ai import (
    AIWorkerClient,
    JobContext,
    ProviderCapabilities,
    WorkerConfig,
)

from .config import Settings
from .inference import ModelPaths, Rtv4X3DObservationProvider
from .pipeline import AnalysisPipeline, PipelineConfig

LOGGER = logging.getLogger(__name__)


def capabilities(settings: Settings) -> ProviderCapabilities:
    """Declare the exact contract versions and temporary optional outputs."""
    return ProviderCapabilities.model_validate(
        {
            "schema_version": "1.0.0",
            "provider_name": "volleyball-analysis-engine",
            "provider_build_id": settings.provider_build_id,
            "supported_job_schema_versions": ["1.1.0"],
            "supported_result_schema_versions": ["1.0.0"],
            "supported_overlay_formats": ["flatbuffers_v1"],
            "optional_extensions": {"action": True, "group_phase": False, "confidence": True},
            "action_taxonomies": [
                {
                    "taxonomy_id": "volleyball-analysis-engine.rtv4-x3d-actions",
                    "taxonomy_version": "1",
                }
            ],
            "limits": {"max_concurrent_jobs": settings.max_concurrency},
        }
    )


async def run_worker(settings: Settings) -> None:
    """Connect forever, accepting only central-server leased work."""
    settings.validate_online()
    pipeline = build_pipeline(settings)
    if settings.prewarm_models:
        await asyncio.to_thread(pipeline.prepare)
    worker_config = (
        WorkerConfig(
            server_ws_url=settings.server_ws_url,
            token=settings.token,
            workspace=settings.workspace,
            provider_build_id=settings.provider_build_id,
            capabilities=capabilities(settings),
            max_concurrency=settings.max_concurrency,
            instance_id=settings.instance_id,
        )
        if settings.instance_id is not None
        else WorkerConfig(
            server_ws_url=settings.server_ws_url,
            token=settings.token,
            workspace=settings.workspace,
            provider_build_id=settings.provider_build_id,
            capabilities=capabilities(settings),
            max_concurrency=settings.max_concurrency,
        )
    )
    client = AIWorkerClient(worker_config)

    async def handle(context: JobContext) -> None:
        download_started = perf_counter()
        clip_path = await context.download_clip()
        download_seconds = perf_counter() - download_started
        loop = asyncio.get_running_loop()

        def report(progress: float, stage: str) -> None:
            future = asyncio.run_coroutine_threadsafe(
                context.report_progress(progress, stage),
                loop,
            )
            future.result(timeout=15)

        analysis_started = perf_counter()
        bundle = await asyncio.to_thread(
            pipeline.analyze,
            context.job,
            clip_path,
            report,
            context.workspace / "artifacts" if settings.write_debug_artifacts else None,
        )
        analysis_seconds = perf_counter() - analysis_started
        complete_started = perf_counter()
        await context.complete(bundle)
        complete_seconds = perf_counter() - complete_started
        LOGGER.info(
            "job timing download=%.3fs analysis=%.3fs complete=%.3fs total=%.3fs",
            download_seconds,
            analysis_seconds,
            complete_seconds,
            download_seconds + analysis_seconds + complete_seconds,
        )
        await context.report_progress(1.0, "completed")

    await client.run_forever(handle)


def build_pipeline(settings: Settings) -> AnalysisPipeline:
    """Build the single model pipeline shared by online and offline entrypoints."""
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
        detector_stride=settings.detector_stride,
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
    )
    return AnalysisPipeline(provider, PipelineConfig())
