"""Durable outbound worker entrypoint using the central SDK."""

from __future__ import annotations

import asyncio

from volleyball_monitoring_ai import (
    AIWorkerClient,
    JobContext,
    ProviderCapabilities,
    WorkerConfig,
)

from .config import Settings
from .pipeline import AnalysisPipeline, PipelineConfig


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
                    "taxonomy_id": "volleyball-analysis-engine.ball-path-heuristic",
                    "taxonomy_version": "1",
                }
            ],
            "limits": {"max_concurrent_jobs": settings.max_concurrency},
        }
    )


async def run_worker(settings: Settings) -> None:
    """Connect forever, accepting only central-server leased work."""
    pipeline = AnalysisPipeline(
        PipelineConfig(
            fixture_root=settings.fixture_root,
            tracking_variant=settings.tracking_variant,
        )
    )
    worker_config = (
        WorkerConfig(
            server_ws_url=settings.provider_url(),
            token=settings.token,
            workspace=settings.workspace,
            provider_build_id=settings.provider_build_id,
            capabilities=capabilities(settings),
            max_concurrency=settings.max_concurrency,
            instance_id=settings.instance_id,
        )
        if settings.instance_id is not None
        else WorkerConfig(
            server_ws_url=settings.provider_url(),
            token=settings.token,
            workspace=settings.workspace,
            provider_build_id=settings.provider_build_id,
            capabilities=capabilities(settings),
            max_concurrency=settings.max_concurrency,
        )
    )
    client = AIWorkerClient(worker_config)

    async def handle(context: JobContext) -> None:
        await context.download_clip()
        loop = asyncio.get_running_loop()

        def report(progress: float, stage: str) -> None:
            future = asyncio.run_coroutine_threadsafe(
                context.report_progress(progress, stage),
                loop,
            )
            future.result(timeout=15)

        bundle = await asyncio.to_thread(pipeline.analyze, context.job, report)
        await context.complete(bundle)
        await context.report_progress(1.0, "completed")

    await client.run_forever(handle)
