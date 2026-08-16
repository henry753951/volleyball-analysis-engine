"""Online worker, offline inference and environment diagnostics CLI."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import shutil
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import orjson
from volleyball_monitoring_ai import (
    AIJobRequest,
    AnalysisDataBundle,
    OfflineProgressReporter,
    OfflineRunner,
)

from .config import Settings
from .inference import Rtv4X3DObservationProvider
from .worker import build_pipeline, run_worker


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="volleyball-analysis")
    subcommands = parser.add_subparsers(dest="command", required=True)
    worker = subcommands.add_parser(
        "worker", help="connect to the central system and process leased jobs"
    )
    vlm_group = worker.add_mutually_exclusive_group()
    vlm_group.add_argument(
        "--enable-reid-vlm",
        dest="reid_vlm_enabled",
        action="store_true",
        default=None,
        help="enable the optional jersey-number VLM inside ReID feature work",
    )
    vlm_group.add_argument(
        "--disable-reid-vlm",
        dest="reid_vlm_enabled",
        action="store_false",
        help="disable VLM loading and VLM capability advertisement",
    )
    offline = subcommands.add_parser("offline", help="infer one local clip without network access")
    offline.add_argument("--job", type=Path, required=True)
    offline.add_argument("--clip", type=Path, required=True)
    offline.add_argument("--output", type=Path, required=True)
    offline.add_argument("--keypoints", type=Path)
    offline.add_argument("--skip-clip-verification", action="store_true")
    offline.add_argument("--prewarm", action="store_true")
    offline.add_argument("--no-debug-artifacts", action="store_true")
    doctor = subcommands.add_parser("doctor", help="validate models, GPU, FFmpeg and a clip")
    doctor.add_argument("--clip", type=Path)
    doctor.add_argument("--load-models", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Dispatch a command using environment-backed model settings."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    # Signed media URLs contain short-lived credentials and must never enter logs.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    arguments = _parser().parse_args(argv)
    settings = _settings_from_arguments(arguments)
    if arguments.command == "worker":
        asyncio.run(run_worker(settings))
        return
    if arguments.command == "offline":
        asyncio.run(_run_offline(settings, arguments))
        return
    if arguments.command == "doctor":
        _doctor(settings, arguments.clip, load_models=arguments.load_models)


def worker_main() -> None:
    """Run the Provider Work worker console entrypoint."""
    main(["worker", *sys.argv[1:]])


def _settings_from_arguments(arguments: argparse.Namespace) -> Settings:
    """Load environment settings and apply explicit worker CLI overrides."""
    settings = Settings()
    reid_vlm_enabled = getattr(arguments, "reid_vlm_enabled", None)
    if reid_vlm_enabled is not None:
        settings = settings.model_copy(update={"reid_vlm_enabled": reid_vlm_enabled})
    return settings


async def _run_offline(settings: Settings, arguments: argparse.Namespace) -> None:
    pipeline = build_pipeline(settings)
    output_dir = arguments.output.expanduser().resolve()
    prewarm_seconds = 0.0
    if arguments.prewarm:
        started = perf_counter()
        await asyncio.to_thread(pipeline.prepare)
        prewarm_seconds = perf_counter() - started

    def analyze(
        job: AIJobRequest,
        clip_path: Path,
        report: OfflineProgressReporter,
    ) -> AnalysisDataBundle:
        return pipeline.analyze(
            job,
            clip_path,
            report,
            None if arguments.no_debug_artifacts else output_dir,
        )

    def progress(value: float, stage: str) -> None:
        logging.getLogger(__name__).info("%5.1f%% %s", value * 100.0, stage)

    started = perf_counter()
    result = await OfflineRunner(verify_clip=not arguments.skip_clip_verification).run(
        job_path=arguments.job,
        key_points_path=arguments.keypoints,
        clip_path=arguments.clip,
        output_dir=output_dir,
        analyzer=analyze,
        progress=progress,
    )
    wall_seconds = perf_counter() - started
    capture = cv2.VideoCapture(str(arguments.clip.expanduser().resolve()))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    clip_seconds = frames / max(fps, 1e-9)
    benchmark = {
        "schema": "volleyball-analysis-benchmark-v1",
        "frames": frames,
        "source_fps": fps,
        "clip_seconds": clip_seconds,
        "prewarm_seconds": prewarm_seconds,
        "job_wall_seconds": wall_seconds,
        "effective_fps": frames / max(wall_seconds, 1e-9),
        "realtime_factor": clip_seconds / max(wall_seconds, 1e-9),
        "debug_artifacts": not arguments.no_debug_artifacts,
    }
    benchmark_path = result.output_dir / "benchmark.json"
    benchmark_path.write_bytes(orjson.dumps(benchmark, option=orjson.OPT_INDENT_2))
    print(orjson.dumps(benchmark).decode())


def _doctor(settings: Settings, clip: Path | None, *, load_models: bool) -> None:
    pipeline = build_pipeline(settings)
    provider = pipeline.provider
    if not isinstance(provider, Rtv4X3DObservationProvider):
        raise TypeError("doctor requires the RTv4 observation provider")
    paths = provider.paths
    paths.validate()
    report: dict[str, Any] = {
        "assets": "ok",
        "ffmpeg": shutil.which("ffmpeg"),
        "ffprobe": shutil.which("ffprobe"),
        "device": settings.device,
    }
    if report["ffmpeg"] is None or report["ffprobe"] is None:
        raise RuntimeError("ffmpeg and ffprobe must be available on PATH")
    torch = __import__("torch")
    report["torch"] = torch.__version__
    report["cuda_available"] = bool(torch.cuda.is_available())
    if settings.device.startswith("cuda") and not report["cuda_available"]:
        raise RuntimeError("CUDA device requested but unavailable")
    if report["cuda_available"]:
        report["gpu"] = torch.cuda.get_device_name(0)
    if clip is not None:
        capture = cv2.VideoCapture(str(clip.expanduser().resolve(strict=True)))
        if not capture.isOpened():
            raise ValueError(f"cannot open clip: {clip}")
        report["clip"] = {
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": capture.get(cv2.CAP_PROP_FPS),
            "frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        }
        capture.release()
    if load_models:

        def model_progress(value: float, stage: str) -> None:
            logging.getLogger(__name__).info("%5.1f%% %s", value * 100.0, stage)

        provider.prepare(model_progress)
        report["model_strict_load"] = "ok"
        report["streaming_backend"] = provider.effective_backend
    print(json.dumps(report, ensure_ascii=False, indent=2))
