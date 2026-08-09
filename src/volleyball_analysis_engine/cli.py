"""Online worker, offline inference and environment diagnostics CLI."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import shutil
from pathlib import Path
from typing import Any

import cv2
from volleyball_monitoring_ai import (
    AIJobRequest,
    AnalysisBundle,
    OfflineProgressReporter,
    OfflineRunner,
)

from .config import Settings
from .inference import Rtv4X3DObservationProvider
from .worker import build_pipeline, run_worker


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="volleyball-analysis")
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("worker", help="connect to the central system and process leased jobs")
    offline = subcommands.add_parser("offline", help="infer one local clip without network access")
    offline.add_argument("--job", type=Path, required=True)
    offline.add_argument("--clip", type=Path, required=True)
    offline.add_argument("--output", type=Path, required=True)
    offline.add_argument("--keypoints", type=Path)
    offline.add_argument("--skip-clip-verification", action="store_true")
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
    arguments = _parser().parse_args(argv)
    settings = Settings()
    if arguments.command == "worker":
        asyncio.run(run_worker(settings))
        return
    if arguments.command == "offline":
        asyncio.run(_run_offline(settings, arguments))
        return
    if arguments.command == "doctor":
        _doctor(settings, arguments.clip, load_models=arguments.load_models)


def worker_main() -> None:
    """Backward-compatible worker-only console entrypoint."""
    main(["worker"])


async def _run_offline(settings: Settings, arguments: argparse.Namespace) -> None:
    pipeline = build_pipeline(settings)
    output_dir = arguments.output.expanduser().resolve()

    def analyze(
        job: AIJobRequest,
        clip_path: Path,
        report: OfflineProgressReporter,
    ) -> AnalysisBundle:
        return pipeline.analyze(
            job,
            clip_path,
            report,
            output_dir,
        )

    def progress(value: float, stage: str) -> None:
        logging.getLogger(__name__).info("%5.1f%% %s", value * 100.0, stage)

    result = await OfflineRunner(
        verify_clip=not arguments.skip_clip_verification
    ).run(
        job_path=arguments.job,
        key_points_path=arguments.keypoints,
        clip_path=arguments.clip,
        output_dir=output_dir,
        analyzer=analyze,
        progress=progress,
    )
    print(result.output_dir)


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
