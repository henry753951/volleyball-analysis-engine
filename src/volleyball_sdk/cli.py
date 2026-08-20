"""Command-line JSONL export for downstream pipelines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .predictor import VolleyballPredictor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Volleyball multitask checkpoint and export clean JSONL."
    )
    parser.add_argument("--checkpoint", required=True, help="Project checkpoint (.pth/.pt)")
    parser.add_argument("--video", required=True, help="Input video")
    parser.add_argument("--output", required=True, help="Output .jsonl path")
    parser.add_argument("--config", default=None, help="Optional custom inference YAML")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda:0, ...")
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="Output every N source frames. Model temporal jump remains config jump_frame.",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Inference batch size")
    parser.add_argument("--score-threshold", type=float, default=None)
    parser.add_argument("--limit", type=int, default=0, help="0 = full video")
    parser.add_argument("--no-warmup", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    predictor = VolleyballPredictor(
        args.checkpoint,
        config=args.config,
        device=args.device,
        fp16=args.fp16,
        warmup=not args.no_warmup,
    )

    print(json.dumps({"model": predictor.info()}, ensure_ascii=False, indent=2))
    count = 0
    with output.open("w", encoding="utf-8") as handle:
        for result in predictor.predict_video(
            args.video,
            step=args.step,
            batch_size=args.batch_size,
            score_threshold=args.score_threshold,
        ):
            handle.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
            if args.limit > 0 and count >= args.limit:
                break

    print(f"Wrote {count} predictions to {output}")


if __name__ == "__main__":
    main()
