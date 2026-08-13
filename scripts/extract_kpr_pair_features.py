"""Extract plain and COCO-17-prompted Official KPR features in its Python 3.10 runtime."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


def main() -> None:
    """Run Official KPR once and export prompt-free plus prompted part arrays."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--kpr-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    records = json.loads(args.manifest.read_text(encoding="utf-8"))
    kpr_root = args.kpr_root.resolve()
    os.chdir(kpr_root)
    sys.path.insert(0, str(kpr_root))
    from torchreid.scripts.builder import build_config
    from torchreid.tools.feature_extractor import KPRFeatureExtractor

    config = build_config(config_path="configs/kpr/imagenet/kpr_occ_posetrack_test.yaml")
    config.model.compute_complexity = False
    extractor = KPRFeatureExtractor(config, verbose=False)
    plain_embeddings: list[np.ndarray] = []
    plain_visibility: list[np.ndarray] = []
    prompted_embeddings: list[np.ndarray] = []
    prompted_visibility: list[np.ndarray] = []
    prompted_flags: list[bool] = []

    with torch.inference_mode():
        for start in range(0, len(records), args.batch_size):
            batch = records[start : start + args.batch_size]
            plain_samples: list[dict[str, object]] = []
            prompted_samples: list[dict[str, object]] = []
            for record in batch:
                image = cv2.imread(str(record["image_path"]), cv2.IMREAD_COLOR)
                if image is None:
                    raise FileNotFoundError(record["image_path"])
                plain_samples.append({"image": image})
                prompt = record.get("keypoints_xyc")
                prompted = prompt is not None
                prompted_flags.append(prompted)
                prompted_sample: dict[str, object] = {"image": image}
                if prompted:
                    prompted_sample["keypoints_xyc"] = np.asarray(prompt, dtype=np.float32)
                    prompted_sample["negative_kps"] = np.empty((0, 17, 3), dtype=np.float32)
                prompted_samples.append(prompted_sample)
            _, embedding, visibility, _ = extractor(plain_samples)
            plain_embeddings.append(embedding.detach().cpu().float().numpy())
            plain_visibility.append(visibility.detach().cpu().float().numpy())
            _, embedding, visibility, _ = extractor(prompted_samples)
            prompted_embeddings.append(embedding.detach().cpu().float().numpy())
            prompted_visibility.append(visibility.detach().cpu().float().numpy())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        keys=np.asarray([str(record["key"]) for record in records]),
        plain_embeddings=np.concatenate(plain_embeddings, axis=0),
        plain_visibility=np.concatenate(plain_visibility, axis=0),
        prompted_embeddings=np.concatenate(prompted_embeddings, axis=0),
        prompted_visibility=np.concatenate(prompted_visibility, axis=0),
        prompted=np.asarray(prompted_flags),
    )


if __name__ == "__main__":
    main()
