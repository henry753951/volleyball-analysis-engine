"""Frozen multi-backbone descriptors for leakage-safe Nested Part Adaptation."""

from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class NestedReidPaths:
    """External frozen-model assets; none are embedded in the worker image."""

    dinov2_root: Path
    dinov2_checkpoint: Path
    kpr_python: Path
    kpr_root: Path
    kpr_checkpoint: Path
    kpr_bridge: Path

    def validate(self) -> None:
        """Fail early when a frozen descriptor model or runtime is unavailable."""
        required = {
            "DINOv2 source": self.dinov2_root / "hubconf.py",
            "DINOv2 checkpoint": self.dinov2_checkpoint,
            "KPR Python 3.10": self.kpr_python,
            "Official KPR source": self.kpr_root / "torchreid" / "tools" / "feature_extractor.py",
            "Official KPR checkpoint": self.kpr_checkpoint,
            "KPR bridge": self.kpr_bridge,
        }
        missing = [f"{label}: {path}" for label, path in required.items() if not path.exists()]
        if missing:
            raise FileNotFoundError("missing Nested Part Adaptation assets:\n" + "\n".join(missing))

    def validate_dino(self) -> None:
        """Validate only the DINO assets needed by the DINO modality."""
        required = (self.dinov2_root / "hubconf.py", self.dinov2_checkpoint)
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError("missing DINO assets:\n" + "\n".join(missing))

    def validate_kpr(self) -> None:
        """Validate only the isolated KPR assets needed by KPR modalities."""
        required = (
            self.kpr_python,
            self.kpr_root / "torchreid" / "tools" / "feature_extractor.py",
            self.kpr_checkpoint,
            self.kpr_bridge,
        )
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError("missing KPR assets:\n" + "\n".join(missing))


def _normalize(value: NDArray[np.floating[Any]]) -> NDArray[np.float32]:
    vector = np.asarray(value, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    return np.asarray(vector / norm, dtype=np.float32) if norm > 1e-12 else vector


class NestedPartDescriptorExtractor:
    """Compute DINOv2, Sports OSNet and Official KPR descriptors for selected crops."""

    def __init__(self, paths: NestedReidPaths, *, device: str, batch_size: int = 64) -> None:
        """Configure the frozen descriptor stack without loading weights yet."""
        self.paths = paths
        self.device_name = device
        self.batch_size = max(1, batch_size)
        self._torch: Any = None
        self._device: Any = None
        self._dino: Any = None
        self._kpr_runtime_validated = False

    def prepare(self) -> None:
        """Load descriptor models only; person pose is persisted by base analysis."""
        self.paths.validate()
        self.prepare_dino()
        self.validate_kpr_runtime()

    def prepare_dino(self) -> None:
        """Load only DINO so a KPR deployment problem cannot suppress DINO evidence."""
        if self._dino is not None:
            return
        self.paths.validate_dino()
        import torch

        self._torch = torch
        self._device = torch.device(self.device_name)
        hub: Any = torch.hub
        dino = hub.load(
            str(self.paths.dinov2_root.resolve()),
            "dinov2_vits14_reg",
            source="local",
            pretrained=False,
        )
        state = torch.load(
            self.paths.dinov2_checkpoint,
            map_location="cpu",
            weights_only=True,
        )
        dino.load_state_dict(state, strict=True)
        self._dino = dino.eval().to(self._device)

    def validate_kpr_runtime(self) -> None:
        """Validate KPR without loading DINO or pose."""
        if self._kpr_runtime_validated:
            return
        self.paths.validate_kpr()
        completed = subprocess.run(
            [str(self.paths.kpr_python), "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
        if "Python 3.10" not in f"{completed.stdout}{completed.stderr}":
            raise RuntimeError("Official KPR bridge must run in the isolated Python 3.10 runtime")
        self._kpr_runtime_validated = True

    def encode_saved_pose(
        self,
        *,
        keys: list[tuple[int, int]],
        crops: list[NDArray[np.uint8]],
        prompts: list[list[list[float]] | None],
    ) -> tuple[
        NDArray[np.float32],
        NDArray[np.float32],
        NDArray[np.float32],
        NDArray[np.bool_],
    ]:
        """Encode selected crops while reusing persisted COCO-17 pose prompts.

        This is the Provider Work v2 path. It intentionally prepares DINO/KPR
        without loading or invoking the pose model; pose is base-analysis evidence.
        """
        if len(keys) != len(crops) or len(crops) != len(prompts):
            raise ValueError("saved-pose descriptor inputs must have equal lengths")
        if not crops:
            raise ValueError("saved-pose descriptor extraction requires at least one crop")
        self.prepare()
        dino = self._encode_dino(crops)
        kpr, kpr_prompt, prompted = self._encode_kpr(keys, crops, prompts)
        return dino, kpr, kpr_prompt, prompted

    def encode_dino_crops(self, crops: list[NDArray[np.uint8]]) -> NDArray[np.float32]:
        """Encode DINO independently of KPR and pose availability."""
        if not crops:
            raise ValueError("DINO descriptor extraction requires at least one crop")
        self.prepare_dino()
        return self._encode_dino(crops)

    def encode_kpr_crops(
        self,
        *,
        keys: list[tuple[int, int]],
        crops: list[NDArray[np.uint8]],
        prompts: list[list[list[float]] | None],
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.bool_]]:
        """Encode KPR and KPR-prompt from persisted pose without pose inference."""
        if len(keys) != len(crops) or len(crops) != len(prompts):
            raise ValueError("KPR saved-pose inputs must have equal lengths")
        if not crops:
            raise ValueError("KPR descriptor extraction requires at least one crop")
        self.validate_kpr_runtime()
        return self._encode_kpr(keys, crops, prompts)

    def _encode_dino(self, crops: list[NDArray[np.uint8]]) -> NDArray[np.float32]:
        torch = self._torch
        mean = torch.tensor((0.485, 0.456, 0.406), device=self._device).view(1, 3, 1, 1)
        std = torch.tensor((0.229, 0.224, 0.225), device=self._device).view(1, 3, 1, 1)
        output: list[NDArray[np.float32]] = []
        with (
            torch.inference_mode(),
            torch.autocast(
                device_type=self._device.type,
                dtype=torch.float16,
                enabled=self._device.type == "cuda",
            ),
        ):
            for start in range(0, len(crops), self.batch_size):
                tensors: list[Any] = []
                for crop in crops[start : start + self.batch_size]:
                    rgb = cv2.cvtColor(cv2.resize(crop, (224, 224)), cv2.COLOR_BGR2RGB)
                    tensors.append(torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().div_(255))
                batch = torch.stack(tensors).to(self._device, non_blocking=True)
                encoded = self._dino((batch - mean) / std).float().cpu().numpy()
                output.extend(_normalize(value) for value in encoded)
        return np.stack(output)

    def _encode_kpr(
        self,
        rows: list[tuple[Any, ...]],
        crops: list[NDArray[np.uint8]],
        prompts: list[list[list[float]] | None],
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.bool_]]:
        with tempfile.TemporaryDirectory(prefix="vollyai-kpr-") as directory_name:
            directory = Path(directory_name)
            manifest: list[dict[str, object]] = []
            for index, (row, crop, prompt) in enumerate(zip(rows, crops, prompts, strict=True)):
                image_path = directory / f"{index:05d}.jpg"
                if not cv2.imwrite(str(image_path), crop):
                    raise OSError(f"failed to write KPR crop: {image_path}")
                manifest.append(
                    {
                        "key": f"{row[0]}:{row[1]}",
                        "image_path": str(image_path),
                        "keypoints_xyc": prompt,
                    }
                )
            manifest_path = directory / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            output_path = directory / "features.npz"
            subprocess.run(
                [
                    str(self.paths.kpr_python),
                    str(self.paths.kpr_bridge),
                    "--manifest",
                    str(manifest_path),
                    "--kpr-root",
                    str(self.paths.kpr_root),
                    "--output",
                    str(output_path),
                    "--batch-size",
                    str(min(32, self.batch_size)),
                ],
                check=True,
            )
            with np.load(output_path) as cache:
                plain = np.asarray(cache["plain_embeddings"], dtype=np.float32)
                prompted_values = np.asarray(cache["prompted_embeddings"], dtype=np.float32)
                prompted = np.asarray(cache["prompted"], dtype=np.bool_)
                # Part visibility is applied before temporal aggregation. Prompt failures
                # are already honest prompt-free samples in the isolated bridge.
                plain *= np.asarray(cache["plain_visibility"], dtype=np.float32)[..., None]
                prompted_values *= np.asarray(cache["prompted_visibility"], dtype=np.float32)[
                    ..., None
                ]
            return plain, prompted_values, prompted

    @staticmethod
    def _aggregate_kpr(values: NDArray[np.float32], indices: list[int]) -> NDArray[np.float32]:
        embeddings = values[indices, 1:]
        parts: list[NDArray[np.float32]] = []
        for part_index in range(embeddings.shape[1]):
            part = embeddings[:, part_index]
            visible = np.linalg.norm(part, axis=1) > 1e-12
            parts.append(
                _normalize(part[visible].mean(axis=0))
                if np.any(visible)
                else np.zeros(512, dtype=np.float32)
            )
        return np.concatenate(parts)
