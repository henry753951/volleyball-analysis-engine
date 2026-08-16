"""Frozen multi-backbone descriptors for leakage-safe Nested Part Adaptation."""

from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
from numpy.typing import NDArray

from .records import ReIdDescriptorSet, ReIdEmbeddingModel, ReIdFeatureSnapshot
from .reid_features import cached_checkpoint_sha256


@dataclass(frozen=True, slots=True)
class NestedReidPaths:
    """External frozen-model assets; none are embedded in the worker image."""

    dinov2_root: Path
    dinov2_checkpoint: Path
    pose_checkpoint: Path
    kpr_python: Path
    kpr_root: Path
    kpr_checkpoint: Path
    kpr_bridge: Path

    def validate(self, *, require_pose: bool = True) -> None:
        """Fail early when a frozen model or isolated runtime is unavailable."""
        required = {
            "DINOv2 source": self.dinov2_root / "hubconf.py",
            "DINOv2 checkpoint": self.dinov2_checkpoint,
            "KPR Python 3.10": self.kpr_python,
            "Official KPR source": self.kpr_root / "torchreid" / "tools" / "feature_extractor.py",
            "Official KPR checkpoint": self.kpr_checkpoint,
            "KPR bridge": self.kpr_bridge,
        }
        if require_pose:
            required["COCO-17 pose checkpoint"] = self.pose_checkpoint
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
        self._pose: Any = None
        self._kpr_runtime_validated = False

    def prepare(self, *, load_pose: bool = True) -> None:
        """Load the in-process frozen models and validate the isolated KPR runtime."""
        self.paths.validate(require_pose=load_pose)
        import torch

        if self._dino is None:
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
        if load_pose and self._pose is None:
            from ultralytics import YOLO

            self._pose = cast("Any", YOLO(str(self.paths.pose_checkpoint.resolve())))
        if not self._kpr_runtime_validated:
            completed = subprocess.run(
                [str(self.paths.kpr_python), "--version"],
                check=True,
                capture_output=True,
                text=True,
            )
            if "Python 3.10" not in f"{completed.stdout}{completed.stderr}":
                raise RuntimeError(
                    "Official KPR bridge must run in the isolated Python 3.10 runtime"
                )
            self._kpr_runtime_validated = True

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

    def recipe_metadata(
        self,
        *,
        osnet_model: ReIdEmbeddingModel | None = None,
    ) -> dict[str, Any]:
        """Describe the exact frozen components used to create all four modalities."""
        return {
            "name": "nested-part-adaptation",
            "version": "1.0.0",
            "selection_protocol": "past-only-nested-leave-one-clip-out",
            "roster_contract": "fixed-six-per-team",
            "modalities": [
                {
                    "name": "dino",
                    "model": "dinov2_vits14_reg",
                    "dimension": 384,
                    "checkpoint_sha256": cached_checkpoint_sha256(self.paths.dinov2_checkpoint),
                },
                {
                    "name": "osnet",
                    "model": "sports-osnet-x1.0",
                    "dimension": 512,
                    "checkpoint_sha256": osnet_model.checkpoint_sha256
                    if osnet_model is not None
                    else None,
                    "preprocess_version": osnet_model.preprocess_version
                    if osnet_model is not None
                    else None,
                },
                {
                    "name": "kpr",
                    "model": "official-kpr-occ-posetrack",
                    "dimension": 4096,
                    "checkpoint_sha256": cached_checkpoint_sha256(self.paths.kpr_checkpoint),
                },
                {
                    "name": "kpr_prompt",
                    "model": "official-kpr-occ-posetrack+coco17-prompt",
                    "dimension": 4096,
                    "checkpoint_sha256": cached_checkpoint_sha256(self.paths.kpr_checkpoint),
                    "pose_checkpoint_sha256": cached_checkpoint_sha256(self.paths.pose_checkpoint),
                    "pose_failure": "prompt-free-fallback",
                },
            ],
        }

    def enrich(self, snapshot: ReIdFeatureSnapshot) -> ReIdFeatureSnapshot:
        """Attach four tracklet-level descriptors without changing run-local TIDs."""
        self.prepare()
        rows = [
            (feature.track_id, sample.frame_index, sample.crop_jpeg, sample.osnet_embedding)
            for feature in snapshot.features
            for sample in feature.samples
            if sample.crop_jpeg is not None
        ]
        if not rows:
            return replace(snapshot, descriptor_sets=())
        crops = [self._decode_crop(row[2]) for row in rows]
        dino = self._encode_dino(crops)
        prompts = self._pose_prompts(crops)
        kpr, kpr_prompt, prompted = self._encode_kpr(rows, crops, prompts)
        by_track: dict[int, list[int]] = {}
        for index, row in enumerate(rows):
            by_track.setdefault(row[0], []).append(index)
        descriptor_sets: list[ReIdDescriptorSet] = []
        for feature in snapshot.features:
            indices = by_track.get(feature.track_id, [])
            if not indices:
                continue
            descriptor_sets.append(
                ReIdDescriptorSet(
                    track_id=feature.track_id,
                    dino=tuple(float(value) for value in _normalize(dino[indices].mean(axis=0))),
                    osnet=tuple(
                        float(value)
                        for value in _normalize(
                            np.asarray([rows[index][3] for index in indices]).mean(axis=0)
                        )
                    ),
                    kpr=tuple(float(value) for value in self._aggregate_kpr(kpr, indices)),
                    kpr_prompt=tuple(
                        float(value) for value in self._aggregate_kpr(kpr_prompt, indices)
                    ),
                    prompt_coverage=float(np.mean(prompted[indices])),
                )
            )
        return replace(
            snapshot,
            descriptor_sets=tuple(descriptor_sets),
            descriptor_recipe=self.recipe_metadata(osnet_model=snapshot.embedding_model),
        )

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
        self.prepare(load_pose=False)
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

    @staticmethod
    def _decode_crop(encoded: bytes) -> NDArray[np.uint8]:
        crop = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        if crop is None:
            raise ValueError("selected ReID crop cannot be decoded")
        return np.asarray(crop, dtype=np.uint8)

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

    def _pose_prompts(self, crops: list[NDArray[np.uint8]]) -> list[list[list[float]] | None]:
        results = self._pose.predict(
            source=crops,
            device=self.device_name,
            imgsz=640,
            conf=0.15,
            batch=self.batch_size,
            half=self._device.type == "cuda",
            verbose=False,
        )
        return [
            self._choose_pose(result, crop.shape[1], crop.shape[0])
            for crop, result in zip(crops, results, strict=True)
        ]

    @staticmethod
    def _choose_pose(result: Any, width: int, height: int) -> list[list[float]] | None:
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        if not len(boxes):
            return None
        xy = result.keypoints.xy.detach().cpu().numpy()
        confidence = result.keypoints.conf.detach().cpu().numpy()
        center = np.asarray((width / 2, height / 2), dtype=np.float32)
        centers = np.stack(
            ((boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2), axis=1
        )
        areas = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
        scores = areas / max(width * height, 1) - 0.2 * np.linalg.norm(
            (centers - center) / np.asarray((width, height)), axis=1
        )
        selected = int(np.argmax(scores))
        keypoints = np.concatenate((xy[selected], confidence[selected, :, None]), axis=1).astype(
            np.float32
        )
        keypoints[:, 0] = np.clip(keypoints[:, 0], 0, max(width - 1e-3, 0))
        keypoints[:, 1] = np.clip(keypoints[:, 1], 0, max(height - 1e-3, 0))
        return keypoints.tolist() if int(np.count_nonzero(keypoints[:, 2] >= 0.3)) >= 4 else None

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
