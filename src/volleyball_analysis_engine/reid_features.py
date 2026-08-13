"""Sparse run-local ReID feature aggregation and result serialization."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from math import isfinite
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .records import (
    CourtSide,
    FrameObservation,
    ReIdEmbeddingModel,
    ReIdFeatureSnapshot,
    ReIdTrackFeature,
)

REID_FEATURE_SCHEMA_VERSION = "1.0.0"
SPORTS_OSNET_NAME = "sports-osnet"
SPORTS_OSNET_DIMENSION = 512
SPORTS_OSNET_PREPROCESS_VERSION = "roi-align-rgb-imagenet-v1"
REID_MIN_OBSERVATIONS = 12
REID_MIN_BBOX_HEIGHT_PX = 28.0
REID_PROTOTYPE_SAMPLES = 4
_MIN_VECTOR_NORM = 1e-12
_COURT_SIDES: tuple[CourtSide, ...] = ("left", "right", "unknown")


@dataclass(slots=True)
class _RankedFeature:
    embedding: NDArray[np.float64]
    quality: float
    frame_index: int


@dataclass(slots=True)
class _RunningFeature:
    candidates: list[_RankedFeature]
    sample_count: int
    first_frame_index: int
    last_frame_index: int
    quality_sum: float


class ReIdFeatureAccumulator:
    """Collect eligible observations and build a calibrated clip-local prototype."""

    def __init__(
        self,
        *,
        dimension: int = SPORTS_OSNET_DIMENSION,
        min_observations: int = REID_MIN_OBSERVATIONS,
        prototype_samples: int = REID_PROTOTYPE_SAMPLES,
    ) -> None:
        """Configure the fixed embedding dimension for this tracker run."""
        self.dimension = dimension
        self.min_observations = max(1, min_observations)
        self.prototype_samples = max(1, prototype_samples)
        self._features: dict[int, _RunningFeature] = {}
        self._cannot_links: dict[int, set[int]] = defaultdict(set)

    def observe(
        self,
        *,
        track_id: int,
        frame_index: int,
        embedding: NDArray[np.float32],
        quality: float,
        selection_quality: float | None = None,
    ) -> None:
        """Retain one eligible sample for later quality-temporal selection."""
        vector = np.asarray(embedding, dtype=np.float64)
        if vector.shape != (self.dimension,):
            message = (
                f"ReID embedding dimension mismatch: expected {self.dimension}, got {vector.shape}"
            )
            raise ValueError(message)
        if not bool(np.all(np.isfinite(vector))):
            return
        norm = float(np.linalg.norm(vector))
        if norm <= _MIN_VECTOR_NORM:
            return
        normalized = vector / norm
        bounded_quality = min(1.0, max(0.0, quality)) if isfinite(quality) else 0.0
        ranking_quality = (
            max(0.0, selection_quality)
            if selection_quality is not None and isfinite(selection_quality)
            else bounded_quality
        )
        feature = self._features.get(track_id)
        if feature is None:
            self._features[track_id] = _RunningFeature(
                candidates=[_RankedFeature(normalized.copy(), ranking_quality, frame_index)],
                sample_count=1,
                first_frame_index=frame_index,
                last_frame_index=frame_index,
                quality_sum=bounded_quality,
            )
            return
        feature.sample_count += 1
        feature.last_frame_index = frame_index
        feature.quality_sum += bounded_quality
        feature.candidates = _bounded_temporal_candidates(
            [
                *feature.candidates,
                _RankedFeature(normalized.copy(), ranking_quality, frame_index),
            ],
            first_frame_index=feature.first_frame_index,
            last_frame_index=feature.last_frame_index,
            bins=self.prototype_samples,
        )

    def observe_co_visibility(self, track_ids: Iterable[int]) -> None:
        """Record symmetric cannot-links for real detections visible in one frame."""
        unique = sorted(set(track_ids))
        for track_id in unique:
            self._cannot_links[track_id].update(
                candidate for candidate in unique if candidate != track_id
            )

    def snapshot(self) -> tuple[ReIdTrackFeature, ...]:
        """Return prototypes built from four quality-temporal observations."""
        output: list[ReIdTrackFeature] = []
        eligible_track_ids = {
            track_id
            for track_id, feature in self._features.items()
            if feature.sample_count >= self.min_observations
        }
        for track_id, feature in sorted(self._features.items()):
            if track_id not in eligible_track_ids:
                continue
            selected = _bounded_temporal_candidates(
                feature.candidates,
                first_frame_index=feature.first_frame_index,
                last_frame_index=feature.last_frame_index,
                bins=self.prototype_samples,
                per_bin=1,
            )
            embedding_sum = np.sum(
                np.stack([sample.embedding for sample in selected]),
                axis=0,
            )
            norm = float(np.linalg.norm(embedding_sum))
            if norm <= _MIN_VECTOR_NORM:
                continue
            prototype = embedding_sum / norm
            output.append(
                ReIdTrackFeature(
                    track_id=track_id,
                    prototype=tuple(float(value) for value in prototype),
                    sample_count=feature.sample_count,
                    first_frame_index=feature.first_frame_index,
                    last_frame_index=feature.last_frame_index,
                    mean_quality=feature.quality_sum / feature.sample_count,
                    cannot_link_track_ids=tuple(
                        sorted(self._cannot_links[track_id] & eligible_track_ids)
                    ),
                )
            )
        return tuple(output)


def _bounded_temporal_candidates(
    samples: list[_RankedFeature],
    *,
    first_frame_index: int,
    last_frame_index: int,
    bins: int,
    per_bin: int = 2,
) -> list[_RankedFeature]:
    """Keep a bounded set of high-quality samples distributed over time."""
    span = max(1, last_frame_index - first_frame_index + 1)
    grouped: list[list[_RankedFeature]] = [[] for _ in range(bins)]
    for sample in samples:
        bin_index = min(
            bins - 1,
            (sample.frame_index - first_frame_index) * bins // span,
        )
        grouped[bin_index].append(sample)
    selected = [
        sample
        for group in grouped
        for sample in sorted(
            group,
            key=lambda item: (item.quality, item.frame_index),
            reverse=True,
        )[:per_bin]
    ]
    return sorted(selected, key=lambda item: item.frame_index)


def sports_osnet_embedding_model(checkpoint: Path) -> ReIdEmbeddingModel:
    """Describe the exact Sports OSNet checkpoint and preprocessing pipeline."""
    return ReIdEmbeddingModel(
        name=SPORTS_OSNET_NAME,
        checkpoint_sha256=cached_checkpoint_sha256(checkpoint),
        preprocess_version=SPORTS_OSNET_PREPROCESS_VERSION,
        dimension=SPORTS_OSNET_DIMENSION,
        distance="cosine",
    )


def cached_checkpoint_sha256(checkpoint: Path) -> str:
    """Hash a model file once per resolved path, size, and modification time."""
    resolved = checkpoint.resolve(strict=True)
    stat = resolved.stat()
    return _checkpoint_sha256(str(resolved), stat.st_size, stat.st_mtime_ns)


@lru_cache(maxsize=8)
def _checkpoint_sha256(path: str, size: int, modified_ns: int) -> str:
    del size, modified_ns
    digest = sha256()
    with Path(path).open("rb") as checkpoint_file:
        for block in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_reid_feature_bank(
    snapshot: ReIdFeatureSnapshot,
    frames: list[FrameObservation],
    *,
    map_frame: Callable[[int], int],
) -> dict[str, Any]:
    """Build the strict v1 clip feature-bank extension using projected court sides."""
    if snapshot.schema_version != REID_FEATURE_SCHEMA_VERSION:
        message = f"unsupported ReID feature snapshot: {snapshot.schema_version}"
        raise ValueError(message)
    model = snapshot.embedding_model
    side_by_track = resolve_track_court_sides(frames)
    features_by_side: dict[CourtSide, list[dict[str, Any]]] = {side: [] for side in _COURT_SIDES}
    for feature in snapshot.features:
        if len(feature.prototype) != model.dimension:
            message = (
                f"ReID prototype dimension mismatch for track {feature.track_id}: "
                f"expected {model.dimension}, got {len(feature.prototype)}"
            )
            raise ValueError(message)
        side = side_by_track.get(feature.track_id, "unknown")
        features_by_side[side].append(
            {
                "provisional_gid": f"clip:{side}:{feature.track_id}",
                "track_id": feature.track_id,
                "first_frame_index": str(map_frame(feature.first_frame_index)),
                "last_frame_index": str(map_frame(feature.last_frame_index)),
                "sample_count": feature.sample_count,
                "mean_quality": feature.mean_quality,
                "prototype": list(feature.prototype),
                "cannot_link_track_ids": list(feature.cannot_link_track_ids),
            }
        )
    return {
        "schema_version": REID_FEATURE_SCHEMA_VERSION,
        "scope": "clip",
        "embedding_model": {
            "name": model.name,
            "checkpoint_sha256": model.checkpoint_sha256,
            "preprocess_version": model.preprocess_version,
            "dimension": model.dimension,
            "distance": model.distance,
        },
        "side_feature_banks": [
            {"court_side": side, "features": features_by_side[side]} for side in _COURT_SIDES
        ],
    }


def resolve_track_court_sides(frames: list[FrameObservation]) -> dict[int, CourtSide]:
    """Resolve the dominant projected side using volley-reid tracklet semantics."""
    counts: dict[int, dict[CourtSide, int]] = defaultdict(lambda: defaultdict(int))
    for frame in frames:
        for player in frame.players:
            counts[player.track_id][player.court_side] += 1

    resolved: dict[int, CourtSide] = {}
    for track_id, side_counts in counts.items():
        left_count = side_counts["left"]
        right_count = side_counts["right"]
        if left_count == right_count:
            resolved[track_id] = "unknown"
        else:
            resolved[track_id] = "left" if left_count > right_count else "right"
    return resolved
