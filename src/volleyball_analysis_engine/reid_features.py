"""Sparse run-local ReID feature aggregation and result serialization."""

from __future__ import annotations

from base64 import b64encode
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from itertools import combinations
from math import isfinite
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from .records import (
    CourtSide,
    FrameObservation,
    ReIdDescriptorSet,
    ReIdEmbeddingModel,
    ReIdFeatureSnapshot,
    ReIdTrackFeature,
    ReIdTrackSample,
)

REID_FEATURE_SCHEMA_VERSION = "2.0.0"
REID_SNAPSHOT_SCHEMA_VERSION = "1.0.0"
SPORTS_OSNET_NAME = "sports-osnet"
SPORTS_OSNET_DIMENSION = 512
SPORTS_OSNET_PREPROCESS_VERSION = "roi-align-rgb-imagenet-v1"
REID_MIN_OBSERVATIONS = 12
REID_MIN_BBOX_HEIGHT_PX = 28.0
REID_PROTOTYPE_SAMPLES = 4
_MIN_VECTOR_NORM = 1e-12
_DUPLICATE_MIN_COVERAGE = 0.70
_DUPLICATE_MIN_IOU = 0.40
_DUPLICATE_MIN_APPEARANCE = 0.84
_DUPLICATE_MIN_SHARED_FRAMES = 3
_MAX_ON_COURT_PLAYERS_PER_SIDE = 6
_EXPECTED_DESCRIPTOR_DIMENSIONS = {
    "dino": 384,
    "osnet": 512,
    "kpr": 4096,
    "kpr_prompt": 4096,
}


@dataclass(slots=True)
class _RankedFeature:
    embedding: NDArray[np.float64]
    quality: float
    frame_index: int
    crop_jpeg: bytes | None = None


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
        crop_bgr: NDArray[np.uint8] | None = None,
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
        candidate = _RankedFeature(normalized.copy(), ranking_quality, frame_index)
        feature = self._features.get(track_id)
        if feature is None:
            candidate.crop_jpeg = _encode_crop(crop_bgr)
            self._features[track_id] = _RunningFeature(
                candidates=[candidate],
                sample_count=1,
                first_frame_index=frame_index,
                last_frame_index=frame_index,
                quality_sum=bounded_quality,
            )
            return
        feature.sample_count += 1
        feature.last_frame_index = frame_index
        feature.quality_sum += bounded_quality
        selected = _bounded_temporal_candidates(
            [
                *feature.candidates,
                candidate,
            ],
            first_frame_index=feature.first_frame_index,
            last_frame_index=feature.last_frame_index,
            bins=self.prototype_samples,
        )
        if any(item is candidate for item in selected):
            candidate.crop_jpeg = _encode_crop(crop_bgr)
        feature.candidates = selected

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
                    samples=tuple(
                        ReIdTrackSample(
                            frame_index=sample.frame_index,
                            quality=min(1.0, max(0.0, sample.quality)),
                            osnet_embedding=tuple(float(value) for value in sample.embedding),
                            crop_jpeg=sample.crop_jpeg,
                        )
                        for sample in selected
                    ),
                )
            )
        return tuple(output)


def _encode_crop(crop_bgr: NDArray[np.uint8] | None) -> bytes | None:
    """JPEG-encode a bounded selected crop without retaining source frames."""
    if crop_bgr is None or crop_bgr.size == 0:
        return None
    ok, encoded = cv2.imencode(".jpg", crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, 94])
    return bytes(encoded) if ok else None


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
    descriptor_recipe: dict[str, Any],
) -> dict[str, Any]:
    """Build fixed-roster v2 tracklets with all Nested Part Adaptation modalities."""
    if snapshot.schema_version != REID_SNAPSHOT_SCHEMA_VERSION:
        message = f"unsupported ReID feature snapshot: {snapshot.schema_version}"
        raise ValueError(message)
    selected_by_frame = _select_on_court_roster(frames)
    side_by_track = resolve_track_court_sides(frames, selected_by_frame=selected_by_frame)
    features = {feature.track_id: feature for feature in snapshot.features}
    descriptors = {descriptor.track_id: descriptor for descriptor in snapshot.descriptor_sets}
    if set(features) != set(descriptors):
        missing = sorted(set(features) ^ set(descriptors))
        message = f"Nested Part descriptors must cover every eligible ReID track: {missing}"
        raise ValueError(message)
    for descriptor in descriptors.values():
        for name, dimension in _EXPECTED_DESCRIPTOR_DIMENSIONS.items():
            if len(getattr(descriptor, name)) != dimension:
                message = (
                    f"{name} descriptor for track {descriptor.track_id} "
                    f"must have {dimension} values"
                )
                raise ValueError(message)
    all_track_ids = {
        player.track_id
        for frame in frames
        for player in frame.players
        if player.track_id in selected_by_frame[frame.frame_index]
    }
    eligible_features = {
        track_id: feature for track_id, feature in features.items() if track_id in all_track_ids
    }
    canonical = _canonical_track_ids(
        frames,
        eligible_features,
        side_by_track,
        selected_by_frame=selected_by_frame,
    )
    canonical.update(
        (track_id, track_id) for track_id in all_track_ids if track_id not in canonical
    )
    _validate_fixed_roster_capacity(
        frames,
        canonical,
        side_by_track,
        all_track_ids,
        selected_by_frame=selected_by_frame,
    )
    groups: dict[int, list[int]] = defaultdict(list)
    for track_id in all_track_ids:
        groups[canonical.get(track_id, track_id)].append(track_id)
    co_visible: dict[int, set[int]] = defaultdict(set)
    for frame in frames:
        visible = {
            canonical[player.track_id]
            for player in frame.players
            if player.track_id in selected_by_frame[frame.frame_index]
        }
        for track_id in visible:
            co_visible[track_id].update(visible - {track_id})
    tracklets: list[dict[str, Any]] = []
    for canonical_track_id, track_ids in sorted(groups.items()):
        group_features = [features[track_id] for track_id in track_ids if track_id in features]
        group_descriptors = [
            descriptors[track_id] for track_id in track_ids if track_id in descriptors
        ]
        observations = [
            player
            for frame in frames
            for player in frame.players
            if player.track_id in track_ids
            and player.track_id in selected_by_frame[frame.frame_index]
        ]
        side = side_by_track.get(canonical_track_id, "unknown")
        canonical_ids = set(groups)
        # Cannot-link is defined after active-six court filtering.  Raw detector
        # co-visibility may contain bench players or false positives and can
        # otherwise create an impossible seven-node clique for six roster slots.
        cannot_links = co_visible[canonical_track_id] & canonical_ids
        sample_count = (
            sum(feature.sample_count for feature in group_features)
            if group_features
            else len(observations)
        )
        court_positions = [
            player.court_pos
            for frame in frames
            for player in frame.players
            if player.track_id in track_ids
            and player.track_id in selected_by_frame[frame.frame_index]
            and player.court_pos is not None
        ]
        median_court_pos = (
            [
                float(np.median([position[0] for position in court_positions])),
                float(np.median([position[1] for position in court_positions])),
            ]
            if court_positions
            else None
        )
        tracklets.append(
            {
                "canonical_track_id": canonical_track_id,
                "track_ids": sorted(track_ids),
                "court_side": side,
                "median_court_pos": median_court_pos,
                "first_frame_index": str(
                    map_frame(min(player.frame_index for player in observations))
                ),
                "last_frame_index": str(
                    map_frame(max(player.frame_index for player in observations))
                ),
                "sample_count": sample_count,
                "mean_quality": (
                    sum(feature.mean_quality * feature.sample_count for feature in group_features)
                    / sample_count
                    if group_features
                    else sum(player.confidence or 0.0 for player in observations) / sample_count
                ),
                "prompt_coverage": (
                    sum(
                        descriptor.prompt_coverage * features[descriptor.track_id].sample_count
                        for descriptor in group_descriptors
                    )
                    / sample_count
                    if group_descriptors
                    else 0.0
                ),
                "descriptors": _encoded_descriptor_set(group_descriptors),
                "cannot_link_canonical_track_ids": sorted(cannot_links),
            }
        )
    return {
        "schema_version": REID_FEATURE_SCHEMA_VERSION,
        "scope": "clip",
        "identity_contract": "fixed-six-per-team",
        "slots_per_team": 6,
        "descriptor_recipe": descriptor_recipe,
        "tracklets": tracklets,
    }


def _validate_fixed_roster_capacity(
    frames: list[FrameObservation],
    canonical: dict[int, int],
    sides: dict[int, CourtSide],
    eligible_track_ids: set[int],
    *,
    selected_by_frame: dict[int, set[int]],
) -> None:
    """Reject impossible fixed-roster frames instead of inventing a seventh slot."""
    for frame in frames:
        by_side: dict[CourtSide, set[int]] = defaultdict(set)
        for player in frame.players:
            if (
                player.track_id not in eligible_track_ids
                or player.track_id not in selected_by_frame[frame.frame_index]
            ):
                continue
            side = sides.get(player.track_id, "unknown")
            if side == "unknown":
                continue
            by_side[side].add(canonical.get(player.track_id, player.track_id))
        for side in ("left", "right"):
            if len(by_side[side]) > _MAX_ON_COURT_PLAYERS_PER_SIDE:
                message = (
                    f"fixed roster violation at frame {frame.frame_index}: "
                    f"{len(by_side[side])} co-visible {side} tracklets"
                )
                raise ValueError(message)


def _encoded_descriptor_set(
    descriptors: list[ReIdDescriptorSet],
) -> dict[str, str] | None:
    if not descriptors:
        return None
    encoded: dict[str, str] = {}
    for name in ("dino", "osnet", "kpr", "kpr_prompt"):
        value = _descriptor_bytes([getattr(descriptor, name) for descriptor in descriptors])
        if value is None:
            return None
        encoded[name] = value
    return encoded


def _descriptor_bytes(values: list[tuple[float, ...]]) -> str | None:
    matrix = np.asarray(values, dtype=np.float32)
    matrix = matrix[np.linalg.norm(matrix, axis=1) > _MIN_VECTOR_NORM]
    if matrix.shape[0] == 0:
        return None
    prototype = matrix.mean(axis=0)
    norm = float(np.linalg.norm(prototype))
    if norm <= _MIN_VECTOR_NORM:
        return None
    return b64encode(np.asarray(prototype / norm, dtype="<f4").tobytes()).decode("ascii")


def _canonical_track_ids(
    frames: list[FrameObservation],
    features: dict[int, ReIdTrackFeature],
    sides: dict[int, CourtSide],
    *,
    selected_by_frame: dict[int, set[int]],
) -> dict[int, int]:
    """Port volley-reid's strict duplicate-alias rule before fixed-slot assignment."""
    observations: dict[int, dict[int, Any]] = defaultdict(dict)
    side_frame_counts: dict[tuple[CourtSide, int], int] = defaultdict(int)
    for frame in frames:
        for player in frame.players:
            if (
                player.track_id not in features
                or player.track_id not in selected_by_frame[frame.frame_index]
            ):
                continue
            observations[player.track_id][frame.frame_index] = player
            side_frame_counts[(sides.get(player.track_id, "unknown"), frame.frame_index)] += 1
    track_ids = sorted(features)
    parent = {track_id: track_id for track_id in track_ids}

    def find(track_id: int) -> int:
        while parent[track_id] != track_id:
            parent[track_id] = parent[parent[track_id]]
            track_id = parent[track_id]
        return track_id

    for first, second in combinations(track_ids, 2):
        if sides.get(first, "unknown") != sides.get(second, "unknown"):
            continue
        shared_frames = sorted(set(observations[first]) & set(observations[second]))
        if not shared_frames:
            continue
        coverage = len(shared_frames) / min(len(observations[first]), len(observations[second]))
        if coverage < _DUPLICATE_MIN_COVERAGE:
            continue
        median_iou = float(
            np.median(
                [
                    _bbox_iou(
                        observations[first][frame_index].frame_bbox,
                        observations[second][frame_index].frame_bbox,
                    )
                    for frame_index in shared_frames
                ]
            )
        )
        appearance = float(np.dot(features[first].prototype, features[second].prototype))
        over_capacity = any(
            side_frame_counts[(sides.get(first, "unknown"), frame_index)]
            > _MAX_ON_COURT_PLAYERS_PER_SIDE
            for frame_index in shared_frames
        )
        if (
            median_iou >= _DUPLICATE_MIN_IOU
            and appearance >= _DUPLICATE_MIN_APPEARANCE
            and (len(shared_frames) >= _DUPLICATE_MIN_SHARED_FRAMES or over_capacity)
        ):
            first_root, second_root = find(first), find(second)
            parent[max(first_root, second_root)] = min(first_root, second_root)
    return {track_id: find(track_id) for track_id in track_ids}


def _bbox_iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def _select_on_court_roster(frames: list[FrameObservation]) -> dict[int, set[int]]:
    """Mirror volley-reid's per-frame six-most-central roster restriction."""
    dominant_sides = _raw_track_court_sides(frames)
    selected_by_frame: dict[int, set[int]] = {}
    for frame in frames:
        selected: set[int] = set()
        by_side: dict[CourtSide, list[Any]] = defaultdict(list)
        for player in frame.players:
            # Roster capacity is a track-level constraint.  A single noisy court
            # projection must not put the same stable-side roster into a seventh
            # physical slot for one frame.
            side = dominant_sides.get(player.track_id, "unknown")
            if side == "unknown":
                side = player.court_side
            by_side[side].append(player)
        for side, players in by_side.items():
            if side == "unknown" or len(players) <= _MAX_ON_COURT_PLAYERS_PER_SIDE:
                kept = players
            else:
                kept = sorted(
                    players,
                    key=lambda player: (
                        0.0
                        if player.court_pos is None
                        else max(
                            0.0,
                            -player.court_pos[0],
                            player.court_pos[0] - 1.0,
                            -player.court_pos[1],
                            player.court_pos[1] - 1.0,
                        ),
                        -(player.confidence if player.confidence is not None else -1.0),
                        player.track_id,
                    ),
                )[:_MAX_ON_COURT_PLAYERS_PER_SIDE]
            selected.update(player.track_id for player in kept)
        selected_by_frame[frame.frame_index] = selected
    return selected_by_frame


def _raw_track_court_sides(frames: list[FrameObservation]) -> dict[int, CourtSide]:
    """Resolve stable side evidence before per-frame active-six filtering."""
    counts: dict[int, dict[CourtSide, int]] = defaultdict(lambda: defaultdict(int))
    for frame in frames:
        for player in frame.players:
            counts[player.track_id][player.court_side] += 1
    resolved: dict[int, CourtSide] = {}
    for track_id, side_counts in counts.items():
        left_count = side_counts["left"]
        right_count = side_counts["right"]
        resolved[track_id] = (
            "unknown"
            if left_count == right_count
            else "left"
            if left_count > right_count
            else "right"
        )
    return resolved


def resolve_track_court_sides(
    frames: list[FrameObservation],
    *,
    selected_by_frame: dict[int, set[int]] | None = None,
) -> dict[int, CourtSide]:
    """Resolve the dominant projected side using volley-reid tracklet semantics."""
    include_excluded_tracks = selected_by_frame is None
    if selected_by_frame is None:
        selected_by_frame = _select_on_court_roster(frames)
    counts: dict[int, dict[CourtSide, int]] = defaultdict(lambda: defaultdict(int))
    for frame in frames:
        for player in frame.players:
            if player.track_id not in selected_by_frame[frame.frame_index]:
                continue
            counts[player.track_id][player.court_side] += 1

    resolved: dict[int, CourtSide] = {}
    for track_id, side_counts in counts.items():
        left_count = side_counts["left"]
        right_count = side_counts["right"]
        if left_count == right_count:
            resolved[track_id] = "unknown"
        else:
            resolved[track_id] = "left" if left_count > right_count else "right"
    if include_excluded_tracks:
        for track_id, side in _raw_track_court_sides(frames).items():
            resolved.setdefault(track_id, side)
    return resolved
