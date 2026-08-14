"""Strongly typed provider-owned observations used by the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

CourtSide = Literal["left", "right", "unknown"]
ReIdDistance = Literal["cosine"]
COURT_MIDLINE_X = 0.5


@dataclass(frozen=True, slots=True)
class CourtKeypoint:
    """One detected court landmark with optional canonical metre coordinate."""

    index: int
    frame_pos_px: tuple[float, float] | None
    confidence: float | None
    world_pos_m: tuple[float, float] | None


@dataclass(frozen=True, slots=True)
class CourtFrame:
    """Court detector output for one source frame."""

    frame_index: int
    available: bool
    keypoints: tuple[CourtKeypoint, ...]


@dataclass(frozen=True, slots=True)
class PlayerObservation:
    """Normalized player observation owned by the AI subsystem."""

    frame_index: int
    source_track_id: int
    track_id: int
    frame_bbox: tuple[float, float, float, float]
    frame_foot_pos: tuple[float, float]
    court_pos: tuple[float, float] | None
    confidence: float | None = None

    @property
    def court_side(self) -> CourtSide:
        """Return the detected side without changing the projected coordinate."""
        if self.court_pos is None:
            return "unknown"
        return "left" if self.court_pos[0] < COURT_MIDLINE_X else "right"

    def with_identity(self, track_id: int) -> PlayerObservation:
        """Return the observation with an analysis-local canonical identity."""
        return replace(self, track_id=track_id)


@dataclass(frozen=True, slots=True)
class BallObservation:
    """Normalized ball position inferred from the canonical clip."""

    frame_index: int
    frame_pos: tuple[float, float]
    confidence: float | None = None


@dataclass(frozen=True, slots=True)
class ActionObservation:
    """Provider-owned action prediction attached to an analysis-local track."""

    frame_index: int
    track_id: int
    label: str
    confidence: float | None


@dataclass(frozen=True, slots=True)
class FrameObservation:
    """All projected observations at one canonical clip frame."""

    frame_index: int
    players: tuple[PlayerObservation, ...]
    homography_available: bool


@dataclass(frozen=True, slots=True)
class ReIdEmbeddingModel:
    """Versioned appearance extractor metadata carried with one feature snapshot."""

    name: str
    checkpoint_sha256: str
    preprocess_version: str
    dimension: int
    distance: ReIdDistance


@dataclass(frozen=True, slots=True)
class ReIdTrackFeature:
    """One bounded multi-frame appearance sample set for a run-local track."""

    track_id: int
    prototype: tuple[float, ...]
    sample_count: int
    first_frame_index: int
    last_frame_index: int
    mean_quality: float
    cannot_link_track_ids: tuple[int, ...]
    samples: tuple[ReIdTrackSample, ...] = ()


@dataclass(frozen=True, slots=True)
class ReIdTrackSample:
    """One selected quality-temporal crop and its frozen Sports OSNet descriptor."""

    frame_index: int
    quality: float
    osnet_embedding: tuple[float, ...]
    crop_jpeg: bytes | None = None


@dataclass(frozen=True, slots=True)
class ReIdDescriptorSet:
    """Nested Part Adaptation descriptors aggregated over one tracklet."""

    track_id: int
    dino: tuple[float, ...]
    osnet: tuple[float, ...]
    kpr: tuple[float, ...]
    kpr_prompt: tuple[float, ...]
    prompt_coverage: float


@dataclass(frozen=True, slots=True)
class ReIdFeatureSnapshot:
    """Versioned sparse feature snapshot emitted after a canonical clip."""

    schema_version: Literal["1.0.0"]
    embedding_model: ReIdEmbeddingModel
    features: tuple[ReIdTrackFeature, ...]
    descriptor_sets: tuple[ReIdDescriptorSet, ...] = ()
    descriptor_recipe: dict[str, object] | None = None
