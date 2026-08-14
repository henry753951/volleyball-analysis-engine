"""Typed records for the downstream roster identity service.

This package consumes finished `AnalysisResult` artifacts and assigns a match-scoped
roster identity (team + jersey number) to each clip-local canonical track.  It never
rewrites clip-local track IDs; the mapping is emitted as a separate artifact so a clip
can be re-analysed without invalidating identities that were already reviewed.

Identity evidence is the jersey number read by a candidate-constrained VLM, not an
appearance embedding.  Same-team players wear identical kit, so appearance features
cannot separate them; the printed number is the only per-player signal in the frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

CourtSide = Literal["left", "right", "unknown"]
IdentityStatus = Literal["auto_assigned", "human_review", "unknown"]
COURT_MIDLINE_X = 0.5


@dataclass(frozen=True, slots=True)
class RosterCandidate:
    """One player the VLM is allowed to choose from."""

    roster_player_id: str
    jersey_number: int
    team_id: str
    jersey_description: str


@dataclass(frozen=True, slots=True)
class TrackletObservation:
    """One frame of a canonical tracklet, in canonical clip coordinates."""

    frame_index: int
    frame_bbox: tuple[float, float, float, float]
    court_pos: tuple[float, float] | None
    confidence: float | None


@dataclass(slots=True)
class CanonicalTracklet:
    """A clip-local canonical track, the unit this service assigns an identity to.

    Mirrors one entry of `AnalysisResult.extensions.fixed_roster_reid.tracklets` when the
    upstream bank is present, and is rebuilt from `tracks.jsonl` when it is not.
    """

    clip_id: str
    canonical_track_id: int
    track_ids: tuple[int, ...]
    court_side: CourtSide
    observations: tuple[TrackletObservation, ...]
    cannot_link_canonical_track_ids: tuple[int, ...] = ()
    median_court_pos: tuple[float, float] | None = None
    mean_confidence: float = 0.0
    mean_height_px: float = 0.0

    @property
    def key(self) -> str:
        """Stable identifier of this tracklet within the match."""
        return f"{self.clip_id}#{self.canonical_track_id}"

    @property
    def first_frame_index(self) -> int:
        """Canonical clip frame the tracklet starts on."""
        return self.observations[0].frame_index

    @property
    def last_frame_index(self) -> int:
        """Canonical clip frame the tracklet ends on."""
        return self.observations[-1].frame_index

    @property
    def sample_count(self) -> int:
        """Number of observed frames, used to weight how much this reading counts."""
        return len(self.observations)


@dataclass(frozen=True, slots=True)
class SelectedFrame:
    """A frame chosen because the jersey number is likely to be legible in it."""

    frame_index: int
    bbox_px: tuple[int, int, int, int]
    quality: float
    frontality: float
    torso_px: float
    sharpness: float
    completeness: float
    pose_ok: bool
    full_path: str = ""
    torso_path: str = ""


@dataclass(frozen=True, slots=True)
class JerseyReading:
    """Raw candidate-constrained VLM output for one tracklet."""

    roster_player_id: str | None
    jersey_number: str | None
    decision: Literal["candidate", "unknown"]
    confidence: Literal["high", "medium", "low"]
    evidence: tuple[str, ...] = ()
    # Ranked best-first, top choice included.  The assignment stage needs somewhere to fall
    # back to when two co-visible tracklets claim the same player.
    ranking: tuple[str, ...] = ()
    raw_response: str = ""
    latency_s: float = 0.0


@dataclass(slots=True)
class IdentityAssignment:
    """Final identity for one canonical tracklet, with the evidence that produced it."""

    clip_id: str
    canonical_track_id: int
    track_ids: tuple[int, ...]
    roster_player_id: str | None
    status: IdentityStatus
    method: str
    rule: str = ""
    court_side: CourtSide = "unknown"
    selected_frame_indices: tuple[int, ...] = ()
    jersey: JerseyReading | None = None
    quality_flags: list[str] = field(default_factory=list[str])
