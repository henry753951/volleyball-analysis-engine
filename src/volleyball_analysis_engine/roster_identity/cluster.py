"""Group a clip's tracklets by appearance, then name each group from its jersey readings.

A tracker fragments one player into several disjoint spans, so a clip routinely holds far
more tracklets than players - 32 tracklets for 12 people in the development match.  Reading
each fragment independently throws away the fact that they are the same person: one clear
view of the number should name every fragment of that player, and one misread should be
outvoted by the rest.

Appearance answers "are these two the same person", which is a much easier question than
"which of fifteen players is this".  Within one clip the lighting, camera and scale are
fixed, so even a generic person-ReID embedding separates well enough to be useful, and the
co-visibility graph supplies hard negatives for free.

This module is deliberately agnostic about where the embeddings come from: the engine's own
`extensions.fixed_roster_reid` descriptors and a locally extracted OSNet vector are equally
acceptable inputs.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .records import CanonicalTracklet, JerseyReading

CONFIDENCE_WEIGHT = {"high": 1.0, "medium": 0.6, "low": 0.3}


COURT_LENGTH_M = 18.0
COURT_WIDTH_M = 9.0


@dataclass(frozen=True, slots=True)
class ClusterSettings:
    """`similarity_threshold` is the combined link score below which two groups stay apart.

    Appearance alone is a weak linker here.  Measured within a clip, a *different* player who
    is on screen at the same moment scores higher (0.763) than the *same* player seen in a
    different phase of the rally (0.694 for other pairs): the embedding partly encodes when
    the frame was taken, not who is in it.  Where the tracker actually breaks - one player
    split into consecutive spans - re-entry position is the stronger cue, which is why the
    engine's own clip-local ReID is nearest-reentry in court space.

    Measured pairwise AUC on the development match: court distance 0.867, appearance 0.826,
    time gap 0.691, combined 0.874.
    """

    similarity_threshold: float = 0.75
    enable_cannot_link: bool = True
    max_group_size: int = 0  # 0 = unlimited
    appearance_weight: float = 1.0
    reentry_weight: float = 0.0
    # A player cannot cross the court faster than a sprint, so a gap that would require an
    # impossible speed rules the merge out regardless of how alike the two crops look.
    max_speed_m_per_s: float = 9.0
    fps: float = 50.0
    # Distance in metres at which the re-entry cue has decayed to 1/e.
    reentry_decay_m: float = 2.0


def _endpoints(
    tracklet: CanonicalTracklet,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """First and last projected court position, or None when the court was never solved."""
    seen = [o.court_pos for o in tracklet.observations if o.court_pos is not None]
    return (seen[0], seen[-1]) if seen else None


def reentry_score(
    a: CanonicalTracklet, b: CanonicalTracklet, settings: ClusterSettings
) -> tuple[float, bool, bool]:
    """How plausible it is that `b` is `a` re-entering after the tracker dropped it.

    Returns (score in 0..1, kinematically possible, cue available).  The court is not solved
    on every frame, so many pairs have no re-entry evidence at all.  Those must fall back to
    appearance alone: scoring a missing cue as zero would penalise the pair for the court
    detector's gaps, which is how an earlier version silently suppressed most merges.
    """
    first, second = (a, b) if a.last_frame_index <= b.last_frame_index else (b, a)
    ends, starts = _endpoints(first), _endpoints(second)
    if ends is None or starts is None:
        return 0.0, True, False
    exit_pos, entry_pos = ends[1], starts[0]
    metres = float(
        np.hypot(
            (exit_pos[0] - entry_pos[0]) * COURT_LENGTH_M,
            (exit_pos[1] - entry_pos[1]) * COURT_WIDTH_M,
        )
    )
    gap_frames = max(0, second.first_frame_index - first.last_frame_index)
    seconds = max(gap_frames, 1) / settings.fps
    possible = metres <= settings.max_speed_m_per_s * seconds
    return float(np.exp(-metres / settings.reentry_decay_m)), possible, True


@dataclass(frozen=True, slots=True)
class _PairEvidence:
    """Pre-computed pairwise terms, indexed by position in the tracklet list."""

    similarity: np.ndarray
    reentry: np.ndarray
    feasible: np.ndarray
    has_cue: np.ndarray
    forbidden: set[tuple[int, int]]


def _forbidden_pairs(
    tracklets: list[CanonicalTracklet], index: dict[str, int], *, enabled: bool
) -> set[tuple[int, int]]:
    """Position pairs that may never merge because they share a frame."""
    forbidden: set[tuple[int, int]] = set()
    if not enabled:
        return forbidden
    by_id = {t.canonical_track_id: t for t in tracklets}
    for tracklet in tracklets:
        if tracklet.key not in index:
            continue
        for other in tracklet.cannot_link_canonical_track_ids:
            neighbour = by_id.get(other)
            if neighbour is None or neighbour.key not in index:
                continue
            a, b = index[tracklet.key], index[neighbour.key]
            forbidden.add((min(a, b), max(a, b)))
    return forbidden


def _pair_evidence(
    ordered: list[CanonicalTracklet],
    embeddings: dict[str, np.ndarray],
    tracklets: list[CanonicalTracklet],
    index: dict[str, int],
    settings: ClusterSettings,
) -> _PairEvidence:
    vectors = np.stack([embeddings[t.key] for t in ordered]).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-10
    size = len(ordered)
    reentry = np.zeros((size, size), dtype=np.float32)
    feasible = np.ones((size, size), dtype=bool)
    has_cue = np.zeros((size, size), dtype=bool)
    if settings.reentry_weight > 0:
        for i in range(size):
            for j in range(i + 1, size):
                score, possible, available = reentry_score(ordered[i], ordered[j], settings)
                reentry[i, j] = reentry[j, i] = score
                feasible[i, j] = feasible[j, i] = possible
                has_cue[i, j] = has_cue[j, i] = available
    return _PairEvidence(
        similarity=vectors @ vectors.T,
        reentry=reentry,
        feasible=feasible,
        has_cue=has_cue,
        forbidden=_forbidden_pairs(tracklets, index, enabled=settings.enable_cannot_link),
    )


def _link_score(
    left: set[int], right: set[int], evidence: _PairEvidence, settings: ClusterSettings
) -> float | None:
    """Average-linkage score, or None when the merge is ruled out outright."""
    pairs = [(a, b) for a in left for b in right]
    for a, b in pairs:
        if (min(a, b), max(a, b)) in evidence.forbidden or not evidence.feasible[a][b]:
            return None
    # Weights are renormalised per pair, so a pair with no re-entry cue is judged on
    # appearance alone instead of being dragged down by a missing term.
    values = [
        (
            settings.appearance_weight * evidence.similarity[a, b]
            + (settings.reentry_weight if evidence.has_cue[a][b] else 0.0)
            * evidence.reentry[a][b]
        )
        / max(
            1e-9,
            settings.appearance_weight
            + (settings.reentry_weight if evidence.has_cue[a][b] else 0.0),
        )
        for a, b in pairs
    ]
    return float(np.mean(values))


def _best_merge(
    groups: list[set[int]], evidence: _PairEvidence, settings: ClusterSettings
) -> tuple[int, int] | None:
    """Highest-scoring admissible merge, or None when nothing clears the threshold."""
    best_score, best_pair = settings.similarity_threshold, None
    for i, left in enumerate(groups):
        if not left:
            continue
        for j in range(i + 1, len(groups)):
            right = groups[j]
            if not right:
                continue
            if settings.max_group_size and len(left) + len(right) > settings.max_group_size:
                continue
            score = _link_score(left, right, evidence, settings)
            if score is not None and score > best_score:
                best_score, best_pair = score, (i, j)
    return best_pair


def cluster_clip(
    tracklets: list[CanonicalTracklet],
    embeddings: dict[str, np.ndarray],
    settings: ClusterSettings,
) -> dict[str, int]:
    """Group one clip's tracklets by identity.  Returns key -> group id.

    Constrained average-linkage agglomeration: two groups may never merge if any pair across
    them is co-visible, because that pair is provably two different people no matter how
    alike they look.
    """
    ordered = [t for t in tracklets if t.key in embeddings]
    if not ordered:
        return {}
    index = {t.key: position for position, t in enumerate(ordered)}
    evidence = _pair_evidence(ordered, embeddings, tracklets, index, settings)

    groups: list[set[int]] = [{position} for position in range(len(ordered))]
    while True:
        best_pair = _best_merge(groups, evidence, settings)
        if best_pair is None:
            break
        i, j = best_pair
        groups[i] |= groups[j]
        groups[j] = set()

    assignment: dict[str, int] = {}
    for group_id, members in enumerate(group for group in groups if group):
        for position in members:
            assignment[ordered[position].key] = group_id
    return assignment


def vote_group_identity(
    member_keys: list[str],
    readings: dict[str, JerseyReading],
    weights: dict[str, float],
) -> tuple[str | None, float, dict[str, float]]:
    """Weighted vote over a group's jersey readings.

    Returns the winner, its margin over the runner-up, and the full tally.  The margin is
    what a caller should gate on: a unanimous group and a two-to-one split are very
    different levels of evidence even though both have a winner.
    """
    tally: dict[str, float] = defaultdict(float)
    for key in member_keys:
        reading = readings.get(key)
        if reading is None or not reading.roster_player_id:
            continue
        tally[reading.roster_player_id] += weights.get(key, 1.0) * CONFIDENCE_WEIGHT.get(
            reading.confidence, 0.3
        )
    if not tally:
        return None, 0.0, {}
    ranked = sorted(tally.items(), key=lambda item: -item[1])
    winner, top = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
    total = sum(tally.values())
    margin = (top - runner_up) / total if total else 0.0
    return winner, margin, dict(tally)


def reading_weight(tracklet: CanonicalTracklet) -> float:
    """How much a tracklet's own reading should count in its group's vote.

    A longer tracklet offers more frames to pick a legible view from.  Saturates quickly:
    past a couple of seconds extra length adds little.
    """
    return float(min(1.0, tracklet.sample_count / 150.0) ** 0.5)
