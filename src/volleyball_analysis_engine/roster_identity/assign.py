"""Resolve per-tracklet jersey readings into a set of identities consistent within a clip.

Reading each tracklet independently ignores the one constraint that is free and never
wrong: **a player cannot be in two places at once**.  Two tracklets visible in the same
frame must be different people, so if both claim the same shirt at least one is wrong.

Measured on the development match, that constraint separates errors sharply: tracklets in a
clash were wrong 34% of the time, tracklets with no clash only 7%.

The constraint is strictly **pairwise**, and getting that wrong is expensive.  A clip
routinely contains far more tracklets than players - the tracker fragments one player into
several disjoint spans - so in the development match 32 tracklets belonged to just 12
people.  Treating the clip as a one-to-one assignment (a Hungarian solve over tracklets and
squad numbers) therefore rejects the majority of perfectly legal repeats: it dropped
coverage from 76% to 46% and accuracy from 61% to 38%.  What is actually needed is a list
colouring of the co-visibility graph, where one colour may repeat on non-adjacent nodes.

Greedy in order of evidence strength is used: the most reliable claims take their first
choice, weaker ones fall back to the model's own alternatives, and anything left with no
consistent option abstains.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .records import CanonicalTracklet, IdentityAssignment, JerseyReading

# Score for the model's first, second and third choice.  Deliberately coarse: a VLM's own
# confidence wording is not calibrated, and inventing precise numbers would imply an
# accuracy the model does not have.  Only the ordering is trusted.
RANK_SCORES = (1.0, 0.55, 0.30)
CONFIDENCE_ORDER = {"high": 3, "medium": 2, "low": 1}


@dataclass(frozen=True, slots=True)
class AssignmentSettings:
    """`min_score` is the rank score below which a tracklet goes to a human instead."""

    min_score: float = 0.30
    enable_cannot_link: bool = True
    require_high_confidence: bool = False


def _strength(
    tracklet: CanonicalTracklet, reading: JerseyReading | None
) -> tuple[int, int, int]:
    """Sort key: the claims most likely to be right get to pick first."""
    if reading is None:
        return (0, 0, 0)
    return (
        CONFIDENCE_ORDER.get(reading.confidence, 0),
        1 if reading.ranking else 0,
        tracklet.sample_count,
    )


def _first_available(
    reading: JerseyReading,
    *,
    allowed: set[str],
    blocked: set[str],
    settings: AssignmentSettings,
) -> tuple[str | None, float]:
    """Best choice the model offered that is still free, or (None, 0) if there is none."""
    for position, player in enumerate(reading.ranking[: len(RANK_SCORES)]):
        if player not in allowed:
            continue
        score = RANK_SCORES[position]
        if score < settings.min_score:
            break
        if settings.enable_cannot_link and player in blocked:
            continue
        return player, score
    return None, 0.0


def assign_clip(
    tracklets: list[CanonicalTracklet],
    readings: dict[str, JerseyReading],
    candidates_for: dict[str, list[str]],
    settings: AssignmentSettings,
) -> dict[str, tuple[str | None, float, str]]:
    """Assign every tracklet of one clip.  Returns key -> (player, score, rule)."""
    resolved: dict[str, tuple[str | None, float, str]] = {}
    taken: dict[int, set[str]] = defaultdict(set)  # canonical_track_id -> claimed players
    def strength_of(tracklet: CanonicalTracklet) -> tuple[int, int, int]:
        return _strength(tracklet, readings.get(tracklet.key))

    order = sorted(tracklets, key=strength_of, reverse=True)
    for tracklet in order:
        reading = readings.get(tracklet.key)
        if reading is None or not reading.ranking:
            resolved[tracklet.key] = (None, 0.0, "no_reading")
            continue
        if settings.require_high_confidence and reading.confidence != "high":
            resolved[tracklet.key] = (None, 0.0, "low_confidence")
            continue
        chosen, score = _first_available(
            reading,
            allowed=set(candidates_for.get(tracklet.key) or reading.ranking),
            # Players already claimed by a tracklet this one shares a frame with.
            blocked=taken[tracklet.canonical_track_id],
            settings=settings,
        )
        if chosen is None:
            resolved[tracklet.key] = (None, 0.0, "no_consistent_identity")
            continue
        rule = "top_choice" if chosen == reading.ranking[0] else "resolved_clash"
        resolved[tracklet.key] = (chosen, score, rule)
        for neighbour in tracklet.cannot_link_canonical_track_ids:
            taken[neighbour].add(chosen)
    return resolved


def apply_assignments(
    assignments: list[IdentityAssignment],
    resolved: dict[str, tuple[str | None, float, str]],
) -> list[IdentityAssignment]:
    """Rewrite identities in place from the constrained solution."""
    by_key: dict[str, list[IdentityAssignment]] = defaultdict(list)
    for assignment in assignments:
        by_key[f"{assignment.clip_id}#{assignment.canonical_track_id}"].append(assignment)
    for key, (player, _score, rule) in resolved.items():
        for assignment in by_key.get(key, []):
            assignment.roster_player_id = player
            assignment.status = "auto_assigned" if player else "human_review"
            assignment.rule = rule
            assignment.method = "jersey_vlm+cannot_link"
    return assignments
