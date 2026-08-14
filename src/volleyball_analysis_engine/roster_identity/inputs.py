"""Load canonical tracklets from a finished offline analysis directory.

Two sources are supported, in priority order:

1. `AnalysisResult.extensions.fixed_roster_reid` — the interchange bank the engine emits
   for exactly this purpose.  It already merges clip-local tracks into canonical IDs and
   carries court sides and cannot-link constraints.
2. `tracks.jsonl` alone — for artifacts produced before the bank existed.  The equivalent
   structure is rebuilt here: one canonical track per `track_id`, court side from the
   projected court position, and cannot-links from temporal co-visibility.

Rebuilding is lossless for what this service needs.  The bank's appearance descriptors are
deliberately ignored: identity here comes from the jersey number, not from an embedding.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any, cast

from .records import COURT_MIDLINE_X, CanonicalTracklet, CourtSide, TrackletObservation


def _bbox(values: list[float]) -> tuple[float, float, float, float]:
    """Return the normalized xyxy box as a fixed-length tuple."""
    x1, y1, x2, y2 = (float(value) for value in values)
    return (x1, y1, x2, y2)


def _court_side(court_positions: list[tuple[float, float]]) -> CourtSide:
    """Side of the net the track spent most of its life on."""
    if not court_positions:
        return "unknown"
    left = sum(1 for position in court_positions if position[0] < COURT_MIDLINE_X)
    if left * 2 == len(court_positions):
        return "unknown"
    return "left" if left * 2 > len(court_positions) else "right"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _cannot_links(
    by_frame: dict[int, set[int]], canonical_of: dict[int, int]
) -> dict[int, set[int]]:
    """Canonical tracks seen in the same frame cannot be the same person.

    This is the one identity constraint that is free and never wrong: a player cannot be in
    two places at once.  It is far more reliable than any appearance score.
    """
    links: dict[int, set[int]] = defaultdict(set)
    for track_ids in by_frame.values():
        visible = {canonical_of.get(track_id, track_id) for track_id in track_ids}
        for canonical_track_id in visible:
            links[canonical_track_id].update(visible - {canonical_track_id})
    return links


def load_tracklets(
    result_dir: Path,
    *,
    clip_id: str,
    frame_height: int,
    min_samples: int = 25,
) -> list[CanonicalTracklet]:
    """Return canonical tracklets for one analysed clip."""
    bank = _load_bank(result_dir)
    observations, by_frame = _load_observations(result_dir)
    canonical_of = _canonical_map(bank, observations)

    grouped: dict[int, list[TrackletObservation]] = defaultdict(list)
    for track_id, rows in observations.items():
        grouped[canonical_of.get(track_id, track_id)].extend(rows)

    links = _cannot_links(by_frame, canonical_of)
    members: dict[int, set[int]] = defaultdict(set)
    for track_id in observations:
        members[canonical_of.get(track_id, track_id)].add(track_id)

    tracklets: list[CanonicalTracklet] = []
    for canonical_track_id, rows in sorted(grouped.items()):
        rows.sort(key=lambda row: row.frame_index)
        if len(rows) < min_samples:
            continue
        court_positions = [row.court_pos for row in rows if row.court_pos is not None]
        confidences = [row.confidence for row in rows if row.confidence is not None]
        heights = [
            (row.frame_bbox[3] - row.frame_bbox[1]) * frame_height for row in rows
        ]
        tracklets.append(
            CanonicalTracklet(
                clip_id=clip_id,
                canonical_track_id=canonical_track_id,
                track_ids=tuple(sorted(members[canonical_track_id])),
                court_side=_court_side(court_positions),
                observations=tuple(rows),
                cannot_link_canonical_track_ids=tuple(
                    sorted(links[canonical_track_id])
                ),
                median_court_pos=(
                    (
                        median(position[0] for position in court_positions),
                        median(position[1] for position in court_positions),
                    )
                    if court_positions
                    else None
                ),
                mean_confidence=(
                    sum(confidences) / len(confidences) if confidences else 0.0
                ),
                mean_height_px=sum(heights) / len(heights) if heights else 0.0,
            )
        )
    return tracklets


def _load_bank(result_dir: Path) -> dict[str, Any] | None:
    """Return `extensions.fixed_roster_reid` when the upstream artifact carries it."""
    path = result_dir / "analysis-result.json"
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        payload: dict[str, Any] = json.load(handle)
    extensions: dict[str, Any] = payload.get("extensions") or {}
    bank: Any = extensions.get("fixed_roster_reid")
    if not isinstance(bank, dict):
        return None
    typed = cast("dict[str, Any]", bank)
    return typed if typed.get("tracklets") else None


def _canonical_map(
    bank: dict[str, Any] | None, observations: dict[int, list[TrackletObservation]]
) -> dict[int, int]:
    """track_id -> canonical_track_id.  Identity mapping when no bank is available."""
    if bank is None:
        return {track_id: track_id for track_id in observations}
    mapping: dict[int, int] = {}
    entries: list[dict[str, Any]] = bank["tracklets"]
    for tracklet in entries:
        canonical_track_id = int(tracklet["canonical_track_id"])
        members: list[int] = tracklet.get("track_ids") or [canonical_track_id]
        for track_id in members:
            mapping[int(track_id)] = canonical_track_id
    for track_id in observations:
        mapping.setdefault(track_id, track_id)
    return mapping


def _load_observations(
    result_dir: Path,
) -> tuple[dict[int, list[TrackletObservation]], dict[int, set[int]]]:
    observations: dict[int, list[TrackletObservation]] = defaultdict(list)
    by_frame: dict[int, set[int]] = defaultdict(set)
    for row in _read_jsonl(result_dir / "tracks.jsonl"):
        frame_index = int(row["frame_index"])
        players: list[dict[str, Any]] = row.get("players") or []
        for player in players:
            track_id = int(player["track_id"])
            court_pos: list[float] | None = player.get("court_pos")
            observations[track_id].append(
                TrackletObservation(
                    frame_index=frame_index,
                    frame_bbox=_bbox(player["frame_bbox"]),
                    court_pos=(
                        (float(court_pos[0]), float(court_pos[1])) if court_pos else None
                    ),
                    confidence=(
                        float(player["confidence"])
                        if player.get("confidence") is not None
                        else None
                    ),
                )
            )
            by_frame[frame_index].add(track_id)
    return observations, by_frame


def clip_fps(result_dir: Path, default: float = 50.0) -> float:
    """Canonical clip frame rate, used to convert second-based settings into frames."""
    path = result_dir / "inference-manifest.json"
    if not path.exists():
        return default
    with path.open(encoding="utf-8") as handle:
        manifest: dict[str, Any] = json.load(handle)
    return float(manifest.get("fps") or default)
