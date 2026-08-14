"""Assign a match roster identity to every canonical tracklet of a set of clips.

Pipeline per tracklet::

    canonical tracklet
    -> frame quality selection (frontality-weighted)
    -> pose-guided torso crops
    -> upscaled contact sheet
    -> candidate-constrained VLM jersey reading
    -> constrained assignment (co-visibility cannot-links)
    -> roster identity or explicit abstention

Abstaining is a correct outcome.  A wrong identity silently corrupts every statistic
derived from it, while an abstention costs one human click.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2

from .assign import AssignmentSettings, assign_clip
from .frames import FrameSelectionSettings, PoseEstimator, select_frames
from .inputs import load_tracklets
from .records import CanonicalTracklet, IdentityAssignment, JerseyReading, SelectedFrame
from .roster import Roster
from .sheet import SheetSettings, build_contact_sheet
from .vlm import JerseyIdentifier, VlmSettings

ROSTER_IDENTITY_SCHEMA_VERSION = "1.0.0"
# Mirrors the engine's own `analysis_version` style: what produced the artifact, and which
# revision of the method, so a stored mapping can be traced back to how it was made.
ROSTER_IDENTITY_BUILD_ID = "candidate-constrained-jersey-vlm-0.1.0"


@dataclass(frozen=True, slots=True)
class ClipSource:
    """One analysed clip: the engine's output directory plus its canonical video."""

    clip_id: str
    result_dir: Path
    video_path: Path


@dataclass(frozen=True, slots=True)
class ServiceSettings:
    """Everything the service needs beyond the caller-supplied roster."""

    min_samples: int = 25
    candidate_scope: str = "all"  # "all" | "team"
    frame_selection: FrameSelectionSettings = field(default_factory=FrameSelectionSettings)
    sheet: SheetSettings = field(default_factory=SheetSettings)
    vlm: VlmSettings = field(default_factory=VlmSettings)
    assignment: AssignmentSettings = field(default_factory=AssignmentSettings)


def collect_tracklets(
    sources: list[ClipSource], min_samples: int
) -> dict[str, list[CanonicalTracklet]]:
    """Canonical tracklets per clip, keyed by clip id."""
    collected: dict[str, list[CanonicalTracklet]] = {}
    for source in sources:
        capture = cv2.VideoCapture(str(source.video_path))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        capture.release()
        collected[source.clip_id] = load_tracklets(
            source.result_dir,
            clip_id=source.clip_id,
            frame_height=height,
            min_samples=min_samples,
        )
    return collected


def prepare_sheets(
    sources: list[ClipSource],
    tracklets: dict[str, list[CanonicalTracklet]],
    workspace: Path,
    pose: PoseEstimator,
    settings: ServiceSettings,
) -> dict[str, list[SelectedFrame]]:
    """Decode, score and tile frames for every tracklet.  Returns selections by key."""
    selections: dict[str, list[SelectedFrame]] = {}
    sheet_dir = workspace / "sheets"
    sheet_dir.mkdir(parents=True, exist_ok=True)
    for source in sources:
        for tracklet in tracklets[source.clip_id]:
            frames = select_frames(
                tracklet,
                source.video_path,
                workspace / "frames" / tracklet.key.replace("#", "_"),
                pose=pose,
                settings=settings.frame_selection,
            )
            selections[tracklet.key] = frames
            if not frames:
                continue
            sheet = build_contact_sheet(frames, settings.sheet)
            if sheet is not None:
                sheet.save(sheet_dir / f"{tracklet.key.replace('#', '_')}.jpg", quality=94)
    return selections


def read_jerseys(
    tracklets: dict[str, list[CanonicalTracklet]],
    selections: dict[str, list[SelectedFrame]],
    roster: Roster,
    workspace: Path,
    identifier: JerseyIdentifier,
    *,
    team_of: dict[str, str] | None = None,
    on_court: dict[str, dict[str, list[int]]] | None = None,
) -> tuple[dict[str, JerseyReading], dict[str, list[str]]]:
    """One VLM call per tracklet.  Returns readings and the candidate list each one saw.

    `team_of` narrows the candidate list to a single team when the caller already knows it;
    `on_court` narrows it further to the players actually on court for that clip.  Both are
    caller-supplied match metadata, never inferred here.
    """
    from PIL import Image

    sheet_dir = workspace / "sheets"
    readings: dict[str, JerseyReading] = {}
    candidates_for: dict[str, list[str]] = {}
    for clip_tracklets in tracklets.values():
        for tracklet in clip_tracklets:
            frames = selections.get(tracklet.key) or []
            sheet_path = sheet_dir / f"{tracklet.key.replace('#', '_')}.jpg"
            if not frames or not sheet_path.exists():
                continue
            team_id = (team_of or {}).get(tracklet.key)
            candidates = roster.candidates(
                team_id, (on_court or {}).get(tracklet.clip_id)
            )
            candidates_for[tracklet.key] = [
                candidate.roster_player_id for candidate in candidates
            ]
            with Image.open(sheet_path) as image:
                readings[tracklet.key] = identifier.identify(
                    image.convert("RGB"),
                    candidates,
                    roster.team(team_id).display_name if team_id else None,
                )
    return readings, candidates_for


def assign_identities(
    tracklets: dict[str, list[CanonicalTracklet]],
    selections: dict[str, list[SelectedFrame]],
    readings: dict[str, JerseyReading],
    candidates_for: dict[str, list[str]],
    settings: AssignmentSettings,
) -> list[IdentityAssignment]:
    """Turn per-tracklet readings into a set of identities consistent within each clip."""
    assignments: list[IdentityAssignment] = []
    for clip_tracklets in tracklets.values():
        resolved = assign_clip(clip_tracklets, readings, candidates_for, settings)
        for tracklet in clip_tracklets:
            reading = readings.get(tracklet.key)
            frames = selections.get(tracklet.key) or []
            if reading is None:
                assignments.append(
                    IdentityAssignment(
                        clip_id=tracklet.clip_id,
                        canonical_track_id=tracklet.canonical_track_id,
                        track_ids=tracklet.track_ids,
                        roster_player_id=None,
                        status="unknown",
                        method="none",
                        rule="no_usable_frames",
                        court_side=tracklet.court_side,
                        quality_flags=["no_usable_frames"],
                    )
                )
                continue
            player, _score, rule = resolved.get(tracklet.key, (None, 0.0, "no_reading"))
            assignments.append(
                IdentityAssignment(
                    clip_id=tracklet.clip_id,
                    canonical_track_id=tracklet.canonical_track_id,
                    track_ids=tracklet.track_ids,
                    roster_player_id=player,
                    status="auto_assigned" if player else "human_review",
                    method="jersey_vlm+cannot_link"
                    if settings.enable_cannot_link
                    else "jersey_vlm",
                    rule=rule,
                    court_side=tracklet.court_side,
                    selected_frame_indices=tuple(frame.frame_index for frame in frames),
                    jersey=reading,
                )
            )
    return assignments


def write_artifact(
    path: Path,
    *,
    match_id: str,
    assignments: list[IdentityAssignment],
    settings: ServiceSettings,
) -> None:
    """Emit the identity mapping as its own artifact.

    Clip-local track IDs are never rewritten, so a clip can be re-analysed without
    invalidating identities that a human has already confirmed.
    """
    payload = {
        "schema_version": ROSTER_IDENTITY_SCHEMA_VERSION,
        "scope": "match",
        "match_id": match_id,
        "producer": {
            "name": "volleyball-analysis-engine.roster_identity",
            "build_id": ROSTER_IDENTITY_BUILD_ID,
        },
        "method": {
            "identity_evidence": "jersey_number",
            "vlm_model_id": settings.vlm.model_id,
            "candidate_scope": settings.candidate_scope,
        },
        "identity_contract": "team_and_jersey_number",
        "assignments": [
            {
                "clip_id": assignment.clip_id,
                "canonical_track_id": assignment.canonical_track_id,
                "track_ids": list(assignment.track_ids),
                "roster_player_id": assignment.roster_player_id,
                "status": assignment.status,
                "method": assignment.method,
                "rule": assignment.rule,
                "court_side": assignment.court_side,
                "selected_frame_indices": [
                    str(index) for index in assignment.selected_frame_indices
                ],
                "jersey": (
                    {
                        key: value
                        for key, value in asdict(assignment.jersey).items()
                        if key != "raw_response"
                    }
                    if assignment.jersey
                    else None
                ),
                "quality_flags": assignment.quality_flags,
            }
            for assignment in assignments
        ],
        "summary": {
            "assignment_count": len(assignments),
            "auto_assigned_count": sum(
                1 for item in assignments if item.status == "auto_assigned"
            ),
            "human_review_count": sum(
                1 for item in assignments if item.status == "human_review"
            ),
            "unknown_count": sum(1 for item in assignments if item.status == "unknown"),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=1)
