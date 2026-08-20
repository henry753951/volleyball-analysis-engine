"""Stream a legacy predictions JSON export into Provider Work analysis jobs.

This module is deliberately independent from the normal inference worker.  It
indexes the source file once, converts one submitted rally at a time, and never
loads the multi-gigabyte export into memory.
"""

# The importer is an intentionally defensive adapter around an untyped legacy
# JSON format.  Keep those boundary conversions here instead of weakening the
# normal inference pipeline's stricter lint policy.
# ruff: noqa: ANN401, C901, EM101, EM102, PERF401, PLR0912, PLR0915, PLR2004, SLF001, TRY003

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from uuid import NAMESPACE_URL, uuid5

import numpy as np
import orjson
from numpy.typing import NDArray
from volleyball_monitoring_ai import (
    AIJobRequest,
    AnalysisDomainData,
    ProviderAnalysisJobRequest,
    ProviderResultArtifact,
    ProviderWorkCapabilities,
    ProviderWorkContext,
    ProviderWorkerClient,
    ProviderWorkerConfig,
    build_analysis_data,
    validate_passthrough,
)

from .association import HitAssociation, associate_hit
from .contact_detection import ContactProposal, detect_contact_proposals
from .evidence_artifacts import build_analysis_evidence_artifacts
from .geometry import estimate_homography, project_normalized_frame_point
from .inference import HarmonicMeanTracker, normalize_frame_bbox
from .multitask_provider import COURT60_WORLD_XY
from .pipeline import AnalysisPipeline, resolve_track_court_sides
from .records import (
    ActionObservation,
    BallObservation,
    CourtFrame,
    CourtKeypoint,
    FrameObservation,
    PersonPoseObservation,
    PlayerObservation,
)

PLAN_SCHEMA_VERSION = 1
INDEX_SCHEMA_VERSION = 1
IMPORTER_BUILD_ID = "predictions-json-import-v3"
ACTION_TAXONOMY_ID = "volleyball-inference-sdk.actions"
ACTION_TAXONOMY_VERSION = "legacy-export-v1"
POSE_RECIPE_NAMESPACE = "predictions-json/coco17-normalized-video-v1"
CONTACT_ACTIVITY_PHASES = frozenset({"pass", "set", "spike"})

_FRAME_RE = re.compile(rb'"frame_index":(\d+)')
_GROUP_RE = re.compile(
    rb'"group_activity":\{"id":(\d+),"name":"([^"]+)","confidence":([-+0-9.eE]+)\}'
)


@dataclass(frozen=True, slots=True)
class SourceMetadata:
    """The stable subset of export metadata required by the importer."""

    width: int
    height: int
    fps: float
    frame_count: int
    duration_sec: float
    checkpoint: str | None


@dataclass(frozen=True, slots=True)
class Segment:
    """One half-open source frame interval that becomes one submitted rally."""

    segment_index: int
    source_start_frame: int
    source_end_frame_exclusive: int
    start_time_us: int
    end_time_us: int


@dataclass(frozen=True, slots=True)
class PredictionIndex:
    """Compact random-access index for a line-oriented predictions export."""

    source_path: Path
    source_size: int
    source_mtime_ns: int
    metadata: SourceMetadata
    offsets: NDArray[np.uint64]
    activity_ids: NDArray[np.uint8]
    activity_confidences: NDArray[np.float32]
    activity_names: dict[int, str]


@dataclass(frozen=True, slots=True)
class ConvertedAnalysis:
    """Callback-ready analysis bytes and evidence artifacts for one job."""

    analysis_id: str
    domain: AnalysisDomainData
    artifacts: tuple[ProviderResultArtifact, ...]
    source_frame_count: int


@dataclass(frozen=True, slots=True)
class ContactPhaseCandidate:
    """One stable legacy group-activity phase interpreted as a contact."""

    frame_index: int
    end_frame_index: int
    label: str
    confidence: float


def _json_fragment(line: bytes) -> Any:
    _, value = line.split(b":", 1)
    return orjson.loads(value.strip().rstrip(b","))


def _metadata_from_header(header: dict[str, Any]) -> SourceMetadata:
    video = cast("dict[str, Any]", header.get("video", {}))
    return SourceMetadata(
        width=int(video["width"]),
        height=int(video["height"]),
        fps=float(video["fps"]),
        frame_count=int(video["source_frame_count"]),
        duration_sec=float(video["source_duration_sec"]),
        checkpoint=cast("str | None", header.get("checkpoint")),
    )


def build_prediction_index(
    source_path: Path,
    index_path: Path,
    *,
    report: Callable[[int], None] | None = None,
) -> PredictionIndex:
    """Scan the export once and persist byte offsets plus group-activity labels."""
    source_path = source_path.resolve()
    stat = source_path.stat()
    header: dict[str, Any] = {}
    offsets: list[int] = []
    activity_ids: list[int] = []
    activity_confidences: list[float] = []
    activity_names: dict[int, str] = {}

    with source_path.open("rb") as handle:
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            stripped = line.lstrip()
            if stripped.startswith(b'"video"'):
                header["video"] = _json_fragment(stripped)
                continue
            if stripped.startswith(b'"checkpoint"'):
                header["checkpoint"] = _json_fragment(stripped)
                continue
            if not stripped.startswith(b'{"frame_index"'):
                continue
            frame_match = _FRAME_RE.search(stripped)
            group_match = _GROUP_RE.search(stripped)
            if frame_match is None or group_match is None:
                raise ValueError(f"prediction frame at byte {offset} is missing required fields")
            frame_index = int(frame_match.group(1))
            if frame_index != len(offsets):
                raise ValueError(
                    "prediction frames must be contiguous: "
                    f"expected {len(offsets)}, got {frame_index}"
                )
            activity_id = int(group_match.group(1))
            if not 0 <= activity_id <= 255:
                raise ValueError(f"group activity id {activity_id} does not fit uint8")
            offsets.append(offset)
            activity_ids.append(activity_id)
            activity_confidences.append(float(group_match.group(3)))
            activity_names[activity_id] = group_match.group(2).decode("utf-8")
            if report is not None and len(offsets) % 10_000 == 0:
                report(len(offsets))

    metadata = _metadata_from_header(header)
    if len(offsets) != metadata.frame_count:
        raise ValueError(
            f"source metadata declares {metadata.frame_count} frames, indexed {len(offsets)}"
        )
    index_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        index_path,
        schema_version=np.asarray([INDEX_SCHEMA_VERSION], dtype=np.uint16),
        source_path=np.asarray([str(source_path)]),
        source_size=np.asarray([stat.st_size], dtype=np.uint64),
        source_mtime_ns=np.asarray([stat.st_mtime_ns], dtype=np.uint64),
        width=np.asarray([metadata.width], dtype=np.uint32),
        height=np.asarray([metadata.height], dtype=np.uint32),
        fps=np.asarray([metadata.fps], dtype=np.float64),
        frame_count=np.asarray([metadata.frame_count], dtype=np.uint64),
        duration_sec=np.asarray([metadata.duration_sec], dtype=np.float64),
        checkpoint=np.asarray([metadata.checkpoint or ""]),
        offsets=np.asarray(offsets, dtype=np.uint64),
        activity_ids=np.asarray(activity_ids, dtype=np.uint8),
        activity_confidences=np.asarray(activity_confidences, dtype=np.float32),
        activity_names=np.asarray([json.dumps(activity_names, sort_keys=True)]),
    )
    return load_prediction_index(index_path)


def load_prediction_index(index_path: Path) -> PredictionIndex:
    """Load and validate a compact predictions index."""
    with np.load(index_path, allow_pickle=False) as raw:
        if int(raw["schema_version"][0]) != INDEX_SCHEMA_VERSION:
            raise ValueError("unsupported predictions index schema")
        source_path = Path(str(raw["source_path"][0]))
        source_size = int(raw["source_size"][0])
        source_mtime_ns = int(raw["source_mtime_ns"][0])
        checkpoint = str(raw["checkpoint"][0]) or None
        metadata = SourceMetadata(
            width=int(raw["width"][0]),
            height=int(raw["height"][0]),
            fps=float(raw["fps"][0]),
            frame_count=int(raw["frame_count"][0]),
            duration_sec=float(raw["duration_sec"][0]),
            checkpoint=checkpoint,
        )
        offsets = raw["offsets"].astype(np.uint64, copy=True)
        activity_ids = raw["activity_ids"].astype(np.uint8, copy=True)
        activity_confidences = raw["activity_confidences"].astype(np.float32, copy=True)
        names_payload = json.loads(str(raw["activity_names"][0]))
    stat = source_path.stat()
    if stat.st_size != source_size or stat.st_mtime_ns != source_mtime_ns:
        raise ValueError("predictions source changed after the index was created")
    if not (len(offsets) == len(activity_ids) == len(activity_confidences) == metadata.frame_count):
        raise ValueError("predictions index arrays have inconsistent lengths")
    return PredictionIndex(
        source_path=source_path,
        source_size=source_size,
        source_mtime_ns=source_mtime_ns,
        metadata=metadata,
        offsets=offsets,
        activity_ids=activity_ids,
        activity_confidences=activity_confidences,
        activity_names={int(key): str(value) for key, value in names_payload.items()},
    )


def _runs(mask: NDArray[np.bool_]) -> Iterator[tuple[bool, int, int]]:
    if not len(mask):
        return
    start = 0
    current = bool(mask[0])
    for index in range(1, len(mask)):
        value = bool(mask[index])
        if value == current:
            continue
        yield current, start, index
        start = index
        current = value
    yield current, start, len(mask)


def _majority(mask: NDArray[np.bool_], window: int) -> NDArray[np.bool_]:
    if window <= 1:
        return mask.copy()
    if window % 2 == 0:
        window += 1
    counts = np.convolve(mask.astype(np.uint8), np.ones(window, dtype=np.uint16), mode="same")
    return np.asarray(counts >= (window // 2 + 1), dtype=np.bool_)


def detect_segments(
    index: PredictionIndex,
    *,
    smooth_seconds: float = 0.5,
    max_gap_seconds: float = 1.0,
    min_active_seconds: float = 0.75,
    padding_before_seconds: float = 2.0,
    padding_after_seconds: float = 1.0,
) -> list[Segment]:
    """Turn play labels into rallies while treating winpoint labels as terminal states."""
    fps = index.metadata.fps
    inactive_names = {"non_rally", "l_winpoint", "r_winpoint"}
    playing_ids = {
        activity_id
        for activity_id, name in index.activity_names.items()
        if name not in inactive_names
    }
    if not playing_ids:
        raise ValueError("prediction export has no playable group-activity labels")
    active = np.isin(index.activity_ids, np.asarray(sorted(playing_ids), dtype=np.uint8))
    active = _majority(active, max(1, round(smooth_seconds * fps)))
    max_gap_frames = max(0, round(max_gap_seconds * fps))
    for value, start, end in list(_runs(active)):
        if not value and start > 0 and end < len(active) and end - start <= max_gap_frames:
            active[start:end] = True
    min_active_frames = max(1, round(min_active_seconds * fps))
    for value, start, end in list(_runs(active)):
        if value and end - start < min_active_frames:
            active[start:end] = False

    padding_before = max(0, round(padding_before_seconds * fps))
    padding_after = max(0, round(padding_after_seconds * fps))
    padded: list[tuple[int, int]] = []
    for value, start, end in _runs(active):
        if value:
            padded.append((max(0, start - padding_before), min(len(active), end + padding_after)))
    merged: list[tuple[int, int]] = []
    for start, end in padded:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return [
        Segment(
            segment_index=segment_index,
            source_start_frame=start,
            source_end_frame_exclusive=end,
            start_time_us=round(start * 1_000_000 / fps),
            end_time_us=round(end * 1_000_000 / fps),
        )
        for segment_index, (start, end) in enumerate(merged)
    ]


def create_plan(index_path: Path, plan_path: Path, **segment_options: float) -> dict[str, Any]:
    """Create a resumable JSON plan from an existing compact index."""
    index = load_prediction_index(index_path)
    segments = detect_segments(index, **segment_options)
    payload: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "index_path": str(index_path.resolve()),
        "predictions": {
            "path": str(index.source_path),
            "size": index.source_size,
            "mtime_ns": index.source_mtime_ns,
            "width": index.metadata.width,
            "height": index.metadata.height,
            "fps": index.metadata.fps,
            "frame_count": index.metadata.frame_count,
            "duration_sec": index.metadata.duration_sec,
            "checkpoint": index.metadata.checkpoint,
        },
        "segment_options": segment_options,
        "segments": [
            {
                "segment_index": segment.segment_index,
                "source_start_frame": segment.source_start_frame,
                "source_end_frame_exclusive": segment.source_end_frame_exclusive,
                "start_time_us": str(segment.start_time_us),
                "end_time_us": str(segment.end_time_us),
                "rally_id": None,
                "submission_id": None,
            }
            for segment in segments
        ],
    }
    save_plan(plan_path, payload)
    return payload


def load_plan(plan_path: Path) -> dict[str, Any]:
    """Load a plan and verify its referenced source index still matches."""
    payload = cast("dict[str, Any]", json.loads(plan_path.read_text(encoding="utf-8")))
    if payload.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported predictions import plan schema")
    load_prediction_index(Path(str(payload["index_path"])))
    return payload


def save_plan(plan_path: Path, payload: dict[str, Any]) -> None:
    """Atomically checkpoint submission IDs so interrupted runs can resume."""
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = plan_path.with_suffix(plan_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(plan_path)


def _source_rows(index: PredictionIndex, start: int, end: int) -> Iterator[dict[str, Any]]:
    if start < 0 or end > index.metadata.frame_count or start >= end:
        raise ValueError("invalid source frame range")
    with index.source_path.open("rb") as handle:
        handle.seek(int(index.offsets[start]))
        for expected in range(start, end):
            line = handle.readline().rstrip(b",\r\n")
            row = cast("dict[str, Any]", orjson.loads(line))
            if int(row.get("frame_index", -1)) != expected:
                raise ValueError(f"source frame mismatch at {expected}")
            yield row


def _destination_frame(source_frame: int, start: int, end: int, total_frames: int) -> int:
    source_count = end - start
    if source_count <= 1 or total_frames <= 1:
        return 0
    return round((source_frame - start) * (total_frames - 1) / (source_count - 1))


def _unit(value: Any) -> float:
    return min(1.0, max(0.0, float(value)))


def _bounded_bbox(values: NDArray[np.float32]) -> tuple[float, float, float, float]:
    left, right = sorted((_unit(values[0]), _unit(values[2])))
    top, bottom = sorted((_unit(values[1]), _unit(values[3])))
    return left, top, right, bottom


def _pose_observation(
    detection: dict[str, Any] | None,
    *,
    frame_index: int,
    track_id: int,
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
) -> PersonPoseObservation:
    raw_pose = detection.get("human_pose") if detection is not None else None
    keypoints: tuple[tuple[float, float, float], ...] | None = None
    status = "NO_USABLE_BBOX"
    pose_points: list[dict[str, Any]] = []
    if isinstance(raw_pose, list):
        for raw_point in cast("list[Any]", raw_pose):
            if isinstance(raw_point, dict):
                pose_points.append(cast("dict[str, Any]", raw_point))
    if len(pose_points) == 17:
        ordered = sorted(pose_points, key=lambda point: int(point.get("index", -1)))
        keypoints = tuple(
            (_unit(point["x"] / width), _unit(point["y"] / height), _unit(point["visibility"]))
            for point in ordered
        )
        status = "AVAILABLE" if sum(point[2] >= 0.35 for point in keypoints) >= 5 else "LOW_QUALITY"
    return PersonPoseObservation(
        frame_index=frame_index,
        track_id=track_id,
        bbox_source="DETECTOR" if detection is not None else "TRACKER_PROPAGATED",
        frame_bbox=bbox,
        crop_transform=(1.0 / width, 1.0 / height, bbox[0], bbox[1]),
        status=cast("Any", status),
        keypoints=keypoints,
    )


def _activity_runs(rows: list[tuple[int, str, float]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for frame_index, label, confidence in rows:
        if result and result[-1]["label"] == label:
            result[-1]["end_frame_index"] = str(frame_index)
            result[-1]["mean_confidence_sum"] += confidence
            result[-1]["sample_count"] += 1
            continue
        result.append(
            {
                "label": label,
                "start_frame_index": str(frame_index),
                "end_frame_index": str(frame_index),
                "mean_confidence_sum": confidence,
                "sample_count": 1,
            }
        )
    for item in result:
        item["mean_confidence"] = item.pop("mean_confidence_sum") / item.pop("sample_count")
    return result


def _contact_phase(label: str) -> str | None:
    phase = label.rsplit("_", 1)[-1]
    return phase if phase in CONTACT_ACTIVITY_PHASES else None


def _contact_phase_candidates(
    rows: list[tuple[int, str, float]],
    *,
    fps: float,
) -> list[ContactPhaseCandidate]:
    """Keep stable pass/set/spike phases and reject isolated legacy noise."""
    if not rows:
        return []
    minimum_frames = max(3, round(fps * 0.12))
    bridge_frames = max(2, round(fps * 0.35))
    cluster_gap_frames = max(3, round(fps * 1.0))
    unique_rows = sorted({frame: (label, confidence) for frame, label, confidence in rows}.items())
    raw_runs: list[ContactPhaseCandidate] = []
    run_start = unique_rows[0][0]
    run_end = run_start
    run_label = unique_rows[0][1][0]
    confidences = [unique_rows[0][1][1]]
    for frame, (label, confidence) in unique_rows[1:]:
        if frame == run_end + 1 and label == run_label:
            run_end = frame
            confidences.append(confidence)
            continue
        if (
            _contact_phase(run_label) is not None
            and run_end - run_start + 1 >= minimum_frames
            and sum(confidences) / len(confidences) >= 0.60
        ):
            raw_runs.append(
                ContactPhaseCandidate(
                    frame_index=run_start,
                    end_frame_index=run_end,
                    label=run_label,
                    confidence=sum(confidences) / len(confidences),
                )
            )
        run_start = run_end = frame
        run_label = label
        confidences = [confidence]
    if (
        _contact_phase(run_label) is not None
        and run_end - run_start + 1 >= minimum_frames
        and sum(confidences) / len(confidences) >= 0.60
    ):
        raw_runs.append(
            ContactPhaseCandidate(
                frame_index=run_start,
                end_frame_index=run_end,
                label=run_label,
                confidence=sum(confidences) / len(confidences),
            )
        )

    merged: list[ContactPhaseCandidate] = []
    for item in raw_runs:
        previous = merged[-1] if merged else None
        if (
            previous is not None
            and previous.label == item.label
            and item.frame_index - previous.end_frame_index - 1 <= bridge_frames
        ):
            previous_frames = previous.end_frame_index - previous.frame_index + 1
            item_frames = item.end_frame_index - item.frame_index + 1
            merged[-1] = ContactPhaseCandidate(
                frame_index=previous.frame_index,
                end_frame_index=item.end_frame_index,
                label=item.label,
                confidence=(previous.confidence * previous_frames + item.confidence * item_frames)
                / (previous_frames + item_frames),
            )
        else:
            merged.append(item)

    clusters: list[list[ContactPhaseCandidate]] = []
    for item in merged:
        if not clusters or item.frame_index - clusters[-1][-1].end_frame_index > cluster_gap_frames:
            clusters.append([item])
        else:
            clusters[-1].append(item)
    return [
        item
        for cluster in clusters
        if any(_contact_phase(item.label) in {"set", "spike"} for item in cluster)
        for item in cluster
    ]


def _nearest_action(
    actions: dict[tuple[int, int], ActionObservation],
    *,
    frame_index: int | None,
    track_id: int,
    radius: int,
) -> ActionObservation | None:
    if frame_index is None:
        return None
    candidates = [
        action
        for (candidate_frame, candidate_track), action in actions.items()
        if candidate_track == track_id and abs(candidate_frame - frame_index) <= radius
    ]
    return (
        None
        if not candidates
        else min(
            candidates,
            key=lambda action: (
                abs(action.frame_index - frame_index),
                -(action.confidence or 0.0),
            ),
        )
    )


def _actor_payload(
    association: HitAssociation,
    *,
    actions: dict[tuple[int, int], ActionObservation],
    action_radius: int,
) -> dict[str, Any] | None:
    player = association.player
    observation_frame = association.observation_frame
    if player is None or observation_frame is None:
        return None
    action = _nearest_action(
        actions,
        frame_index=observation_frame,
        track_id=player.track_id,
        radius=action_radius,
    )
    return {
        "track_id": player.track_id,
        "observation_frame_index": str(observation_frame),
        "association_confidence": association.confidence,
        "frame_bbox": {
            "x1": player.frame_bbox[0],
            "y1": player.frame_bbox[1],
            "x2": player.frame_bbox[2],
            "y2": player.frame_bbox[3],
        },
        "frame_foot_pos": {"x": player.frame_foot_pos[0], "y": player.frame_foot_pos[1]},
        "court_pos": (
            None
            if player.court_pos is None
            else {"x": player.court_pos[0], "y": player.court_pos[1]}
        ),
        "action": (
            None
            if action is None
            else {
                "label": action.label,
                "taxonomy_id": ACTION_TAXONOMY_ID,
                "taxonomy_version": ACTION_TAXONOMY_VERSION,
                "confidence": action.confidence,
                "attributes": {"source": "legacy-predictions-json"},
            }
        ),
    }


def _candidate_semantic(candidate: ContactPhaseCandidate) -> dict[str, str]:
    side = "left" if candidate.label.startswith("l_") else "right"
    phase = _contact_phase(candidate.label)
    kind = {"pass": "receive", "set": "contact", "spike": "spike"}.get(phase or "", "contact")
    return {
        "court_side": side,
        "phase": phase or "contact",
        "ball_event_kind": kind,
    }


def _build_contact_events(
    *,
    job: Any,
    phase_candidates: list[ContactPhaseCandidate],
    proposals: list[ContactProposal],
    balls: dict[int, BallObservation],
    players: dict[int, tuple[PlayerObservation, ...]],
    actions: dict[tuple[int, int], ActionObservation],
    poses: dict[int, tuple[PersonPoseObservation, ...]],
    homographies: dict[int, NDArray[np.float64]],
    fps: float,
    frame_width: int,
    frame_height: int,
) -> list[dict[str, Any]]:
    if job.key_points:
        anchors = [int(point.clip_frame_index) for point in job.key_points]
        action_radius = max(3, round(fps * 0.20))
        events: list[dict[str, Any]] = []
        for sequence_index, point in enumerate(job.key_points):
            anchor = anchors[sequence_index]
            previous_anchor = anchors[sequence_index - 1] if sequence_index > 0 else -1
            next_anchor = (
                anchors[sequence_index + 1]
                if sequence_index + 1 < len(anchors)
                else int(job.clip.video.total_frames)
            )
            association = associate_hit(
                anchor_frame=anchor,
                previous_anchor_frame=previous_anchor,
                next_anchor_frame=next_anchor,
                is_terminal=point.is_terminal,
                balls=balls,
                players=players,
                actions=actions,
                poses=poses,
                frame_width=frame_width,
                frame_height=frame_height,
                action_search_radius=action_radius,
            )
            actor = _actor_payload(
                association,
                actions=actions,
                action_radius=action_radius,
            )
            representative = cast("Any", AnalysisPipeline)._representative_position(
                association.player,
                association.ball,
                association.observation_frame,
                homographies,
                frame_width=frame_width,
                frame_height=frame_height,
            )
            events.append(
                {
                    "key_point_id": point.key_point_id,
                    "source_key_point_id": point.key_point_id,
                    "anchor_origin": "human_anchor",
                    "detection_confidence": None,
                    "sequence_index": sequence_index,
                    "marker_kind": point.marker_kind,
                    "is_terminal": point.is_terminal,
                    "anchor_frame_index": point.clip_frame_index,
                    "resolved_frame_index": (
                        None
                        if association.observation_frame is None
                        else str(association.observation_frame)
                    ),
                    "association_state": "resolved_single" if actor is not None else "no_player",
                    "actors": [] if actor is None else [actor],
                    "actor_candidates": [],
                    "ball": (
                        {"state": "missing"}
                        if association.ball is None
                        else {
                            "state": "observed",
                            "sample_frame_index": str(association.ball.frame_index),
                            "frame_pos": {
                                "x": association.ball.frame_pos[0],
                                "y": association.ball.frame_pos[1],
                            },
                            "confidence": association.ball.confidence,
                        }
                    ),
                    "representative_court_positions": (
                        [] if representative is None else [representative]
                    ),
                    "quality_flags": ["human_anchor_passthrough", association.mode],
                    "extensions": {
                        "authoritative_clip_pts": point.clip_pts,
                        "authoritative_clip_time_us": point.clip_time_us,
                        "hitter_association": association.evidence,
                    },
                }
            )
        return events
    if not phase_candidates:
        return []
    proposal_radius = max(4, round(fps * 0.35))
    used_proposals: set[int] = set()
    aligned: list[tuple[ContactPhaseCandidate, int, ContactProposal | None]] = []
    for candidate in phase_candidates:
        available = [
            proposal
            for proposal in proposals
            if proposal.frame_index not in used_proposals
            and abs(proposal.frame_index - candidate.frame_index) <= proposal_radius
        ]
        proposal = (
            None
            if not available
            else min(
                available,
                key=lambda item: (
                    abs(item.frame_index - candidate.frame_index),
                    -item.confidence,
                ),
            )
        )
        anchor = candidate.frame_index if proposal is None else proposal.frame_index
        if proposal is not None:
            used_proposals.add(proposal.frame_index)
        aligned.append((candidate, anchor, proposal))
    aligned.sort(key=lambda item: item[1])
    deduplicated: list[tuple[ContactPhaseCandidate, int, ContactProposal | None]] = []
    for item in aligned:
        if not deduplicated or deduplicated[-1][1] != item[1]:
            deduplicated.append(item)
            continue
        previous = deduplicated[-1]
        previous_score = previous[0].confidence + (
            previous[2].confidence if previous[2] is not None else 0.0
        )
        item_score = item[0].confidence + (item[2].confidence if item[2] is not None else 0.0)
        if item_score > previous_score:
            deduplicated[-1] = item
    aligned = deduplicated

    action_radius = max(3, round(fps * 0.20))
    events: list[dict[str, Any]] = []
    for sequence_index, (candidate, anchor, proposal) in enumerate(aligned):
        previous_anchor = aligned[sequence_index - 1][1] if sequence_index > 0 else -1
        next_anchor = (
            aligned[sequence_index + 1][1]
            if sequence_index + 1 < len(aligned)
            else int(job.clip.video.total_frames)
        )
        association = associate_hit(
            anchor_frame=anchor,
            previous_anchor_frame=previous_anchor,
            next_anchor_frame=next_anchor,
            is_terminal=False,
            balls=balls,
            players=players,
            actions=actions,
            poses=poses,
            frame_width=frame_width,
            frame_height=frame_height,
            action_search_radius=action_radius,
        )
        actor = _actor_payload(
            association,
            actions=actions,
            action_radius=action_radius,
        )
        representative = cast("Any", AnalysisPipeline)._representative_position(
            association.player,
            association.ball,
            association.observation_frame,
            homographies,
            frame_width=frame_width,
            frame_height=frame_height,
        )
        semantic = _candidate_semantic(candidate)
        detection_confidence = candidate.confidence
        if proposal is not None:
            detection_confidence = (detection_confidence + proposal.confidence) / 2.0
        key_point_id = str(
            uuid5(
                NAMESPACE_URL,
                f"legacy-predictions-contact:{job.rally_submission_id}:{anchor}:v1",
            )
        )
        events.append(
            {
                "key_point_id": key_point_id,
                "source_key_point_id": None,
                "anchor_origin": "ai_detected",
                "detection_confidence": _unit(detection_confidence),
                "sequence_index": sequence_index,
                "marker_kind": "contact",
                "is_terminal": False,
                "anchor_frame_index": str(anchor),
                "resolved_frame_index": (
                    None
                    if association.observation_frame is None
                    else str(association.observation_frame)
                ),
                "association_state": "resolved_single" if actor is not None else "no_player",
                "actors": [] if actor is None else [actor],
                "actor_candidates": [],
                "ball": (
                    {"state": "missing"}
                    if association.ball is None
                    else {
                        "state": "observed",
                        "sample_frame_index": str(association.ball.frame_index),
                        "frame_pos": {
                            "x": association.ball.frame_pos[0],
                            "y": association.ball.frame_pos[1],
                        },
                        "confidence": association.ball.confidence,
                    }
                ),
                "representative_court_positions": (
                    [] if representative is None else [representative]
                ),
                "quality_flags": [
                    "legacy_group_activity_transition",
                    association.mode,
                    *(("ball_flight_aligned",) if proposal is not None else ()),
                ],
                "extensions": {
                    "detection": {
                        "method": "legacy_group_activity_transition_v1",
                        "group_activity_label": candidate.label,
                        "court_side": semantic["court_side"],
                        "phase": semantic["phase"],
                        "ball_event_kind": semantic["ball_event_kind"],
                        "phase_start_frame_index": str(candidate.frame_index),
                        "phase_end_frame_index": str(candidate.end_frame_index),
                        "ball_proposal_frame_index": (
                            None if proposal is None else str(proposal.frame_index)
                        ),
                    },
                    "hitter_association": association.evidence,
                },
            }
        )
    return events


def convert_segment(
    *,
    index: PredictionIndex,
    segment: dict[str, Any],
    job: AIJobRequest,
) -> ConvertedAnalysis:
    """Convert only the source range corresponding to one Provider Work job."""
    start = int(segment["source_start_frame"])
    end = int(segment["source_end_frame_exclusive"])
    total_frames = int(job.clip.video.total_frames)
    tracker = HarmonicMeanTracker(
        max_lost_frames=120,
        max_geometry_lost_frames=120,
        max_prediction_frames=2,
    )
    frame_records: dict[int, dict[str, Any]] = {}
    ball_positions: dict[int, dict[str, Any]] = {}
    ball_observations: dict[int, BallObservation] = {}
    court_keypoints: dict[int, list[dict[str, Any]]] = {}
    homographies: dict[int, NDArray[np.float64]] = {}
    poses: dict[int, tuple[PersonPoseObservation, ...]] = {}
    players_by_frame: dict[int, tuple[PlayerObservation, ...]] = {}
    actions: dict[tuple[int, int], ActionObservation] = {}
    track_stats: dict[int, dict[str, Any]] = defaultdict(
        lambda: {"first": total_frames, "last": 0, "confidence_sum": 0.0, "count": 0}
    )
    activity_samples: list[tuple[int, str, float]] = []

    for row in _source_rows(index, start, end):
        source_frame = int(row["frame_index"])
        frame_index = _destination_frame(source_frame, start, end, total_frames)
        raw_court = row.get("court")
        visible_points: list[dict[str, Any]] = []
        projection_points: list[CourtKeypoint] = []
        if isinstance(raw_court, dict):
            court = cast("dict[str, Any]", raw_court)
            candidate_points = court.get("keypoints")
            raw_points: list[Any] = (
                cast("list[Any]", candidate_points)
                if court.get("valid") and isinstance(candidate_points, list)
                else []
            )
            for point in raw_points:
                if not isinstance(point, dict):
                    continue
                point_data = cast("dict[str, Any]", point)
                point_index = int(point_data["index"])
                confidence = _unit(point_data.get("visibility", 0.0))
                x_px = float(point_data.get("x", -1.0))
                y_px = float(point_data.get("y", -1.0))
                x = x_px / index.metadata.width
                y = y_px / index.metadata.height
                world_pos = (
                    COURT60_WORLD_XY[point_index]
                    if 0 <= point_index < len(COURT60_WORLD_XY)
                    else None
                )
                projection_points.append(
                    CourtKeypoint(
                        index=point_index,
                        frame_pos_px=(x_px, y_px),
                        confidence=confidence,
                        world_pos_m=world_pos,
                    )
                )
                if confidence >= 0.35 and 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
                    visible_points.append(
                        {
                            "keypoint_id": point_index,
                            "frame_pos": {"x": x, "y": y},
                            "confidence": confidence,
                        }
                    )
        if visible_points:
            court_keypoints[frame_index] = visible_points
        if projection_points:
            homography = estimate_homography(
                CourtFrame(
                    frame_index=frame_index,
                    available=True,
                    keypoints=tuple(projection_points),
                ),
                confidence_threshold=0.35,
            )
            if homography is not None:
                homographies[frame_index] = homography
        else:
            homography = None

        detections: list[dict[str, Any]] = []
        raw_detections = row.get("detections")
        if isinstance(raw_detections, list):
            for raw_detection in cast("list[Any]", raw_detections):
                if not isinstance(raw_detection, dict):
                    continue
                detection = cast("dict[str, Any]", raw_detection)
                if detection.get("class_name") == "person":
                    detections.append(detection)
        boxes = np.asarray(
            [
                normalize_frame_bbox(
                    cast("tuple[float, float, float, float]", tuple(item["bbox_xyxy"])),
                    width=index.metadata.width,
                    height=index.metadata.height,
                )
                for item in detections
            ],
            dtype=np.float32,
        ).reshape((-1, 4))
        scores = np.asarray([float(item["confidence"]) for item in detections], dtype=np.float32)
        tracked = tracker.update(frame_index, boxes, scores, None)
        players: list[dict[str, Any]] = []
        frame_players: list[PlayerObservation] = []
        frame_poses: list[PersonPoseObservation] = []
        for item in tracked:
            bbox = _bounded_bbox(item.bbox)
            detection = detections[item.detection_index] if item.detection_index >= 0 else None
            frame_foot_pos = ((bbox[0] + bbox[2]) / 2.0, bbox[3])
            court_pos = (
                None
                if homography is None
                else project_normalized_frame_point(
                    frame_foot_pos,
                    homography,
                    frame_width=index.metadata.width,
                    frame_height=index.metadata.height,
                )
            )
            player_observation = PlayerObservation(
                frame_index=frame_index,
                source_track_id=item.track_id,
                track_id=item.track_id,
                frame_bbox=bbox,
                frame_foot_pos=frame_foot_pos,
                court_pos=court_pos,
                confidence=_unit(item.score),
            )
            player: dict[str, Any] = {
                "track_id": item.track_id,
                "confidence": _unit(item.score),
                "frame_bbox": {"x1": bbox[0], "y1": bbox[1], "x2": bbox[2], "y2": bbox[3]},
                "frame_foot_pos": {"x": frame_foot_pos[0], "y": frame_foot_pos[1]},
                "court_pos": (
                    None if court_pos is None else {"x": court_pos[0], "y": court_pos[1]}
                ),
                "action_label": None,
                "action_confidence": None,
            }
            action = detection.get("action") if detection is not None else None
            if isinstance(action, dict):
                action_data = cast("dict[str, Any]", action)
                if isinstance(action_data.get("name"), str):
                    player["action_label"] = action_data["name"]
                    player["action_confidence"] = _unit(action_data.get("confidence", 0.0))
                    actions[(frame_index, item.track_id)] = ActionObservation(
                        frame_index=frame_index,
                        track_id=item.track_id,
                        label=str(action_data["name"]),
                        confidence=_unit(action_data.get("confidence", 0.0)),
                    )
            players.append(player)
            frame_players.append(player_observation)
            frame_poses.append(
                _pose_observation(
                    detection,
                    frame_index=frame_index,
                    track_id=item.track_id,
                    bbox=bbox,
                    width=index.metadata.width,
                    height=index.metadata.height,
                )
            )
            stats = track_stats[item.track_id]
            stats["first"] = min(stats["first"], frame_index)
            stats["last"] = max(stats["last"], frame_index)
            stats["confidence_sum"] += _unit(item.score)
            stats["count"] += 1
        frame_records[frame_index] = {"frame_index": frame_index, "players": players}
        players_by_frame[frame_index] = tuple(frame_players)
        poses[frame_index] = tuple(frame_poses)

        raw_ball = row.get("ball")
        if isinstance(raw_ball, dict):
            ball = cast("dict[str, Any]", raw_ball)
            center = ball.get("center_xy")
            center_values = cast("list[Any]", center) if isinstance(center, list) else []
            if ball.get("visible") and len(center_values) == 2:
                x_value, y_value = center_values
                if x_value is not None and y_value is not None:
                    ball_positions[frame_index] = {
                        "x": _unit(float(x_value) / index.metadata.width),
                        "y": _unit(float(y_value) / index.metadata.height),
                        "confidence": _unit(ball.get("confidence", 0.0)),
                    }
                    ball_observations[frame_index] = BallObservation(
                        frame_index=frame_index,
                        frame_pos=(
                            _unit(float(x_value) / index.metadata.width),
                            _unit(float(y_value) / index.metadata.height),
                        ),
                        confidence=_unit(ball.get("confidence", 0.0)),
                    )

        activity = row.get("group_activity")
        if isinstance(activity, dict):
            activity_data = cast("dict[str, Any]", activity)
            activity_samples.append(
                (
                    frame_index,
                    str(activity_data.get("name", "unknown")),
                    _unit(activity_data.get("confidence", 0.0)),
                )
            )

    track_sides = resolve_track_court_sides(
        [
            FrameObservation(
                frame_index=frame_index,
                players=players,
                homography_available=frame_index in homographies,
            )
            for frame_index, players in sorted(players_by_frame.items())
        ]
    )
    tracks = [
        {
            "track_id": track_id,
            "court_side": track_sides.get(track_id, "unknown"),
            "first_frame_index": str(stats["first"]),
            "last_frame_index": str(stats["last"]),
            "mean_confidence": stats["confidence_sum"] / stats["count"],
            "metadata": {"identity_basis": "geometry_only_import_tracking"},
        }
        for track_id, stats in sorted(track_stats.items())
    ]
    activity_counts = Counter(label for _, label, _ in activity_samples)
    fps = int(job.clip.video.fps.num) / int(job.clip.video.fps.den)
    phase_candidates = _contact_phase_candidates(activity_samples, fps=fps)
    ball_proposals = detect_contact_proposals(
        ball_observations,
        start_frame=0,
        end_frame=max(0, total_frames - 1),
        fps=fps,
    )
    contact_events = _build_contact_events(
        job=job,
        phase_candidates=phase_candidates,
        proposals=ball_proposals,
        balls=ball_observations,
        players=players_by_frame,
        actions=actions,
        poses=poses,
        homographies=homographies,
        fps=fps,
        frame_width=index.metadata.width,
        frame_height=index.metadata.height,
    )
    path_segments = cast("Any", AnalysisPipeline)._build_paths(contact_events)
    analysis_id = f"predictions-import:{job.ai_job_id}"
    domain = AnalysisDomainData.model_validate(
        {
            "schema_version": "1.0.0",
            "analysis_id": analysis_id,
            "analysis_version": IMPORTER_BUILD_ID,
            "ai_job_id": job.ai_job_id,
            "rally_submission_id": job.rally_submission_id,
            "rally_id": job.rally_id,
            "match_id": job.match_id,
            "annotation_revision": job.annotation_revision,
            "clip_asset_id": job.clip.clip_asset_id,
            "input_clip_sha256": job.clip.sha256,
            "producer": {"name": "volleyball-analysis-engine", "build_id": IMPORTER_BUILD_ID},
            "tracks": tracks,
            "contact_events": contact_events,
            "path_segments": path_segments,
            "summary": {
                "track_count": len(tracks),
                "contact_event_count": len(contact_events),
                "path_segment_count": len(path_segments),
                "unresolved_event_count": sum(
                    event["association_state"] in {"ambiguous", "unresolved"}
                    for event in contact_events
                ),
                "multiple_event_count": sum(
                    event["association_state"] == "resolved_multiple" for event in contact_events
                ),
                "warnings": [
                    (
                        "contact times are AI-detected from stable legacy group-activity "
                        "transitions and require review"
                    ),
                    "legacy predictions import has no authoritative manual contact anchors",
                    "track identities are geometry-only and analysis-run-local",
                    "court sides and paths are projected from legacy Court60 keypoints",
                ],
            },
            "extensions": {
                "predictions_import": {
                    "schema_version": "1.0.0",
                    "source_file": index.source_path.name,
                    "source_frame_range": {"start": str(start), "end_exclusive": str(end)},
                    "source_checkpoint_declaration": index.metadata.checkpoint,
                    "group_activity_counts": dict(sorted(activity_counts.items())),
                    "group_activity_runs": _activity_runs(activity_samples),
                    "contact_phase_candidates": [
                        {
                            "frame_index": str(candidate.frame_index),
                            "end_frame_index": str(candidate.end_frame_index),
                            "label": candidate.label,
                            "confidence": candidate.confidence,
                        }
                        for candidate in phase_candidates
                    ],
                    "ball_flight_proposal_count": len(ball_proposals),
                    "left_court_side": "left",
                    "right_court_side": "right",
                    "homography_frame_count": len(homographies),
                }
            },
        }
    )
    validate_passthrough(job, domain)
    analysis_data = build_analysis_data(
        job,
        domain=domain,
        frame_records=[frame_records[index] for index in sorted(frame_records)],
        ball_positions=ball_positions,
        court_keypoints=court_keypoints,
        action_taxonomy_id=ACTION_TAXONOMY_ID,
        action_taxonomy_version=ACTION_TAXONOMY_VERSION,
    )
    recipe_hash = hashlib.sha256(
        f"{POSE_RECIPE_NAMESPACE}:{IMPORTER_BUILD_ID}".encode()
    ).hexdigest()
    evidence = build_analysis_evidence_artifacts(
        job=job,
        analysis_run_id=analysis_id,
        analysis_data_bytes=analysis_data,
        poses=poses,
        pose_recipe={
            "namespace": POSE_RECIPE_NAMESPACE,
            "model_name": "legacy-predictions-json-import",
            "checkpoint_sha256": recipe_hash,
            "preprocess_version": IMPORTER_BUILD_ID,
            "keypoint_layout": "COCO_17",
            "coordinate_space": "NORMALIZED_VIDEO",
        },
    )
    return ConvertedAnalysis(
        analysis_id=analysis_id,
        domain=domain,
        artifacts=evidence.artifacts,
        source_frame_count=end - start,
    )


def _analysis_job(
    context: ProviderWorkContext,
    request: ProviderAnalysisJobRequest,
) -> AIJobRequest:
    clips = [item for item in context.work.input_artifacts if item.kind == "CANONICAL_CLIP"]
    if len(clips) != 1:
        raise ValueError("predictions import requires exactly one CANONICAL_CLIP artifact")
    artifact = clips[0]
    return AIJobRequest.model_validate(
        {
            "schema_version": "3.0.0",
            "ai_job_id": request.ai_job_id,
            "rally_submission_id": request.rally_submission_id,
            "rally_id": request.rally_id,
            "match_id": request.match_id,
            "annotation_revision": request.annotation_revision,
            "clip": {
                "clip_asset_id": request.clip.clip_asset_id,
                "download_url": artifact.download_url,
                "download_url_expires_at": artifact.download_url_expires_at,
                "sha256": artifact.sha256,
                "byte_length": artifact.byte_length,
                "content_type": artifact.content_type,
                "video": request.clip.video.model_dump(mode="json"),
            },
            "key_points": [point.model_dump(mode="json") for point in request.key_points],
            "boundaries": [point.model_dump(mode="json") for point in request.boundaries],
            "outcome": request.outcome.model_dump(mode="json"),
            "analysis_plan": {
                "mode": "full",
                "modules": {"court": "run", "tracking": "run", "reid": "run", "contacts": "run"},
                "source_analysis_data": None,
                "preserve_manual_corrections": True,
            },
            "callback": {
                "url": context.work.callback.url,
                "token": context.work.callback.token,
                "expires_at": context.work.callback.expires_at,
            },
        }
    )


class _PlanBoundWorkerClient(ProviderWorkerClient):
    """Reject jobs outside this import plan before the SDK accepts their leases."""

    def __init__(self, config: ProviderWorkerConfig, *, plan_path: Path) -> None:
        super().__init__(config)
        self._plan_path = plan_path

    async def _accept_offer(self, offer: Any, handlers: Any) -> None:
        rally_id = str(offer.work.request.get("rally_id", ""))
        plan = load_plan(self._plan_path)
        allowed = {
            str(item["rally_id"])
            for item in cast("list[dict[str, Any]]", plan["segments"])
            if item.get("rally_id")
        }
        if rally_id not in allowed:
            await self._reject(
                offer,
                "WORK_NOT_BOUND_TO_IMPORT_PLAN",
                "predictions importer only accepts rallies recorded in its plan",
                retryable=True,
            )
            return
        await super()._accept_offer(offer, handlers)


def provider_capabilities(provider_build_id: str) -> ProviderWorkCapabilities:
    """Advertise an ANALYSIS-only worker that consumes the existing clip contract."""
    return ProviderWorkCapabilities.model_validate(
        {
            "schema_version": "3.0.0",
            "provider_name": "volleyball-analysis-engine-predictions-import",
            "provider_build_id": provider_build_id,
            "work_capabilities": [
                {
                    "work_kind": "ANALYSIS",
                    "request_schema_versions": ["1.0.0"],
                    "result_schema_versions": ["1.0.0"],
                    "accepted_input_artifact_kinds": ["CANONICAL_CLIP"],
                    "produced_artifact_kinds": [
                        "ANALYSIS_DATA",
                        "ANALYSIS_EVIDENCE_MANIFEST",
                        "PERSON_POSE_EVIDENCE_MANIFEST",
                        "PERSON_POSE_EVIDENCE_CHUNK",
                        "PLAYER_CROP_SOURCE_MANIFEST",
                    ],
                    "model_recipe_namespaces": [POSE_RECIPE_NAMESPACE],
                    "hardware": {"accelerator": "ANY"},
                    "max_concurrency": 1,
                }
            ],
        }
    )


def create_import_worker(
    *,
    plan_path: Path,
    server_ws_url: str,
    token: str,
    workspace: Path,
    instance_id: str | None = None,
) -> tuple[ProviderWorkerClient, dict[str, Callable[[ProviderWorkContext], Any]]]:
    """Build a Provider Work client and its isolated predictions handler."""
    plan = load_plan(plan_path)
    index = load_prediction_index(Path(str(plan["index_path"])))
    provider_build_id = f"volleyball-analysis-engine/{IMPORTER_BUILD_ID}"
    options: dict[str, Any] = {
        "server_ws_url": server_ws_url,
        "token": token,
        "workspace": workspace,
        "provider_build_id": provider_build_id,
        "capabilities": provider_capabilities(provider_build_id),
    }
    if instance_id:
        options["instance_id"] = instance_id
    client = _PlanBoundWorkerClient(ProviderWorkerConfig(**options), plan_path=plan_path)

    async def handle_analysis(context: ProviderWorkContext) -> None:
        request = ProviderAnalysisJobRequest.model_validate(context.work.request)
        if request.provider_job_id != context.work.provider_job_id:
            raise ValueError("provider analysis request/job ID mismatch")
        current_plan = load_plan(plan_path)
        segment = next(
            (
                item
                for item in cast("list[dict[str, Any]]", current_plan["segments"])
                if item.get("rally_id") == request.rally_id
            ),
            None,
        )
        if segment is None:
            raise ValueError(f"rally {request.rally_id} is not bound to this import plan")
        await context.report_progress(0.05, "reading_predictions_segment")
        converted = await asyncio.to_thread(
            convert_segment,
            index=index,
            segment=segment,
            job=_analysis_job(context, request),
        )
        await context.report_progress(0.9, "uploading_imported_analysis")
        await context.complete(
            result_schema_version="1.0.0",
            artifacts=list(converted.artifacts),
        )
        status_dir = workspace / "status"
        status_dir.mkdir(parents=True, exist_ok=True)
        (status_dir / f"{request.rally_id}.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "rally_id": request.rally_id,
                    "submission_id": request.rally_submission_id,
                    "analysis_id": converted.analysis_id,
                    "source_frame_count": converted.source_frame_count,
                    "completed_at": datetime.now(UTC).isoformat(),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    return client, {"ANALYSIS": handle_analysis}
