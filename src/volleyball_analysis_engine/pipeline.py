"""Reference-backed analysis pipeline with canonical frame alignment."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray
from volleyball_monitoring_ai import (
    AIJobRequest,
    AnalysisBundle,
    AnalysisResult,
    build_tracking_overlay,
    validate_passthrough,
)

from .association import associate_hit, classify_action
from .geometry import estimate_homography, project_normalized_frame_point
from .records import BallObservation, FrameObservation, PlayerObservation
from .reference_data import load_ball_positions, load_court_frames, load_player_frames
from .reid import CourtPositionReidentifier

ProgressReporter = Callable[[float, str], None]


def _noop_progress(_progress: float, _stage: str) -> None:
    """Accept progress updates when no reporter was supplied."""


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Paths and thresholds for the replaceable reference backend."""

    fixture_root: Path
    tracking_variant: str = "sam-deep-eiou"
    court_confidence_threshold: float = 0.25
    reid_max_distance: float = 0.35
    fixture_frame_width: int = 640
    fixture_frame_height: int = 360

    @property
    def tracking_path(self) -> Path:
        """Return the selected recorded player-tracking fixture."""
        variants = {
            "deep-eiou": "tracks-deep-eiou.jsonl",
            "sam-deep-eiou": "tracks-sam-deep-eiou.jsonl",
        }
        try:
            file_name = variants[self.tracking_variant]
        except KeyError as exc:
            msg = f"unknown tracking variant: {self.tracking_variant}"
            raise ValueError(msg) from exc
        return self.fixture_root / "tracking-data" / file_name

    @property
    def court_path(self) -> Path:
        """Return the recorded court-keypoint fixture."""
        return self.fixture_root / "tracking-data" / "court-keypoints.jsonl"

    @property
    def ball_path(self) -> Path:
        """Return the temporary manual ball fixture."""
        return self.fixture_root / "input" / "ball-annotations.manual.json"


class AnalysisPipeline:
    """Produce contract-valid result and overlay artifacts for an incoming job."""

    analysis_version = "reference-geometry-reid-0.1.0"

    def __init__(self, config: PipelineConfig) -> None:
        """Configure one reusable reference-backed analysis pipeline."""
        self.config = config

    def analyze(
        self,
        job: AIJobRequest,
        report: ProgressReporter | None = None,
    ) -> AnalysisBundle:
        """Analyze one canonical clip while preserving every immutable anchor."""
        reporter: ProgressReporter = report or _noop_progress
        reporter(0.10, "loading_reference_data")
        source_players = load_player_frames(self.config.tracking_path)
        source_courts = load_court_frames(self.config.court_path)
        source_balls = load_ball_positions(self.config.ball_path)
        source_last_frame = max(
            max(source_players, default=0),
            max(source_courts, default=0),
            max(source_balls, default=0),
        )
        destination_frames = int(job.clip.video.total_frames)
        if destination_frames < 1:
            msg = "canonical clip must contain at least one frame"
            raise ValueError(msg)

        reporter(0.22, "court_projection")
        frames, homographies = self._project_frames(
            source_players,
            source_courts,
            source_last_frame=source_last_frame,
            destination_frames=destination_frames,
            frame_width=self.config.fixture_frame_width,
            frame_height=self.config.fixture_frame_height,
        )
        balls = self._map_balls(source_balls, source_last_frame, destination_frames)

        reporter(0.38, "player_tracking")
        reporter(0.48, "reidentification")
        frames = CourtPositionReidentifier(
            max_per_side=6,
            max_distance=self.config.reid_max_distance,
        ).apply(frames)
        players_by_frame = {frame.frame_index: frame.players for frame in frames}

        reporter(0.62, "hit_association")
        events = self._build_events(
            job,
            balls,
            players_by_frame,
            homographies,
        )
        self._apply_actions(events)
        tracks = self._build_tracks(frames)
        paths = self._build_paths(events)

        reporter(0.82, "building_artifacts")
        analysis_id = str(uuid4())
        unresolved = sum(
            event["association_state"] in {"ambiguous", "unresolved"} for event in events
        )
        result = AnalysisResult.model_validate(
            {
                "schema_version": "1.0.0",
                "analysis_id": analysis_id,
                "analysis_version": self.analysis_version,
                "ai_job_id": job.ai_job_id,
                "rally_submission_id": job.rally_submission_id,
                "rally_id": job.rally_id,
                "match_id": job.match_id,
                "annotation_revision": job.annotation_revision,
                "clip_asset_id": job.clip.clip_asset_id,
                "input_clip_sha256": job.clip.sha256,
                "producer": {
                    "name": "volleyball-analysis-engine",
                    "build_id": self.analysis_version,
                    "sdk_version": "0.2.0",
                },
                "tracks": tracks,
                "contact_events": events,
                "path_segments": paths,
                "summary": {
                    "track_count": len(tracks),
                    "contact_event_count": len(events),
                    "path_segment_count": len(paths),
                    "unresolved_event_count": unresolved,
                    "multiple_event_count": 0,
                    "warnings": [
                        "ball positions are read from the temporary fixed JSON fixture",
                        "actions use the temporary adjacent A/B court-position heuristic",
                    ],
                },
                "extensions": {
                    "reference_backend": "volleyball-ai-contract-lab/ai-team-handoff",
                    "court_detection": "recorded-keypoints+randsac-homography",
                    "tracking": self.config.tracking_variant,
                    "reid": "nearest-reentry-in-unclamped-2d-court-space",
                    "action_source": "adjacent-a-b-court-position-heuristic",
                    "fixture_source_frame_count": source_last_frame + 1,
                    "canonical_frame_count": destination_frames,
                },
            }
        )
        validate_passthrough(job, result)
        overlay = build_tracking_overlay(
            job,
            analysis_id=analysis_id,
            analysis_version=self.analysis_version,
            frame_records=self._overlay_records(frames),
            ball_positions={
                frame_index: {"x": ball.frame_pos[0], "y": ball.frame_pos[1]}
                for frame_index, ball in balls.items()
            },
        )
        reporter(0.94, "callback")
        return AnalysisBundle(result=result, overlay_bytes=overlay)

    def _project_frames(
        self,
        source_players: dict[int, tuple[PlayerObservation, ...]],
        source_courts: dict[int, Any],
        *,
        source_last_frame: int,
        destination_frames: int,
        frame_width: int,
        frame_height: int,
    ) -> tuple[list[FrameObservation], dict[int, NDArray[np.float64]]]:
        mapped: dict[int, FrameObservation] = {}
        homographies: dict[int, NDArray[np.float64]] = {}
        last_homography: NDArray[np.float64] | None = None
        for source_frame in sorted(source_players):
            destination_frame = self._map_frame(
                source_frame,
                source_last_frame,
                destination_frames,
            )
            court_frame = source_courts.get(source_frame)
            homography = (
                None
                if court_frame is None
                else estimate_homography(
                    court_frame,
                    confidence_threshold=self.config.court_confidence_threshold,
                )
            )
            if homography is not None:
                last_homography = homography
            effective_homography = homography if homography is not None else last_homography
            if effective_homography is not None:
                homographies[destination_frame] = effective_homography
            players = tuple(
                PlayerObservation(
                    frame_index=destination_frame,
                    source_track_id=player.source_track_id,
                    track_id=player.track_id,
                    frame_bbox=player.frame_bbox,
                    frame_foot_pos=player.frame_foot_pos,
                    court_pos=(
                        player.court_pos
                        if effective_homography is None
                        else project_normalized_frame_point(
                            player.frame_foot_pos,
                            effective_homography,
                            frame_width=frame_width,
                            frame_height=frame_height,
                        )
                    ),
                    confidence=player.confidence,
                )
                for player in source_players[source_frame]
            )
            mapped[destination_frame] = FrameObservation(
                frame_index=destination_frame,
                players=players,
                homography_available=effective_homography is not None,
            )
        return [mapped[index] for index in sorted(mapped)], homographies

    @classmethod
    def _map_balls(
        cls,
        source: dict[int, BallObservation],
        source_last_frame: int,
        destination_frames: int,
    ) -> dict[int, BallObservation]:
        mapped: dict[int, BallObservation] = {}
        for source_frame, ball in source.items():
            frame_index = cls._map_frame(source_frame, source_last_frame, destination_frames)
            mapped[frame_index] = BallObservation(frame_index=frame_index, frame_pos=ball.frame_pos)
        return mapped

    @staticmethod
    def _map_frame(frame: int, source_last_frame: int, destination_frames: int) -> int:
        if source_last_frame <= 0 or destination_frames <= 1:
            return 0
        return min(
            destination_frames - 1,
            max(0, round(frame * (destination_frames - 1) / source_last_frame)),
        )

    def _build_events(
        self,
        job: AIJobRequest,
        balls: dict[int, BallObservation],
        players: dict[int, tuple[PlayerObservation, ...]],
        homographies: dict[int, NDArray[np.float64]],
    ) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        total_frames = int(job.clip.video.total_frames)
        for index, point in enumerate(job.key_points):
            anchor = int(point.clip_frame_index)
            next_anchor = (
                int(job.key_points[index + 1].clip_frame_index)
                if index + 1 < len(job.key_points)
                else total_frames
            )
            association = associate_hit(
                anchor_frame=anchor,
                next_anchor_frame=next_anchor,
                is_terminal=point.is_terminal,
                balls=balls,
                players=players,
                frame_width=self.config.fixture_frame_width,
                frame_height=self.config.fixture_frame_height,
            )
            actor = self._actor(
                association.player, association.observation_frame, association.confidence
            )
            representative = self._representative_position(
                association.player,
                association.ball,
                association.observation_frame,
                homographies,
                frame_width=self.config.fixture_frame_width,
                frame_height=self.config.fixture_frame_height,
                terminal=point.is_terminal,
            )
            events.append(
                {
                    "key_point_id": point.key_point_id,
                    "sequence_index": point.sequence_index,
                    "marker_kind": point.marker_kind,
                    "is_terminal": point.is_terminal,
                    # This exact passthrough is the PTS/frame alignment boundary.
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
                            "confidence": 1.0,
                        }
                    ),
                    "representative_court_positions": (
                        [] if representative is None else [representative]
                    ),
                    "quality_flags": ["fixture_ball_data", association.mode],
                    "extensions": {
                        "authoritative_clip_pts": point.clip_pts,
                        "authoritative_clip_time_us": point.clip_time_us,
                    },
                }
            )
        return events

    @staticmethod
    def _actor(
        player: PlayerObservation | None,
        observation_frame: int | None,
        confidence: float | None,
    ) -> dict[str, Any] | None:
        if player is None or observation_frame is None:
            return None
        return {
            "track_id": player.track_id,
            "observation_frame_index": str(observation_frame),
            "association_confidence": confidence,
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
        }

    @staticmethod
    def _representative_position(
        player: PlayerObservation | None,
        ball: BallObservation | None,
        observation_frame: int | None,
        homographies: dict[int, NDArray[np.float64]],
        *,
        frame_width: int,
        frame_height: int,
        terminal: bool,
    ) -> dict[str, Any] | None:
        if player is not None and player.court_pos is not None:
            return {
                "track_id": player.track_id,
                "basis": "player_footprint_proxy",
                "court_pos": {"x": player.court_pos[0], "y": player.court_pos[1]},
                "confidence": 0.85,
            }
        if not terminal or ball is None or observation_frame is None:
            return None
        homography = homographies.get(observation_frame)
        if homography is None and homographies:
            nearest_frame = min(homographies, key=lambda frame: abs(frame - observation_frame))
            homography = homographies[nearest_frame]
        if homography is None:
            return None
        court_pos = project_normalized_frame_point(
            ball.frame_pos,
            homography,
            frame_width=frame_width,
            frame_height=frame_height,
        )
        return {
            "track_id": None,
            "basis": "terminal_projection",
            "court_pos": {"x": court_pos[0], "y": court_pos[1]},
            "confidence": 0.65,
        }

    @staticmethod
    def _apply_actions(events: list[dict[str, Any]]) -> None:
        for index, event in enumerate(events[:-1]):
            actors = cast("list[dict[str, Any]]", event["actors"])
            if len(actors) != 1:
                continue
            positions = cast("list[dict[str, Any]]", event["representative_court_positions"])
            next_positions = cast(
                "list[dict[str, Any]]",
                events[index + 1]["representative_court_positions"],
            )
            start = AnalysisPipeline._court_tuple(positions)
            end = AnalysisPipeline._court_tuple(next_positions)
            classification = classify_action(
                start,
                end,
                is_service=event["marker_kind"] == "service",
            )
            if classification is None:
                continue
            label, crosses_court = classification
            actors[0]["action"] = {
                "label": label,
                "taxonomy_id": "volleyball-analysis-engine.ball-path-heuristic",
                "taxonomy_version": "1",
                "confidence": 0.72,
                "attributes": {
                    "heuristic": True,
                    "source": "adjacent_event_court_positions",
                    "crosses_court": crosses_court,
                },
            }

    @staticmethod
    def _court_tuple(positions: list[dict[str, Any]]) -> tuple[float, float] | None:
        if not positions:
            return None
        court = cast("dict[str, Any]", positions[0]["court_pos"])
        return float(court["x"]), float(court["y"])

    @staticmethod
    def _build_tracks(frames: list[FrameObservation]) -> list[dict[str, Any]]:
        observations: dict[int, list[PlayerObservation]] = defaultdict(list)
        for frame in frames:
            for player in frame.players:
                observations[player.track_id].append(player)
        tracks: list[dict[str, Any]] = []
        for track_id, players in sorted(observations.items()):
            frames_seen = [player.frame_index for player in players]
            side_counts: dict[str, int] = defaultdict(int)
            for player in players:
                side_counts[player.court_side] += 1
            court_side = max(side_counts, key=side_counts.__getitem__, default="unknown")
            confidence_values = [
                player.confidence for player in players if player.confidence is not None
            ]
            tracks.append(
                {
                    "track_id": track_id,
                    "court_side": court_side,
                    "first_frame_index": str(min(frames_seen)),
                    "last_frame_index": str(max(frames_seen)),
                    "mean_confidence": (
                        None
                        if not confidence_values
                        else sum(confidence_values) / len(confidence_values)
                    ),
                    "metadata": {
                        "reid_basis": "2d_court_position",
                        "source_track_ids": sorted({player.source_track_id for player in players}),
                    },
                }
            )
        return tracks

    @staticmethod
    def _build_paths(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        paths: list[dict[str, Any]] = []
        for index, (start, end) in enumerate(pairwise(events)):
            start_positions = cast("list[dict[str, Any]]", start["representative_court_positions"])
            end_positions = cast("list[dict[str, Any]]", end["representative_court_positions"])
            paths.append(
                {
                    "sequence_index": index,
                    "start_key_point_id": start["key_point_id"],
                    "end_key_point_id": end["key_point_id"],
                    "start_frame_index": start["anchor_frame_index"],
                    "end_frame_index": end["anchor_frame_index"],
                    "start_court_positions": start_positions,
                    "end_court_positions": end_positions,
                    "render_state": (
                        "complete" if start_positions and end_positions else "unavailable"
                    ),
                    "is_terminal_segment": bool(end["is_terminal"]),
                    "quality_flags": ["fixture_ball_data", "canonical_anchor_frames"],
                }
            )
        return paths

    @staticmethod
    def _overlay_records(frames: list[FrameObservation]) -> list[dict[str, Any]]:
        return [
            {
                "frame_index": frame.frame_index,
                "players": [
                    {
                        "track_id": player.track_id,
                        "frame_bbox": {
                            "x1": player.frame_bbox[0],
                            "y1": player.frame_bbox[1],
                            "x2": player.frame_bbox[2],
                            "y2": player.frame_bbox[3],
                        },
                        "frame_foot_pos": {
                            "x": player.frame_foot_pos[0],
                            "y": player.frame_foot_pos[1],
                        },
                        "court_pos": (
                            None
                            if player.court_pos is None
                            else {"x": player.court_pos[0], "y": player.court_pos[1]}
                        ),
                    }
                    for player in frame.players
                ],
            }
            for frame in frames
        ]
