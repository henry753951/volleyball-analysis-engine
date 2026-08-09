"""Model-backed analysis pipeline with canonical frame alignment."""

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

from .artifacts import write_inference_artifacts
from .association import associate_hit
from .geometry import estimate_homography, project_normalized_frame_point
from .inference import ObservationProvider
from .records import ActionObservation, BallObservation, FrameObservation, PlayerObservation
from .reid import CourtPositionReidentifier

ProgressReporter = Callable[[float, str], None]


def _noop_progress(_progress: float, _stage: str) -> None:
    """Accept progress updates when no reporter was supplied."""


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Geometry and identity thresholds independent of a specific model backend."""

    court_confidence_threshold: float = 0.25
    reid_max_distance: float = 0.35


class AnalysisPipeline:
    """Produce contract-valid result and overlay artifacts for an incoming job."""

    analysis_version = "rtv4-x3d-court-reid-visual-v5-0.3.0"

    def __init__(
        self,
        provider: ObservationProvider,
        config: PipelineConfig | None = None,
    ) -> None:
        """Configure one reusable model-backed analysis pipeline."""
        self.provider = provider
        self.config = config or PipelineConfig()

    def analyze(
        self,
        job: AIJobRequest,
        clip_path: Path,
        report: ProgressReporter | None = None,
        artifact_dir: Path | None = None,
    ) -> AnalysisBundle:
        """Analyze one canonical clip while preserving every immutable anchor."""
        reporter: ProgressReporter = report or _noop_progress
        inferred = self.provider.infer(clip_path, job, reporter)
        source_last_frame = inferred.frame_count - 1
        destination_frames = int(job.clip.video.total_frames)
        if destination_frames < 1:
            msg = "canonical clip must contain at least one frame"
            raise ValueError(msg)

        reporter(0.72, "court_projection")
        frames, homographies = self._project_frames(
            inferred.players,
            inferred.courts,
            source_last_frame=source_last_frame,
            destination_frames=destination_frames,
            frame_width=inferred.frame_width,
            frame_height=inferred.frame_height,
        )
        balls = self._map_balls(inferred.balls, source_last_frame, destination_frames)

        reporter(0.76, "court_reidentification")
        frames = CourtPositionReidentifier(
            max_per_side=6,
            max_distance=self.config.reid_max_distance,
        ).apply(frames)
        players_by_frame = {frame.frame_index: frame.players for frame in frames}

        reporter(0.82, "hit_association")
        events = self._build_events(
            job,
            balls,
            players_by_frame,
            homographies,
            actions=inferred.actions,
            frame_width=inferred.frame_width,
            frame_height=inferred.frame_height,
        )
        tracks = self._build_tracks(frames)
        paths = self._build_paths(events)

        reporter(0.87, "building_wire_artifacts")
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
                    "sdk_version": "0.3.0",
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
                    "warnings": [],
                },
                "extensions": {
                    "inference_source": "canonical_clip",
                    "court_detection": "YOLO26-pose+ransac-homography",
                    "tracking": "harmonic-mean-eiou+OSNet",
                    "reid": "nearest-reentry-in-unclamped-2d-court-space",
                    "action_source": "RT-DETRv4-X3D",
                    "provider_metadata": inferred.metadata,
                    "decoded_source_frame_count": source_last_frame + 1,
                    "canonical_frame_count": destination_frames,
                },
            }
        )
        validate_passthrough(job, result)
        if artifact_dir is not None:
            reporter(0.91, "writing_visual_v5_artifacts")
            write_inference_artifacts(
                output_dir=artifact_dir,
                clip_path=clip_path,
                job=job,
                result=result,
                frames=frames,
                balls=balls,
                courts=inferred.courts,
                actions=inferred.actions,
                fps=inferred.fps,
                frame_width=inferred.frame_width,
                frame_height=inferred.frame_height,
            )
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
        reporter(0.98, "analysis_bundle_ready")
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
            mapped[frame_index] = BallObservation(
                frame_index=frame_index,
                frame_pos=ball.frame_pos,
                confidence=ball.confidence,
            )
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
        *,
        actions: dict[tuple[int, int], ActionObservation],
        frame_width: int,
        frame_height: int,
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
                frame_width=frame_width,
                frame_height=frame_height,
            )
            actor = self._actor(
                association.player, association.observation_frame, association.confidence
            )
            representative = self._representative_position(
                association.player,
                association.ball,
                association.observation_frame,
                homographies,
                frame_width=frame_width,
                frame_height=frame_height,
                terminal=point.is_terminal,
            )
            if actor is not None and association.player is not None:
                action = self._nearest_action(
                    actions,
                    frame_index=association.observation_frame,
                    track_id=association.player.source_track_id,
                )
                if action is not None:
                    actor["action"] = {
                        "label": action.label,
                        "taxonomy_id": "volleyball-analysis-engine.rtv4-x3d-actions",
                        "taxonomy_version": "1",
                        "confidence": action.confidence,
                        "attributes": {"source": "RT-DETRv4-X3D"},
                    }
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
                            "confidence": association.ball.confidence,
                        }
                    ),
                    "representative_court_positions": (
                        [] if representative is None else [representative]
                    ),
                    "quality_flags": [association.mode],
                    "extensions": {
                        "authoritative_clip_pts": point.clip_pts,
                        "authoritative_clip_time_us": point.clip_time_us,
                    },
                }
            )
        return events

    @staticmethod
    def _nearest_action(
        actions: dict[tuple[int, int], ActionObservation],
        *,
        frame_index: int | None,
        track_id: int,
        radius: int = 3,
    ) -> ActionObservation | None:
        if frame_index is None:
            return None
        candidates = [
            action
            for (candidate_frame, candidate_track), action in actions.items()
            if candidate_track == track_id and abs(candidate_frame - frame_index) <= radius
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda action: abs(action.frame_index - frame_index))

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
