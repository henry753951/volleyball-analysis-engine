"""Model-backed analysis pipeline with canonical frame alignment."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, replace
from itertools import pairwise
from pathlib import Path
from typing import Any, cast
from uuid import NAMESPACE_URL, uuid4, uuid5

import numpy as np
from numpy.typing import NDArray
from volleyball_monitoring_ai import (
    AIJobRequest,
    AnalysisDataBundle,
    AnalysisDomainData,
    build_analysis_data,
    validate_passthrough,
)
from volleyball_monitoring_ai.models import KeyPointInput

from .artifacts import write_inference_artifacts
from .association import associate_hit
from .contact_detection import detect_contact_proposals
from .evidence_artifacts import AnalysisEvidenceArtifacts, build_analysis_evidence_artifacts
from .geometry import estimate_homography, project_normalized_frame_point
from .inference import ObservationProvider
from .records import (
    ActionObservation,
    BallObservation,
    CourtFrame,
    FrameObservation,
    PersonPoseObservation,
    PlayerObservation,
)

ProgressReporter = Callable[[float, str], None]


def _noop_progress(_progress: float, _stage: str) -> None:
    """Accept progress updates when no reporter was supplied."""


def _resolve_track_court_sides(frames: list[FrameObservation]) -> dict[int, str]:
    """Resolve a run-local track's stable court side without roster assumptions."""
    counts: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for frame in frames:
        for player in frame.players:
            counts[player.track_id][player.court_side] += 1
    resolved: dict[int, str] = {}
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


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Geometry and association thresholds independent of a specific model backend."""

    court_confidence_threshold: float = 0.25
    association_search_seconds: float = 0.25


@dataclass(frozen=True, slots=True)
class _EventSpec:
    anchor: int
    key_point_id: str
    source_key_point_id: str | None
    anchor_origin: str
    marker_kind: str
    is_terminal: bool
    detection_confidence: float | None
    point: KeyPointInput | None
    detection: dict[str, str | float] | None


@dataclass(frozen=True, slots=True)
class AnalysisProviderWorkResult:
    """AnalysisData and every immutable output required by Provider Work v2."""

    bundle: AnalysisDataBundle
    evidence: AnalysisEvidenceArtifacts


@dataclass(frozen=True, slots=True)
class _AnalysisExecution:
    bundle: AnalysisDataBundle
    evidence: AnalysisEvidenceArtifacts | None


class AnalysisPipeline:
    """Produce one contract-valid AnalysisData artifact for an incoming job."""

    analysis_version = "volleyball-multitask-v2-contact-0.9.0"

    def __init__(
        self,
        provider: ObservationProvider,
        config: PipelineConfig | None = None,
    ) -> None:
        """Configure one reusable model-backed analysis pipeline."""
        self.provider = provider
        self.config = config or PipelineConfig()

    def prepare(self, report: ProgressReporter | None = None) -> None:
        """Load persistent model state before accepting a worker lease."""
        prepare = getattr(self.provider, "prepare", None)
        if callable(prepare):
            prepare(report or _noop_progress)

    def analyze(
        self,
        job: AIJobRequest,
        clip_path: Path,
        report: ProgressReporter | None = None,
        artifact_dir: Path | None = None,
    ) -> AnalysisDataBundle:
        """Run the offline AnalysisData-only entrypoint without identity matching."""
        return self._analyze(
            job,
            clip_path,
            report=report,
            artifact_dir=artifact_dir,
            include_evidence=False,
        ).bundle

    def analyze_provider_work(
        self,
        job: AIJobRequest,
        clip_path: Path,
        report: ProgressReporter | None = None,
        artifact_dir: Path | None = None,
    ) -> AnalysisProviderWorkResult:
        """Run base analysis and emit independently reusable every-frame evidence."""
        execution = self._analyze(
            job,
            clip_path,
            report=report,
            artifact_dir=artifact_dir,
            include_evidence=True,
        )
        if execution.evidence is None:
            message = "provider analysis did not create required evidence"
            raise RuntimeError(message)
        return AnalysisProviderWorkResult(bundle=execution.bundle, evidence=execution.evidence)

    def _analyze(
        self,
        job: AIJobRequest,
        clip_path: Path,
        *,
        report: ProgressReporter | None,
        artifact_dir: Path | None,
        include_evidence: bool,
    ) -> _AnalysisExecution:
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
        actions = self._map_actions(inferred.actions, source_last_frame, destination_frames)
        poses = self._map_poses(inferred.poses, source_last_frame, destination_frames)

        players_by_frame = {frame.frame_index: frame.players for frame in frames}

        reporter(0.82, "hit_association")
        events, contact_suggestions = self._build_events(
            job,
            balls,
            players_by_frame,
            homographies,
            actions=actions,
            poses=poses,
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
        group_activity_by_frame = {
            self._map_frame(source_frame, source_last_frame, destination_frames): {
                "label": activity.label,
                "confidence": activity.confidence,
            }
            for source_frame, activity in sorted(inferred.group_activities.items())
        }
        extensions: dict[str, object] = {
            "inference_source": "canonical_clip",
            "court_detection": inferred.metadata.get("court_detector", "unknown"),
            "tracking": inferred.metadata.get("tracker", "unknown"),
            "action_source": inferred.metadata.get("action_source", "unknown"),
            "group_activity": {
                "status": "stored_not_interpreted",
                "taxonomy": inferred.metadata.get("group_activity_taxonomy"),
                "frames": [
                    {"frame_index": frame_index, **value}
                    for frame_index, value in sorted(group_activity_by_frame.items())
                ],
            },
            "provider_metadata": inferred.metadata,
            "contact_suggestions": contact_suggestions,
            "decoded_source_frame_count": source_last_frame + 1,
            "canonical_frame_count": destination_frames,
            "provider_work_boundary": "base-analysis-without-identity",
        }
        domain = AnalysisDomainData.model_validate(
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
                    "sdk_version": "0.5.0",
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
                "extensions": extensions,
            }
        )
        validate_passthrough(job, domain)
        if artifact_dir is not None:
            reporter(0.91, "writing_visual_v5_artifacts")
            write_inference_artifacts(
                output_dir=artifact_dir,
                clip_path=clip_path,
                job=job,
                domain=domain,
                frames=frames,
                balls=balls,
                courts=inferred.courts,
                actions=actions,
                fps=inferred.fps,
                frame_width=inferred.frame_width,
                frame_height=inferred.frame_height,
            )
        analysis_data = build_analysis_data(
            job,
            domain=domain,
            frame_records=self._analysis_frame_records(frames, actions),
            ball_positions={
                frame_index: {
                    "x": ball.frame_pos[0],
                    "y": ball.frame_pos[1],
                    "confidence": ball.confidence,
                }
                for frame_index, ball in balls.items()
            },
            court_keypoints=self._analysis_court_keypoints(
                inferred.courts,
                source_last_frame=source_last_frame,
                destination_frames=destination_frames,
                frame_width=inferred.frame_width,
                frame_height=inferred.frame_height,
            ),
            action_taxonomy_id="volleyball-inference-sdk.actions",
            action_taxonomy_version="2.0",
        )
        evidence: AnalysisEvidenceArtifacts | None = None
        if include_evidence:
            raw_pose_recipe = inferred.metadata.get("person_pose_recipe")
            candidate_recipe = (
                cast("dict[object, object]", raw_pose_recipe)
                if isinstance(raw_pose_recipe, dict)
                else {}
            )
            if not candidate_recipe or not all(
                isinstance(key, str) and isinstance(value, str)
                for key, value in candidate_recipe.items()
            ):
                message = "Provider Work analysis requires an immutable person pose recipe"
                raise ValueError(message)
            pose_recipe = {
                key: value
                for key, value in candidate_recipe.items()
                if isinstance(key, str) and isinstance(value, str)
            }
            evidence = build_analysis_evidence_artifacts(
                job=job,
                analysis_run_id=analysis_id,
                analysis_data_bytes=analysis_data,
                poses=poses,
                pose_recipe=pose_recipe,
            )
        reporter(0.98, "analysis_data_ready")
        return _AnalysisExecution(
            bundle=AnalysisDataBundle(domain=domain, analysis_data_bytes=analysis_data),
            evidence=evidence,
        )

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
                homographies[destination_frame] = homography
            players = tuple(
                PlayerObservation(
                    frame_index=destination_frame,
                    source_track_id=player.source_track_id,
                    track_id=player.track_id,
                    frame_bbox=player.frame_bbox,
                    frame_foot_pos=player.frame_foot_pos,
                    court_pos=(
                        player.court_pos
                        if homography is None
                        else project_normalized_frame_point(
                            player.frame_foot_pos,
                            homography,
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
                homography_available=homography is not None,
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

    @classmethod
    def _map_actions(
        cls,
        source: dict[tuple[int, int], ActionObservation],
        source_last_frame: int,
        destination_frames: int,
    ) -> dict[tuple[int, int], ActionObservation]:
        """Map provider frames into the canonical clip frame domain."""
        mapped: dict[tuple[int, int], ActionObservation] = {}
        for (source_frame, track_id), action in source.items():
            frame_index = cls._map_frame(source_frame, source_last_frame, destination_frames)
            candidate = ActionObservation(
                frame_index=frame_index,
                track_id=track_id,
                label=action.label,
                confidence=action.confidence,
            )
            key = (frame_index, track_id)
            previous = mapped.get(key)
            if previous is None or (candidate.confidence or 0.0) > (previous.confidence or 0.0):
                mapped[key] = candidate
        return mapped

    @classmethod
    def _map_poses(
        cls,
        source: dict[int, tuple[PersonPoseObservation, ...]],
        source_last_frame: int,
        destination_frames: int,
    ) -> dict[int, tuple[PersonPoseObservation, ...]]:
        """Map evidence without interpolating or inventing a missing observation."""
        mapped: dict[int, tuple[PersonPoseObservation, ...]] = dict.fromkeys(
            range(destination_frames), ()
        )
        selected_distance: dict[int, float] = {}
        for source_frame, observations in sorted(source.items()):
            frame_index = cls._map_frame(
                source_frame,
                source_last_frame,
                destination_frames,
            )
            ideal_source = (
                0.0
                if destination_frames <= 1
                else frame_index * source_last_frame / (destination_frames - 1)
            )
            distance = abs(source_frame - ideal_source)
            if distance >= selected_distance.get(frame_index, float("inf")):
                continue
            selected_distance[frame_index] = distance
            mapped[frame_index] = tuple(
                replace(observation, frame_index=frame_index) for observation in observations
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
        poses: dict[int, tuple[PersonPoseObservation, ...]],
        frame_width: int,
        frame_height: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        events: list[dict[str, Any]] = []
        total_frames = int(job.clip.video.total_frames)
        fps = int(job.clip.video.fps.num) / int(job.clip.video.fps.den)
        action_search_radius = max(3, round(fps * self.config.association_search_seconds))
        human_specs = [
            _EventSpec(
                anchor=int(point.clip_frame_index),
                key_point_id=point.key_point_id,
                source_key_point_id=point.key_point_id,
                anchor_origin="human_anchor",
                marker_kind=point.marker_kind,
                is_terminal=point.is_terminal,
                detection_confidence=None,
                point=point,
                detection=None,
            )
            for point in job.key_points
        ]
        if job.boundaries:
            first_anchor = int(job.boundaries[0].clip_frame_index)
            last_anchor = int(job.boundaries[-1].clip_frame_index)
        else:
            first_anchor = min(int(point.clip_frame_index) for point in job.key_points)
            last_anchor = max(int(point.clip_frame_index) for point in job.key_points)
        proposals = (
            []
            if human_specs
            else detect_contact_proposals(
                balls,
                start_frame=first_anchor,
                end_frame=last_anchor,
                fps=fps,
                protected_frames=set(),
            )
        )
        proposal_specs = [
            _EventSpec(
                anchor=proposal.frame_index,
                key_point_id=str(
                    uuid5(
                        NAMESPACE_URL,
                        f"volleyball-contact:{job.rally_submission_id}:{proposal.frame_index}:v2",
                    )
                ),
                source_key_point_id=None,
                anchor_origin="ai_detected",
                marker_kind="contact",
                is_terminal=False,
                detection_confidence=proposal.confidence,
                point=None,
                detection={
                    "method": "piecewise_ball_flight_v2",
                    "direction_change": proposal.direction_change,
                    "acceleration": proposal.acceleration,
                    "speed_ratio": proposal.speed_ratio,
                    "model_improvement": proposal.model_improvement,
                    "prediction_error": proposal.prediction_error,
                },
            )
            for proposal in proposals
        ]
        human_anchors = sorted(spec.anchor for spec in human_specs)
        contact_suggestions: list[dict[str, Any]] = []
        validated_proposal_ids: set[str] = set()
        for spec in proposal_specs:
            previous_anchor = max(
                (anchor for anchor in human_anchors if anchor < spec.anchor),
                default=-1,
            )
            next_anchor = min(
                (anchor for anchor in human_anchors if anchor > spec.anchor),
                default=total_frames,
            )
            association = associate_hit(
                anchor_frame=spec.anchor,
                previous_anchor_frame=previous_anchor,
                next_anchor_frame=next_anchor,
                is_terminal=False,
                balls=balls,
                players=players,
                actions=actions,
                poses=poses,
                frame_width=frame_width,
                frame_height=frame_height,
                action_search_radius=action_search_radius,
            )
            actor = self._actor(
                association.player,
                association.observation_frame,
                association.confidence,
            )
            validated = actor is not None
            if validated:
                validated_proposal_ids.add(spec.key_point_id)
            contact_suggestions.append(
                {
                    "frame_index": str(spec.anchor),
                    "confidence": spec.detection_confidence,
                    "marker_kind": spec.marker_kind,
                    "validation_state": (
                        "player_contact_supported" if validated else "trajectory_only"
                    ),
                    "resolved_frame_index": (
                        None
                        if association.observation_frame is None
                        else str(association.observation_frame)
                    ),
                    "association_mode": association.mode,
                    "association_confidence": association.confidence,
                    "association_evidence": association.evidence,
                    "actor": actor,
                    "detection": spec.detection,
                }
            )
        validated_proposals = [
            spec for spec in proposal_specs if spec.key_point_id in validated_proposal_ids
        ]
        # An explicit contact list is an operator override. Boundary-only jobs
        # receive physics proposals; jobs carrying any preserved/manual contact
        # keep that exact sequence and do not silently add new AI contacts.
        specs = human_specs or validated_proposals
        specs.sort(key=lambda item: (item.anchor, item.anchor_origin != "human_anchor"))
        for index, spec in enumerate(specs):
            anchor = spec.anchor
            previous_anchor = specs[index - 1].anchor if index > 0 else -1
            next_anchor = specs[index + 1].anchor if index + 1 < len(specs) else total_frames
            association = associate_hit(
                anchor_frame=anchor,
                previous_anchor_frame=previous_anchor,
                next_anchor_frame=next_anchor,
                is_terminal=spec.is_terminal,
                balls=balls,
                players=players,
                actions=actions,
                poses=poses,
                frame_width=frame_width,
                frame_height=frame_height,
                action_search_radius=action_search_radius,
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
                terminal=spec.is_terminal,
            )
            if actor is not None and association.player is not None:
                action = self._nearest_action(
                    actions,
                    frame_index=association.observation_frame,
                    track_id=association.player.source_track_id,
                    radius=action_search_radius,
                )
                if action is not None:
                    actor["action"] = {
                        "label": action.label,
                        "taxonomy_id": "volleyball-inference-sdk.actions",
                        "taxonomy_version": "2.0",
                        "confidence": action.confidence,
                        "attributes": {"source": "volleyball-multitask-v2"},
                    }
            event_extensions: dict[str, Any] = (
                {
                    "authoritative_clip_pts": spec.point.clip_pts,
                    "authoritative_clip_time_us": spec.point.clip_time_us,
                }
                if spec.point is not None
                else {"detection": spec.detection}
            )
            event_extensions["hitter_association"] = association.evidence
            events.append(
                {
                    "key_point_id": spec.key_point_id,
                    "source_key_point_id": spec.source_key_point_id,
                    "anchor_origin": spec.anchor_origin,
                    "detection_confidence": spec.detection_confidence,
                    "sequence_index": index,
                    "marker_kind": spec.marker_kind,
                    "is_terminal": spec.is_terminal,
                    # Human frames remain exact passthrough; detected frames stay canonical.
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
                        association.mode,
                        *(
                            ("trajectory_contact_proposal",)
                            if spec.anchor_origin == "ai_detected"
                            else ()
                        ),
                    ],
                    "extensions": event_extensions,
                }
            )
        return events, contact_suggestions

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
        court_sides = _resolve_track_court_sides(frames)
        observations: dict[int, list[PlayerObservation]] = defaultdict(list)
        for frame in frames:
            for player in frame.players:
                observations[player.track_id].append(player)
        tracks: list[dict[str, Any]] = []
        for track_id, players in sorted(observations.items()):
            frames_seen = [player.frame_index for player in players]
            confidence_values = [
                player.confidence for player in players if player.confidence is not None
            ]
            tracks.append(
                {
                    "track_id": track_id,
                    "court_side": court_sides[track_id],
                    "first_frame_index": str(min(frames_seen)),
                    "last_frame_index": str(max(frames_seen)),
                    "mean_confidence": (
                        None
                        if not confidence_values
                        else sum(confidence_values) / len(confidence_values)
                    ),
                    "metadata": {
                        "identity_basis": "run_local_tracker_id",
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
                    "quality_flags": ["canonical_anchor_frames"],
                }
            )
        return paths

    @staticmethod
    def _analysis_frame_records(
        frames: list[FrameObservation],
        actions: dict[tuple[int, int], ActionObservation],
    ) -> list[dict[str, Any]]:
        return [
            {
                "frame_index": frame.frame_index,
                "players": [
                    {
                        "track_id": player.track_id,
                        "confidence": player.confidence,
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
                        "action_label": (
                            actions[(frame.frame_index, player.source_track_id)].label
                            if (frame.frame_index, player.source_track_id) in actions
                            else None
                        ),
                        "action_confidence": (
                            actions[(frame.frame_index, player.source_track_id)].confidence
                            if (frame.frame_index, player.source_track_id) in actions
                            else None
                        ),
                    }
                    for player in frame.players
                ],
            }
            for frame in frames
        ]

    @classmethod
    def _analysis_court_keypoints(
        cls,
        courts: dict[int, CourtFrame],
        *,
        source_last_frame: int,
        destination_frames: int,
        frame_width: int,
        frame_height: int,
    ) -> dict[int, list[dict[str, Any]]]:
        """Map only court poses that belong to the exact canonical frame."""
        mapped: dict[int, list[dict[str, Any]]] = {}
        for source_frame, court in sorted(courts.items()):
            if not court.available:
                continue
            frame_index = cls._map_frame(
                source_frame,
                source_last_frame,
                destination_frames,
            )
            mapped[frame_index] = [
                {
                    "keypoint_id": keypoint.index,
                    "frame_pos": {
                        "x": keypoint.frame_pos_px[0] / frame_width,
                        "y": keypoint.frame_pos_px[1] / frame_height,
                    },
                    "confidence": keypoint.confidence,
                }
                for keypoint in court.keypoints
                if keypoint.frame_pos_px is not None
            ]

        return mapped
