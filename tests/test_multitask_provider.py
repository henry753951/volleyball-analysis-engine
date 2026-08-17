"""Geometry invariants for the Volleyball Court60 provider adapter."""

from volleyball_analysis_engine.config import Settings
from volleyball_analysis_engine.multitask_provider import (
    COURT60_WORLD_XY,
    VolleyballMultitaskObservationProvider,
)
from volleyball_analysis_engine.worker import build_pipeline


def test_court60_world_points_preserve_base_anchors_and_dense_edge_order() -> None:
    assert len(COURT60_WORLD_XY) == 60
    assert COURT60_WORLD_XY[:10] == (
        (0.0, 0.0),
        (6.0, 0.0),
        (9.0, 0.0),
        (12.0, 0.0),
        (18.0, 0.0),
        (18.0, 9.0),
        (12.0, 9.0),
        (9.0, 9.0),
        (6.0, 9.0),
        (0.0, 9.0),
    )
    assert COURT60_WORLD_XY[10:15] == (
        (1.0, 0.0),
        (2.0, 0.0),
        (3.0, 0.0),
        (4.0, 0.0),
        (5.0, 0.0),
    )


def test_worker_pipeline_has_no_standalone_court_or_pose_provider() -> None:
    settings = Settings()

    pipeline = build_pipeline(settings)

    assert isinstance(pipeline.provider, VolleyballMultitaskObservationProvider)
    assert "court_model" not in Settings.model_fields
    assert "pose_checkpoint" not in Settings.model_fields
    assert "person_pose_enabled" not in Settings.model_fields
