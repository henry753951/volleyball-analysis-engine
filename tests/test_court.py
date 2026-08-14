"""Frame completeness tests for batched court-line inference."""

from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from volley_court import CourtKeypoint as ModelCourtKeypoint
from volley_court import CourtLayout, CourtLineModel

from volleyball_analysis_engine.config import Settings
from volleyball_analysis_engine.court import (
    POSE36_CANONICAL_POINTS,
    CourtLineEstimator,
    CourtVideoProcessor,
    interpolate_short_court_gaps,
)
from volleyball_analysis_engine.records import CourtFrame, CourtKeypoint


class FakeCourtModel:
    """Record every frame passed through the batched model boundary."""

    def __init__(self, layouts: list[CourtLayout] | None = None) -> None:
        """Initialize call counters."""
        self.batch_sizes: list[int] = []
        self.frame_markers: list[list[int]] = []
        self.include_layout_calls: list[bool] = []
        self.attach_count = 0
        self._layouts = list(layouts or [])

    def predict_many(
        self,
        frames: list[np.ndarray[Any, Any]],
        *,
        include_layout: bool,
    ) -> list[SimpleNamespace]:
        """Return one direct layout result for every supplied frame."""
        assert include_layout is True
        self.batch_sizes.append(len(frames))
        self.frame_markers.append([int(frame[0, 0, 0]) for frame in frames])
        self.include_layout_calls.append(include_layout)
        layouts = [
            self._layouts.pop(0) if self._layouts else self._empty_layout()
            for _ in frames
        ]
        return [SimpleNamespace(layout=layout) for layout in layouts]

    def predict(
        self,
        frame: np.ndarray[Any, Any],
        *,
        include_layout: bool,
    ) -> SimpleNamespace:
        """Return one direct layout through the single-frame 0.3 API."""
        return self.predict_many([frame], include_layout=include_layout)[0]

    def attach_layout(
        self,
        result: SimpleNamespace,
    ) -> SimpleNamespace:
        """Expose the 0.3 signature and prove direct layouts are not reattached."""
        self.attach_count += 1
        return result

    @staticmethod
    def _empty_layout() -> CourtLayout:
        return CourtLayout(
            status="abstained",
            score=0.0,
            reason="test",
            keypoints=(),
            candidate_keypoints=(),
            matched_line_count=0,
            hypothesis_margin=0.0,
            semantic_alignment=None,
            homography=None,
        )


def _layout(
    *,
    status: str = "ok",
    point_count: int = 36,
    shift: float = 0.0,
) -> CourtLayout:
    points = tuple(
        ModelCourtKeypoint(
            index,
            10.0 * y + 100.0 + shift,
            10.0 * x + 50.0,
            0.9,
            True,
            "test",
        )
        for index, (x, y) in enumerate(POSE36_CANONICAL_POINTS[:point_count])
    )
    return CourtLayout(
        status=cast("Any", status),
        score=0.9,
        reason="test",
        keypoints=points,
        candidate_keypoints=points,
        matched_line_count=7,
        hypothesis_margin=0.5,
        semantic_alignment=1.0,
        homography=((0.0, 10.0, 100.0), (10.0, 0.0, 50.0), (0.0, 0.0, 1.0)),
    )


def test_full_frame_court_batches_do_not_drop_partial_tail() -> None:
    model = FakeCourtModel([_layout(), _layout(), _layout()])
    processor = CourtVideoProcessor(
        cast("CourtLineModel", model),
        batch_size=2,
        layout_every=1,
        refresh_every=120,
        track_every=1,
        max_hold_frames=30,
    )
    frames = [np.full((240, 360, 3), marker, dtype=np.uint8) for marker in (4, 8, 15)]

    assert processor.submit(41, frames[0]) == {}
    first_batch = processor.submit(7, frames[1])
    assert list(first_batch) == [41, 7]
    assert [court.frame_index for court in first_batch.values()] == [41, 7]
    assert processor.submit(99, frames[2]) == {}
    tail = processor.finish()
    assert list(tail) == [99]
    assert tail[99].frame_index == 99

    assert model.batch_sizes == [2, 1]
    assert model.frame_markers == [[4, 8], [15]]
    assert model.include_layout_calls == [True, True]
    assert model.attach_count == 0
    assert processor.timing.source_frames == 3
    assert processor.timing.inference_frames == 3
    assert processor.timing.layout_attempts == 3
    assert processor.timing.batches == 2
    assert processor.max_hold_frames == 120


@pytest.mark.parametrize("status", ["ambiguous", "abstained"])
def test_non_ok_layout_status_is_never_accepted(status: str) -> None:
    model = FakeCourtModel([_layout(status=status)])
    processor = CourtVideoProcessor(
        cast("CourtLineModel", model),
        batch_size=1,
        layout_every=1,
        refresh_every=120,
        track_every=1,
        max_hold_frames=30,
    )

    assert processor.submit(5, np.zeros((240, 360, 3), dtype=np.uint8)) == {}
    assert processor.timing.accepted_frames == 0
    assert processor.timing.rejected_layouts == 1
    assert processor.timing.layout_abstentions == 1


def test_non_ok_sparse_refresh_does_not_render_the_held_layout() -> None:
    model = FakeCourtModel([_layout(), _layout(status="ambiguous")])
    processor = CourtVideoProcessor(
        cast("CourtLineModel", model),
        batch_size=16,
        layout_every=2,
        refresh_every=2,
        track_every=1,
        max_hold_frames=30,
    )
    frame = np.zeros((240, 360, 3), dtype=np.uint8)

    assert list(processor.submit(0, frame)) == [0]
    assert list(processor.submit(1, frame)) == [1]
    assert processor.submit(2, frame) == {}
    assert processor.timing.accepted_frames == 2
    assert processor.timing.rejected_layouts == 1


def test_ok_layout_requires_all_36_pose_keypoints() -> None:
    duplicate_id = _layout()
    duplicate_id = replace(
        duplicate_id,
        keypoints=(*duplicate_id.keypoints[:-1], replace(duplicate_id.keypoints[-1], id=0)),
    )
    model = FakeCourtModel([_layout(point_count=35), duplicate_id, _layout()])
    processor = CourtVideoProcessor(
        cast("CourtLineModel", model),
        batch_size=3,
        layout_every=1,
        refresh_every=120,
        track_every=1,
        max_hold_frames=30,
    )
    frame = np.zeros((240, 360, 3), dtype=np.uint8)

    assert processor.submit(20, frame) == {}
    assert processor.submit(21, frame) == {}
    output = processor.submit(22, frame)

    assert list(output) == [22]
    assert len(output[22].keypoints) == 36
    assert processor.timing.accepted_frames == 1
    assert processor.timing.rejected_layouts == 2


def test_v3_defaults_and_v2_environment_rollback_use_direct_layout_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str | None, dict[str, object]]] = []

    def from_pretrained(
        model: str | None = None,
        **config: object,
    ) -> CourtLineModel:
        calls.append((model, config))
        return cast("CourtLineModel", FakeCourtModel())

    monkeypatch.delenv("VOLLYAI_COURT_MODEL", raising=False)
    monkeypatch.delenv("VOLLYAI_COURT_IMGSZ", raising=False)
    monkeypatch.delenv("VOLLYAI_COURT_BATCH_SIZE", raising=False)
    monkeypatch.delenv("VOLLYAI_COURT_DECODER", raising=False)
    monkeypatch.setattr(CourtLineModel, "from_pretrained", staticmethod(from_pretrained))

    defaults = Settings()
    CourtLineEstimator(
        defaults.court_model,
        device="cuda:0",
        image_size=defaults.court_imgsz,
        decoder=defaults.court_decoder,
        batch_size=defaults.court_batch_size,
        layout_every=1,
        refresh_every=120,
        track_every=1,
        max_hold_frames=180,
    )
    monkeypatch.setenv("VOLLYAI_COURT_MODEL", "v2")
    rollback = Settings()
    CourtLineEstimator(
        rollback.court_model,
        device="cuda:0",
        image_size=rollback.court_imgsz,
        decoder=rollback.court_decoder,
        batch_size=rollback.court_batch_size,
        layout_every=1,
        refresh_every=120,
        track_every=1,
        max_hold_frames=180,
    )

    assert defaults.court_model == "v3"
    assert defaults.court_imgsz == 512
    assert defaults.court_batch_size == 16
    assert defaults.court_decoder == "auto"
    assert rollback.court_model == "v2"
    assert [model for model, _ in calls] == ["v3", "v2"]
    assert all(config["include_layout"] is True for _, config in calls)


def test_abstained_current_frame_does_not_render_stale_layout() -> None:
    model = FakeCourtModel()
    processor = CourtVideoProcessor(
        cast("CourtLineModel", model),
        batch_size=1,
        layout_every=1,
        refresh_every=120,
        track_every=1,
        max_hold_frames=30,
    )
    previous = CourtLayout(
        status="ok",
        score=0.9,
        reason="previous frame",
        keypoints=tuple(
            ModelCourtKeypoint(index, float(index), float(index), 0.9, True, "test")
            for index in range(36)
        ),
        candidate_keypoints=(),
        matched_line_count=7,
        hypothesis_margin=0.5,
        semantic_alignment=1.0,
        homography=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    )
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    tracker = cast("Any", processor)._tracker  # noqa: SLF001
    tracker.update(previous, width=16, height=16, frame=frame)

    assert processor.submit(0, frame) == {}


def test_short_court_gap_is_interpolated_from_both_neighbor_frames() -> None:
    world = ((0, 0), (6, 0), (18, 0), (0, 9), (6, 9), (18, 9))
    image = ((0, 0), (60, 0), (180, 0), (0, 90), (60, 90), (180, 90))

    def court(frame_index: int, shift: float) -> CourtFrame:
        return CourtFrame(
            frame_index,
            True,
            tuple(
                CourtKeypoint(index, (x + shift, y), 0.9, (wx, wy))
                for index, ((x, y), (wx, wy)) in enumerate(zip(image, world, strict=True))
            ),
        )

    courts = {10: court(10, 0.0), 12: court(12, 2.0)}

    assert interpolate_short_court_gaps(courts) == 1
    assert courts[11].keypoints[0].frame_pos_px == (1.0, 0.0)


def test_layout_orientation_lock_corrects_a_both_axis_recovery_flip() -> None:
    def layout(*, flipped: bool) -> CourtLayout:
        points = tuple(
            ModelCourtKeypoint(
                index,
                20.0 * (9.0 - x if flipped else x) + 100.0,
                10.0 * (18.0 - y if flipped else y) + 50.0,
                0.9,
                True,
                "test",
            )
            for index, (x, y) in enumerate(POSE36_CANONICAL_POINTS)
        )
        homography = (
            (-20.0, 0.0, 280.0),
            (0.0, -10.0, 230.0),
            (0.0, 0.0, 1.0),
        ) if flipped else (
            (20.0, 0.0, 100.0),
            (0.0, 10.0, 50.0),
            (0.0, 0.0, 1.0),
        )
        return CourtLayout(
            status="ok",
            score=0.9,
            reason="test",
            keypoints=points,
            candidate_keypoints=points,
            matched_line_count=7,
            hypothesis_margin=0.5,
            semantic_alignment=1.0,
            homography=homography,
        )

    reference = layout(flipped=False)
    corrected, orientation = CourtVideoProcessor._align_orientation(  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001
        layout(flipped=True),
        reference,
    )

    assert orientation == "both"
    assert [(point.x, point.y) for point in corrected.keypoints] == [
        (point.x, point.y) for point in reference.keypoints
    ]
    assert corrected.homography == reference.homography


def test_initial_layout_is_oriented_like_the_video_screen() -> None:
    points = tuple(
        ModelCourtKeypoint(
            index,
            20.0 * (18.0 - y) + 100.0,
            10.0 * x + 50.0,
            0.9,
            True,
            "test",
        )
        for index, (x, y) in enumerate(POSE36_CANONICAL_POINTS)
    )
    layout = CourtLayout(
        status="ok",
        score=0.9,
        reason="test",
        keypoints=points,
        candidate_keypoints=points,
        matched_line_count=7,
        hypothesis_margin=0.5,
        semantic_alignment=1.0,
        homography=None,
    )

    orientation = CourtVideoProcessor._screen_canonical_orientation(  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001
        layout
    )

    assert orientation == "near_far"


def test_recovery_orientation_correction_never_flips_output_sides() -> None:
    def layout(*, flipped: bool) -> CourtLayout:
        points = tuple(
            ModelCourtKeypoint(
                index,
                20.0 * (9.0 - x if flipped else x) + 100.0,
                10.0 * (18.0 - y if flipped else y) + 50.0,
                0.9,
                True,
                "test",
            )
            for index, (x, y) in enumerate(POSE36_CANONICAL_POINTS)
        )
        homography = (
            (-20.0, 0.0, 280.0),
            (0.0, -10.0, 230.0),
            (0.0, 0.0, 1.0),
        ) if flipped else (
            (20.0, 0.0, 100.0),
            (0.0, 10.0, 50.0),
            (0.0, 0.0, 1.0),
        )
        return CourtLayout(
            status="ok",
            score=0.9,
            reason="test",
            keypoints=points,
            candidate_keypoints=points,
            matched_line_count=7,
            hypothesis_margin=0.5,
            semantic_alignment=1.0,
            homography=homography,
        )

    model = FakeCourtModel([layout(flipped=False), layout(flipped=True)])
    processor = CourtVideoProcessor(
        cast("CourtLineModel", model),
        batch_size=1,
        layout_every=1,
        refresh_every=120,
        track_every=1,
        max_hold_frames=30,
    )
    frame = np.zeros((300, 400, 3), dtype=np.uint8)

    first = processor.submit(0, frame)[0]
    recovered = processor.submit(1, frame)[1]

    assert [point.frame_pos_px for point in recovered.keypoints] == [
        point.frame_pos_px for point in first.keypoints
    ]
    assert processor.timing.orientation_corrections == 1


def test_sparse_refresh_uses_the_same_clip_orientation_lock() -> None:
    reference = _layout()
    flipped_points = tuple(
        ModelCourtKeypoint(
            index,
            10.0 * (18.0 - y) + 100.0,
            10.0 * (9.0 - x) + 50.0,
            0.9,
            True,
            "test",
        )
        for index, (x, y) in enumerate(POSE36_CANONICAL_POINTS)
    )
    flipped = replace(
        reference,
        keypoints=flipped_points,
        candidate_keypoints=flipped_points,
        homography=(
            (0.0, -10.0, 280.0),
            (-10.0, 0.0, 140.0),
            (0.0, 0.0, 1.0),
        ),
    )
    model = FakeCourtModel([reference, flipped])
    processor = CourtVideoProcessor(
        cast("CourtLineModel", model),
        batch_size=16,
        layout_every=2,
        refresh_every=2,
        track_every=1,
        max_hold_frames=30,
    )
    frame = np.zeros((400, 400, 3), dtype=np.uint8)

    first = processor.submit(0, frame)[0]
    processor.submit(1, frame)
    recovered = processor.submit(2, frame)[2]

    assert [point.frame_pos_px for point in recovered.keypoints] == [
        point.frame_pos_px for point in first.keypoints
    ]
    assert processor.timing.orientation_corrections == 1
