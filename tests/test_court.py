"""Frame completeness tests for batched court-line inference."""

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
from volley_court import CourtKeypoint as ModelCourtKeypoint
from volley_court import CourtLayout, CourtLineModel

from volleyball_analysis_engine.court import (
    POSE36_CANONICAL_POINTS,
    CourtVideoProcessor,
    interpolate_short_court_gaps,
)
from volleyball_analysis_engine.records import CourtFrame, CourtKeypoint


class FakeCourtModel:
    """Record every frame passed through the batched model boundary."""

    def __init__(self) -> None:
        """Initialize call counters."""
        self.batch_sizes: list[int] = []
        self.attach_count = 0

    def predict_many(
        self,
        frames: list[np.ndarray[Any, Any]],
        *,
        include_layout: bool,
    ) -> list[SimpleNamespace]:
        """Return one empty layout result for every supplied frame."""
        assert include_layout is False
        self.batch_sizes.append(len(frames))
        return [SimpleNamespace(layout=self._empty_layout()) for _ in frames]

    def attach_layout(
        self,
        result: SimpleNamespace,
        *,
        prior_layout: CourtLayout | None = None,
    ) -> SimpleNamespace:
        """Record layout fitting attempts."""
        del prior_layout
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


def test_full_frame_court_batches_do_not_drop_partial_tail() -> None:
    model = FakeCourtModel()
    processor = CourtVideoProcessor(
        cast("CourtLineModel", model),
        batch_size=2,
        layout_every=1,
        refresh_every=120,
        track_every=1,
        max_hold_frames=30,
    )
    frame = np.zeros((16, 16, 3), dtype=np.uint8)

    assert processor.submit(0, frame) == {}
    assert processor.submit(1, frame) == {}
    assert processor.submit(2, frame) == {}
    assert processor.finish() == {}

    assert model.batch_sizes == [2, 1]
    assert model.attach_count == 3
    assert processor.timing.source_frames == 3
    assert processor.timing.inference_frames == 3
    assert processor.timing.layout_attempts == 3
    assert processor.timing.batches == 2
    assert processor.max_hold_frames == 120


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
