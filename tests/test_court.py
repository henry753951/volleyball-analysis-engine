"""Frame completeness tests for batched court-line inference."""

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
from volley_court import CourtLineModel

from volleyball_analysis_engine.court import CourtVideoProcessor


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
        return [SimpleNamespace(layout=None) for _ in frames]

    def attach_layout(self, result: SimpleNamespace) -> SimpleNamespace:
        """Record layout fitting attempts."""
        self.attach_count += 1
        return result


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
    assert processor.timing.batches == 2
