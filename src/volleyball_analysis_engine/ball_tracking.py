"""Lightweight temporal identity tracking for detector-owned ball candidates."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from .records import BallObservation

MINIMUM_VELOCITY_SAMPLES = 2


@dataclass(frozen=True, slots=True)
class BallTrackingConfig:
    """Normalized-coordinate gating for one broadcast ball trajectory."""

    base_gate: float = 0.045
    gate_growth_per_frame: float = 0.012
    maximum_missed_frames: int = 5
    history_size: int = 8


class BallTrajectoryTracker:
    """Select a temporally consistent ball candidate without inventing observations."""

    def __init__(self, config: BallTrackingConfig | None = None) -> None:
        """Initialize bounded trajectory history."""
        self.config = config or BallTrackingConfig()
        self._history: deque[BallObservation] = deque(maxlen=self.config.history_size)
        self._missed_frames = 0

    def reset(self) -> None:
        """Forget the active trajectory after a cut or sustained detection loss."""
        self._history.clear()
        self._missed_frames = 0

    def update(
        self,
        frame_index: int,
        candidates: Sequence[tuple[float, float]],
        confidences: Sequence[float],
    ) -> BallObservation | None:
        """Return the current observed candidate that best agrees with recent motion."""
        if len(candidates) != len(confidences):
            message = "ball candidates and confidences must have equal length"
            raise ValueError(message)
        if not candidates:
            self._register_miss()
            return None

        if not self._history:
            selected = max(range(len(candidates)), key=confidences.__getitem__)
            return self._accept(frame_index, candidates, confidences, selected)

        prediction = self._predict(frame_index)
        gap = max(1, frame_index - self._history[-1].frame_index)
        speed = float(np.linalg.norm(self._velocity()))
        gate = (
            self.config.base_gate
            + self.config.gate_growth_per_frame * max(0, gap - 1)
            + 0.75 * speed * gap
        )
        distances = [float(np.linalg.norm(np.asarray(point) - prediction)) for point in candidates]
        eligible = [index for index, distance in enumerate(distances) if distance <= gate]
        if not eligible:
            self._register_miss()
            return None
        selected = min(
            eligible,
            key=lambda index: distances[index] / max(gate, 1e-9)
            + 0.35 * (1.0 - float(confidences[index])),
        )
        return self._accept(frame_index, candidates, confidences, selected)

    def _accept(
        self,
        frame_index: int,
        candidates: Sequence[tuple[float, float]],
        confidences: Sequence[float],
        index: int,
    ) -> BallObservation:
        point = candidates[index]
        observation = BallObservation(
            frame_index=frame_index,
            frame_pos=(float(point[0]), float(point[1])),
            confidence=float(confidences[index]),
        )
        self._history.append(observation)
        self._missed_frames = 0
        return observation

    def _register_miss(self) -> None:
        self._missed_frames += 1
        if self._missed_frames > self.config.maximum_missed_frames:
            self.reset()

    def _predict(self, frame_index: int) -> np.ndarray:
        latest = self._history[-1]
        delta = max(0, frame_index - latest.frame_index)
        return np.asarray(latest.frame_pos, dtype=np.float64) + self._velocity() * delta

    def _velocity(self) -> np.ndarray:
        if len(self._history) < MINIMUM_VELOCITY_SAMPLES:
            return np.zeros(2, dtype=np.float64)
        rows = list(self._history)[-5:]
        velocities: list[np.ndarray] = []
        for first, second in pairwise(rows):
            delta = second.frame_index - first.frame_index
            if delta > 0:
                velocities.append(
                    (np.asarray(second.frame_pos) - np.asarray(first.frame_pos)) / delta
                )
        return (
            np.median(np.asarray(velocities, dtype=np.float64), axis=0)
            if velocities
            else np.zeros(2, dtype=np.float64)
        )
