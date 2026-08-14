"""Physics-informed ball-flight change-point proposal detection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .records import BallObservation

MIN_BALL_CONFIDENCE = 0.2
MINIMUM_SIDE_SAMPLES = 4
MINIMUM_MODEL_IMPROVEMENT = 0.32
MINIMUM_PREDICTION_ERROR = 0.004


@dataclass(frozen=True, slots=True)
class ContactProposal:
    """One reviewable discontinuity between two fitted ball-flight segments."""

    frame_index: int
    confidence: float
    direction_change: float
    acceleration: float
    speed_ratio: float
    model_improvement: float
    prediction_error: float


@dataclass(frozen=True, slots=True)
class _FlightFit:
    coefficients: NDArray[np.float64]
    rmse: float


def _design(times: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.column_stack((np.ones_like(times), times, times * times))


def _fit_flight(samples: list[BallObservation], *, origin: int, fps: float) -> _FlightFit:
    times = np.asarray(
        [(sample.frame_index - origin) / fps for sample in samples],
        dtype=np.float64,
    )
    positions = np.asarray([sample.frame_pos for sample in samples], dtype=np.float64)
    design = _design(times)
    weights = np.asarray([sample.confidence or 0.0 for sample in samples], dtype=np.float64)
    weights = np.clip(weights, 0.2, 1.0)
    coefficients = np.zeros((3, 2), dtype=np.float64)
    for _iteration in range(4):
        root_weights = np.sqrt(weights)[:, None]
        coefficients = np.linalg.lstsq(
            design * root_weights,
            positions * root_weights,
            rcond=None,
        )[0]
        residuals = np.linalg.norm(positions - design @ coefficients, axis=1)
        scale = max(float(np.median(residuals)) * 1.4826, 5e-4)
        robust = np.minimum(1.0, 1.5 * scale / np.maximum(residuals, 1e-9))
        weights = np.clip(weights * robust, 0.05, 1.0)
    residuals = np.linalg.norm(positions - design @ coefficients, axis=1)
    return _FlightFit(coefficients, float(np.sqrt(np.mean(residuals * residuals))))


def _predict(fit: _FlightFit, frames: list[int], *, origin: int, fps: float) -> NDArray[np.float64]:
    times = np.asarray([(frame - origin) / fps for frame in frames], dtype=np.float64)
    return _design(times) @ fit.coefficients


def _velocity(fit: _FlightFit, *, frame: int, origin: int, fps: float) -> NDArray[np.float64]:
    time = (frame - origin) / fps
    return fit.coefficients[1] + 2.0 * fit.coefficients[2] * time


def _continuous_segments(
    samples: list[BallObservation],
    *,
    maximum_gap: int,
) -> list[list[BallObservation]]:
    segments: list[list[BallObservation]] = []
    current: list[BallObservation] = []
    for sample in samples:
        if current and sample.frame_index - current[-1].frame_index > maximum_gap:
            segments.append(current)
            current = []
        current.append(sample)
    if current:
        segments.append(current)
    return segments


def _candidate(
    window: list[BallObservation],
    split: int,
    *,
    fps: float,
) -> ContactProposal | None:
    before = window[:split]
    after = window[split:]
    if len(before) < MINIMUM_SIDE_SAMPLES or len(after) < MINIMUM_SIDE_SAMPLES:
        return None
    center = after[0]
    origin = center.frame_index
    whole_fit = _fit_flight(window, origin=origin, fps=fps)
    before_fit = _fit_flight(before, origin=origin, fps=fps)
    after_fit = _fit_flight(after, origin=origin, fps=fps)
    split_rmse = float(
        np.sqrt(
            (before_fit.rmse**2 * len(before) + after_fit.rmse**2 * len(after))
            / (len(before) + len(after))
        )
    )
    improvement = 1.0 - split_rmse / max(whole_fit.rmse, 1e-9)

    future = after[: min(3, len(after))]
    if not future:
        return None
    predicted = _predict(
        before_fit,
        [sample.frame_index for sample in future],
        origin=origin,
        fps=fps,
    )
    observed = np.asarray([sample.frame_pos for sample in future], dtype=np.float64)
    prediction_error = float(np.median(np.linalg.norm(predicted - observed, axis=1)))
    adaptive_error = max(MINIMUM_PREDICTION_ERROR, before_fit.rmse * 3.5)

    before_velocity = _velocity(before_fit, frame=origin, origin=origin, fps=fps)
    after_velocity = _velocity(after_fit, frame=origin, origin=origin, fps=fps)
    before_speed = float(np.linalg.norm(before_velocity))
    after_speed = float(np.linalg.norm(after_velocity))
    acceleration = float(np.linalg.norm(after_velocity - before_velocity))
    denominator = max(before_speed * after_speed, 1e-9)
    direction_change = float(
        np.clip(1.0 - float(np.dot(before_velocity, after_velocity)) / denominator, 0.0, 2.0)
    )
    speed_ratio = abs(after_speed - before_speed) / max(before_speed, after_speed, 1e-9)
    if improvement < MINIMUM_MODEL_IMPROVEMENT or prediction_error < adaptive_error:
        return None

    confidence = float(
        np.clip(
            0.42 * min(improvement / 0.65, 1.0)
            + 0.33 * min(prediction_error / (adaptive_error * 3.0), 1.0)
            + 0.15 * min(acceleration / 1.0, 1.0)
            + 0.10 * (center.confidence or 0.0),
            0.0,
            1.0,
        )
    )
    return ContactProposal(
        center.frame_index,
        confidence,
        direction_change,
        acceleration,
        speed_ratio,
        improvement,
        prediction_error,
    )


def detect_contact_proposals(
    balls: dict[int, BallObservation],
    *,
    start_frame: int,
    end_frame: int,
    fps: float,
    protected_frames: set[int] | None = None,
) -> list[ContactProposal]:
    """Detect flight-model breakpoints while rejecting smooth trajectory apexes."""
    protected = protected_frames or set()
    samples = [
        ball
        for frame, ball in sorted(balls.items())
        if start_frame < frame < end_frame
        and (ball.confidence or 0.0) >= MIN_BALL_CONFIDENCE
    ]
    maximum_gap = max(2, round(fps * 0.05))
    radius = max(6, round(fps * 0.18))
    proposals: list[ContactProposal] = []
    for segment in _continuous_segments(samples, maximum_gap=maximum_gap):
        if len(segment) < 2 * MINIMUM_SIDE_SAMPLES + 1:
            continue
        for split in range(MINIMUM_SIDE_SAMPLES, len(segment) - MINIMUM_SIDE_SAMPLES + 1):
            left = max(0, split - radius)
            right = min(len(segment), split + radius + 1)
            proposal = _candidate(segment[left:right], split - left, fps=fps)
            if proposal is not None:
                proposals.append(proposal)

    minimum_gap = max(4, round(fps * 0.12))
    selected: list[ContactProposal] = []
    for proposal in sorted(proposals, key=lambda item: (-item.confidence, item.frame_index)):
        # A flight discontinuity can lag a labelled contact by several noisy samples.
        # Do not duplicate a trusted human event with a nearby review suggestion.
        protected_gap = max(minimum_gap, round(fps * 0.30))
        if any(abs(proposal.frame_index - frame) <= protected_gap for frame in protected):
            continue
        if any(
            abs(proposal.frame_index - current.frame_index) <= minimum_gap
            for current in selected
        ):
            continue
        selected.append(proposal)
    return sorted(selected, key=lambda item: item.frame_index)
