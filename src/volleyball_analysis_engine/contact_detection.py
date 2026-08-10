"""Deterministic ball-trajectory contact proposal detection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .records import BallObservation

MIN_BALL_CONFIDENCE = 0.2
MINIMUM_SAMPLES = 5
MIN_DIRECTION_CHANGE = 0.12
MIN_SPEED_RATIO = 0.3


@dataclass(frozen=True, slots=True)
class ContactProposal:
    """One reviewable contact-time proposal in canonical clip frames."""

    frame_index: int
    confidence: float
    direction_change: float
    acceleration: float
    speed_ratio: float


def detect_contact_proposals(
    balls: dict[int, BallObservation],
    *,
    start_frame: int,
    end_frame: int,
    fps: float,
    protected_frames: set[int] | None = None,
) -> list[ContactProposal]:
    """Find abrupt trajectory changes without changing canonical frame order.

    The detector deliberately emits proposals rather than authoritative contacts. It uses
    centered velocity vectors, acceleration, direction change and detector confidence, then
    applies temporal non-maximum suppression. Sparse/gapped observations are not bridged.
    """
    protected = protected_frames or set()
    samples = [
        ball
        for frame, ball in sorted(balls.items())
        if start_frame < frame < end_frame
        and (ball.confidence or 0.0) >= MIN_BALL_CONFIDENCE
    ]
    if len(samples) < MINIMUM_SAMPLES:
        return []

    raw: list[tuple[int, float, float, float, float]] = []
    for index in range(2, len(samples) - 2):
        left, center, right = samples[index - 2], samples[index], samples[index + 2]
        before_frames = center.frame_index - left.frame_index
        after_frames = right.frame_index - center.frame_index
        max_bridge = max(4, round(fps * 0.1))
        if (
            before_frames < 1
            or after_frames < 1
            or before_frames > max_bridge
            or after_frames > max_bridge
        ):
            continue
        before = (np.asarray(center.frame_pos) - np.asarray(left.frame_pos)) / before_frames
        after = (np.asarray(right.frame_pos) - np.asarray(center.frame_pos)) / after_frames
        before_speed = float(np.linalg.norm(before))
        after_speed = float(np.linalg.norm(after))
        acceleration = float(np.linalg.norm(after - before))
        denominator = max(before_speed * after_speed, 1e-9)
        direction_change = float(np.clip(1.0 - float(np.dot(before, after)) / denominator, 0, 2))
        speed_ratio = abs(after_speed - before_speed) / max(before_speed, after_speed, 1e-9)
        raw.append(
            (
                center.frame_index,
                acceleration,
                direction_change,
                speed_ratio,
                center.confidence or 0.0,
            )
        )
    if not raw:
        return []

    accelerations = np.asarray([item[1] for item in raw], dtype=np.float64)
    median = float(np.median(accelerations))
    mad = float(np.median(np.abs(accelerations - median)))
    threshold = max(median + 2.5 * max(mad, 1e-5), float(np.quantile(accelerations, 0.8)))
    scale = max(threshold, 1e-5)
    proposals: list[ContactProposal] = []
    for frame, acceleration, direction_change, speed_ratio, detector_confidence in raw:
        if acceleration < threshold or (
            direction_change < MIN_DIRECTION_CHANGE and speed_ratio < MIN_SPEED_RATIO
        ):
            continue
        confidence = float(
            np.clip(
                0.5 * min(acceleration / (scale * 1.5), 1.0)
                + 0.3 * min(direction_change / 0.8, 1.0)
                + 0.1 * min(speed_ratio / 0.8, 1.0)
                + 0.1 * detector_confidence,
                0,
                1,
            )
        )
        proposals.append(
            ContactProposal(frame, confidence, direction_change, acceleration, speed_ratio)
        )

    minimum_gap = max(4, round(fps * 0.12))
    selected: list[ContactProposal] = []
    for proposal in sorted(proposals, key=lambda item: (-item.confidence, item.frame_index)):
        if any(abs(proposal.frame_index - frame) < minimum_gap for frame in protected):
            continue
        if any(
            abs(proposal.frame_index - current.frame_index) < minimum_gap
            for current in selected
        ):
            continue
        selected.append(proposal)
    return sorted(selected, key=lambda item: item.frame_index)
