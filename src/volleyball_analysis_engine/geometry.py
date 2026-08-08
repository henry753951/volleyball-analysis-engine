"""Court detector geometry and canonical court projection."""

from __future__ import annotations

from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from .records import CourtFrame

COURT_WIDTH_M = 18.0
COURT_HEIGHT_M = 9.0
MIN_HOMOGRAPHY_POINTS = 6


def estimate_homography(
    frame: CourtFrame,
    *,
    confidence_threshold: float = 0.25,
) -> NDArray[np.float64] | None:
    """Solve image-pixel to canonical-court-metre homography using RANSAC."""
    image_points: list[tuple[float, float]] = []
    world_points: list[tuple[float, float]] = []
    for point in frame.keypoints:
        if (
            point.frame_pos_px is None
            or point.world_pos_m is None
            or point.confidence is None
            or point.confidence < confidence_threshold
        ):
            continue
        image_points.append(point.frame_pos_px)
        world_points.append(point.world_pos_m)
    if len(image_points) < MIN_HOMOGRAPHY_POINTS:
        return None
    matrix, inliers = cast(
        "tuple[NDArray[np.float64] | None, NDArray[np.uint8] | None]",
        cv2.findHomography(
            np.asarray(image_points, dtype=np.float64),
            np.asarray(world_points, dtype=np.float64),
            cv2.RANSAC,
            1.0,
        ),
    )
    if matrix is None or inliers is None or int(inliers.sum()) < MIN_HOMOGRAPHY_POINTS:
        return None
    return np.asarray(matrix, dtype=np.float64)


def project_normalized_frame_point(
    position: tuple[float, float],
    homography: NDArray[np.float64],
    *,
    frame_width: int,
    frame_height: int,
) -> tuple[float, float]:
    """Project normalized video coordinates into unclamped canonical court coordinates."""
    point_px = np.asarray(
        [[[position[0] * frame_width, position[1] * frame_height]]],
        dtype=np.float64,
    )
    court_m = cv2.perspectiveTransform(point_px, homography).reshape(2)
    return float(court_m[0] / COURT_WIDTH_M), float(court_m[1] / COURT_HEIGHT_M)
