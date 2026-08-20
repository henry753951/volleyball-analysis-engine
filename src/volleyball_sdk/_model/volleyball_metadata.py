"""Canonical task metadata for the Volleyball multitask model.

Court annotations use 60 points:
- points 0..9: manually verified semantic anchors;
- points 10..59: five deterministic 1/6..5/6 interpolation points on
  each of the ten perimeter edges, in clockwise schema order.

The dense points are useful supervision for court-line evidence and final local
refinement, while the first ten anchors remain the independent geometric basis.
"""

from __future__ import annotations

PERSON_CLASS_ID = 0
BALL_CLASS_ID = 1
COURT_CLASS_ID = 2
ACTION_IGNORE_INDEX = -1
NON_RALLY_GROUP_ID = 8

SCHEMA_NAMES = ("person", "ball", "court")
ACTION_NAMES = (
    "waiting",
    "setting",
    "digging",
    "falling",
    "spiking",
    "blocking",
    "jumping",
    "moving",
    "standing",
)
GROUP_ACTIVITY_NAMES = (
    "r_set",
    "r_spike",
    "r_pass",
    "r_winpoint",
    "l_set",
    "l_spike",
    "l_pass",
    "l_winpoint",
    "non_rally",
)
HUMAN_KEYPOINT_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

COURT_BASE_KEYPOINT_NAMES = (
    "LE_Far",
    "LA_Far",
    "C_Far",
    "RA_Far",
    "RE_Far",
    "RE_Near",
    "RA_Near",
    "C_Near",
    "LA_Near",
    "LE_Near",
)

# 0-based base-point pairs.  Each edge owns five dense interior samples.
COURT_DENSE_EDGE_BASE_PAIRS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (8, 9),
    (9, 0),
)

_dense_names: list[str] = []
for first, second in COURT_DENSE_EDGE_BASE_PAIRS:
    for fraction in range(1, 6):
        _dense_names.append(
            f"{COURT_BASE_KEYPOINT_NAMES[first]}__"
            f"{COURT_BASE_KEYPOINT_NAMES[second]}__{fraction}of6"
        )
COURT_KEYPOINT_NAMES = COURT_BASE_KEYPOINT_NAMES + tuple(_dense_names)
del _dense_names

# Canonical normalized court coordinates corresponding to the ten base points.
# These encode only projective topology, not camera/viewpoint priors.
COURT_CANONICAL_BASE_XY = (
    (0.0, 0.0),
    (1.0 / 3.0, 0.0),
    (0.5, 0.0),
    (2.0 / 3.0, 0.0),
    (1.0, 0.0),
    (1.0, 1.0),
    (2.0 / 3.0, 1.0),
    (0.5, 1.0),
    (1.0 / 3.0, 1.0),
    (0.0, 1.0),
)

# new_index -> old_index after horizontal image flip.
HUMAN_HORIZONTAL_FLIP_INDEX = (
    0,
    2,
    1,
    4,
    3,
    6,
    5,
    8,
    7,
    10,
    9,
    12,
    11,
    14,
    13,
    16,
    15,
)
COURT_BASE_HORIZONTAL_FLIP_INDEX = (4, 3, 2, 1, 0, 9, 8, 7, 6, 5)
GROUP_HORIZONTAL_FLIP_INDEX = (4, 5, 6, 7, 0, 1, 2, 3, 8)


def _build_court_dense_flip_index() -> tuple[int, ...]:
    """Return the 60-point semantic permutation for an image horizontal flip."""

    # Map semantic dense point (edge, fraction) to its global dense index.
    lookup: dict[tuple[int, int, int], int] = {}
    for edge_index, (first, second) in enumerate(COURT_DENSE_EDGE_BASE_PAIRS):
        for fraction in range(1, 6):
            lookup[(first, second, fraction)] = 10 + edge_index * 5 + (fraction - 1)

    result = list(COURT_BASE_HORIZONTAL_FLIP_INDEX)
    for first, second in COURT_DENSE_EDGE_BASE_PAIRS:
        mapped_first = COURT_BASE_HORIZONTAL_FLIP_INDEX[first]
        mapped_second = COURT_BASE_HORIZONTAL_FLIP_INDEX[second]
        for fraction in range(1, 6):
            direct = lookup.get((mapped_first, mapped_second, fraction))
            if direct is not None:
                result.append(direct)
                continue
            reverse = lookup.get((mapped_second, mapped_first, 6 - fraction))
            if reverse is None:
                raise RuntimeError("Failed to construct Court60 flip permutation")
            result.append(reverse)
    if sorted(result) != list(range(60)):
        raise RuntimeError("Court60 horizontal flip mapping is not a permutation")
    return tuple(result)


COURT_HORIZONTAL_FLIP_INDEX = _build_court_dense_flip_index()

# Dense 60-point graph exactly matching annotation60.  The first 60 edges trace
# the ten perimeter segments via the five inserted samples.  The final three
# lines are the inner attack/center structural lines.
_dense_skeleton: list[tuple[int, int]] = []
for edge_index, (first, second) in enumerate(COURT_DENSE_EDGE_BASE_PAIRS):
    samples = [first] + [10 + edge_index * 5 + offset for offset in range(5)] + [second]
    _dense_skeleton.extend(zip(samples[:-1], samples[1:]))
_dense_skeleton.extend(((1, 8), (2, 7), (3, 6)))
COURT_PCP_EDGES = tuple(_dense_skeleton)
COURT_DENSE_SKELETON = COURT_PCP_EDGES
del _dense_skeleton

# Uniform custom-court sigmas.  There is no standard COCO calibration for this
# schema, so these remain explicit and easy to recalibrate later.
COURT_OKS_SIGMAS = (0.025,) * 60

HUMAN_PCP_EDGES = (
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (5, 11),
    (6, 12),
    (5, 6),
    (11, 12),
)

COCO_HUMAN_KEYPOINT_SIGMAS = (
    0.026,
    0.025,
    0.025,
    0.035,
    0.035,
    0.079,
    0.079,
    0.072,
    0.072,
    0.062,
    0.062,
    0.107,
    0.107,
    0.087,
    0.087,
    0.089,
    0.089,
)

__all__ = [
    "ACTION_IGNORE_INDEX",
    "ACTION_NAMES",
    "BALL_CLASS_ID",
    "COCO_HUMAN_KEYPOINT_SIGMAS",
    "COURT_BASE_HORIZONTAL_FLIP_INDEX",
    "COURT_BASE_KEYPOINT_NAMES",
    "COURT_CANONICAL_BASE_XY",
    "COURT_CLASS_ID",
    "COURT_DENSE_EDGE_BASE_PAIRS",
    "COURT_DENSE_SKELETON",
    "COURT_HORIZONTAL_FLIP_INDEX",
    "COURT_KEYPOINT_NAMES",
    "COURT_OKS_SIGMAS",
    "COURT_PCP_EDGES",
    "GROUP_ACTIVITY_NAMES",
    "GROUP_HORIZONTAL_FLIP_INDEX",
    "HUMAN_HORIZONTAL_FLIP_INDEX",
    "HUMAN_KEYPOINT_NAMES",
    "HUMAN_PCP_EDGES",
    "NON_RALLY_GROUP_ID",
    "PERSON_CLASS_ID",
    "SCHEMA_NAMES",
]
