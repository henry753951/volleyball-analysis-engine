"""Clean inference SDK for the Volleyball multitask checkpoint."""

from .predictor import SCHEMA_VERSION, VolleyballPredictor

# Friendly YOLO-style public name. It is the same stable inference wrapper.
Volleyball = VolleyballPredictor
from ._model.volleyball_metadata import (
    ACTION_NAMES,
    COURT_KEYPOINT_NAMES,
    GROUP_ACTIVITY_NAMES,
    HUMAN_KEYPOINT_NAMES,
    SCHEMA_NAMES,
)

__all__ = [
    "ACTION_NAMES",
    "COURT_KEYPOINT_NAMES",
    "GROUP_ACTIVITY_NAMES",
    "HUMAN_KEYPOINT_NAMES",
    "SCHEMA_NAMES",
    "SCHEMA_VERSION",
    "Volleyball",
    "VolleyballPredictor",
]
