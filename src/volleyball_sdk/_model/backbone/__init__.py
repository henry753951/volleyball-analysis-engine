"""Inference-only backbone registry."""

from .temporal_fusion import TemporalPyramidFusion
from .x3d import X3DBackbone

__all__ = ["TemporalPyramidFusion", "X3DBackbone"]
