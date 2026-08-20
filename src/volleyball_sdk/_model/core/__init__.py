"""Minimal registry/config surface for deployment."""

from .workspace import GLOBAL_CONFIG, create, register
from .inference_config import InferenceConfig

__all__ = ["GLOBAL_CONFIG", "InferenceConfig", "create", "register"]
