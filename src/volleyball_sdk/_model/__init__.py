"""Private model implementation used by :mod:`volleyball_sdk`."""

# Importing these modules performs the registry decorators needed by the YAML
# factory. Nothing training-related is imported here.
from . import backbone, rtv4

__all__ = ["backbone", "rtv4"]
