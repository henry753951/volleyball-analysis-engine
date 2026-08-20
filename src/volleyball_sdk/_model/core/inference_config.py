"""Minimal YAML-backed object factory used only for inference."""

from __future__ import annotations

import copy
from pathlib import Path

from .workspace import create
from .yaml_utils import load_config, merge_config


class InferenceConfig:
    """Load only model/postprocessor/runtime settings.

    This intentionally contains no optimizer, dataloader, criterion, evaluator,
    EMA, AMP scaler, or training state.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"Config not found: {self.path}")
        self.yaml_cfg = load_config(str(self.path))
        self._model = None
        self._postprocessor = None

    @property
    def global_cfg(self):
        return merge_config(copy.deepcopy(self.yaml_cfg), inplace=False, overwrite=False)

    @property
    def model(self):
        if self._model is None:
            self._model = create(self.yaml_cfg["model"], self.global_cfg)
        return self._model

    @property
    def postprocessor(self):
        if self._postprocessor is None:
            self._postprocessor = create(self.yaml_cfg["postprocessor"], self.global_cfg)
        return self._postprocessor

    @property
    def runtime(self) -> dict:
        runtime = dict(self.yaml_cfg.get("runtime", {}) or {})
        required = {
            "frame_num",
            "jump_frame",
            "sampling_mode",
            "input_size",
            "normalize_mean",
            "normalize_std",
            "score_threshold",
        }
        missing = sorted(required - set(runtime))
        if missing:
            raise KeyError(f"Missing runtime config fields: {', '.join(missing)}")
        return runtime
