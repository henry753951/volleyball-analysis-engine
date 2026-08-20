"""PyTorchVideo X3D feature-pyramid backbones."""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn

from ..core import register

__all__ = ["X3DBackbone", "X3D_xs"]


def _build_x3d(model_name: str, pretrained: bool):
    try:
        from pytorchvideo.models.hub import x3d_m, x3d_s, x3d_xs
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "PyTorchVideo is required for X3D. Install requirements.txt before "
            "constructing the model. For a dependency-light ablation, use "
            "TorchvisionVideoBackbone from configs/ablation/."
        ) from error
    factories = {"x3d_xs": x3d_xs, "x3d_s": x3d_s, "x3d_m": x3d_m}
    return factories[model_name](pretrained=bool(pretrained))


@register()
class X3DBackbone(nn.Module):
    """Return three X3D feature levels as ``[B,C,T,H,W]`` tensors.

    X3D-XS is the default because it has the best latency margin for the
    downstream high-resolution detector.  X3D-S and X3D-M can be selected from
    YAML without changing the detector interface.
    """

    def __init__(
        self,
        feat_strides=(4, 8, 16),
        model_name="x3d_xs",
        pretrained=True,
        checkpoint=None,
        freeze_bn=True,
        expected_channels=(24, 48, 96),
    ):
        super().__init__()
        if model_name not in {"x3d_xs", "x3d_s", "x3d_m"}:
            raise ValueError(f"Unsupported X3D variant: {model_name!r}")
        if tuple(int(value) for value in feat_strides) != (4, 8, 16):
            raise ValueError("X3DBackbone supports feature strides [4, 8, 16].")

        self.model_name = str(model_name)
        self.freeze_bn = bool(freeze_bn)
        self.feat_strides = (4, 8, 16)
        self.out_channels = tuple(int(value) for value in expected_channels)
        if len(self.out_channels) != 3 or any(value <= 0 for value in self.out_channels):
            raise ValueError("expected_channels must contain three positive values")

        full_model = _build_x3d(self.model_name, bool(pretrained))
        stage_count = int(math.log2(self.feat_strides[-1]))
        self.stages = nn.ModuleList(list(full_model.blocks[:stage_count]))
        self.num_levels = len(self.feat_strides)

        if checkpoint:
            self.load_external_checkpoint(checkpoint)
        if self.freeze_bn:
            self._set_bn_eval()

    def _set_bn_eval(self):
        for module in self.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self.freeze_bn:
            self._set_bn_eval()
        return self

    def load_external_checkpoint(self, checkpoint):
        """Load exact-shape X3D tensors from legacy or full-model checkpoints."""

        path = Path(checkpoint).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"X3D checkpoint not found: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(payload, dict):
            if isinstance(payload.get("ema"), dict):
                payload = payload["ema"].get("module", payload["ema"])
            elif isinstance(payload.get("model"), dict):
                payload = payload["model"]
            elif isinstance(payload.get("state_dict"), dict):
                payload = payload["state_dict"]
        if not isinstance(payload, dict):
            raise TypeError("Unsupported checkpoint format for X3D.")

        own = self.state_dict()
        matched = {}
        prefixes = (
            "module.backbone.",
            "backbone.",
            "model.backbone.",
            "module.",
        )
        for key, value in payload.items():
            candidates = [key]
            for prefix in prefixes:
                if key.startswith(prefix):
                    candidates.append(key[len(prefix) :])
            expanded = list(candidates)
            for candidate in candidates:
                # Legacy wrapper stored truncated blocks as model.<index> while
                # this wrapper names them stages.<index>.
                if candidate.startswith("model."):
                    expanded.append("stages." + candidate[len("model.") :])
                if candidate.startswith("stages."):
                    expanded.append("model." + candidate[len("stages.") :])
            for candidate in expanded:
                if candidate in own and own[candidate].shape == value.shape:
                    matched[candidate] = value
                    break
        if not matched:
            raise RuntimeError(f"No matching X3D tensors found in {path}.")
        missing, unexpected = self.load_state_dict(matched, strict=False)
        print(
            f"[X3D] loaded {len(matched)} tensors from {path}; "
            f"remaining={len(missing)}, unexpected={len(unexpected)}"
        )

    def forward(self, video):
        if video.ndim != 5:
            raise ValueError(f"X3D expects [B,C,T,H,W], got {tuple(video.shape)}")
        features = []
        x = video
        for index, stage in enumerate(self.stages):
            x = stage(x)
            if index >= len(self.stages) - self.num_levels:
                features.append(x)
        actual_channels = tuple(int(feature.shape[1]) for feature in features)
        if actual_channels != self.out_channels:
            raise RuntimeError(
                f"Unexpected {self.model_name} stage channels {actual_channels}; "
                f"expected {self.out_channels}. Set expected_channels explicitly "
                "only after verifying the installed PyTorchVideo version."
            )
        return features


@register()
class X3D_xs(X3DBackbone):
    """Backward-compatible registry name used by the original YAML."""

    def __init__(
        self,
        feat_strides=(4, 8, 16),
        model_name="x3d_xs",
        pretrained=True,
        checkpoint=None,
        freeze_bn=True,
        expected_channels=(24, 48, 96),
    ):
        super().__init__(
            feat_strides=feat_strides,
            model_name=model_name,
            pretrained=pretrained,
            checkpoint=checkpoint,
            freeze_bn=freeze_bn,
            expected_channels=expected_channels,
        )
