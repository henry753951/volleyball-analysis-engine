"""Video backbone + temporal adapter + HybridEncoder + D-FINE multitask model."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..core import register

__all__ = ["RTv4"]


@register()
class RTv4(nn.Module):
    """Connect replaceable video components while preserving public model I/O."""

    __inject__ = ["backbone", "temporal_fusion", "encoder", "decoder"]

    def __init__(
        self,
        backbone: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
        temporal_fusion: nn.Module | None = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.temporal_fusion = temporal_fusion
        self.encoder = encoder
        self.decoder = decoder

    @staticmethod
    def _temporal_stride_from_targets(targets, batch_size: int, device):
        """Return one source-frame jump per clip when targets provide it.

        Validation and deployment may call the model without targets, in which
        case the temporal adapter's configured default is used.  During
        training this path lets one batch contain multiple temporal sampling
        rates without changing the public ``model(video, targets)`` interface.
        """

        if not isinstance(targets, (list, tuple)) or len(targets) != batch_size:
            return None
        values = []
        for target in targets:
            if not isinstance(target, dict) or "temporal_stride" not in target:
                return None
            value = target["temporal_stride"]
            value = torch.as_tensor(value, device=device, dtype=torch.float32).reshape(-1)
            if value.numel() != 1:
                raise ValueError("Each target temporal_stride must be a scalar")
            values.append(value[0])
        stride = torch.stack(values)
        if not bool(torch.isfinite(stride).all()) or bool((stride <= 0).any()):
            raise ValueError("Target temporal_stride values must be finite and positive")
        return stride

    def forward(self, video, targets=None):
        multiscale_features = self.backbone(video)
        if self.temporal_fusion is None:
            # Backward-compatible fallback for old YAML files.
            features_2d = [
                feature.flatten(1, 2) if feature.ndim == 5 else feature
                for feature in multiscale_features
            ]
        else:
            temporal_stride = self._temporal_stride_from_targets(
                targets, int(video.shape[0]), video.device
            )
            features_2d = self.temporal_fusion(multiscale_features, temporal_stride=temporal_stride)
        encoded = self.encoder(features_2d)

        # Court evidence must retain center-frame spatial information.  Passing
        # the raw first backbone level avoids temporal averaging and keeps the
        # added compute small (the court branch is depthwise/lightweight).
        raw_court_feature = multiscale_features[0]
        if raw_court_feature.ndim == 5:
            center_mode = getattr(self.temporal_fusion, "center_mode", "middle")
            center_index = (
                raw_court_feature.shape[2] - 1
                if center_mode == "last"
                else raw_court_feature.shape[2] // 2
            )
            raw_court_feature = raw_court_feature[:, :, center_index]
        elif raw_court_feature.ndim != 4:
            raise ValueError(f"Unexpected raw court feature shape {tuple(raw_court_feature.shape)}")
        return self.decoder(encoded, targets, center_feature=raw_court_feature)

    def deploy(self):
        self.eval()
        for module in self.modules():
            if hasattr(module, "convert_to_deploy"):
                module.convert_to_deploy()
        return self
