"""Lightweight temporal adapters between a 3D video backbone and a 2D neck.

The legacy project reshaped ``[B, C, T, H, W]`` into ``[B, C*T, H, W]``.
That bridge is fast, but it hard-codes the clip length into the neck channels and
provides no explicit temporal order, motion, or frame-importance modelling.

This module keeps the public model I/O unchanged while making temporal fusion a
small, replaceable component.  ``TemporalPyramidFusion`` is the recommended
adapter; ``TemporalFlatten`` is retained for exact legacy ablations.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core import register

__all__ = ["TemporalFlatten", "TemporalPyramidFusion"]


def _valid_group_count(channels: int, requested: int) -> int:
    """Return the largest useful GroupNorm group count that divides channels."""

    requested = max(1, min(int(requested), int(channels)))
    for groups in range(requested, 0, -1):
        if channels % groups == 0:
            return groups
    return 1


@register()
class TemporalFlatten(nn.Module):
    """Exact legacy ``C*T`` bridge.

    This exists only for checkpoint/ablation compatibility.  Its output channel
    count depends on the runtime clip length, so the following encoder must be
    configured with ``in_channels[level] * frame_num``.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        features: Sequence[torch.Tensor],
        temporal_stride: float | torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        del temporal_stride
        outputs: list[torch.Tensor] = []
        for level, feature in enumerate(features):
            if feature.ndim == 5:
                outputs.append(feature.flatten(1, 2))
            elif feature.ndim == 4:
                outputs.append(feature)
            else:
                raise ValueError(
                    f"TemporalFlatten level {level} expects 4D/5D features, "
                    f"got {tuple(feature.shape)}"
                )
        return outputs


class _TemporalLevelFusion(nn.Module):
    """One parameter-efficient temporal fusion block for a feature level."""

    def __init__(
        self,
        channels: int,
        *,
        temporal_kernel_size: int,
        norm_groups: int,
        dropout: float,
        use_motion_branch: bool,
        use_temporal_position: bool,
        max_temporal_stride: float,
        attention_temperature: float,
        center_mode: str,
        residual_scale_init: float,
    ) -> None:
        super().__init__()
        channels = int(channels)
        kernel = int(temporal_kernel_size)
        if kernel <= 0 or kernel % 2 == 0:
            raise ValueError("temporal_kernel_size must be a positive odd integer")
        if attention_temperature <= 0:
            raise ValueError("attention_temperature must be positive")
        if max_temporal_stride <= 0:
            raise ValueError("max_temporal_stride must be positive")
        if center_mode not in {"middle", "last"}:
            raise ValueError("center_mode must be 'middle' or 'last'")
        if not 0.0 <= float(residual_scale_init) <= 1.0:
            raise ValueError("residual_scale_init must be in [0,1]")

        groups = _valid_group_count(channels, norm_groups)
        self.channels = channels
        self.center_mode = center_mode
        self.use_motion_branch = bool(use_motion_branch)
        self.use_temporal_position = bool(use_temporal_position)
        self.max_temporal_stride = float(max_temporal_stride)
        self.attention_temperature = float(attention_temperature)

        self.pre_norm = nn.GroupNorm(groups, channels)
        self.temporal_depthwise = nn.Conv3d(
            channels,
            channels,
            kernel_size=(kernel, 1, 1),
            padding=(kernel // 2, 0, 0),
            groups=channels,
            bias=False,
        )
        self.temporal_pointwise = nn.Conv3d(channels, channels, 1, bias=False)
        self.temporal_norm = nn.GroupNorm(groups, channels)

        if self.use_motion_branch:
            self.motion_depthwise = nn.Conv3d(
                channels,
                channels,
                kernel_size=(kernel, 1, 1),
                padding=(kernel // 2, 0, 0),
                groups=channels,
                bias=False,
            )
            self.motion_pointwise = nn.Conv3d(channels, channels, 1, bias=False)
            self.motion_scale = nn.Parameter(torch.zeros(1, channels, 1, 1, 1))
        else:
            self.motion_depthwise = None
            self.motion_pointwise = None
            self.register_parameter("motion_scale", None)

        if self.use_temporal_position:
            position_hidden = max(16, min(64, channels // 2))
            self.position_mlp = nn.Sequential(
                nn.Linear(2, position_hidden),
                nn.SiLU(inplace=True),
                nn.Linear(position_hidden, channels),
            )
            # Start as a no-op so old pretrained spatial features are not
            # disturbed at initialization.
            nn.init.zeros_(self.position_mlp[-1].weight)
            nn.init.zeros_(self.position_mlp[-1].bias)
        else:
            self.position_mlp = None

        self.frame_score = nn.Conv1d(channels, 1, kernel_size=1, bias=True)
        self.output_gate = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.output_norm = nn.GroupNorm(groups, channels)
        self.output_scale = nn.Parameter(
            torch.full((1, channels, 1, 1), float(residual_scale_init))
        )
        self.dropout = nn.Dropout2d(float(dropout)) if dropout > 0 else nn.Identity()

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        # A residual-preserving initialization: temporal filtering begins as an
        # identity, motion is gated off, and a small LayerScale-style residual
        # perturbs the contextual center feature rather than renormalizing it.
        nn.init.zeros_(self.temporal_depthwise.weight)
        center = self.temporal_depthwise.kernel_size[0] // 2
        with torch.no_grad():
            self.temporal_depthwise.weight[:, 0, center, 0, 0] = 1.0
        nn.init.dirac_(self.temporal_pointwise.weight)
        nn.init.zeros_(self.frame_score.weight)
        nn.init.zeros_(self.frame_score.bias)
        nn.init.zeros_(self.output_gate.weight)
        nn.init.zeros_(self.output_gate.bias)
        if self.motion_depthwise is not None:
            # The branch input is already an adjacent-frame difference.  Start
            # its depthwise filter as temporal identity while keeping the
            # learnable output scale at zero.  This preserves an exact no-op at
            # initialization *and* gives ``motion_scale`` a non-zero gradient;
            # zero-initializing both the filter and scale would permanently
            # deadlock the entire motion branch.
            nn.init.zeros_(self.motion_depthwise.weight)
            motion_center = self.motion_depthwise.kernel_size[0] // 2
            with torch.no_grad():
                self.motion_depthwise.weight[:, 0, motion_center, 0, 0] = 1.0
            nn.init.dirac_(self.motion_pointwise.weight)

    def _temporal_positions(
        self,
        temporal_length: int,
        temporal_stride: float | torch.Tensor,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.center_mode == "last":
            offsets = torch.arange(-(temporal_length - 1), 1, device=device, dtype=torch.float32)
        else:
            center = (temporal_length - 1) / 2.0
            offsets = torch.arange(temporal_length, device=device, dtype=torch.float32) - center
        stride = torch.as_tensor(temporal_stride, device=device, dtype=torch.float32).reshape(-1)
        if stride.numel() == 1:
            stride = stride.expand(batch_size)
        elif stride.numel() != batch_size:
            raise ValueError(
                "temporal_stride must be scalar or contain one value per clip; "
                f"got {stride.numel()} values for batch_size={batch_size}"
            )
        if not bool(torch.isfinite(stride).all()) or bool((stride <= 0).any()):
            raise ValueError("temporal_stride values must be finite and positive")
        absolute = offsets.unsqueeze(0) * stride.unsqueeze(1)
        signed = absolute / self.max_temporal_stride
        magnitude = absolute.abs() / self.max_temporal_stride
        return torch.stack((signed, magnitude), dim=-1).to(dtype=dtype)

    @staticmethod
    def _temporal_difference(x: torch.Tensor) -> torch.Tensor:
        if x.shape[2] <= 1:
            return torch.zeros_like(x)
        difference = x[:, :, 1:] - x[:, :, :-1]
        # Keep the temporal length and align each difference with its later
        # frame. The first slot has no predecessor and therefore carries zero.
        return F.pad(difference, (0, 0, 0, 0, 1, 0))

    def forward(self, x: torch.Tensor, temporal_stride: float | torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            return x
        if x.ndim != 5:
            raise ValueError(f"Expected [B,C,T,H,W], got {tuple(x.shape)}")
        if x.shape[1] != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {int(x.shape[1])}")

        residual = x
        mixed = self.temporal_pointwise(self.temporal_depthwise(self.pre_norm(x)))
        mixed = self.temporal_norm(mixed)

        if self.position_mlp is not None:
            positions = self._temporal_positions(
                x.shape[2],
                temporal_stride,
                batch_size=x.shape[0],
                device=x.device,
                dtype=x.dtype,
            )
            position_embedding = self.position_mlp(positions).permute(0, 2, 1)
            mixed = mixed + position_embedding[:, :, :, None, None]

        if self.motion_depthwise is not None:
            motion = self._temporal_difference(residual)
            motion = self.motion_pointwise(self.motion_depthwise(motion))
            mixed = mixed + torch.tanh(self.motion_scale) * motion

        # A learned soft selection over frames.  Spatial averaging makes this
        # branch tiny and keeps inference cost nearly independent of H/W.
        frame_tokens = mixed.mean(dim=(-1, -2))
        frame_logits = self.frame_score(frame_tokens).squeeze(1)
        frame_weights = torch.softmax(frame_logits / self.attention_temperature, dim=-1)
        pooled = (mixed * frame_weights[:, None, :, None, None]).sum(dim=2)

        center_index = x.shape[2] - 1 if self.center_mode == "last" else x.shape[2] // 2
        center_feature = residual[:, :, center_index]
        gate = torch.sigmoid(self.output_gate(torch.cat((center_feature, pooled), dim=1)))
        temporal_delta = self.output_norm(pooled - center_feature)
        output = center_feature + torch.tanh(self.output_scale) * gate * temporal_delta
        return self.dropout(output)


@register()
class TemporalPyramidFusion(nn.Module):
    """Fuse every 3D backbone level into an order-aware 2D feature map.

    Parameters are deliberately level-local and depthwise in time.  The module
    accepts any odd clip length at runtime; unlike flattening, changing
    ``frame_num`` does not require changing the encoder channel dimensions.
    ``temporal_stride`` represents the source-frame jump used by the sampler
    and is included in the learned temporal positional encoding.
    """

    def __init__(
        self,
        in_channels=(24, 48, 96),
        temporal_stride=2.0,
        temporal_kernel_size=3,
        norm_groups=8,
        dropout=0.0,
        use_motion_branch=True,
        use_temporal_position=True,
        max_temporal_stride=8.0,
        attention_temperature=1.0,
        center_mode="middle",
        residual_scale_init=0.10,
    ) -> None:
        super().__init__()
        channels = tuple(int(value) for value in in_channels)
        if not channels or any(value <= 0 for value in channels):
            raise ValueError("in_channels must contain positive channel counts")
        if float(temporal_stride) <= 0:
            raise ValueError("temporal_stride must be positive")
        self.in_channels = channels
        self.out_channels = channels
        self.temporal_stride = float(temporal_stride)
        self.levels = nn.ModuleList(
            [
                _TemporalLevelFusion(
                    value,
                    temporal_kernel_size=int(temporal_kernel_size),
                    norm_groups=int(norm_groups),
                    dropout=float(dropout),
                    use_motion_branch=bool(use_motion_branch),
                    use_temporal_position=bool(use_temporal_position),
                    max_temporal_stride=float(max_temporal_stride),
                    attention_temperature=float(attention_temperature),
                    center_mode=str(center_mode),
                    residual_scale_init=float(residual_scale_init),
                )
                for value in channels
            ]
        )

    def set_temporal_stride(self, temporal_stride: float) -> None:
        """Update source-frame spacing for controlled inference ablations."""

        if float(temporal_stride) <= 0:
            raise ValueError("temporal_stride must be positive")
        self.temporal_stride = float(temporal_stride)

    def forward(
        self,
        features: Sequence[torch.Tensor],
        temporal_stride: float | torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        if len(features) != len(self.levels):
            raise ValueError(f"Expected {len(self.levels)} feature levels, got {len(features)}")
        stride = self.temporal_stride if temporal_stride is None else temporal_stride
        return [block(feature, stride) for block, feature in zip(self.levels, features)]

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, temporal_stride={self.temporal_stride}, "
            f"levels={len(self.levels)}"
        )
