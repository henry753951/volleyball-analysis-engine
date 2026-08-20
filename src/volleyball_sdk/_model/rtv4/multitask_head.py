"""Modular Volleyball multitask heads.

Human pose remains query/box-relative because each person is an object instance.
Court pose is intentionally different: one scene-level evidence branch predicts
10 spatial anchor heatmaps and one court-line heatmap from the center frame,
fits a confidence-weighted ridge-stabilized homography, expands the ten anchors to Court60 and
performs lightweight local point refinement.

There is deliberately NO separate Court State classifier.  Court validity is
computed from image evidence itself (anchor peaks, line response and projective
consistency) and is calibrated in the postprocessor.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core import register
from ..volleyball_metadata import (
    COURT_CANONICAL_BASE_XY,
    COURT_DENSE_EDGE_BASE_PAIRS,
    PERSON_CLASS_ID,
)
from .keypoint_ops import decode_keypoints_relative

__all__ = ["QueryMultitaskHead"]


class _BranchMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        *,
        dropout: float = 0.0,
        final_bias: float = 0.0,
        zero_final: bool = False,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, output_dim),
        )
        nn.init.xavier_uniform_(self.net[1].weight)
        nn.init.zeros_(self.net[1].bias)
        if zero_final:
            nn.init.zeros_(self.net[-1].weight)
        else:
            nn.init.normal_(self.net[-1].weight, std=0.01)
        nn.init.constant_(self.net[-1].bias, float(final_bias))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _PoseBranch(nn.Module):
    """BBox-relative human pose plus feature-aligned point refinement."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        feature_channels: int,
        pose_hidden_dim: int,
        feature_dim: int,
        num_keypoints: int,
        refinement_scale: float,
        dropout: float,
        visibility_prior: float,
        enable_refinement: bool,
    ) -> None:
        super().__init__()
        self.num_keypoints = int(num_keypoints)
        self.refinement_scale = float(refinement_scale)
        self.enable_refinement = bool(enable_refinement)
        self.coarse = _BranchMLP(
            hidden_dim,
            pose_hidden_dim,
            self.num_keypoints * 2,
            dropout=dropout,
            zero_final=True,
        )
        prior = min(max(float(visibility_prior), 1.0e-4), 1.0 - 1.0e-4)
        visibility_bias = math.log(prior / (1.0 - prior))
        self.visibility = _BranchMLP(
            hidden_dim,
            pose_hidden_dim,
            self.num_keypoints,
            dropout=dropout,
            final_bias=visibility_bias,
        )
        if self.enable_refinement:
            self.sample_projection = nn.Sequential(
                nn.Linear(feature_channels, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.SiLU(inplace=True),
            )
            self.query_projection = nn.Sequential(
                nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, feature_dim)
            )
            self.keypoint_embedding = nn.Parameter(torch.empty(self.num_keypoints, feature_dim))
            self.context_norm = nn.LayerNorm(feature_dim)
            self.offset_head = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.SiLU(inplace=True),
                nn.Linear(feature_dim, 2),
            )
            self.visibility_refine = nn.Sequential(
                nn.Linear(feature_dim, max(16, feature_dim // 2)),
                nn.SiLU(inplace=True),
                nn.Linear(max(16, feature_dim // 2), 1),
            )
            nn.init.normal_(self.keypoint_embedding, std=0.02)
            nn.init.normal_(self.offset_head[-1].weight, std=1.0e-3)
            nn.init.zeros_(self.offset_head[-1].bias)
            nn.init.normal_(self.visibility_refine[-1].weight, std=1.0e-3)
            nn.init.zeros_(self.visibility_refine[-1].bias)
        else:
            self.sample_projection = None
            self.query_projection = None
            self.register_parameter("keypoint_embedding", None)
            self.context_norm = None
            self.offset_head = None
            self.visibility_refine = None

    @staticmethod
    def _sample_points(feature: torch.Tensor, points_xy: torch.Tensor) -> torch.Tensor:
        batch, queries, keypoints, _ = points_xy.shape
        grid = points_xy.mul(2.0).sub(1.0).reshape(batch, queries * keypoints, 1, 2)
        sampled = F.grid_sample(
            feature, grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        return (
            sampled.squeeze(-1).transpose(1, 2).reshape(batch, queries, keypoints, feature.shape[1])
        )

    def forward(self, query_features, boxes, feature, *, refine: bool):
        batch, queries = query_features.shape[:2]
        coarse = torch.sigmoid(self.coarse(query_features)).reshape(
            batch, queries, self.num_keypoints, 2
        )
        visibility = self.visibility(query_features)
        keypoints = coarse
        if refine and self.enable_refinement:
            if feature is None:
                raise ValueError("Human pose refinement requires a feature map")
            absolute = decode_keypoints_relative(coarse, boxes).clamp(0.0, 1.0)
            sampled = self.sample_projection(self._sample_points(feature, absolute))
            context = (
                sampled
                + self.query_projection(query_features)[:, :, None, :]
                + self.keypoint_embedding[None, None, :, :]
            )
            context = self.context_norm(context)
            offset = torch.tanh(self.offset_head(context)) * self.refinement_scale
            keypoints = (coarse + offset).clamp(0.0, 1.0)
            visibility = visibility + self.visibility_refine(context).squeeze(-1)
        return keypoints, visibility, coarse


def _group_norm(channels: int) -> nn.GroupNorm:
    groups = min(8, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(max(1, groups), channels)


class _CourtEvidenceBranch(nn.Module):
    """Scene-level Court60 from explicit spatial evidence, without state head."""

    def __init__(
        self,
        *,
        input_channels: int,
        context_channels: int,
        evidence_dim: int = 32,
        refine_dim: int = 96,
        heatmap_temperature: float = 0.35,
        refinement_scale: float = 0.025,
        anchor_support_center: float = 2.5,
        anchor_support_temperature: float = 0.5,
        anchor_spread_scale: float = 0.020,
        geometry_score_scale: float = 0.035,
        homography_ridge: float = 1.0e-2,
        homography_prior_strength: float = 1.0,
        projection_denominator_floor: float = 0.02,
        projection_margin: float = 2.0,
    ) -> None:
        super().__init__()
        if input_channels <= 0 or context_channels <= 0 or evidence_dim <= 0 or refine_dim <= 0:
            raise ValueError("Court evidence dimensions must be positive")
        if not 0.05 <= heatmap_temperature <= 2.0:
            raise ValueError("heatmap_temperature must be in [0.05,2]")
        if anchor_support_temperature <= 0:
            raise ValueError("anchor_support_temperature must be positive")
        if anchor_spread_scale <= 0:
            raise ValueError("anchor_spread_scale must be positive")
        self.input_channels = int(input_channels)
        self.context_channels = int(context_channels)
        self.evidence_dim = int(evidence_dim)
        self.refine_dim = int(refine_dim)
        self.heatmap_temperature = float(heatmap_temperature)
        self.refinement_scale = float(refinement_scale)
        self.anchor_support_center = float(anchor_support_center)
        self.anchor_support_temperature = float(anchor_support_temperature)
        self.anchor_spread_scale = float(anchor_spread_scale)
        self.geometry_score_scale = float(geometry_score_scale)
        self.homography_ridge = float(homography_ridge)
        self.homography_prior_strength = float(homography_prior_strength)
        self.projection_denominator_floor = float(projection_denominator_floor)
        self.projection_margin = float(projection_margin)
        if self.homography_ridge <= 0:
            raise ValueError("homography_ridge must be positive")
        if self.homography_prior_strength < 0:
            raise ValueError("homography_prior_strength must be non-negative")
        if self.projection_denominator_floor <= 0:
            raise ValueError("projection_denominator_floor must be positive")
        if self.projection_margin <= 0:
            raise ValueError("projection_margin must be positive")

        # Very cheap spatial head: depthwise evidence extraction followed by a
        # small pointwise projection.  It retains HxW instead of pooling the
        # image into one scene vector.
        self.evidence = nn.Sequential(
            nn.Conv2d(
                self.input_channels,
                self.input_channels,
                kernel_size=3,
                padding=1,
                groups=self.input_channels,
                bias=False,
            ),
            _group_norm(self.input_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.input_channels, self.evidence_dim, kernel_size=1, bias=False),
            _group_norm(self.evidence_dim),
            nn.SiLU(inplace=True),
        )
        # Cheap semantic context comes from the already-computed lowest-
        # resolution projected D-FINE feature (normally stride 16). It remains
        # spatial, so it can disambiguate which similar-looking line junction is
        # LE/LA/C/... without reverting to global pooled coordinate regression.
        self.context_projection = nn.Sequential(
            nn.Conv2d(self.context_channels, self.evidence_dim, kernel_size=1, bias=False),
            _group_norm(self.evidence_dim),
            nn.SiLU(inplace=True),
        )
        self.context_scale = nn.Parameter(torch.tensor(0.10, dtype=torch.float32))
        self.anchor_head = nn.Conv2d(self.evidence_dim, 10, kernel_size=1)
        self.line_head = nn.Conv2d(self.evidence_dim, 1, kernel_size=1)

        self.sample_projection = nn.Sequential(
            nn.Linear(self.evidence_dim, self.refine_dim),
            nn.LayerNorm(self.refine_dim),
            nn.SiLU(inplace=True),
        )
        self.coordinate_projection = nn.Linear(2, self.refine_dim)
        self.point_embedding = nn.Parameter(torch.empty(60, self.refine_dim))
        self.refine_norm = nn.LayerNorm(self.refine_dim)
        self.offset_head = nn.Sequential(
            nn.Linear(self.refine_dim, self.refine_dim),
            nn.SiLU(inplace=True),
            nn.Linear(self.refine_dim, 2),
        )
        nn.init.normal_(self.point_embedding, std=0.02)
        nn.init.normal_(self.offset_head[-1].weight, std=1.0e-3)
        nn.init.zeros_(self.offset_head[-1].bias)
        # Start conservative: negative court samples should quickly push these
        # logits down, while positive heatmap losses establish peaks.
        nn.init.constant_(self.anchor_head.bias, -2.0)
        nn.init.constant_(self.line_head.bias, -2.0)

        self.register_buffer(
            "canonical_base",
            torch.tensor(COURT_CANONICAL_BASE_XY, dtype=torch.float32),
            persistent=False,
        )

    @staticmethod
    def _coordinate_grid(height: int, width: int, reference: torch.Tensor) -> torch.Tensor:
        y = (torch.arange(height, device=reference.device, dtype=reference.dtype) + 0.5) / height
        x = (torch.arange(width, device=reference.device, dtype=reference.dtype) + 0.5) / width
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack((xx, yy), dim=-1).reshape(-1, 2)

    def _softargmax(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Keep the coordinate expectation in FP32. This branch is tiny compared
        # with X3D/D-FINE, while softmax/einsum precision directly controls the
        # conditioning of the downstream projective solve.
        batch, keypoints, height, width = logits.shape
        with torch.autocast(device_type=logits.device.type, enabled=False):
            logits32 = logits.float()
            flat = logits32.flatten(2)
            distribution = torch.softmax(flat / float(self.heatmap_temperature), dim=-1)
            grid = self._coordinate_grid(height, width, logits32)
            coordinates = torch.einsum("bkn,nc->bkc", distribution, grid)
            confidence = logits32.sigmoid().flatten(2).amax(dim=-1)
        return coordinates, confidence

    def _weighted_homography(
        self, source: torch.Tensor, destination: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        """Stable confidence-weighted 8-DoF homography in an FP32 island.

        Early Court heatmaps are often nearly uniform, which makes the raw DLT
        system rank-deficient.  We therefore use three stabilizers:

        1. the complete normal-equation solve stays outside AMP;
        2. Tikhonov regularization is toward the identity homography rather than
           toward the zero vector;
        3. regularization automatically becomes stronger when anchor confidence
           is weak, then relaxes as spatial evidence becomes reliable.
        """

        batch, count, _ = destination.shape
        with torch.autocast(device_type=destination.device.type, enabled=False):
            destination32 = destination.float()
            weights32 = weights.float().clamp(0.0, 1.0)
            src = source.float().to(destination32.device).unsqueeze(0).expand(batch, -1, -1)
            x, y = src.unbind(dim=-1)
            u, v = destination32.unbind(dim=-1)
            ones = torch.ones_like(x)
            zeros = torch.zeros_like(x)
            row_u = torch.stack((x, y, ones, zeros, zeros, zeros, -u * x, -u * y), dim=-1)
            row_v = torch.stack((zeros, zeros, zeros, x, y, ones, -v * x, -v * y), dim=-1)
            a = torch.stack((row_u, row_v), dim=2).reshape(batch, count * 2, 8)
            b = torch.stack((u, v), dim=2).reshape(batch, count * 2, 1)

            # Do not let weak/untrained heatmaps make the system numerically
            # empty. Their contribution remains tiny, while the identity prior
            # carries the degenerate initial state safely.
            row_weight = weights32.clamp_min(1.0e-4).sqrt()
            row_weight = row_weight.unsqueeze(-1).repeat(1, 1, 2)
            row_weight = row_weight.reshape(batch, count * 2, 1)
            aw = a * row_weight
            bw = b * row_weight
            normal = aw.transpose(1, 2) @ aw
            rhs = aw.transpose(1, 2) @ bw

            identity_solution = (
                destination32.new_tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
                .view(1, 8, 1)
                .expand(batch, -1, -1)
            )
            mean_confidence = weights32.mean(dim=1, keepdim=True).unsqueeze(-1)
            adaptive = 1.0 + 4.0 * (1.0 - mean_confidence).square()
            ridge = float(self.homography_ridge) * adaptive
            eye = torch.eye(8, device=normal.device, dtype=torch.float32).unsqueeze(0)
            regularizer = ridge * eye
            rhs = rhs + regularizer @ (identity_solution * float(self.homography_prior_strength))
            solution = torch.linalg.solve(normal + regularizer, rhs).squeeze(-1)
            solution = torch.nan_to_num(solution, nan=0.0, posinf=4.0, neginf=-4.0)
            h = torch.cat((solution, solution.new_ones((batch, 1))), dim=-1)
        return h.reshape(batch, 3, 3)

    def _project(self, points: torch.Tensor, homography: torch.Tensor) -> torch.Tensor:
        # Projective division is another numerically sensitive geometry op. Keep
        # it in FP32, preserve denominator sign, and bound only catastrophic
        # initialization-time extrapolations. The default [-2, 3] normalized
        # range still permits points multiple image widths outside the frame.
        with torch.autocast(device_type=homography.device.type, enabled=False):
            h32 = homography.float()
            batch = h32.shape[0]
            points32 = points.float().to(h32.device).unsqueeze(0).expand(batch, -1, -1)
            ones = torch.ones_like(points32[..., :1])
            homogeneous = torch.cat((points32, ones), dim=-1)
            projected = torch.einsum("bij,bkj->bki", h32, homogeneous)
            denominator = projected[..., 2:3]
            sign = torch.where(
                denominator >= 0.0,
                torch.ones_like(denominator),
                -torch.ones_like(denominator),
            )
            safe = sign * denominator.abs().clamp_min(float(self.projection_denominator_floor))
            xy = projected[..., :2] / safe
            lower = -float(self.projection_margin)
            upper = 1.0 + float(self.projection_margin)
            xy = torch.nan_to_num(xy, nan=0.5, posinf=upper, neginf=lower).clamp(lower, upper)
        return xy

    @staticmethod
    def _expand_base_to_60(base: torch.Tensor) -> torch.Tensor:
        dense = [base]
        for first, second in COURT_DENSE_EDGE_BASE_PAIRS:
            a = base[:, first]
            b = base[:, second]
            samples = [a + (b - a) * (numerator / 6.0) for numerator in range(1, 6)]
            dense.append(torch.stack(samples, dim=1))
        return torch.cat(dense, dim=1)

    @staticmethod
    def _expand_confidence_to_60(confidence: torch.Tensor) -> torch.Tensor:
        dense = [confidence]
        for first, second in COURT_DENSE_EDGE_BASE_PAIRS:
            edge_conf = torch.minimum(confidence[:, first], confidence[:, second])
            dense.append(edge_conf[:, None].expand(-1, 5))
        return torch.cat(dense, dim=1)

    def _anchor_geometry_evidence(
        self, points: torch.Tensor, confidence: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Continuous anchor evidence without a hard visible-point count gate.

        ``support_score`` is a smooth effective-anchor-count statistic, while
        ``spread_score`` measures whether confident anchors span two image
        dimensions rather than collapsing onto one short/collinear region.
        ``peak_score`` keeps the confidence tied to actual heatmap peaks.

        This intentionally allows three strong, well-spread anchors plus line
        evidence to remain usable, while several collinear or weak peaks receive
        a low score.  No Court State classifier is introduced.
        """

        with torch.autocast(device_type=points.device.type, enabled=False):
            confidence32 = confidence.float().clamp(0.0, 1.0)
            points32 = points.float()
            topk = min(3, confidence32.shape[1])
            peak_score = confidence32.topk(topk, dim=1).values.mean(dim=1)

            squared = confidence32.square()
            effective_count = squared.sum(dim=1) / squared.amax(dim=1).clamp_min(1.0e-6)
            support_score = torch.sigmoid(
                (effective_count - float(self.anchor_support_center))
                / float(self.anchor_support_temperature)
            )

            weights = squared / squared.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
            centroid = (points32 * weights.unsqueeze(-1)).sum(dim=1, keepdim=True)
            centered = points32 - centroid
            covariance = torch.einsum("bk,bki,bkj->bij", weights, centered, centered)
            # For a 2x2 covariance det = a*d-b*c; spelling it out avoids a
            # second linalg backward path and is exact for this case.
            determinant = (
                covariance[:, 0, 0] * covariance[:, 1, 1]
                - covariance[:, 0, 1] * covariance[:, 1, 0]
            ).clamp_min(0.0)
            spread_area = determinant.sqrt()
            spread_score = 1.0 - torch.exp(-spread_area / float(self.anchor_spread_scale))
            spread_score = spread_score.clamp(0.0, 1.0)

            anchor_score = (
                peak_score.clamp_min(1.0e-6)
                * support_score.clamp_min(1.0e-6)
                * spread_score.clamp_min(1.0e-6)
            ).pow(1.0 / 3.0)
        return anchor_score, peak_score, support_score, spread_score

    @staticmethod
    def _sample_feature(feature: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        # Off-image geometry is valid for Court60, but local image refinement is
        # intentionally zero there.
        in_frame = (
            (points[..., 0] >= 0.0)
            & (points[..., 0] <= 1.0)
            & (points[..., 1] >= 0.0)
            & (points[..., 1] <= 1.0)
        )
        grid = points.clamp(0.0, 1.0).mul(2.0).sub(1.0).unsqueeze(2)
        sampled = (
            F.grid_sample(feature, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
            .squeeze(-1)
            .transpose(1, 2)
        )
        return sampled, in_frame

    def forward(
        self, center_feature: torch.Tensor, context_feature: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if center_feature.ndim != 4 or center_feature.shape[1] != self.input_channels:
            raise ValueError(
                "Court center feature must be [B,C,H,W] with C="
                f"{self.input_channels}, got {tuple(center_feature.shape)}"
            )
        if context_feature.ndim != 4 or context_feature.shape[1] != self.context_channels:
            raise ValueError(
                "Court context feature must be [B,C,H,W] with C="
                f"{self.context_channels}, got {tuple(context_feature.shape)}"
            )
        evidence = self.evidence(center_feature)
        context = self.context_projection(context_feature)
        context = F.interpolate(
            context, size=evidence.shape[-2:], mode="bilinear", align_corners=False
        )
        evidence = evidence + self.context_scale.to(evidence.dtype) * context
        anchor_logits = self.anchor_head(evidence)
        line_logits = self.line_head(evidence)
        observed_base, anchor_confidence = self._softargmax(anchor_logits)

        homography = self._weighted_homography(
            self.canonical_base, observed_base, anchor_confidence
        )
        homography_base = self._project(self.canonical_base, homography)

        # High-confidence visible anchors follow the spatial peak; low-confidence
        # anchors are completed by the common projective geometry.
        blend = anchor_confidence.unsqueeze(-1)
        base_geometry = blend * observed_base + (1.0 - blend) * homography_base
        dense_geometry = self._expand_base_to_60(base_geometry)

        sampled, in_frame = self._sample_feature(evidence, dense_geometry)
        context = (
            self.sample_projection(sampled)
            + self.coordinate_projection(dense_geometry)
            + self.point_embedding[None, :, :]
        )
        context = self.refine_norm(context)
        offset = torch.tanh(self.offset_head(context)) * self.refinement_scale
        offset = offset * in_frame.unsqueeze(-1).to(offset.dtype)
        refined = dense_geometry + offset

        line_probability = line_logits.sigmoid()
        line_sample, line_in_frame = self._sample_feature(line_probability, dense_geometry)
        line_sample = line_sample.squeeze(-1)
        dense_anchor_conf = self._expand_confidence_to_60(anchor_confidence)
        point_confidence = (
            dense_anchor_conf.clamp_min(1.0e-6) * line_sample.clamp_min(1.0e-6)
        ).sqrt()
        point_confidence = point_confidence * line_in_frame.to(point_confidence.dtype)

        valid_count = line_in_frame.sum(dim=1).clamp_min(1)
        line_score = (line_sample * line_in_frame.to(line_sample.dtype)).sum(
            dim=1
        ) / valid_count.to(line_sample.dtype)
        (
            anchor_score,
            anchor_peak_score,
            anchor_support_score,
            anchor_spread_score,
        ) = self._anchor_geometry_evidence(observed_base, anchor_confidence)
        reprojection_error = torch.linalg.vector_norm(observed_base - homography_base, dim=-1)
        geometry_error = (reprojection_error * anchor_confidence).sum(
            dim=1
        ) / anchor_confidence.sum(dim=1).clamp_min(1.0e-4)
        geometry_score = torch.exp(-geometry_error / max(self.geometry_score_scale, 1.0e-6)).clamp(
            0.0, 1.0
        )
        court_evidence_score = (
            anchor_score.clamp_min(1.0e-6)
            * line_score.clamp_min(1.0e-6)
            * geometry_score.clamp_min(1.0e-6)
        ).pow(1.0 / 3.0)

        return {
            "pred_court_anchor_logits": anchor_logits,
            "pred_court_line_logits": line_logits,
            "pred_court_anchor_keypoints": observed_base,
            "pred_court_anchor_confidence": anchor_confidence,
            "pred_court_anchor_score": anchor_score,
            "pred_court_anchor_peak_score": anchor_peak_score,
            "pred_court_anchor_support_score": anchor_support_score,
            "pred_court_anchor_spread_score": anchor_spread_score,
            "pred_court_homography": homography,
            "pred_court_homography_keypoints": homography_base,
            "pred_court_geometry_keypoints": dense_geometry,
            "pred_court_keypoints": refined,
            "pred_court_visibility": point_confidence,
            "pred_court_line_score": line_score,
            "pred_court_geometry_score": geometry_score,
            "pred_court_evidence_score": court_evidence_score,
        }


@register()
class QueryMultitaskHead(nn.Module):
    """Query semantic heads + one scene-level court evidence branch."""

    __share__ = ["num_actions", "num_group_classes", "use_focal_loss"]

    def __init__(
        self,
        hidden_dim=384,
        feature_channels=384,
        num_actions=9,
        num_group_classes=9,
        use_focal_loss=True,
        num_human_keypoints=17,
        num_court_keypoints=60,
        action_hidden_dim=256,
        group_hidden_dim=256,
        pose_hidden_dim=256,
        pose_feature_dim=96,
        pose_refine_level=0,
        pose_refinement=True,
        human_refinement_scale=0.15,
        dropout=0.05,
        actionness_prior=0.01,
        human_visibility_prior=0.80,
        group_attention_temperature=1.0,
        detach_group_person_prior=True,
        court_center_feature_channels=24,
        court_evidence_dim=32,
        court_refine_dim=96,
        court_heatmap_temperature=0.35,
        court_refinement_scale=0.025,
        court_anchor_support_center=2.5,
        court_anchor_support_temperature=0.5,
        court_anchor_spread_scale=0.020,
        court_geometry_score_scale=0.035,
        court_homography_ridge=1.0e-2,
        court_homography_prior_strength=1.0,
        court_projection_denominator_floor=0.02,
        court_projection_margin=2.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.feature_channels = int(feature_channels)
        self.num_actions = int(num_actions)
        self.num_group_classes = int(num_group_classes)
        self.use_focal_loss = bool(use_focal_loss)
        self.num_human_keypoints = int(num_human_keypoints)
        self.num_court_keypoints = int(num_court_keypoints)
        if self.num_court_keypoints not in (0, 60):
            raise ValueError("Evidence-based court branch uses exactly 60 points")
        self.pose_refine_level = int(pose_refine_level)
        self.pose_refinement = bool(pose_refinement)
        self.group_attention_temperature = float(group_attention_temperature)
        self.detach_group_person_prior = bool(detach_group_person_prior)

        self.action_head = (
            _BranchMLP(
                self.hidden_dim,
                int(action_hidden_dim),
                self.num_actions,
                dropout=float(dropout),
            )
            if self.num_actions > 0
            else None
        )
        actionness_bias = math.log(actionness_prior / (1.0 - actionness_prior))
        self.actionness_head = (
            _BranchMLP(
                self.hidden_dim,
                max(64, int(action_hidden_dim) // 2),
                1,
                dropout=float(dropout),
                final_bias=actionness_bias,
            )
            if self.num_actions > 0
            else None
        )
        self.group_head = (
            _BranchMLP(
                self.hidden_dim,
                int(group_hidden_dim),
                self.num_group_classes,
                dropout=float(dropout),
            )
            if self.num_group_classes > 0
            else None
        )
        self.group_attention = (
            _BranchMLP(
                self.hidden_dim,
                max(64, int(group_hidden_dim) // 2),
                1,
                dropout=float(dropout),
                zero_final=True,
            )
            if self.num_group_classes > 0
            else None
        )
        self.human_pose = (
            _PoseBranch(
                hidden_dim=self.hidden_dim,
                feature_channels=self.feature_channels,
                pose_hidden_dim=int(pose_hidden_dim),
                feature_dim=int(pose_feature_dim),
                num_keypoints=self.num_human_keypoints,
                refinement_scale=float(human_refinement_scale),
                dropout=float(dropout),
                visibility_prior=float(human_visibility_prior),
                enable_refinement=self.pose_refinement,
            )
            if self.num_human_keypoints > 0
            else None
        )
        self.court_evidence = (
            _CourtEvidenceBranch(
                input_channels=int(court_center_feature_channels),
                context_channels=self.feature_channels,
                evidence_dim=int(court_evidence_dim),
                refine_dim=int(court_refine_dim),
                heatmap_temperature=float(court_heatmap_temperature),
                refinement_scale=float(court_refinement_scale),
                anchor_support_center=float(court_anchor_support_center),
                anchor_support_temperature=float(court_anchor_support_temperature),
                anchor_spread_scale=float(court_anchor_spread_scale),
                geometry_score_scale=float(court_geometry_score_scale),
                homography_ridge=float(court_homography_ridge),
                homography_prior_strength=float(court_homography_prior_strength),
                projection_denominator_floor=float(court_projection_denominator_floor),
                projection_margin=float(court_projection_margin),
            )
            if self.num_court_keypoints > 0
            else None
        )

    def _pose_feature(self, projected_features: Sequence[torch.Tensor] | None):
        if not self.pose_refinement:
            return None
        if projected_features is None:
            raise ValueError("projected_features are required for human pose refinement")
        feature = projected_features[self.pose_refine_level]
        if feature.ndim != 4 or feature.shape[1] != self.feature_channels:
            raise ValueError(
                f"Human pose feature must be [B,{self.feature_channels},H,W], "
                f"got {tuple(feature.shape)}"
            )
        return feature

    def forward(
        self,
        query_features: torch.Tensor,
        object_logits: torch.Tensor,
        boxes: torch.Tensor,
        projected_features: Sequence[torch.Tensor] | None = None,
        *,
        center_feature: torch.Tensor | None = None,
        refine_pose: bool = True,
        include_court: bool = True,
    ) -> dict[str, torch.Tensor]:
        if query_features.ndim != 3:
            raise ValueError("query_features must be [B,Q,D]")
        output: dict[str, torch.Tensor] = {}
        if self.action_head is not None:
            output["pred_actions"] = self.action_head(query_features)
            output["pred_actionness"] = self.actionness_head(query_features).squeeze(-1)

        if self.group_head is not None:
            query_logits = self.group_head(query_features)
            raw_attention = self.group_attention(query_features).squeeze(-1)
            if self.use_focal_loss:
                person_prior = object_logits[..., PERSON_CLASS_ID].sigmoid()
            else:
                person_prior = object_logits.softmax(dim=-1)[..., PERSON_CLASS_ID]
            person_prior = person_prior.clamp_min(1.0e-6)
            if self.detach_group_person_prior:
                person_prior = person_prior.detach()
            attention = torch.softmax(
                (raw_attention + person_prior.log()) / self.group_attention_temperature,
                dim=1,
            )
            output["pred_group_logits"] = query_logits
            output["pred_group_attention"] = attention
            output["pred_group_frame_logits"] = (query_logits * attention.unsqueeze(-1)).sum(dim=1)

        if self.human_pose is not None:
            pose_feature = self._pose_feature(projected_features) if refine_pose else None
            keypoints, visibility, coarse = self.human_pose(
                query_features, boxes, pose_feature, refine=bool(refine_pose)
            )
            output["pred_human_keypoints"] = keypoints
            output["pred_human_visibility"] = visibility
            output["pred_human_keypoints_coarse"] = coarse

        if include_court and self.court_evidence is not None:
            if center_feature is None:
                raise ValueError("Court evidence branch requires center_feature")
            if projected_features is None or len(projected_features) == 0:
                raise ValueError("Court evidence branch requires projected context features")
            output.update(self.court_evidence(center_feature, projected_features[-1]))
        return output
