"""Inference post-processing for the Volleyball X3D multitask model."""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision

from ..core import register
from ..volleyball_metadata import COURT_CLASS_ID, PERSON_CLASS_ID
from .keypoint_ops import decode_keypoints_relative

__all__ = ["PostProcessor"]


@register()
class PostProcessor(nn.Module):
    """Convert query outputs to image-space detection and semantic results.

    The enhanced head may provide a learned person-prior attention aggregate
    through ``pred_group_frame_logits``.  Legacy checkpoints fall back to the
    original thresholded arithmetic mean, preserving their inference behavior.
    """

    __share__ = [
        "num_classes",
        "num_actions",
        "num_group_classes",
        "use_focal_loss",
    ]

    def __init__(
        self,
        num_classes=3,
        num_actions=9,
        num_group_classes=9,
        use_focal_loss=True,
        num_top_queries=300,
        group_person_score_threshold=0.25,
        group_max_person_queries=18,
        human_pose_ap_topk=20,
        court_pose_ap_topk=1,
        pose_pck_threshold=0.05,
        pose_pcp_threshold=0.50,
        court_pose_conf_threshold=0.35,
        court_detection_conf_threshold=0.15,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.num_actions = int(num_actions)
        self.num_group_classes = int(num_group_classes)
        self.use_focal_loss = bool(use_focal_loss)
        self.num_top_queries = int(num_top_queries)
        self.group_person_score_threshold = float(group_person_score_threshold)
        self.group_max_person_queries = int(group_max_person_queries)
        self.human_pose_ap_topk = int(human_pose_ap_topk)
        self.court_pose_ap_topk = int(court_pose_ap_topk)
        self.pose_pck_threshold = float(pose_pck_threshold)
        self.pose_pcp_threshold = float(pose_pcp_threshold)
        self.court_pose_conf_threshold = float(court_pose_conf_threshold)
        self.court_detection_conf_threshold = float(court_detection_conf_threshold)
        self.deploy_mode = False

    def _select_group_person_queries(self, object_logits):
        if self.use_focal_loss:
            person_scores = object_logits.sigmoid()[:, PERSON_CLASS_ID]
        else:
            person_scores = object_logits.softmax(dim=-1)[:, PERSON_CLASS_ID]
        selected = torch.nonzero(
            person_scores >= self.group_person_score_threshold, as_tuple=False
        ).flatten()
        if selected.numel() == 0:
            selected = person_scores.argmax().view(1)
        if self.group_max_person_queries > 0 and selected.numel() > self.group_max_person_queries:
            _, order = torch.topk(person_scores[selected], self.group_max_person_queries)
            selected = selected[order]
        return selected

    def _frame_group_logits(self, object_logits, group_evidence):
        return torch.stack(
            [
                evidence[self._select_group_person_queries(logits)].mean(dim=0)
                for logits, evidence in zip(object_logits, group_evidence)
            ],
            dim=0,
        )

    def forward(self, outputs, orig_target_sizes):
        logits = outputs["pred_logits"]
        boxes_norm = outputs["pred_boxes"]
        batch_size, query_count = logits.shape[:2]
        sizes_wh = orig_target_sizes.to(boxes_norm).view(batch_size, 1, 2)
        boxes_xyxy = torchvision.ops.box_convert(boxes_norm, in_fmt="cxcywh", out_fmt="xyxy").clamp(
            0.0, 1.0
        )
        boxes_px = boxes_xyxy * sizes_wh.repeat(1, 1, 2)

        if self.use_focal_loss:
            probabilities = logits.sigmoid()
            top_count = min(self.num_top_queries, query_count * self.num_classes)
            scores, flat_indices = torch.topk(probabilities.flatten(1), top_count, dim=-1)
            labels = flat_indices.remainder(self.num_classes)
            query_indices = flat_indices.div(self.num_classes, rounding_mode="floor")
        else:
            probabilities = logits.softmax(dim=-1)
            query_scores, query_labels = probabilities.max(dim=-1)
            top_count = min(self.num_top_queries, query_count)
            scores, query_indices = torch.topk(query_scores, top_count, dim=-1)
            labels = query_labels.gather(1, query_indices)

        selected_boxes_px = boxes_px.gather(1, query_indices.unsqueeze(-1).expand(-1, -1, 4))
        selected_boxes_norm = boxes_norm.gather(1, query_indices.unsqueeze(-1).expand(-1, -1, 4))
        person_output = labels == PERSON_CLASS_ID
        court_output = labels == COURT_CLASS_ID

        action_labels = action_scores = None
        if "pred_actions" in outputs:
            action_logits = outputs["pred_actions"].gather(
                1,
                query_indices.unsqueeze(-1).expand(-1, -1, outputs["pred_actions"].shape[-1]),
            )
            action_probabilities = action_logits.softmax(dim=-1)
            action_scores, action_labels = action_probabilities.max(dim=-1)
            if "pred_actionness" in outputs:
                selected_actionness = outputs["pred_actionness"].gather(1, query_indices).sigmoid()
                action_scores = action_scores * selected_actionness
            else:
                selected_actionness = torch.ones_like(action_scores)
            action_labels = torch.where(
                person_output, action_labels, torch.full_like(action_labels, -1)
            )
            action_scores = torch.where(
                person_output, action_scores, torch.zeros_like(action_scores)
            )

        selected_group_labels = selected_group_scores = None
        frame_group_logits = frame_group_probabilities = None
        if "pred_group_logits" in outputs:
            selected_group_logits = outputs["pred_group_logits"].gather(
                1,
                query_indices.unsqueeze(-1).expand(-1, -1, outputs["pred_group_logits"].shape[-1]),
            )
            selected_group_probabilities = selected_group_logits.softmax(dim=-1)
            selected_group_scores, selected_group_labels = selected_group_probabilities.max(dim=-1)
            selected_group_labels = torch.where(
                person_output,
                selected_group_labels,
                torch.full_like(selected_group_labels, -1),
            )
            selected_group_scores = torch.where(
                person_output,
                selected_group_scores,
                torch.zeros_like(selected_group_scores),
            )
            if "pred_group_frame_logits" in outputs:
                frame_group_logits = outputs["pred_group_frame_logits"].float()
            else:
                frame_group_logits = self._frame_group_logits(logits, outputs["pred_group_logits"])
            frame_group_probabilities = frame_group_logits.softmax(dim=-1)

        human_keypoints_px = None
        human_visibility = None
        if "pred_human_keypoints" in outputs:
            relative = outputs["pred_human_keypoints"].gather(
                1,
                query_indices[:, :, None, None].expand(
                    -1,
                    -1,
                    outputs["pred_human_keypoints"].shape[2],
                    2,
                ),
            )
            human_keypoints_px = (
                decode_keypoints_relative(relative, selected_boxes_norm).clamp(0.0, 1.0)
                * sizes_wh[:, :, None, :]
            )
            human_keypoints_px = torch.where(
                person_output[:, :, None, None],
                human_keypoints_px,
                torch.zeros_like(human_keypoints_px),
            )
            if "pred_human_visibility" in outputs:
                human_visibility = (
                    outputs["pred_human_visibility"]
                    .gather(
                        1,
                        query_indices[:, :, None].expand(
                            -1, -1, outputs["pred_human_visibility"].shape[-1]
                        ),
                    )
                    .sigmoid()
                )
                human_visibility = torch.where(
                    person_output[:, :, None],
                    human_visibility,
                    torch.zeros_like(human_visibility),
                )

        # Scene-level Court60.  No separate state classifier exists: validity
        # is derived from detector confidence + actual spatial court evidence.
        court_keypoints_px = None
        court_visibility = None
        court_valid = None
        court_pose_score = None
        court_combined_score = None
        court_evidence_score = None
        court_line_score = None
        court_geometry_score = None
        court_anchor_confidence = None
        court_anchor_score = None
        court_anchor_peak_score = None
        court_anchor_support_score = None
        court_anchor_spread_score = None
        court_detection_score = probabilities[..., COURT_CLASS_ID].amax(dim=1)
        if "pred_court_keypoints" in outputs:
            scene_court = outputs["pred_court_keypoints"].float()
            if scene_court.ndim != 3 or scene_court.shape[1:] != (60, 2):
                raise ValueError(
                    f"Evidence Court60 output must be [B,60,2], got {tuple(scene_court.shape)}"
                )
            court_keypoints_px = scene_court * sizes_wh
            court_visibility = outputs.get("pred_court_visibility")
            if court_visibility is None:
                court_visibility = scene_court.new_ones((batch_size, 60))
            else:
                court_visibility = court_visibility.float().clamp(0.0, 1.0)
            court_evidence_score = (
                outputs.get("pred_court_evidence_score", scene_court.new_zeros(batch_size))
                .float()
                .clamp(0.0, 1.0)
            )
            court_line_score = (
                outputs.get("pred_court_line_score", scene_court.new_zeros(batch_size))
                .float()
                .clamp(0.0, 1.0)
            )
            court_geometry_score = (
                outputs.get("pred_court_geometry_score", scene_court.new_zeros(batch_size))
                .float()
                .clamp(0.0, 1.0)
            )
            court_anchor_confidence = (
                outputs.get(
                    "pred_court_anchor_confidence",
                    scene_court.new_zeros((batch_size, 10)),
                )
                .float()
                .clamp(0.0, 1.0)
            )
            court_anchor_score = (
                outputs.get("pred_court_anchor_score", scene_court.new_zeros(batch_size))
                .float()
                .clamp(0.0, 1.0)
            )
            court_anchor_peak_score = (
                outputs.get("pred_court_anchor_peak_score", scene_court.new_zeros(batch_size))
                .float()
                .clamp(0.0, 1.0)
            )
            court_anchor_support_score = (
                outputs.get("pred_court_anchor_support_score", scene_court.new_zeros(batch_size))
                .float()
                .clamp(0.0, 1.0)
            )
            court_anchor_spread_score = (
                outputs.get("pred_court_anchor_spread_score", scene_court.new_zeros(batch_size))
                .float()
                .clamp(0.0, 1.0)
            )
            # No hard visible-anchor count gate exists. Pose confidence already
            # fuses heatmap peak strength, smooth effective support, spatial
            # spread, line response and homography consistency. Detector
            # confidence remains an independent end-to-end sanity check.
            court_pose_score = court_evidence_score
            court_combined_score = (
                court_detection_score.clamp_min(1.0e-8) * court_pose_score.clamp_min(1.0e-8)
            ).sqrt()
            court_valid = (court_pose_score >= self.court_pose_conf_threshold) & (
                court_detection_score >= self.court_detection_conf_threshold
            )

        if self.deploy_mode:
            return labels, selected_boxes_px, scores

        results = []
        for batch_index in range(batch_size):
            result = {
                "labels": labels[batch_index],
                "boxes": selected_boxes_px[batch_index],
                "scores": scores[batch_index],
                "query_indices": query_indices[batch_index],
            }
            if action_labels is not None:
                result["action_labels"] = action_labels[batch_index]
                result["action_scores"] = action_scores[batch_index]
                result["actionness_scores"] = selected_actionness[batch_index]
            if selected_group_labels is not None:
                result["query_group_labels"] = selected_group_labels[batch_index]
                result["query_group_scores"] = selected_group_scores[batch_index]
                result["group_logits"] = frame_group_logits[batch_index]
                result["group_probabilities"] = frame_group_probabilities[batch_index]
                group_score, group_label = frame_group_probabilities[batch_index].max(dim=-1)
                result["group_label"] = group_label
                result["group_score"] = group_score
            if human_keypoints_px is not None:
                result["human_keypoints"] = human_keypoints_px[batch_index]
                if human_visibility is not None:
                    result["human_visibility"] = human_visibility[batch_index]
            if court_keypoints_px is not None:
                raw_points = court_keypoints_px[batch_index]
                raw_visibility = court_visibility[batch_index]
                is_valid = bool(court_valid[batch_index].item())
                result["court_valid"] = is_valid
                result["court_pose_score"] = court_pose_score[batch_index]
                result["court_combined_score"] = court_combined_score[batch_index]
                result["court_detection_score"] = court_detection_score[batch_index]
                result["court_evidence_score"] = court_evidence_score[batch_index]
                result["court_line_score"] = court_line_score[batch_index]
                result["court_geometry_score"] = court_geometry_score[batch_index]
                result["court_anchor_confidence"] = court_anchor_confidence[batch_index]
                result["court_anchor_score"] = court_anchor_score[batch_index]
                result["court_anchor_peak_score"] = court_anchor_peak_score[batch_index]
                result["court_anchor_support_score"] = court_anchor_support_score[batch_index]
                result["court_anchor_spread_score"] = court_anchor_spread_score[batch_index]
                result["court_keypoints_raw"] = raw_points
                # Deployment SDK keeps the model's raw visibility even when the
                # final validity gate rejects the court. This is an extra output
                # only; it does not change any score, threshold, or prediction.
                result["court_visibility_raw"] = raw_visibility
                if is_valid:
                    result["court_keypoints"] = raw_points
                    result["court_visibility"] = raw_visibility
                else:
                    result["court_keypoints"] = raw_points.new_zeros((0, 2))
                    result["court_visibility"] = raw_visibility.new_zeros((0,))
            results.append(result)
        return results

    def deploy(self):
        self.eval()
        self.deploy_mode = True
        return self
