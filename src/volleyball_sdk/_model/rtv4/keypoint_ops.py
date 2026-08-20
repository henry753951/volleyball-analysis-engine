"""Keypoint coordinate helpers shared by the dataset, loss, metrics and export."""

from __future__ import annotations

import torch
from torch import Tensor


def box_cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    cx, cy, width, height = boxes.unbind(-1)
    return torch.stack(
        (cx - width * 0.5, cy - height * 0.5, cx + width * 0.5, cy + height * 0.5),
        dim=-1,
    )


def encode_keypoints_relative(
    keypoints_xy: Tensor,
    boxes_xyxy: Tensor,
    *,
    eps: float = 1.0e-6,
    clamp: bool = True,
) -> Tensor:
    """Encode absolute xy points relative to their owning xyxy boxes."""

    top_left = boxes_xyxy[..., None, :2]
    size = (boxes_xyxy[..., 2:] - boxes_xyxy[..., :2]).clamp_min(eps)
    relative = (keypoints_xy - top_left) / size[..., None, :]
    return relative.clamp(0.0, 1.0) if clamp else relative


def decode_keypoints_relative(
    relative_keypoints: Tensor,
    boxes_cxcywh: Tensor,
    *,
    clamp_boxes: bool = False,
) -> Tensor:
    """Decode bbox-relative xy points into normalized image coordinates."""

    boxes_xyxy = box_cxcywh_to_xyxy(boxes_cxcywh)
    if clamp_boxes:
        boxes_xyxy = boxes_xyxy.clamp(0.0, 1.0)
    top_left = boxes_xyxy[..., None, :2]
    size = (boxes_xyxy[..., 2:] - boxes_xyxy[..., :2]).clamp_min(1.0e-6)
    return top_left + relative_keypoints * size[..., None, :]


def normalized_keypoints_to_pixels(
    normalized_keypoints: Tensor,
    image_sizes_wh: Tensor,
) -> Tensor:
    """Convert normalized xy coordinates to pixels."""

    scale = image_sizes_wh.to(normalized_keypoints).view(image_sizes_wh.shape[0], 1, 1, 2)
    return normalized_keypoints * scale


__all__ = [
    "box_cxcywh_to_xyxy",
    "decode_keypoints_relative",
    "encode_keypoints_relative",
    "normalized_keypoints_to_pixels",
]
