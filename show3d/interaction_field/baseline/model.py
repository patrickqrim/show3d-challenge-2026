# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""InterField baseline model for the SHOW3D interaction-field challenge.

A single-image regressor in the spirit of the InterField model from ARCTIC
(Fan et al., CVPR 2023): a ResNet-50 image encoder followed by an MLP head that
regresses, for each hand, the ``(21, 3)`` hand-anchored interaction field (the
vector from each of the 21 hand joints to the nearest object-surface point).

The field target lives in 3D world space, but a hand crop cannot reveal the
world's (gravity-aligned) orientation, so the model regresses the field in the
**camera frame** -- where direction is tied to image appearance -- and callers
rotate it back to world with the per-frame ``R_world_from_camera`` (the
translation cancels because the field is a difference of two world points).
See ``predict.py`` for the camera->world step at inference time.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights

# Fixed 21-joint UmeTrack skeleton per hand; two hands (left, right).
NUM_HANDS: int = 2
NUM_JOINTS: int = 21
OUT_DIM: int = NUM_HANDS * NUM_JOINTS * 3  # 126

# ImageNet statistics for the pretrained ResNet backbone.
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


class InterFieldModel(nn.Module):
    """ResNet-50 encoder + MLP field head.

    Input:  images ``(B, 3, H, W)``, ImageNet-normalized.
    Output: fields ``(B, 2, 21, 3)`` in millimeters (camera frame), ordered
            ``[left_to_object, right_to_object]``.
    """

    def __init__(
        self,
        pretrained: bool = True,
        hidden_dim: int = 512,
        dropout: float = 0.5,
        num_hands: int = NUM_HANDS,
    ) -> None:
        super().__init__()
        self.num_hands = num_hands
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet50(weights=weights)
        feat_dim = backbone.fc.in_features  # 2048
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_hands * NUM_JOINTS * 3),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(images)
        out = self.head(feats)
        return out.view(-1, self.num_hands, NUM_JOINTS, 3)


def masked_field_loss(
    pred: torch.Tensor, target: torch.Tensor, hand_mask: torch.Tensor
) -> torch.Tensor:
    """Mean per-joint endpoint (L2) error over the valid hands.

    ``pred``/``target`` are ``(B, 2, 21, 3)``; ``hand_mask`` is ``(B, 2)`` with 1
    for hands that have a valid ground-truth field. This is exactly the training
    surrogate for the evaluator's ADE, so we optimize the metric directly.
    """
    per_joint = torch.linalg.norm(pred - target, dim=-1)  # (B, 2, 21)
    per_hand = per_joint.mean(dim=-1)  # (B, 2)
    denom = hand_mask.sum().clamp_min(1.0)
    return (per_hand * hand_mask).sum() / denom
