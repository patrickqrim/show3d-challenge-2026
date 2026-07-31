# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""InterField reference baseline for the SHOW3D interaction-field challenge.

A single-image ResNet-50 regressor (InterField, after ARCTIC) that predicts the
per-hand ``(21, 3)`` interaction field from a headset0 frame. See ``README.md``
in this directory for the end-to-end train / predict / evaluate / visualize flow.
"""

from .model import InterFieldModel, masked_field_loss, NUM_HANDS, NUM_JOINTS, OUT_DIM

__all__ = [
    "InterFieldModel",
    "masked_field_loss",
    "NUM_HANDS",
    "NUM_JOINTS",
    "OUT_DIM",
]
