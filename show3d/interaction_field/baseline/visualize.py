# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Visualize the InterField baseline: predicted vs. ground-truth interaction field.

    python -m show3d.interaction_field.baseline.visualize \
        --checkpoint checkpoints/interfield.pt \
        --root /path/to/show3d --manifest manifest.jsonl \
        --index 0 --out baseline_field.png

Renders two panels for one frame: (left) the 3D hand skeleton + object surface
with ground-truth field arrows (crimson) and the model's predicted arrows
(orange); (right) the egocentric headset0 frame with the projected object, hand
skeleton, and predicted joint->object endpoints. With no --root/--manifest it
falls back to the bundled synthetic scene.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from matplotlib.figure import Figure

from ... import camera, viz
from .. import make_manifest_rows, Show3DInteractionFieldDataset
from ..demo import build_synthetic_scene
from ...dataset import Show3DPaths, Show3DFrameRef
from .model import IMAGENET_MEAN, IMAGENET_STD, InterFieldModel
from .predict import camera_to_world, load_model

PRED_COLOR = "darkorange"
GT_COLOR = viz.FIELD_COLOR  # crimson


def _preprocess(image_rgb: np.ndarray, image_size: int) -> torch.Tensor:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (image_size, image_size), interpolation=cv2.INTER_AREA)
    arr = np.repeat((gray.astype(np.float32) / 255.0)[None], 3, axis=0)
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)[:, None, None]
    std = np.asarray(IMAGENET_STD, dtype=np.float32)[:, None, None]
    return torch.from_numpy((arr - mean) / std)


@torch.no_grad()
def predict_world_fields(model, image_rgb, rotation, device, image_size=224):
    """Return predicted world-frame fields {left, right}, each (21, 3)."""
    img = _preprocess(image_rgb, image_size)[None].to(device)
    pred_cam = model(img).cpu().numpy()[0]  # (2, 21, 3)
    return {
        "left_to_object": camera_to_world(pred_cam[0], rotation),
        "right_to_object": camera_to_world(pred_cam[1], rotation),
    }


def _build_dataset(root, manifest, scene_index, fps):
    """Build a decode-enabled dataset from a sample- OR scene-level manifest.

    A sample manifest (rows with ``frame_index``) is used directly. A scene
    manifest (like ``train_manifest_202607.jsonl``: subject/scene/object_alias) is
    expanded for one chosen scene by reading the headset0 video length and
    subsampling at ``fps``.
    """
    import json

    rows = [json.loads(l) for l in open(manifest) if l.strip()]
    if rows and "frame_index" in rows[0]:
        return Show3DInteractionFieldDataset(
            root, manifest, multiview=False, decode_images=True, load_labels=True,
        )
    row = rows[scene_index]
    subject_id, scene_id = row["subject_id"], row["scene_id"]
    paths = Show3DPaths(root)
    frame_ref = Show3DFrameRef(subject_id=subject_id, scene_id=scene_id, frame_index=0)
    cap = cv2.VideoCapture(str(paths.headset_path(frame_ref, 0)))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    samples = make_manifest_rows(
        subject_id, scene_id, num_frames, sampling_fps=fps
    )
    print(f"[baseline viz] scene {subject_id}/{scene_id}: {len(samples)} sampled frames")
    return Show3DInteractionFieldDataset(
        root, samples=samples, multiview=False, decode_images=True, load_labels=True,
    )


def _find_frame(dataset, index):
    """Return a valid example: honor --index if valid, else first valid frame."""
    if index is not None:
        example = dataset[index]
        if example.is_valid:
            return example
    for i in range(len(dataset)):
        example = dataset[i]
        if example.is_valid and example.views.get("headset0") is not None:
            if example.views["headset0"].calibration is not None:
                return example
    raise RuntimeError("no valid headset0 frame with an extrinsic found")


def render(example, pred_fields, out_path):
    view = example.views["headset0"]
    calibration = view.calibration
    fig = Figure(figsize=(16, 7))

    # ---- left: 3D GT vs predicted arrows ----
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    extent = []
    surface = viz.object_surface(example)
    if surface is not None:
        viz.draw_object_3d(ax3d, surface)
        extent.append(surface)
    field_lbl = pred_lbl = False
    _HAND_FIELD = {"left hand": "left_to_object", "right hand": "right_to_object"}
    for label, color, joints, field in viz.present_hands(example):
        viz.draw_hand_skeleton_3d(ax3d, joints, color, label)
        extent.append(joints)
        if field is not None:
            viz.draw_field_3d(
                ax3d, joints, field, color=GT_COLOR,
                label=None if field_lbl else "GT field",
            )
            field_lbl = True
            extent.append(joints + field)
        pred = pred_fields.get(_HAND_FIELD[label])
        if pred is not None:
            viz.draw_field_3d(
                ax3d, joints, pred, color=PRED_COLOR,
                label=None if pred_lbl else "predicted field",
            )
            pred_lbl = True
            extent.append(joints + pred)
    viz._finish_3d(ax3d, extent, f"Interaction field (GT vs pred): {example.sample.sample_id}")

    # ---- right: 2D overlay with predicted endpoints ----
    ax2d = fig.add_subplot(1, 2, 2)
    image = view.image
    if image is not None:
        ax2d.imshow(image)
    else:
        ax2d.set_facecolor("0.9")
    if surface is not None:
        obj_uv, obj_valid = camera.project_to_image(surface, calibration)
        viz.draw_object_2d(ax2d, obj_uv, obj_valid)
    for label, color, joints, _field in viz.present_hands(example):
        joints_uv, valid = camera.project_to_image(joints, calibration)
        if bool(valid.any()):
            viz.draw_hand_skeleton_2d(ax2d, joints_uv, valid, color, label)
        pred = pred_fields.get(_HAND_FIELD[label])
        if pred is not None:
            endpoints = joints + pred
            end_uv, end_valid = camera.project_to_image(endpoints, calibration)
            both = valid & end_valid
            for j in range(joints_uv.shape[0]):
                if both[j]:
                    ax2d.plot(
                        [joints_uv[j, 0], end_uv[j, 0]],
                        [joints_uv[j, 1], end_uv[j, 1]],
                        c=PRED_COLOR, linewidth=1.0,
                    )
    ax2d.set_xlim(0, calibration.image_width)
    ax2d.set_ylim(calibration.image_height, 0)
    ax2d.set_aspect("equal")
    ax2d.axis("off")
    ax2d.set_title("headset0 overlay: predicted joint->object")
    ax2d.legend(loc="upper right", fontsize="small")

    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"[baseline viz] {example.sample.sample_id} -> {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Visualize the InterField baseline")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--root", type=Path, default=None)
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--index", type=int, default=None,
                   help="frame position in the dataset (default: first valid)")
    p.add_argument("--scene-index", type=int, default=0,
                   help="which scene to expand when given a scene-level manifest")
    p.add_argument("--fps", type=float, default=10.0,
                   help="sampling fps when expanding a scene manifest")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--out", type=Path, default=Path("baseline_field.png"))
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)

    def run(root, manifest):
        dataset = _build_dataset(root, manifest, args.scene_index, args.fps)
        example = _find_frame(dataset, args.index)
        view = example.views["headset0"]
        rot = np.asarray(view.calibration.t_world_from_camera, dtype=np.float64)[:3, :3]
        preds = predict_world_fields(
            model, view.image, rot, device, image_size=args.image_size
        )
        render(example, preds, args.out)

    if args.root is not None and args.manifest is not None:
        run(args.root, args.manifest)
    else:
        with tempfile.TemporaryDirectory() as td:
            manifest = build_synthetic_scene(Path(td))
            print("No --root/--manifest; visualizing the synthetic scene.")
            run(Path(td), manifest)


if __name__ == "__main__":
    main()
