# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Run the trained InterField baseline and write a challenge ``predictions.jsonl``.

Two input modes:

  * ``--frames-dir`` (fast): predict over frames already extracted by
    ``show3d.extract_images`` -- used to score the held-out split.
  * ``--manifest`` (canonical): decode headset0 frames straight from the MP4s via
    ``Show3DInteractionFieldDataset`` -- the exact path a participant runs at
    submission time.

Either way the model predicts in the camera frame and this module rotates the
field back to world with the per-frame ``R_world_from_camera`` before writing the
submission, so the output matches the evaluator's world-space convention. Both
hands are always predicted (the baseline never abstains).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .. import PredictionRecord, write_submission_jsonl
from .data import collate, InterFieldFrameDataset, _read_rotation_world_from_camera
from .model import InterFieldModel, NUM_JOINTS


def load_model(checkpoint: str | Path, device: torch.device) -> InterFieldModel:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model = InterFieldModel(pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def camera_to_world(field_cam: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """(21,3) camera-frame field -> world frame: V_world = R_world_from_camera @ V_cam."""
    return field_cam @ rotation.T


@torch.no_grad()
def predict_frames_dir(
    model, frames_dir, root, subjects, device, image_size=224, batch_size=256, workers=16
) -> list[PredictionRecord]:
    ds = InterFieldFrameDataset(
        frames_dir, root, subjects=subjects, image_size=image_size,
        train=False, require_labels=False,
    )
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=workers,
        pin_memory=True, collate_fn=collate,
    )
    paths = ds.paths
    calib_cache: dict = {}
    records: list[PredictionRecord] = []
    from ...dataset import Show3DFrameRef

    for images, _, _, metas in loader:
        images = images.to(device, non_blocking=True)
        pred_cam = model(images).cpu().numpy()  # (B,2,21,3) camera frame
        for i, meta in enumerate(metas):
            frame_ref = Show3DFrameRef(
                subject_id=meta["subject_id"], scene_id=meta["scene_id"],
                frame_index=meta["frame_index"],
            )
            calib_path = paths.camera_calibration_path(frame_ref, "headset0")
            rot = _read_rotation_world_from_camera(
                calib_path, meta["frame_index"], calib_cache
            )
            if rot is None:
                continue  # synthesized/untracked frame: no world mapping
            left = camera_to_world(pred_cam[i, 0], rot)
            right = camera_to_world(pred_cam[i, 1], rot)
            records.append(
                PredictionRecord(
                    sample_id=meta["sample_id"],
                    fields={
                        "left_to_object": left.astype(np.float64),
                        "right_to_object": right.astype(np.float64),
                    },
                )
            )
    return records


@torch.no_grad()
def predict_manifest(
    model, root, manifest, device, image_size=224
) -> list[PredictionRecord]:
    """Canonical path: decode headset0 frames from the MP4s and predict."""
    import cv2

    from .. import Show3DInteractionFieldDataset
    from .model import IMAGENET_MEAN, IMAGENET_STD

    ds = Show3DInteractionFieldDataset(
        root, manifest, multiview=False, decode_images=True, load_labels=False,
    )
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)[:, None, None]
    std = np.asarray(IMAGENET_STD, dtype=np.float32)[:, None, None]
    records: list[PredictionRecord] = []
    for i in range(len(ds)):
        example = ds[i]
        view = example.views.get("headset0")
        if view is None or view.image is None or view.calibration is None:
            continue
        rot = view.calibration.t_world_from_camera
        if rot is None:
            continue
        rot = np.asarray(rot, dtype=np.float64)[:3, :3]
        gray = cv2.cvtColor(view.image, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (image_size, image_size), interpolation=cv2.INTER_AREA)
        arr = np.repeat((gray.astype(np.float32) / 255.0)[None], 3, axis=0)
        arr = (arr - mean) / std
        img = torch.from_numpy(arr)[None].to(device)
        pred_cam = model(img).cpu().numpy()[0]  # (2,21,3)
        records.append(
            PredictionRecord(
                sample_id=example.sample.sample_id,
                fields={
                    "left_to_object": camera_to_world(pred_cam[0], rot),
                    "right_to_object": camera_to_world(pred_cam[1], rot),
                },
            )
        )
    return records


def main() -> None:
    p = argparse.ArgumentParser(description="Run the InterField baseline predictor")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--frames-dir", type=Path, default=None)
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--subjects", nargs="+", default=None, help="frames-dir mode filter")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--out", type=Path, default=Path("predictions.jsonl"))
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)

    if args.frames_dir is not None:
        records = predict_frames_dir(
            model, args.frames_dir, args.root, args.subjects, device,
            image_size=args.image_size,
        )
    elif args.manifest is not None:
        records = predict_manifest(
            model, args.root, args.manifest, device, image_size=args.image_size
        )
    else:
        p.error("provide --frames-dir or --manifest")

    write_submission_jsonl(args.out, records)
    print(f"Wrote {len(records)} predictions -> {args.out}")


if __name__ == "__main__":
    main()
