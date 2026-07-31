# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Hand-cropped InterField variant (per-hand, ARCTIC-style).

The full-frame baseline (``train.py``) sees the whole 1024x1280 egocentric frame,
so a hand is only ~15 px across. This variant instead crops a square region
around each hand and resizes it to 224x224, so the hand fills the input -- the
main lever for accuracy, and the setting ARCTIC's InterField actually used.

The crop box comes from the **ground-truth hand pose** (project the 21 world
joints into the image). GT hand pose exists for train subjects and our held-out
split, but is *withheld on the official test set*, so this is an **oracle-box,
upper-bound reference**: to submit on the real test set you would supply boxes
from a hand detector/pose estimator. The full-frame ``train.py`` model remains the
directly-submittable baseline.

Subcommands (all under ``python -m ...baseline.crop_baseline``):

    precompute-boxes  --frames-dir F --root R              # -> F/hand_boxes.jsonl
    train             --frames-dir F --root R --out CKPT   # per-hand ResNet-50
    predict           --frames-dir F --root R --checkpoint CKPT --out preds.jsonl

Score ``preds.jsonl`` with the usual ``evaluate`` module.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ... import camera
from .. import (
    InteractionFieldSample,
    make_sample_id,
    PredictionRecord,
    Show3DInteractionFieldDataset,
    write_submission_jsonl,
)
from ...dataset import Show3DFrameRef, Show3DPaths
from .data import _read_jsonl, _read_rotation_world_from_camera
from .model import IMAGENET_MEAN, IMAGENET_STD, InterFieldModel, NUM_JOINTS
from .predict import camera_to_world

HANDS = ("left_to_object", "right_to_object")
HAND_ATTR = {"left_to_object": "left_hand", "right_to_object": "right_hand"}
BOX_KEY = {"left_to_object": "left_box", "right_to_object": "right_box"}


# ---------------------------------------------------------------------------
# box geometry
# ---------------------------------------------------------------------------
def square_box(box, width, height, margin=0.4, min_size=64):
    """Expand a tight [x0,y0,x1,y1] box to a padded square, clamped to the image."""
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    size = max(x1 - x0, y1 - y0) * (1.0 + 2.0 * margin)
    size = max(size, min_size)
    half = size / 2.0
    x0, y0 = int(round(cx - half)), int(round(cy - half))
    x1, y1 = int(round(cx + half)), int(round(cy + half))
    # clamp while keeping square where possible
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(width, x1), min(height, y1)
    return [x0, y0, x1, y1]


def _crop_and_preprocess(gray, box, image_size, augment_rng=None):
    x0, y0, x1, y1 = box
    crop = gray[y0:y1, x0:x1]
    if crop.size == 0:
        crop = gray
    crop = cv2.resize(crop, (image_size, image_size), interpolation=cv2.INTER_AREA)
    arr = crop.astype(np.float32) / 255.0
    if augment_rng is not None:
        rng = augment_rng
        arr = arr * rng.uniform(0.7, 1.3)
        m = float(arr.mean())
        arr = (arr - m) * rng.uniform(0.7, 1.3) + m
        arr = np.clip(arr, 0.0, 1.0) ** rng.uniform(0.7, 1.4)
        arr = arr + rng.normal(0.0, 0.02, size=arr.shape)
        arr = np.clip(arr, 0.0, 1.0)
    arr = np.repeat(arr[None], 3, axis=0)
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)[:, None, None]
    std = np.asarray(IMAGENET_STD, dtype=np.float32)[:, None, None]
    return ((arr - mean) / std).astype(np.float32)


# ---------------------------------------------------------------------------
# precompute boxes (scene by scene to bound memory)
# ---------------------------------------------------------------------------
def precompute_boxes(frames_dir: Path, root: Path, out: Path | None = None) -> Path:
    out = out or (frames_dir / "hand_boxes.jsonl")
    labels = _read_jsonl(frames_dir / "labels.jsonl")
    by_scene: dict[tuple[str, str], list[tuple[str, int]]] = defaultdict(list)
    for row in labels:
        sid = row["sample_id"]
        key, frame = sid.rsplit(":", 1)
        subj, scene = key.split("/", 1)
        by_scene[(subj, scene)].append((sid, int(frame)))

    n_boxes = 0
    with out.open("w") as fout:
        for (subj, scene), items in sorted(by_scene.items()):
            samples = [
                InteractionFieldSample(
                    sample_id=sid, subject_id=subj, scene_id=scene, frame_index=fr
                )
                for sid, fr in items
            ]
            ds = Show3DInteractionFieldDataset(
                root, samples=samples, multiview=False,
                decode_images=False, load_labels=True,
            )
            for i in range(len(ds)):
                ex = ds[i]
                view = ex.views.get("headset0")
                row: dict = {"sample_id": ex.sample.sample_id}
                if view is None or view.calibration is None \
                        or view.calibration.t_world_from_camera is None:
                    fout.write(json.dumps(row) + "\n")
                    continue
                cal = view.calibration
                for field in HANDS:
                    hand = getattr(ex.frame_data, HAND_ATTR[field])
                    if hand is None or hand.landmarks_world_mm is None:
                        continue
                    uv, valid = camera.project_to_image(hand.landmarks_world_mm, cal)
                    pts = uv[valid]
                    if pts.shape[0] < 2:
                        continue
                    box = [float(pts[:, 0].min()), float(pts[:, 1].min()),
                           float(pts[:, 0].max()), float(pts[:, 1].max())]
                    row[BOX_KEY[field]] = square_box(
                        box, cal.image_width, cal.image_height
                    )
                    n_boxes += 1
                fout.write(json.dumps(row) + "\n")
    print(f"Wrote boxes for {n_boxes} hands -> {out}")
    return out


# ---------------------------------------------------------------------------
# dataset (per hand)
# ---------------------------------------------------------------------------
class CropHandDataset(Dataset):
    def __init__(self, frames_dir, root, *, subjects=None, image_size=224,
                 train=False, require_labels=True):
        self.frames_dir = Path(frames_dir)
        self.paths = Show3DPaths(root)
        self.image_size = image_size
        self.train = train
        self._calib_cache: dict = {}

        labels = {r["sample_id"]: r for r in _read_jsonl(self.frames_dir / "labels.jsonl")}
        boxes = {r["sample_id"]: r for r in _read_jsonl(self.frames_dir / "hand_boxes.jsonl")}
        subject_set = set(subjects) if subjects is not None else None

        self.items: list[tuple[dict, str, list]] = []
        for row in _read_jsonl(self.frames_dir / "index.jsonl"):
            if row.get("view") != "headset0":
                continue
            if subject_set is not None and row["subject_id"] not in subject_set:
                continue
            sid = row["sample_id"]
            box_row = boxes.get(sid)
            if box_row is None:
                continue
            label_row = labels.get(sid)
            for field in HANDS:
                box = box_row.get(BOX_KEY[field])
                if box is None:
                    continue
                if require_labels and (label_row is None or label_row.get(field) is None):
                    continue
                self.items.append((row, field, box))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        row, field, box = self.items[index]
        subj, scene, frame = row["subject_id"], row["scene_id"], int(row["frame_index"])
        gray = cv2.imread(str(self.frames_dir / row["image"]), cv2.IMREAD_GRAYSCALE)
        rng = np.random.default_rng() if self.train else None
        image = torch.from_numpy(_crop_and_preprocess(gray, box, self.image_size, rng))

        ref = Show3DFrameRef(subject_id=subj, scene_id=scene, frame_index=frame)
        R = _read_rotation_world_from_camera(
            self.paths.camera_calibration_path(ref, "headset0"), frame, self._calib_cache
        )
        target = np.zeros((NUM_JOINTS, 3), dtype=np.float32)
        valid = 0.0
        world = self._label_lookup(row["sample_id"], field)
        if world is not None and R is not None:
            target = (np.asarray(world, dtype=np.float64) @ R).astype(np.float32)
            valid = 1.0
        meta = {"sample_id": row["sample_id"], "field": field,
                "subject_id": subj, "scene_id": scene, "frame_index": frame}
        return image, torch.from_numpy(target), torch.tensor(valid), meta

    # labels dict kept for target lookup
    def _label_lookup(self, sample_id, field):
        return self._labels.get(sample_id, {}).get(field)


def _crop_collate(batch):
    images = torch.stack([b[0] for b in batch])
    targets = torch.stack([b[1] for b in batch])
    valids = torch.stack([b[2] for b in batch])
    metas = [b[3] for b in batch]
    return images, targets, valids, metas


# attach a labels dict to the dataset after construction (avoids re-reading per item)
def _attach_labels(ds: CropHandDataset):
    ds._labels = {r["sample_id"]: r
                  for r in _read_jsonl(ds.frames_dir / "labels.jsonl")}
    return ds


ALL_SUBJECTS = ["ASC023", "SPI102", "LWA828", "YZH016", "XXI103",
                "PCW023", "MHA016", "MMO925", "XYZ109", "LYA722"]


@torch.no_grad()
def _val_ade(model, loader, device):
    model.eval()
    tot, n = 0.0, 0
    for images, targets, valids, _ in loader:
        images, targets = images.to(device), targets.to(device)
        pred = model(images).squeeze(1)  # (B,21,3)
        err = torch.linalg.norm(pred - targets, dim=-1).mean(dim=-1)  # (B,)
        v = valids.to(device)
        tot += float((err * v).sum()); n += int(v.sum())
    return tot / max(n, 1)


def train(args):
    device = torch.device("cuda")
    val_subjects = list(args.val_subjects)
    train_subjects = [s for s in ALL_SUBJECTS if s not in val_subjects]
    train_all = not val_subjects
    print(f"Train {train_subjects}\nVal   {val_subjects or '(none: train on all)'}")

    tr = _attach_labels(CropHandDataset(args.frames_dir, args.root,
        subjects=train_subjects, image_size=args.image_size, train=True))
    tl = DataLoader(tr, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                    pin_memory=True, drop_last=True, collate_fn=_crop_collate,
                    persistent_workers=args.workers > 0)
    vl = None
    if not train_all:
        va = _attach_labels(CropHandDataset(args.frames_dir, args.root,
            subjects=val_subjects, image_size=args.image_size, train=False))
        vl = DataLoader(va, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                        pin_memory=True, collate_fn=_crop_collate,
                        persistent_workers=args.workers > 0)
        print(f"Train hand-crops: {len(tr)} | Val hand-crops: {len(va)}")
    else:
        print(f"Train hand-crops: {len(tr)} | (no val: training on all subjects)")

    model = InterFieldModel(pretrained=True, num_hands=1).to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train(); t0 = time.time(); run = 0.0; seen = 0
        for it, (images, targets, valids, _) in enumerate(tl):
            images, targets, valids = images.to(device), targets.to(device), valids.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(images).squeeze(1)
            err = torch.linalg.norm(pred - targets, dim=-1).mean(dim=-1)
            loss = (err * valids).sum() / valids.sum().clamp_min(1.0)
            loss.backward(); opt.step()
            run += float(loss.item()) * images.size(0); seen += images.size(0)
            if it % 50 == 0:
                print(f"  ep{epoch} it{it}/{len(tl)} loss {loss.item():.2f}", flush=True)
        sched.step()
        ade = None if train_all else _val_ade(model, vl, device)
        if ade is None:
            print(f"[epoch {epoch}] train_loss {run/max(seen,1):.2f} mm | {time.time()-t0:.0f}s", flush=True)
        else:
            print(f"[epoch {epoch}] train_loss {run/max(seen,1):.2f} mm | val_ADE {ade:.2f} mm | {time.time()-t0:.0f}s", flush=True)
        state = (model.module if hasattr(model, "module") else model).state_dict()
        ckpt = {"model_state": state, "epoch": epoch, "val_ade": ade,
                "arch": "resnet50_interfield_crop", "num_hands": 1,
                "image_size": args.image_size, "val_subjects": val_subjects}
        torch.save(ckpt, args.out.with_suffix(".last.pt"))
        if train_all:
            torch.save(ckpt, args.out)
        elif ade < best:
            best = ade; torch.save(ckpt, args.out)
            print(f"  -> new best val_ADE {best:.2f} mm, saved {args.out}")
    if train_all:
        print(f"Done (all subjects). Final-epoch model -> {args.out}")
    else:
        print(f"Done. Best val_ADE {best:.2f} mm -> {args.out}")


@torch.no_grad()
def predict(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = InterFieldModel(pretrained=False, num_hands=1).to(device)
    model.load_state_dict(ckpt["model_state"]); model.eval()

    ds = _attach_labels(CropHandDataset(args.frames_dir, args.root,
        subjects=args.subjects, image_size=ckpt.get("image_size", 224),
        train=False, require_labels=False))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True, collate_fn=_crop_collate)
    paths = ds.paths; calib_cache: dict = {}
    fields_by_sample: dict[str, dict] = defaultdict(dict)
    for images, _, _, metas in loader:
        pred = model(images.to(device)).squeeze(1).cpu().numpy()  # (B,21,3) cam
        for i, meta in enumerate(metas):
            ref = Show3DFrameRef(subject_id=meta["subject_id"],
                                 scene_id=meta["scene_id"], frame_index=meta["frame_index"])
            R = _read_rotation_world_from_camera(
                paths.camera_calibration_path(ref, "headset0"), meta["frame_index"], calib_cache)
            if R is None:
                continue
            fields_by_sample[meta["sample_id"]][meta["field"]] = camera_to_world(pred[i], R)
    records = [PredictionRecord(sample_id=sid, fields=f) for sid, f in fields_by_sample.items()]
    write_submission_jsonl(args.out, records)
    print(f"Wrote {len(records)} frame predictions -> {args.out}")


def main():
    p = argparse.ArgumentParser(description="Hand-cropped InterField variant")
    sub = p.add_subparsers(dest="cmd", required=True)

    pb = sub.add_parser("precompute-boxes")
    pb.add_argument("--frames-dir", type=Path, required=True)
    pb.add_argument("--root", type=Path, required=True)
    pb.add_argument("--out", type=Path, default=None)

    tr = sub.add_parser("train")
    tr.add_argument("--frames-dir", type=Path, required=True)
    tr.add_argument("--root", type=Path, required=True)
    tr.add_argument("--val-subjects", nargs="*", default=["XYZ109", "LYA722"],
                    help="held-out val subjects; pass none (--val-subjects) to train on all")
    tr.add_argument("--epochs", type=int, default=25)
    tr.add_argument("--batch-size", type=int, default=256)
    tr.add_argument("--lr", type=float, default=1e-4)
    tr.add_argument("--weight-decay", type=float, default=5e-4)
    tr.add_argument("--workers", type=int, default=16)
    tr.add_argument("--image-size", type=int, default=224)
    tr.add_argument("--out", type=Path, default=Path("checkpoints/interfield_crop.pt"))

    pr = sub.add_parser("predict")
    pr.add_argument("--frames-dir", type=Path, required=True)
    pr.add_argument("--root", type=Path, required=True)
    pr.add_argument("--checkpoint", type=Path, required=True)
    pr.add_argument("--subjects", nargs="+", default=None)
    pr.add_argument("--batch-size", type=int, default=256)
    pr.add_argument("--workers", type=int, default=16)
    pr.add_argument("--out", type=Path, default=Path("predictions_crop.jsonl"))

    args = p.parse_args()
    if args.cmd == "precompute-boxes":
        precompute_boxes(args.frames_dir, args.root, args.out)
    elif args.cmd == "train":
        train(args)
    elif args.cmd == "predict":
        predict(args)


if __name__ == "__main__":
    main()
