# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Train the InterField baseline on SHOW3D (headset0, single view).

    python -m show3d.interaction_field.baseline.train \
        --frames-dir /path/to/frames --root /path/to/show3d \
        --val-subjects XYZ109 LYA722 --epochs 25 --out checkpoints/interfield.pt

Trains on the subjects NOT in ``--val-subjects`` and reports validation ADE (mm)
each epoch. Because the field is rotated into the camera frame with an orthogonal
matrix, camera-frame endpoint error equals the world-space ADE the official
evaluator reports, so the number logged here is directly comparable.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import collate, InterFieldFrameDataset
from .model import InterFieldModel, masked_field_loss

# The 10 training subjects (from train_manifest_202607.jsonl).
ALL_SUBJECTS = [
    "ASC023", "SPI102", "LWA828", "YZH016", "XXI103",
    "PCW023", "MHA016", "MMO925", "XYZ109", "LYA722",
]


@torch.no_grad()
def evaluate_ade(model, loader, device) -> tuple[float, int]:
    """Mean per-joint endpoint error (mm) over valid hands; == world-space ADE."""
    model.eval()
    total_err, total_hands = 0.0, 0
    for images, targets, masks, _ in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        pred = model(images)
        per_joint = torch.linalg.norm(pred - targets, dim=-1)  # (B,2,21)
        per_hand = per_joint.mean(dim=-1)  # (B,2)
        total_err += float((per_hand * masks).sum().item())
        total_hands += int(masks.sum().item())
    ade = total_err / max(total_hands, 1)
    return ade, total_hands


def main() -> None:
    p = argparse.ArgumentParser(description="Train the SHOW3D InterField baseline")
    p.add_argument("--frames-dir", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument(
        "--val-subjects", nargs="*", default=["XYZ109", "LYA722"],
        help="train subjects held out for validation; pass none "
        "(--val-subjects) to train on ALL subjects (releases the final-epoch ckpt)",
    )
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--out", type=Path, default=Path("checkpoints/interfield.pt"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda")
    n_gpu = torch.cuda.device_count()

    val_subjects = list(args.val_subjects)
    train_subjects = [s for s in ALL_SUBJECTS if s not in val_subjects]
    train_all = not val_subjects  # release model: use every training subject
    print(f"Train subjects ({len(train_subjects)}): {train_subjects}")
    print(f"Val subjects   ({len(val_subjects)}): {val_subjects or '(none: train on all)'}")

    train_ds = InterFieldFrameDataset(
        args.frames_dir, args.root, subjects=train_subjects,
        image_size=args.image_size, train=True,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True,
        collate_fn=collate, persistent_workers=args.workers > 0,
    )
    val_loader = None
    if not train_all:
        val_ds = InterFieldFrameDataset(
            args.frames_dir, args.root, subjects=val_subjects,
            image_size=args.image_size, train=False,
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True,
            collate_fn=collate, persistent_workers=args.workers > 0,
        )
        print(f"Train frames: {len(train_ds)} | Val frames: {len(val_ds)}")
    else:
        print(f"Train frames: {len(train_ds)} | (no val: training on all subjects)")

    model = InterFieldModel(pretrained=True).to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
        print(f"DataParallel across {n_gpu} GPUs")

    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    best_ade = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running, seen = 0.0, 0
        for it, (images, targets, masks, _) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            pred = model(images)
            loss = masked_field_loss(pred, targets, masks)
            loss.backward()
            opt.step()
            running += float(loss.item()) * images.size(0)
            seen += images.size(0)
            if it % 50 == 0:
                print(
                    f"  epoch {epoch} it {it}/{len(train_loader)} "
                    f"loss {loss.item():.2f} mm", flush=True
                )
        sched.step()
        train_loss = running / max(seen, 1)
        dt = time.time() - t0
        val_ade = None
        if val_loader is not None:
            val_ade, val_hands = evaluate_ade(model, val_loader, device)
            print(
                f"[epoch {epoch}] train_loss {train_loss:.2f} mm | "
                f"val_ADE {val_ade:.2f} mm ({val_hands} hands) | {dt:.0f}s",
                flush=True,
            )
        else:
            print(f"[epoch {epoch}] train_loss {train_loss:.2f} mm | {dt:.0f}s", flush=True)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_ade": val_ade})

        state = (model.module if hasattr(model, "module") else model).state_dict()
        ckpt = {
            "model_state": state,
            "epoch": epoch,
            "val_ade": val_ade,
            "train_subjects": train_subjects,
            "val_subjects": val_subjects,
            "image_size": args.image_size,
            "arch": "resnet50_interfield",
        }
        torch.save(ckpt, args.out.with_suffix(".last.pt"))
        if train_all:
            # No held-out set: the released model is the final epoch.
            torch.save(ckpt, args.out)
        elif val_ade < best_ade:
            best_ade = val_ade
            torch.save(ckpt, args.out)
            print(f"  -> new best val_ADE {best_ade:.2f} mm, saved {args.out}")

    (args.out.parent / "train_history.json").write_text(json.dumps(history, indent=2))
    if train_all:
        print(f"Done (all subjects). Final-epoch model -> {args.out}")
    else:
        print(f"Done. Best val_ADE {best_ade:.2f} mm -> {args.out}")


if __name__ == "__main__":
    main()
