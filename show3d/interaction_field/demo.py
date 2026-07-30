# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Minimal end-to-end demo for the SHOW3D interaction-field challenge.

It runs the whole loop in one file: load frames from a scene, run a model,
write a submission, and evaluate it against the labels. The "model" here is a
naive stand-in that emits random vectors -- replace ``naive_model`` with your
own predictor to get a real baseline.

Run it with no arguments to use a small synthetic scene (no data download), or
point it at a local SHOW3D mirror::

    python -m show3d.demo                              # synthetic toy scene
    python -m show3d.demo --root DIR --manifest M      # your downloaded data
    python -m show3d.demo --root DIR --manifest M --decode-images   # + pixels
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from ..dataset import FloatArray
from . import (
    evaluate_submission_jsonl,
    EvaluationResult,
    FIELD_NAMES,
    InteractionFieldExample,
    InteractionFieldSample,
    make_sample_id,
    NUM_HAND_LANDMARKS,
    PredictionRecord,
    Show3DInteractionFieldDataset,
    write_manifest_jsonl,
    write_submission_jsonl,
)


def naive_model(
    example: InteractionFieldExample, rng: np.random.Generator
) -> dict[str, FloatArray]:
    """Stand-in model: random ``(NUM_HAND_LANDMARKS, 3)`` vectors per hand field.

    Swap this for your real predictor. A real model would read the loaded
    frame(s) -- ``example.views[name].image`` (with ``decode_images=True``) and
    ``example.views[name].calibration`` -- and regress the per-landmark vectors.
    The hand-anchored fields are fixed size, so a prediction is always
    ``(NUM_HAND_LANDMARKS, 3)`` regardless of the object.
    """
    del example  # a real model uses it; the naive baseline ignores the pixels.
    return {
        field_name: rng.standard_normal((NUM_HAND_LANDMARKS, 3)) * 100.0
        for field_name in FIELD_NAMES
    }


def run_demo(
    root: str | Path,
    manifest_path: str | Path,
    *,
    seed: int = 0,
    decode_images: bool = False,
    verbose: bool = False,
) -> EvaluationResult:
    """Load frames, run ``naive_model`` on each, write a submission, evaluate."""
    dataset = Show3DInteractionFieldDataset(
        root, manifest_path, decode_images=decode_images
    )
    rng = np.random.default_rng(seed)

    predictions: list[PredictionRecord] = []
    num_valid = 0
    first: InteractionFieldExample | None = None
    peek: tuple[InteractionFieldExample, dict[str, FloatArray]] | None = None
    for index in range(len(dataset)):
        example = dataset[index]
        if first is None:
            first = example
        # Predict for every sampled frame (a real model has no labels at test
        # time); the evaluator scores only frames that carry a valid target.
        fields = naive_model(example, rng)
        if example.is_valid:
            num_valid += 1
            if peek is None:
                peek = (example, fields)
        predictions.append(
            PredictionRecord(sample_id=example.sample.sample_id, fields=fields)
        )

    if verbose:
        _print_load_summary(len(dataset), num_valid, first)
        if peek is not None:
            _print_field_peek(*peek)

    submission_path = Path(manifest_path).parent / "submission.jsonl"
    write_submission_jsonl(submission_path, predictions)
    result = evaluate_submission_jsonl(dataset, submission_path)
    if verbose:
        _print_result(result)
    return result


def build_synthetic_scene(root: Path, *, num_frames: int = 6) -> Path:
    """Fabricate a tiny SHOW3D scene (bundled ``mug`` object) plus a manifest, so
    the demo runs without downloading data. Frame 3 is flagged as a headset
    tracking failure to show that invalid frames are skipped in evaluation."""
    subject_id = "S000"
    scene_id = "mug_grab_demo"  # object alias -> "mug" -> bundled mug.glb
    rng = np.random.default_rng(0)

    scene_dir = root / "scenes" / subject_id / scene_id
    (scene_dir / "camera_calibration").mkdir(parents=True)
    (scene_dir / "metadata").mkdir()
    (scene_dir / "headset0.mp4").write_bytes(b"")
    (scene_dir / "headset1.mp4").write_bytes(b"")
    (scene_dir / "metadata" / "frame_info.json").write_text("{}")

    identity = [[1.0 if r == c else 0.0 for c in range(4)] for r in range(4)]
    by_index = {
        str(i): {
            "index": i,
            "T_WorldFromCamera": identity,
            # Frame 3 stands in for a headset-tracking failure.
            "is_synthesized": i == 3,
        }
        for i in range(num_frames)
    }
    for name in ("headset0", "headset1"):
        (scene_dir / "camera_calibration" / f"{name}.json").write_text(
            json.dumps(
                {
                    "ImageSizeX": 1408,
                    "ImageSizeY": 1408,
                    "fx": 600.0,
                    "fy": 600.0,
                    "cx": 704.0,
                    "cy": 704.0,
                    "DistortionModel": "PinholePlane",
                    "T_WorldFromCamera_by_index": by_index,
                }
            )
        )

    object_pose = {
        str(i): {
            "confidence": 0.99,
            "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "t": [[0.0], [0.0], [0.0]],
        }
        for i in range(num_frames)
    }
    _write_json(
        root / "object_pose" / "v1" / "scenes" / subject_id / scene_id,
        "object_pose.json",
        object_pose,
    )

    # Left hand: 21 landmarks scattered near the object; right hand: absent.
    hand_pose = {
        str(i): {
            "hand_poses": {
                "0": {
                    "confidence": 0.99,
                    "landmarks_3d_mm": (
                        rng.standard_normal((NUM_HAND_LANDMARKS, 3)) * 50.0
                    ).tolist(),
                },
                "1": {"confidence": 0.0, "landmarks_3d_mm": None},
            }
        }
        for i in range(num_frames)
    }
    _write_json(
        root / "hand_pose" / "v2" / "scenes" / subject_id / scene_id,
        "hand_pose.json",
        hand_pose,
    )

    manifest_path = root / "manifest.jsonl"
    samples = [
        InteractionFieldSample(
            sample_id=make_sample_id(subject_id, scene_id, i),
            subject_id=subject_id,
            scene_id=scene_id,
            frame_index=i,
            object_alias="mug",
            has_left_hand=True,
        )
        for i in range(num_frames)
    ]
    write_manifest_jsonl(manifest_path, samples)
    return manifest_path


def _write_json(directory: Path, name: str, payload: object) -> None:
    directory.mkdir(parents=True)
    (directory / name).write_text(json.dumps(payload))


def _print_load_summary(
    num_frames: int, num_valid: int, first: InteractionFieldExample | None
) -> None:
    print(
        f"Loaded {num_frames} frames from the manifest "
        f"({num_valid} valid: headset tracked + object/hand pose present)."
    )
    if first is not None:
        cal = next(
            (v.calibration for v in first.views.values() if v.calibration is not None),
            None,
        )
        views = ", ".join(sorted(first.views))
        size = f"{cal.image_width}x{cal.image_height}" if cal is not None else "n/a"
        print(f"  views: {views} | image {size}")
    print()


def _print_field_peek(
    example: InteractionFieldExample, prediction: dict[str, FloatArray]
) -> None:
    """Show one frame's interaction field: GT vs predicted vector per hand."""
    assert example.labels is not None
    print(
        f"Interaction field for {example.sample.sample_id} "
        "(hand joint -> nearest object-surface point, mm):"
    )
    for field_name in FIELD_NAMES:
        target = example.labels.get(field_name)
        if target is None:
            continue
        pred = prediction[field_name]
        t0 = target[0]
        p0 = pred[0]
        print(
            f"  {field_name:16s}: {target.shape[0]}x3 field | "
            f"joint0 gt=({t0[0]:6.0f},{t0[1]:6.0f},{t0[2]:6.0f}) "
            f"pred=({p0[0]:6.0f},{p0[1]:6.0f},{p0[2]:6.0f})"
        )
    print()


def _print_result(result: EvaluationResult) -> None:
    print("Evaluation (interaction field, naive random model):")
    for field_name in FIELD_NAMES:
        metric = result.fields[field_name]
        if metric.ade_mm is None:
            print(f"  {field_name:16s}: no valid targets")
            continue
        acc = " ".join(
            f"acc@{int(t)}mm={v:.2f}"
            for t, v in sorted(metric.accuracy_by_threshold_mm.items())
        )
        print(
            f"  {field_name:16s}: ADE {metric.ade_mm:8.1f} mm | "
            f"{metric.num_points:5d} pts | {acc}"
        )
    mean = result.mean_ade_mm
    print(f"  mean ADE: {mean:.1f} mm" if mean is not None else "  mean ADE: n/a")


def main() -> None:
    parser = argparse.ArgumentParser(description="SHOW3D interaction-field demo")
    parser.add_argument("--root", type=Path, default=None, help="SHOW3D mirror root")
    parser.add_argument(
        "--manifest", type=Path, default=None, help="challenge manifest .jsonl"
    )
    parser.add_argument("--seed", type=int, default=0, help="random-model seed")
    parser.add_argument(
        "--decode-images",
        action="store_true",
        help="decode frame pixels (needs real videos + opencv)",
    )
    args = parser.parse_args()

    print("SHOW3D interaction-field challenge: end-to-end demo")
    print("  load frames -> predict the interaction field -> evaluate\n")

    if args.root is not None and args.manifest is not None:
        run_demo(
            args.root,
            args.manifest,
            seed=args.seed,
            decode_images=args.decode_images,
            verbose=True,
        )
        return

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        manifest_path = build_synthetic_scene(root)
        print("No --root/--manifest given; running on a synthetic toy scene.\n")
        run_demo(root, manifest_path, seed=args.seed, verbose=True)


if __name__ == "__main__":
    main()
