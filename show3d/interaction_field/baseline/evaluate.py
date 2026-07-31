# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Score an InterField submission with the official evaluator, plus the paper's ACC.

    python -m show3d.interaction_field.baseline.evaluate \
        --labels frames/labels.jsonl --predictions predictions.jsonl \
        --subjects XYZ109 LYA722

Reports the challenge metrics (ADE mm, recall, acc@10/50/100 mm) via
``show3d.interaction_field.evaluate_prediction_records`` on the held-out subjects,
and the paper's temporal-smoothness metric ACC (acceleration, m/s^2) computed on
the predicted field over consecutive frames within each scene.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from .. import (
    evaluate_prediction_records,
    FIELD_NAMES,
    read_label_jsonl,
    read_submission_jsonl,
)


def _subject_of(sample_id: str) -> str:
    return sample_id.split("/", 1)[0]


def _scene_and_frame(sample_id: str) -> tuple[str, int]:
    # "SUBJECT/scene_id:frameidx"
    key, frame = sample_id.rsplit(":", 1)
    return key, int(frame)


def acceleration_m_s2(predictions, fps: float) -> float:
    """Mean acceleration magnitude (m/s^2) of the predicted field over time.

    Groups predictions by scene, orders by frame index, and takes the discrete
    second derivative f[t-1] - 2 f[t] + f[t+1] over the *sampled* frames. dt is
    the sampling period (1/fps); fields are in mm, converted to meters.
    """
    by_scene: dict[str, dict[int, dict]] = defaultdict(dict)
    for sid, rec in predictions.items():
        scene, frame = _scene_and_frame(sid)
        by_scene[scene][frame] = rec.fields
    dt = 1.0 / fps
    accels: list[float] = []
    for frames in by_scene.values():
        idx = sorted(frames)
        for field_name in FIELD_NAMES:
            for a, b, c in zip(idx, idx[1:], idx[2:]):
                # require consecutive sampled frames (uniform dt)
                if (b - a) != (c - b):
                    continue
                fa = frames[a].get(field_name)
                fb = frames[b].get(field_name)
                fc = frames[c].get(field_name)
                if fa is None or fb is None or fc is None:
                    continue
                step = (b - a) * dt
                accel = (fa - 2.0 * fb + fc) / (step * step)  # mm/s^2
                accel_m = accel / 1000.0  # m/s^2
                accels.append(float(np.linalg.norm(accel_m, axis=1).mean()))
    return float(np.mean(accels)) if accels else float("nan")


def main() -> None:
    p = argparse.ArgumentParser(description="Score the InterField baseline")
    p.add_argument("--labels", type=Path, required=True, help="reference labels.jsonl")
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--subjects", nargs="+", default=None, help="restrict to subjects")
    p.add_argument("--fps", type=float, default=10.0, help="sampling fps for ACC")
    args = p.parse_args()

    references = read_label_jsonl(args.labels)
    if args.subjects is not None:
        keep = set(args.subjects)
        references = [r for r in references if _subject_of(r.sample_id) in keep]
    predictions = read_submission_jsonl(args.predictions)

    result = evaluate_prediction_records(references, predictions)

    print("SHOW3D interaction-field baseline (InterField, headset0)")
    print(f"  references: {len(references)} frames "
          f"| subjects: {args.subjects or 'all'}")
    for field_name in FIELD_NAMES:
        m = result.fields[field_name]
        if m.recall is None:
            print(f"  {field_name:16s}: no valid targets")
            continue
        acc = " ".join(
            f"acc@{int(t)}mm={v:.3f}"
            for t, v in sorted(m.accuracy_by_threshold_mm.items())
        )
        ade = f"{m.ade_mm:.2f} mm" if m.ade_mm is not None else "n/a"
        predicted = m.num_samples - m.missing_predictions
        print(f"  {field_name:16s}: ADE {ade} | recall {m.recall:.3f} "
              f"({predicted}/{m.num_samples}) | {acc}")
    if result.mean_ade_mm is not None:
        print(f"  mean ADE:    {result.mean_ade_mm:.2f} mm")
    if result.mean_recall is not None:
        print(f"  mean recall: {result.mean_recall:.3f}")
    acc = acceleration_m_s2(predictions, args.fps)
    print(f"  ACC:         {acc:.2f} m/s^2  (temporal smoothness, lower is better)")


if __name__ == "__main__":
    main()
