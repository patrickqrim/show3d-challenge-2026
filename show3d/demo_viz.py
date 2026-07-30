# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Top-level visualization demo for SHOW3D.

Renders one frame's interaction field in 3D -- the object surface, the hand
landmarks, and the nearest-neighbor vectors from each hand joint to the object --
and saves it to a PNG. Needs matplotlib (the end-to-end eval demo in
``show3d.interaction_field.demo`` does not).

    python -m show3d.demo_viz --out field.png                        # synthetic scene
    python -m show3d.demo_viz --root DIR --manifest M --out field.png
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from matplotlib.figure import Figure

from .dataset import default_object_mesh_provider
from .interaction_field import (
    FIELD_NAMES,
    InteractionFieldExample,
    LEFT_TO_OBJECT,
    RIGHT_TO_OBJECT,
    Show3DInteractionFieldDataset,
)
from .interaction_field.demo import build_synthetic_scene

_HAND_COLORS: dict[str, str] = {
    LEFT_TO_OBJECT: "tab:blue",
    RIGHT_TO_OBJECT: "tab:green",
}


def visualize_interaction_field(
    example: InteractionFieldExample, out_path: str | Path
) -> None:
    """Draw one frame's object surface + hand landmarks + field arrows to a PNG."""
    fig = Figure(figsize=(9, 7))
    ax = fig.add_subplot(projection="3d")

    object_pose = example.frame_data.object_pose
    alias = example.sample.object_alias
    mesh = default_object_mesh_provider()(alias) if alias is not None else None
    if mesh is not None and object_pose is not None:
        surface = object_pose.pose_vertices(mesh)
        step = max(1, surface.shape[0] // 3000)  # thin dense meshes for a light plot
        ax.scatter(
            surface[::step, 0],
            surface[::step, 1],
            surface[::step, 2],
            s=1,
            c="0.75",
            label=f"{alias} surface",
        )

    labels = example.labels
    hands = {
        LEFT_TO_OBJECT: example.frame_data.left_hand,
        RIGHT_TO_OBJECT: example.frame_data.right_hand,
    }
    if labels is not None:
        for field_name in FIELD_NAMES:
            field = labels.get(field_name)
            hand = hands[field_name]
            if field is None or hand is None or hand.landmarks_world_mm is None:
                continue
            joints = hand.landmarks_world_mm
            color = _HAND_COLORS[field_name]
            ax.scatter(
                joints[:, 0],
                joints[:, 1],
                joints[:, 2],
                s=20,
                c=color,
                label=field_name,
            )
            # Each arrow points from a hand joint to its nearest object point.
            ax.quiver(
                joints[:, 0],
                joints[:, 1],
                joints[:, 2],
                field[:, 0],
                field[:, 1],
                field[:, 2],
                color=color,
                arrow_length_ratio=0.1,
                linewidth=0.8,
            )

    ax.set_title(f"SHOW3D interaction field: {example.sample.sample_id}")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("z (mm)")
    ax.legend(loc="upper right", fontsize="small")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")


def run_visualization(
    root: str | Path,
    manifest_path: str | Path,
    out_path: str | Path,
    *,
    verbose: bool = False,
) -> Path:
    """Load the first valid frame and visualize its interaction field."""
    dataset = Show3DInteractionFieldDataset(root, manifest_path)
    example: InteractionFieldExample | None = None
    for index in range(len(dataset)):
        candidate = dataset[index]
        if candidate.is_valid:
            example = candidate
            break
    if example is None:
        raise ValueError("No valid frame to visualize (headset tracked + pose present)")

    out = Path(out_path)
    visualize_interaction_field(example, out)
    if verbose:
        print(f"Wrote interaction-field visualization for {example.sample.sample_id}")
        print(f"  -> {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SHOW3D interaction-field visualization demo"
    )
    parser.add_argument("--root", type=Path, default=None, help="SHOW3D mirror root")
    parser.add_argument(
        "--manifest", type=Path, default=None, help="challenge manifest .jsonl"
    )
    parser.add_argument(
        "--out", type=Path, default=Path("interaction_field.png"), help="output PNG"
    )
    args = parser.parse_args()

    print("SHOW3D interaction-field visualization demo\n")
    if args.root is not None and args.manifest is not None:
        run_visualization(args.root, args.manifest, args.out, verbose=True)
        return

    with tempfile.TemporaryDirectory() as td:
        manifest_path = build_synthetic_scene(Path(td))
        print("No --root/--manifest given; visualizing a synthetic toy scene.")
        run_visualization(Path(td), manifest_path, args.out, verbose=True)


if __name__ == "__main__":
    main()
