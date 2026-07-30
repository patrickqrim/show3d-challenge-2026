# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Thin CLI for SHOW3D visualization; the library is :mod:`show3d.viz`.

    python -m show3d.demo_viz --mode field --out field.png
    python -m show3d.demo_viz --mode geometry --root DIR --manifest M.jsonl --out geom.png
    python -m show3d.demo_viz --mode overlay --root DIR --manifest M.jsonl --out overlay.png

With no --root/--manifest it renders a bundled synthetic scene. Modes:
overlay (skeleton + object on the frame), geometry (3D skeleton + object),
field (3D interaction field).
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from . import viz
from .interaction_field.demo import build_synthetic_scene


def main() -> None:
    parser = argparse.ArgumentParser(description="SHOW3D visualization demo")
    parser.add_argument("--root", type=Path, default=None, help="SHOW3D mirror root")
    parser.add_argument(
        "--manifest", type=Path, default=None, help="challenge manifest .jsonl"
    )
    parser.add_argument("--out", type=Path, default=Path("viz.png"), help="output PNG")
    parser.add_argument(
        "--mode", default="field", choices=list(viz.MODES), help="what to draw"
    )
    parser.add_argument(
        "--view",
        default="headset0",
        choices=["headset0", "headset1"],
        help="egocentric view for overlay mode",
    )
    args = parser.parse_args()

    print(f"SHOW3D visualization demo (mode={args.mode})\n")
    if args.root is not None and args.manifest is not None:
        viz.run_visualization(
            args.root,
            args.manifest,
            args.out,
            mode=args.mode,
            view=args.view,
            verbose=True,
        )
        return

    with tempfile.TemporaryDirectory() as td:
        manifest_path = build_synthetic_scene(Path(td))
        print("No --root/--manifest given; visualizing a synthetic toy scene.")
        viz.run_visualization(
            Path(td),
            manifest_path,
            args.out,
            mode=args.mode,
            view=args.view,
            verbose=True,
        )


if __name__ == "__main__":
    main()
