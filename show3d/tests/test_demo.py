# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import tempfile
import unittest
from pathlib import Path

from ..interaction_field import LEFT_TO_OBJECT, NUM_HAND_LANDMARKS, RIGHT_TO_OBJECT
from ..interaction_field.demo import build_synthetic_scene, run_demo


class DemoTest(unittest.TestCase):
    def test_demo_runs_end_to_end_and_scores_valid_frames(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            num_frames = 6
            manifest_path = build_synthetic_scene(root, num_frames=num_frames)

            result = run_demo(root, manifest_path, seed=0)

            # Frame 3 is synthesized (headset tracking failed) -> skipped; each
            # remaining frame contributes NUM_HAND_LANDMARKS left-hand points.
            left = result.fields[LEFT_TO_OBJECT]
            self.assertEqual(left.num_points, NUM_HAND_LANDMARKS * (num_frames - 1))
            self.assertEqual(left.missing_predictions, 0)
            self.assertIsNotNone(left.ade_mm)

            # No right hand in the scene -> nothing to score.
            self.assertEqual(result.fields[RIGHT_TO_OBJECT].num_points, 0)
            self.assertIsNone(result.fields[RIGHT_TO_OBJECT].ade_mm)

            self.assertIsNotNone(result.mean_ade_mm)

    def test_demo_is_deterministic_for_a_fixed_seed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest_path = build_synthetic_scene(root, num_frames=4)
            first = run_demo(root, manifest_path, seed=7)
            second = run_demo(root, manifest_path, seed=7)
            self.assertEqual(
                first.fields[LEFT_TO_OBJECT].ade_mm,
                second.fields[LEFT_TO_OBJECT].ade_mm,
            )


if __name__ == "__main__":
    unittest.main()
