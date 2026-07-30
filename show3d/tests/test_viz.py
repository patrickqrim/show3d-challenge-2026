# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import tempfile
import unittest
from pathlib import Path

from ..interaction_field.demo import build_synthetic_scene

# The visualization demo needs matplotlib; the rest of the package does not, so
# guard the import and skip this test when it is unavailable.
try:
    from ..demo_viz import run_visualization
except ModuleNotFoundError:
    run_visualization = None


@unittest.skipUnless(
    run_visualization is not None, "matplotlib is required for the visualization demo"
)
class VizDemoTest(unittest.TestCase):
    def test_visualization_writes_png(self) -> None:
        assert run_visualization is not None
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest_path = build_synthetic_scene(root, num_frames=4)
            out_path = root / "field.png"

            returned = run_visualization(root, manifest_path, out_path)

            self.assertEqual(returned, out_path)
            self.assertTrue(out_path.exists())
            self.assertGreater(out_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
