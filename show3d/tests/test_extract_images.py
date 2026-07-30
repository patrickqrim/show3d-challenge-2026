# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import json
import tempfile
import unittest
from pathlib import Path
from typing import cast

import cv2
import numpy as np

from ..extract_images import run_extraction, validate_fps
from ..interaction_field.demo import build_synthetic_scene


class ExtractImagesTest(unittest.TestCase):
    def _write_video(
        self,
        path: Path,
        num_frames: int,
        fps: int = 60,
        size: tuple[int, int] = (80, 64),
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        width, height = size
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height)
        )
        self.assertTrue(writer.isOpened(), "mp4v VideoWriter unavailable")
        for i in range(num_frames):
            writer.write(np.full((height, width, 3), (i * 7) % 256, dtype=np.uint8))
        writer.release()

    def _read_index(self, out: Path) -> list[dict[str, object]]:
        return [
            json.loads(line)
            for line in (out / "index.jsonl").read_text().splitlines()
            if line.strip()
        ]

    def test_extract_writes_frames_and_index(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "data"
            scene_dir = root / "scenes" / "S001" / "mug_grab_a1b2"
            for view in ("headset0", "headset1"):
                self._write_video(scene_dir / f"{view}.mp4", num_frames=13, fps=60)

            out = Path(td) / "frames"
            info = run_extraction(
                root, out, fps=10, require_object_pose=False, workers=1
            )

            self.assertEqual(info["num_recordings"], 1)
            rows = self._read_index(out)
            self.assertEqual(len(rows), info["num_frames"])
            self.assertEqual(
                set(rows[0].keys()),
                {"sample_id", "subject_id", "scene_id", "frame_index", "view", "image"},
            )

            # 60 fps sampled to 10 fps -> stride 6; both views identical; frame 0 kept.
            by_view: dict[str, set[int]] = {}
            for row in rows:
                frame_index = cast(int, row["frame_index"])
                self.assertEqual(frame_index % 6, 0)
                self.assertTrue((out / str(row["image"])).exists())
                self.assertTrue(str(row["image"]).endswith(".jpg"))
                by_view.setdefault(str(row["view"]), set()).add(frame_index)
            self.assertEqual(set(by_view), {"headset0", "headset1"})
            self.assertEqual(by_view["headset0"], by_view["headset1"])
            self.assertIn(0, by_view["headset0"])

    def test_single_view_lower_fps_png(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "data"
            scene_dir = root / "scenes" / "S001" / "mug_grab_a1b2"
            self._write_video(scene_dir / "headset0.mp4", num_frames=25, fps=60)
            self._write_video(scene_dir / "headset1.mp4", num_frames=25, fps=60)

            out = Path(td) / "frames"
            info = run_extraction(
                root,
                out,
                fps=5,
                views=("headset0",),
                image_format="png",
                require_object_pose=False,
                workers=1,
            )

            rows = self._read_index(out)
            self.assertTrue(all(row["view"] == "headset0" for row in rows))
            for row in rows:
                self.assertEqual(cast(int, row["frame_index"]) % 12, 0)  # 60 -> 5 fps
                self.assertTrue(str(row["image"]).endswith(".png"))
            self.assertEqual(info["num_frames"], len(rows))
            self.assertEqual(info["format"], "png")

    def test_require_object_pose_filters_to_training_set(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "data"
            # Posed scene (interaction-field training set): has object_pose.json.
            posed = root / "scenes" / "S001" / "mug_grab_aaaa"
            self._write_video(posed / "headset0.mp4", num_frames=13, fps=60)
            op_dir = root / "object_pose" / "v1" / "scenes" / "S001" / "mug_grab_aaaa"
            op_dir.mkdir(parents=True)
            (op_dir / "object_pose.json").write_text("{}")
            # Unposed scene: no object_pose.json -> skipped by default.
            unposed = root / "scenes" / "S001" / "aria_hold_bbbb"
            self._write_video(unposed / "headset0.mp4", num_frames=13, fps=60)

            out = Path(td) / "frames"
            info = run_extraction(root, out, fps=10, views=("headset0",), workers=1)

            self.assertEqual(info["num_recordings"], 1)
            scenes = {row["scene_id"] for row in self._read_index(out)}
            self.assertEqual(scenes, {"mug_grab_aaaa"})

    def test_manifest_selects_exact_scenes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "data"
            for scene in ("mug_grab_aaaa", "aria_hold_bbbb"):
                self._write_video(
                    root / "scenes" / "S001" / scene / "headset0.mp4",
                    num_frames=13,
                    fps=60,
                )
            manifest = Path(td) / "m.jsonl"
            manifest.write_text(
                json.dumps({"subject_id": "S001", "scene_id": "mug_grab_aaaa"}) + "\n"
            )

            out = Path(td) / "frames"
            info = run_extraction(
                root, out, fps=10, views=("headset0",), manifest=manifest, workers=1
            )

            self.assertEqual(info["num_recordings"], 1)
            scenes = {row["scene_id"] for row in self._read_index(out)}
            self.assertEqual(scenes, {"mug_grab_aaaa"})

    def test_save_labels_writes_aligned_deduped_labels(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "data"
            # Synthetic scene ships valid object_pose + hand_pose (left hand only);
            # its mp4s are empty, so overlay real videos to decode.
            build_synthetic_scene(root, num_frames=13)
            scene = root / "scenes" / "S000" / "mug_grab_demo"
            for view in ("headset0", "headset1"):
                self._write_video(scene / f"{view}.mp4", num_frames=13, fps=60)

            out = Path(td) / "frames"
            info = run_extraction(
                root,
                out,
                fps=10,
                require_object_pose=False,
                workers=1,
                save_labels=True,
            )

            # index: per (frame, view) -> 3 frames (0, 6, 12) x 2 views = 6 rows.
            index_sids = {row["sample_id"] for row in self._read_index(out)}
            labels = [
                json.loads(line)
                for line in (out / "labels.jsonl").read_text().splitlines()
                if line.strip()
            ]
            label_sids = [row["sample_id"] for row in labels]
            # One label per FRAME, deduped across the two views, joining to images.
            self.assertEqual(len(label_sids), len(set(label_sids)))
            self.assertEqual(len(labels), 3)
            self.assertEqual(info["num_labels"], 3)
            self.assertTrue(set(label_sids).issubset(index_sids))
            for row in labels:
                self.assertEqual(len(cast(list, row["left_to_object"])), 21)
                self.assertIsNone(row["right_to_object"])

    def test_save_labels_requires_pose_trees(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "data"
            self._write_video(
                root / "scenes" / "S001" / "mug_grab_a1b2" / "headset0.mp4",
                num_frames=13,
                fps=60,
            )
            out = Path(td) / "frames"
            with self.assertRaises(ValueError):
                run_extraction(
                    root,
                    out,
                    fps=10,
                    views=("headset0",),
                    require_object_pose=False,
                    workers=1,
                    save_labels=True,
                )

    def test_validate_fps_accepts_divisors_rejects_others(self) -> None:
        for good in (10, 12, 30, 60, 1):  # divisors of 60
            validate_fps(good, 60.0)
        for bad in (0, -1, 7, 8, 9, 11, 13, 120):  # not positive divisors of 60
            with self.assertRaises(ValueError):
                validate_fps(bad, 60.0)


if __name__ == "__main__":
    unittest.main()
