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

import numpy as np

from ..dataset import (
    ObjectPoseFrame,
    Show3DDataset,
    Show3DFrameRef,
    write_frame_manifest_jsonl,
)


class Show3DDatasetTest(unittest.TestCase):
    def test_generic_frame_loader_reads_paths_and_pose_annotations(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            frame = Show3DFrameRef(
                subject_id="S001",
                scene_id="mug_grab_a1b2",
                frame_index=0,
                object_alias="mug",
            )
            manifest_path = root / "frames.jsonl"
            write_frame_manifest_jsonl(manifest_path, [frame])
            self._write_scene_files(root, frame)

            dataset = Show3DDataset.from_manifest_jsonl(root, manifest_path)
            item = dataset[0]

            self.assertEqual(item.frame, frame)
            self.assertEqual(set(item.views), {"headset0", "headset1"})
            self.assertEqual(item.views["headset0"].video_path.name, "headset0.mp4")
            self.assertEqual(item.views["headset1"].video_path.name, "headset1.mp4")
            self.assertIsNone(item.views["headset0"].image)
            calibration = item.views["headset0"].calibration
            assert calibration is not None
            self.assertEqual(calibration.fx, 450.0)
            self.assertEqual(calibration.image_width, 1024)
            np.testing.assert_allclose(calibration.t_world_from_camera, np.eye(4))
            self.assertIsNotNone(item.views["headset1"].calibration)
            self.assertTrue(item.headset_tracking_valid)
            self.assertEqual(item.frame_info_path.name, "frame_info.json")
            self.assertIsNotNone(item.object_pose)
            self.assertIsNotNone(item.left_hand)
            self.assertIsNotNone(item.right_hand)
            assert item.object_pose is not None
            assert item.left_hand is not None
            assert item.right_hand is not None
            self.assertEqual(item.object_pose.confidence, 0.99)
            np.testing.assert_allclose(item.object_pose.rotation, np.eye(3))
            np.testing.assert_allclose(item.object_pose.translation_mm, np.zeros(3))
            self.assertEqual(item.left_hand.confidence, 0.95)
            self.assertEqual(item.right_hand.confidence, 0.0)

    def test_pose_vertices_applies_world_from_object_transform(self) -> None:
        # 90deg about +z (world-from-object) then translate by (100, 0, 0) mm.
        rotation = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        pose = ObjectPoseFrame(
            confidence=1.0,
            rotation=rotation,
            translation_mm=np.asarray([100.0, 0.0, 0.0]),
        )
        world = pose.pose_vertices(np.asarray([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]))
        np.testing.assert_allclose(
            world, np.asarray([[100.0, 1.0, 0.0], [98.0, 0.0, 0.0]])
        )

    def test_single_view_selection_populates_one_view(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            frame = Show3DFrameRef(
                subject_id="S001",
                scene_id="mug_grab_a1b2",
                frame_index=0,
                object_alias="mug",
            )
            manifest_path = root / "frames.jsonl"
            write_frame_manifest_jsonl(manifest_path, [frame])
            self._write_scene_files(root, frame)

            dataset = Show3DDataset.from_manifest_jsonl(
                root, manifest_path, multiview=False
            )
            self.assertEqual(set(dataset[0].views), {"headset0"})

    def test_synthesized_frame_withholds_pose_and_flags_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            frame = Show3DFrameRef(
                subject_id="S001",
                scene_id="mug_grab_a1b2",
                frame_index=0,
                object_alias="mug",
            )
            manifest_path = root / "frames.jsonl"
            write_frame_manifest_jsonl(manifest_path, [frame])
            self._write_scene_files(root, frame, is_synthesized=True)

            item = Show3DDataset.from_manifest_jsonl(root, manifest_path)[0]

            self.assertFalse(item.headset_tracking_valid)
            calibration = item.views["headset0"].calibration
            assert calibration is not None
            self.assertTrue(calibration.is_synthesized)
            self.assertIsNone(calibration.t_world_from_camera)

    def _write_scene_files(
        self, root: Path, frame: Show3DFrameRef, *, is_synthesized: bool = False
    ) -> None:
        scene_dir = root / "scenes" / frame.subject_id / frame.scene_id
        scene_dir.mkdir(parents=True)
        (scene_dir / "headset0.mp4").write_bytes(b"")
        (scene_dir / "headset1.mp4").write_bytes(b"")

        calibration_dir = scene_dir / "camera_calibration"
        calibration_dir.mkdir()
        identity = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        for name in ("headset0", "headset1"):
            with (calibration_dir / f"{name}.json").open("w") as f:
                json.dump(
                    {
                        "ImageSizeX": 1024,
                        "ImageSizeY": 1280,
                        "fx": 450.0,
                        "fy": 450.0,
                        "cx": 512.0,
                        "cy": 640.0,
                        "DistortionModel": "PinholePlane",
                        "T_WorldFromCamera_by_index": {
                            "0": {
                                "index": 0,
                                "T_WorldFromCamera": identity,
                                "is_synthesized": is_synthesized,
                            }
                        },
                    },
                    f,
                )

        metadata_dir = scene_dir / "metadata"
        metadata_dir.mkdir()
        (metadata_dir / "frame_info.json").write_text("{}")

        object_pose_dir = (
            root / "object_pose" / "v1" / "scenes" / frame.subject_id / frame.scene_id
        )
        object_pose_dir.mkdir(parents=True)
        with (object_pose_dir / "object_pose.json").open("w") as f:
            json.dump(
                {
                    "0": {
                        "confidence": 0.99,
                        "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                        "t": [[0.0], [0.0], [0.0]],
                    }
                },
                f,
            )

        hand_pose_dir = (
            root / "hand_pose" / "v2" / "scenes" / frame.subject_id / frame.scene_id
        )
        hand_pose_dir.mkdir(parents=True)
        with (hand_pose_dir / "hand_pose.json").open("w") as f:
            json.dump(
                {
                    "0": {
                        "hand_poses": {
                            "0": {
                                "confidence": 0.95,
                                "landmarks_3d_mm": [[1.0, 0.0, 0.0]],
                            },
                            "1": {
                                "confidence": 0.0,
                                "landmarks_3d_mm": None,
                            },
                        }
                    }
                },
                f,
            )


if __name__ == "__main__":
    unittest.main()
