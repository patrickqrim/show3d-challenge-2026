# SHOW3D dataset API

Standalone Python starter APIs for the SHOW3D dataset (`facebook/show3d-dataset`).

## Install

```bash
pip install -r requirements.txt
# or: pip install numpy opencv-python   (add matplotlib for the visualization demo)
```

## The dataset

Download SHOW3D from Hugging Face:
[`facebook/show3d-dataset`](https://huggingface.co/datasets/facebook/show3d-dataset).
`show3d.dataset` is the generic, task-agnostic loader: frame references and manifest JSONL
helpers, path resolution, and per-frame object-pose / hand-pose / calibration
loading. Point it at your local mirror:

```python
from show3d.dataset import Show3DDataset

dataset = Show3DDataset.from_manifest_jsonl("/path/to/show3d", "manifest.jsonl")
frame = dataset[0]   # views (video paths + calibration), object_pose, left/right hand
```

The loader expects the released on-disk layout under `root`:

```
<root>/
├── scenes/<subject>/<scene>/
│   ├── headset0.mp4, headset1.mp4              # egocentric videos
│   ├── camera_calibration/headset{0,1}.json   # intrinsics + per-frame pose
│   └── metadata/frame_info.json
├── object_pose/<version>/scenes/<subject>/<scene>/object_pose.json
└── hand_pose/<version>/scenes/<subject>/<scene>/hand_pose.json
```

## Repository layout

```
show3d/
├── dataset.py                    # generic SHOW3D dataloader API
├── demo.py                       # visualization demo (interaction field -> PNG)
├── interaction_field/
│   ├── __init__.py               # interaction-field challenge API
│   └── demo.py                   # end-to-end demo: load -> model -> eval
├── assets/objects/               # bundled object meshes (.glb, HOT3D-derived)
└── tests/                        # unit tests
```

# Interaction Field Estimation

The task is per-frame: given an egocentric frame, predict, for each hand, a
**hand-anchored 3D vector field**: the offset from every one of the 21 hand
joints to the nearest surface point of the manipulated object. The prediction is
therefore fixed size, `(21, 3)` per hand, independent of the object.

`show3d.interaction_field` builds on `show3d.dataset` and owns the task-specific
contracts: the sampled-frame manifest, directed-field label generation, the
submission JSONL schema (`predictions.jsonl`), and the local ADE / accuracy
evaluator.

## Quickstart

Two demos, both runnable with **no data download**: they fall back to a small
synthetic scene, so you see the whole pipeline in one command.

**End-to-end demo** loads frames, runs a model, evaluates:

```bash
python -m show3d.interaction_field.demo
```

```
Loaded 6 frames from the manifest (5 valid: headset tracked + object/hand pose present).
  views: headset0, headset1 | image 1408x1408

Interaction field for S000/mug_grab_demo:000000 (hand joint -> nearest object-surface point, mm):
  left_to_object  : 21x3 field | joint0 gt=(    -0,     7,    -6) pred=(    13,   -13,    64)

Evaluation (interaction field, naive random model):
  left_to_object  : ADE    163.5 mm |   105 pts | acc@10mm=0.00 acc@50mm=0.02 acc@100mm=0.18
  right_to_object : no valid targets
  mean ADE: 163.5 mm
```

**Visualization demo** renders one frame's interaction field to a PNG:

```bash
python -m show3d.demo --out field.png
```

Run either on your own mirror with `--root DIR --manifest MANIFEST.jsonl`.

## Build your baseline

The end-to-end demo is the template. To turn it into a real baseline, edit one
function, `naive_model` in `show3d/interaction_field/demo.py`:

```python
def naive_model(example, rng):
    # example.views[name].image      -> decoded frame (with decode_images=True)
    # example.views[name].calibration -> pinhole intrinsics + world-from-camera
    ...
    return {"left_to_object": left_vecs, "right_to_object": right_vecs}  # each (21, 3)
```

The demo writes `predictions.jsonl` and scores it with the official ADE /
accuracy evaluator.

## Single-view or multi-view

Choose the view mode at construction with the `multiview` bool:

```python
Show3DInteractionFieldDataset(root, manifest)                   # both headsets (default)
Show3DInteractionFieldDataset(root, manifest, multiview=False)  # single-view (headset0)
```

Each frame's `frame_data.views` is a dict keyed by the selected view name. Each
`ViewFrame` has `video_path` (the MP4) and a parsed `calibration`
(`CameraCalibration`: pinhole `fx, fy, cx, cy`, image size, and the 4×4
`t_world_from_camera` for that frame). The interaction-field target is in 3D
world space, so it is view-independent, so the choice only affects the inputs.

Frames decode on demand: pass `decode_images=True` and each `ViewFrame.image`
is the decoded RGB frame `(H, W, 3)` uint8; otherwise `image` is `None` and you
get only `video_path`.

## Frame validity

Some frames have unreliable ground truth: `frame_data.headset_tracking_valid` is
`False` when headset tracking failed and the pose was interpolated (the released
`is_synthesized` flag). On those frames the synthesized `t_world_from_camera` is
withheld (`None`) and the dataset builds no target (`labels=None`), so you can't
train on them by accident. Use `example.is_valid` for the full check: it ANDs
headset tracking with a valid object pose and at least one valid hand.

## Object geometry (self-contained)

The interaction field is the nearest-neighbor vector field between hand landmarks
and the object surface. The object surface for a frame is the object's canonical
mesh posed by the released per-frame `R,t`:

```python
world_mm = object_pose.pose_vertices(canonical_vertices_object_mm)
```

To stay self-contained, `show3d/assets/objects/<alias>.glb` bundles one GLB mesh
per object (object frame, mm) for the 21 challenge objects, so
`Show3DInteractionFieldDataset` builds labels out of the box, with no HOT3D checkout
and no multi-gigabyte per-frame vertices. Pass your own `object_mesh_provider`
to override.

# Reference

## Running the tests

```bash
python -m unittest discover
```

## License

The code and the bundled object meshes are released under Creative Commons
Attribution-NonCommercial 4.0 International (CC BY-NC 4.0); see `LICENSE`. The
bundled meshes are derived from the HOT3D object models (BOP HOT3D release,
`object_models_eval`); see `show3d/assets/objects/ATTRIBUTION.md`.

## Citation

If you use SHOW3D, please cite:

```bibtex
@article{rim2026show3d,
  title   = {SHOW3D: Capturing Scenes of 3D Hands and Objects in the Wild},
  author  = {Rim, Patrick and Harris, Kevin and Copple, Braden and Han, Shangchen and
             Xie, Xu and Shugurov, Ivan and An, Sizhe and Wen, He and
             Wong, Alex and Hodan, Tomas and others},
  journal = {arXiv preprint arXiv:2603.28760},
  year    = {2026}
}
```
