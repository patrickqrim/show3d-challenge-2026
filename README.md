# SHOW3D dataset API

Standalone Python starter APIs for the SHOW3D dataset (`facebook/show3d-dataset`)
and the interaction-field challenge. The core (dataloader + labels + eval) needs
only `numpy` and `opencv-python`; the visualization demo also needs `matplotlib`.

## Install

```bash
pip install -r requirements.txt
# or: pip install numpy opencv-python   (add matplotlib for the visualization demo)
```

## Quickstart

Two demos, both runnable with **no data download** — they fall back to a small
synthetic scene, so you can see the whole pipeline in one command.

**1. End-to-end interaction-field demo** — load frames, run a model, evaluate:

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

**2. Visualization demo** — render one frame's interaction field to a PNG:

```bash
python -m show3d.demo --out field.png
```

Run either on your own mirror with `--root DIR --manifest MANIFEST.jsonl`.

## Build your baseline

The end-to-end demo is the template. To turn it into a real baseline, edit one
function — `naive_model` in `show3d/interaction_field/demo.py`:

```python
def naive_model(example, rng):
    # example.views[name].image      -> decoded frame (with decode_images=True)
    # example.views[name].calibration -> pinhole intrinsics + world-from-camera
    ...
    return {"left_to_object": left_vecs, "right_to_object": right_vecs}  # each (21, 3)
```

The prediction is fixed size — one `(21, 3)` vector field per hand — so it does
not depend on the object. The demo writes `predictions.jsonl` and scores it with
the official ADE / accuracy evaluator.

## Layout

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

## Generic dataset API

`show3d.dataset` only knows the public dataset layout and annotation schemas:

* frame references and manifest JSONL helpers;
* path resolution for videos, calibration, `hand_pose/<version>/`, and
  `object_pose/<version>/`;
* object-pose, hand-pose, and per-frame calibration loading.

Use it for any SHOW3D task.

## Interaction-field challenge API

`show3d.interaction_field` builds on `show3d.dataset` and owns only the
task-specific contracts:

* sampled-frame challenge manifest;
* directed interaction-field label generation;
* submission JSONL schema — write `predictions.jsonl` for the challenge;
* local ADE / accuracy evaluator.

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
world space, so it is view-independent — the choice only affects the inputs.

Frames decode on demand: pass `decode_images=True` and each `ViewFrame.image`
is the decoded RGB frame `(H, W, 3)` uint8; otherwise `image` is `None` and you
get only `video_path`.

## Frame validity

Some frames have unreliable ground truth: `frame_data.headset_tracking_valid` is
`False` when headset tracking failed and the pose was interpolated (the released
`is_synthesized` flag). On those frames the synthesized `t_world_from_camera` is
withheld (`None`) and the interaction-field dataset builds no target
(`labels=None`), so you can't train on them by accident. Use `example.is_valid`
for the full check — it ANDs headset tracking with a valid object pose and at
least one valid hand (i.e. a target actually exists).

## Object geometry (self-contained)

The interaction field is the nearest-neighbor vector field between hand landmarks
and the object surface. The object surface for a frame is the object's canonical
mesh posed by the released per-frame `R,t`:

```python
world_mm = object_pose.pose_vertices(canonical_vertices_object_mm)
```

To stay self-contained, `show3d/assets/objects/<alias>.glb` bundles one GLB mesh
per object (object frame, mm) for the 21 challenge objects, so
`Show3DInteractionFieldDataset` builds labels out of the box — no HOT3D checkout
and no multi-gigabyte per-frame vertices. Pass your own `object_mesh_provider` to
override.

The bundled meshes are the HOT3D object models (BOP HOT3D release,
`object_models_eval`), redistributed here under CC BY-NC 4.0 with attribution —
see `show3d/assets/objects/ATTRIBUTION.md`.

## Running the tests

From the repository root:

```bash
python -m unittest discover
```
