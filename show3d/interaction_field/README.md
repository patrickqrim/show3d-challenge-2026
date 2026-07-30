# Interaction Field Estimation

The SHOW3D Interaction Field Estimation Challenge is hosted at the HANDS workshop
at ECCV 2026. The task is per-frame: given an egocentric frame, predict, for each
hand, a
**hand-anchored 3D vector field**: the offset from
every one of the 21 hand joints to the nearest surface point of the manipulated
object. The prediction is therefore fixed size, `(21, 3)` per hand, independent
of the object.

`show3d.interaction_field` builds on `show3d.dataset` (see the
[top-level README](../../README.md) for install and dataset setup) and owns the
task-specific contracts: the sampled-frame manifest, directed-field label
generation, the submission JSONL schema (`predictions.jsonl`), and the local
ADE / accuracy evaluator.

## Rules

* **Declare your input view** (single-view = headset0 only, multi-view = both
  headsets; see [below](#single-view-or-multi-view)).
* **Declare external training data.** Training on other datasets (for example,
  ARCTIC) is allowed, but submissions with and without external data are ranked
  separately, so you must declare what you used.

## Quickstart

The end-to-end demo runs the whole loop with **no data download** (it falls back
to a small synthetic scene). Load frames, run a model, evaluate:

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

Run it on your own mirror with `--root DIR --manifest MANIFEST.jsonl`.

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
accuracy evaluator. For fast training, pre-extract frames with
`show3d.extract_images --manifest show3d/interaction_field/train_manifest_202607.jsonl`
(the shipped list of training recordings; see the top-level README).

## Submission format

Write one JSON object per line to `predictions.jsonl`, keyed by `sample_id`
(the id is `"<SUBJECT>/<scene_id>:<frame_index:06d>"`):

```json
{"sample_id": "S001/mug_grab_a1b2:000120", "left_to_object": [[x, y, z], ...], "right_to_object": [[x, y, z], ...]}
```

* Each of `left_to_object` / `right_to_object` is a fixed `(21, 3)` array of
  millimeter vectors, one per hand joint (the predicted joint-to-nearest-
  object-surface offset, in world space).
* **Predict both hands.** You do not know at test time which hands have a valid
  target, so a missing or `null` field is scored as a miss (large penalty)
  wherever the target is valid; use `null` only to deliberately abstain.
* Every predicted field must be exactly `(21, 3)`; there is no object-side field.
* Write it with `write_submission_jsonl(path, [PredictionRecord(sample_id=...,
  fields={"left_to_object": arr, "right_to_object": arr})])`.
* Hosted evaluation: upload a zip with `predictions.jsonl` at its root.

## Evaluation

For every field that has a valid target, the endpoint error
`|| prediction_vector - target_vector ||` is computed per joint, in millimeters.
Reported metrics:

* **`mean_ade_mm`**: mean endpoint error across the two hand fields. This is the
  primary leaderboard metric; lower is better.
* **`left_to_object_ade_mm`**, **`right_to_object_ade_mm`**: per-field ADE.

Rules:

* A field is scored only where its target is valid (object pose confident and
  that hand valid); invalid ground-truth fields are ignored.
* A missing prediction for a valid field incurs a large penalty.
* A prediction whose shape is not `(21, 3)` (wrong joint count) is rejected.

`evaluate_submission_jsonl(dataset, "predictions.jsonl")` reproduces this scoring
offline, and additionally reports accuracy at 10 / 50 / 100 mm (the fraction of
joints within each threshold) for your own analysis.

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
world space, so it is view-independent; the choice only affects the inputs.
Declare which setting (single-view or multi-view) you used when you submit.

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
`Show3DInteractionFieldDataset` builds labels out of the box, with no HOT3D
checkout and no multi-gigabyte per-frame vertices. Pass your own
`object_mesh_provider` to override.

The bundled meshes are the HOT3D object models (BOP HOT3D release,
`object_models_eval`); see `show3d/assets/objects/ATTRIBUTION.md`.
