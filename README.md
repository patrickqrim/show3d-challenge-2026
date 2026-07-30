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
`show3d.dataset` is the generic, task-agnostic loader: frame references and
manifest JSONL helpers, path resolution, and per-frame object-pose / hand-pose /
calibration loading. Point it at your local mirror:

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

## Interaction Field Estimation challenge

The Interaction Field Estimation Challenge is hosted at the HANDS workshop at ECCV
2026: predict a per-frame, hand-anchored 3D interaction field. Its API, the
end-to-end demo, a baseline template, and evaluation live in
[`show3d/interaction_field/README.md`](show3d/interaction_field/README.md).

## Visualize a frame

`show3d.demo_viz` renders three views of a frame (needs `matplotlib`); the
projection and drawing primitives live in `show3d.camera` and `show3d.viz`:

```bash
python -m show3d.demo_viz --mode overlay  --out overlay.png    # skeleton + object on the frame
python -m show3d.demo_viz --mode geometry --out geometry.png   # 3D skeleton + object
python -m show3d.demo_viz --mode field    --out field.png      # 3D interaction field
```

Add `--root DIR --manifest MANIFEST.jsonl` to visualize your own mirror (overlay
decodes the frame).

**overlay**: the hand skeleton and object projected onto the egocentric frame.

![overlay](docs/overlay.png)

**geometry**: the hand skeleton and object surface in 3D. **field**: the
interaction field, arrows from each hand joint to the nearest object point.

![geometry](docs/geometry.png)

![field](docs/field.png)

## Extracting frames for training

Random per-frame video seeking is slow, so training straight from the MP4s is
decode-bound. Pre-extract the frames you need, at an fps **you choose**, into a
random-access image store:

```bash
python -m show3d.extract_images --root /path/to/show3d --out frames/ --fps 10 \
    --manifest show3d/interaction_field/train_manifest_202607.jsonl
# --views headset0 | --format png | --quality 80 | --workers 16
```

`--manifest` is the reproducible way to pick scenes: it ships at
`show3d/interaction_field/train_manifest_202607.jsonl` and lists the exact
interaction-field **training** recordings (train subjects only). Without it the
tool walks `<root>/scenes` and keeps scenes that have object-pose GT.

`--fps` is required and has no default: it dominates dataset size and temporal
coverage. It must be a whole number that divides the 60 fps source (1, 2, 3, 4,
5, 6, 10, 12, 15, 20, 30, 60); other values are rejected. The tool decodes each
recording once (sequential, no seeking), writes grayscale JPEGs (or PNG with
`--format png`) to `frames/<subject>/<scene>/<view>/<frame>.jpg` plus an
`index.jsonl` a map-style training `Dataset` can shuffle over.

Storage for the full 468-recording training set at the default JPEG quality
(`--quality 90`). Frame counts are exact; bytes per frame are the measured mean
grayscale JPEG size over a 10-recording sample (about 140 KB at q90):

| fps | frames / view | single-view | both views |
| --- | --- | --- | --- |
| 5  | 69,296  | 9 GiB   | 19 GiB  |
| 10 | 138,579 | 19 GiB  | 37 GiB  |
| 15 | 207,853 | 28 GiB  | 56 GiB  |
| 30 | 415,404 | 56 GiB  | 111 GiB |
| 60 | 830,597 | 111 GiB | 222 GiB |

`--quality 80` is about a third smaller (about 93 KB/frame); `--format png` is
lossless but roughly 3x larger (about 430 KB/frame).

### Labels alongside the frames

Add `--save-labels` and the tool also writes `frames/labels.jsonl`: the per-frame
interaction-field targets, one row per frame (deduped across views) in the
evaluator's reference-label format, keyed by the same `sample_id` as each image.
Training then joins the two by `sample_id` -- `index.jsonl` gives the image
path(s), `labels.jsonl` gives the `(21, 3)` target per hand.

```bash
python -m show3d.extract_images --root /path/to/show3d --out frames/ --fps 10 \
    --manifest show3d/interaction_field/train_manifest_202607.jsonl --save-labels
```

This needs `object_pose/` and `hand_pose/` under `--root` (not just the videos);
the tool errors up front if they are missing. The
[challenge README](show3d/interaction_field/README.md#training-set) walks the
end-to-end training flow.

## Repository layout

```
show3d/
├── dataset.py                    # generic SHOW3D dataloader API
├── demo_viz.py                   # visualization: interaction field -> PNG
├── extract_images.py             # pre-extract frames to images at a chosen fps
├── interaction_field/
│   ├── README.md                 # the challenge: task, demos, baseline, eval
│   ├── train_manifest_202607.jsonl   # exact training recordings (train subjects)
│   ├── __init__.py               # interaction-field challenge API
│   └── demo.py                   # end-to-end demo: load -> model -> eval
├── assets/objects/               # bundled object meshes (.glb, HOT3D-derived)
└── tests/                        # unit tests
```

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
@InProceedings{Rim_2026_CVPR,
    author    = {Rim, Patrick and Harris, Kevin and Copple, Braden and Han, Shangchen and Xie, Xu and Shugurov, Ivan and An, Sizhe and Wen, He and Wong, Alex and Hodan, Tomas and He, Kun},
    title     = {SHOW3D: Capturing Scenes of 3D Hands and Objects in the Wild},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026},
    pages     = {7111-7120}
}
```
