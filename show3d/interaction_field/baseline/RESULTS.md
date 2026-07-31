# InterField baseline — reference results

Reference numbers for the SHOW3D Interaction Field Estimation Challenge (HANDS @
ECCV 2026), produced by the code in this directory. This is a **reference point**
for participants, not a tuned state-of-the-art system.

## Setup

| | |
| --- | --- |
| Model | InterField (after ARCTIC): ResNet-50 image encoder + MLP field head |
| Input | headset0 (single view), full frame → 224², grayscale→3ch, ImageNet-normalized |
| Output | per hand `(21, 3)` joint→nearest-object-surface vector, predicted in the camera frame and rotated to world |
| Train data | SHOW3D headset0, 10 fps → **87,366** labeled frames |
| Released checkpoint | trained on **all 10 train subjects** (final epoch), AdamW lr 1e-4, wd 5e-4, cosine over 20 epochs, batch 256, dropout 0.5, photometric aug; 3× RTX A6000 |
| Reported ADE | from a **leave-2-subjects-out dev run** (`XYZ109 LYA722` held out; 11,948 frames / 20,163 hand-fields) — a cross-subject generalization estimate |
| Checkpoints | `checkpoints/interfield_show3d_headset0.pt` (full frame) · `..._crop.pt` (hand crop) |

**The released checkpoints are trained on all 10 training subjects** (so no data is
wasted). The **ADE numbers below are not test-set numbers** — the official test
subjects (`BBL925 KHE522 SHE109`) have withheld labels, so ADE cannot be computed
on them locally. They come instead from a separate *dev* run that held out two
training subjects (`XYZ109 LYA722`) to estimate cross-subject generalization
("SHOW3D → SHOW3D"). To get an actual test number, run the released checkpoint over
`test_manifest_5fps_202607.jsonl` and submit `predictions.jsonl` to the hosted
evaluator (see [README](README.md#produce-a-test-submission)).

### Two variants

| Variant | Input | Test-time valid? | Checkpoint |
| --- | --- | --- | --- |
| **Full frame** (default) | whole 1024×1280 frame → 224² | ✅ yes — image + calibration only | `interfield_show3d_headset0.pt` |
| **Hand crop** (`crop_baseline.py`) | per-hand square crop → 224² | ⚠️ needs a hand **box** (from a detector) | `interfield_show3d_headset0_crop.pt` |

The hand-crop variant boxes each hand from the **ground-truth hand pose**, which is
withheld on the official test set — so it is an **oracle-box upper-bound
reference**, not a directly-submittable model. The full-frame model is the
submittable baseline.

## Results (held-out subjects, official evaluator)

| Variant | mean ADE (mm) ↓ | recall | acc@10mm | acc@50mm | acc@100mm | ACC (m/s²) ↓ |
| --- | --- | --- | --- | --- | --- | --- |
| Full frame (submittable) | 60.47 | 1.000 | 0.150 | 0.682 | 0.861 | 0.10 |
| Hand crop (oracle box)   | **54.70** | 0.969 | 0.207 | 0.719 | 0.858 | 0.08 |

Full-frame per field: left 60.27 / right 60.68 mm. Hand-crop per field: left 53.90
/ right 55.51 mm. The crop variant's recall is < 1.0 because a hand is only
predicted when its box exists (a few GT hands project outside the frame). ACC is
near the ground-truth field's own 0.083 m/s², i.e. predictions are temporally
stable, not jittery.

The hand crop improves ADE ~10 % — a real but modest gain. Since the hand is
already ~107 px in the 224² full frame, resolution was not the main limit; the
residual error is dominated by **monocular depth ambiguity** in regressing exact
3D vectors (the biggest lever left is multi-view, to resolve depth).

Reference baselines on the same held-out fields, for context:

| Predictor | mean ADE (mm) ↓ |
| --- | --- |
| random (demo `naive_model`) | ~163 |
| predict zero (assume contact) | 82.6 |
| predict train-mean field | 94.2 |
| **InterField (this baseline)** | **60.5** |

The model always predicts both hands, so recall is 1.0 by construction — matching
the challenge guidance that abstaining on a valid target is penalized.

## Reading the numbers

* **This is the challenge's vector metric.** ADE here is the mean L2 **endpoint**
  error of the predicted `(21, 3)` hand→object vectors, in world space — the metric
  the official evaluator computes and the leaderboard uses.
* **Mean ADE is outlier-heavy.** The GT field magnitude is a median of ~32 mm but
  a mean of ~83 mm (frames where the hand is far from the object have large,
  hard-to-regress vectors). The acc@k breakdown is more telling: **~68 % of joints
  land within 5 cm and ~86 % within 10 cm** of the true nearest-object point.

## How to reproduce

See [`README.md`](README.md) for the exact commands (extract → train → predict →
evaluate → visualize). `train_history.json` (next to the checkpoint) has the full
per-epoch curve.

## Where the gains are (measured)

This baseline is deliberately simple. On the levers we tried and what is left:

1. **Hand crop — tried, ~10 %.** Cropping to the hand (`crop_baseline.py`) lowers
   ADE from 60.5 → 54.7 mm. Modest, because the hand is already ~107 px in the full
   224² frame — resolution was not the bottleneck. It also needs a hand box, so it
   is not directly submittable.
2. **Multi-view — the big lever, untried.** The dominant residual error is
   monocular **depth ambiguity** (the along-ray component of each vector). The
   challenge allows both headsets; a stereo pair would constrain depth directly.
   This baseline uses headset0 only (single-view track).
3. **Color.** Trains on the grayscale JPEGs `extract_images` writes; decoding RGB
   from the MP4s adds appearance cues.
4. **Temporal modelling.** Inputs are single frames; a short temporal window would
   exploit the smoothness the ACC metric rewards.
