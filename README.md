# Multi-Label Chest Pathology XAI

A 14-class chest pathology classifier on NIH ChestX-ray14, with HiResCAM
explanations and a FastAPI backend behind a single-screen React workspace.

> **Research prototype. Not a medical device. Not for clinical use.**

![Application demo](docs/figs/demo.gif)

## What it does

- Trains ConvNeXt-Small and DenseNet-121 on 14 pathologies from NIH
  ChestX-ray14.
- Produces HiResCAM overlays with
  `ReLU(sum_channels(gradient * activation))`, so the explanation follows the
  published method rather than a per-element ReLU variant.
- Tunes a decision threshold per disease with Youden's J on the validation
  split only.
- Serves inference from memory through FastAPI and returns an RGBA PNG that the
  frontend composites over the original X-ray.
- Splits by patient ID so follow-up scans cannot leak between train,
  validation, and test.
- Includes a DenseNet-121 demo checkpoint so a fresh clone can start the API
  without downloading 42 GB of images.

## Results

The numbers below are **historical artifacts from an 80/20 patient split**.
They are kept for comparison with the earlier version of the project. The
current code defaults to a **70/10/10** patient split so a held-out test set
exists; rerun training to produce current numbers.

| Model | Parameters | Resolution | Peak val AUROC | Peak val AUPRC |
|---|---|---:|---:|---:|
| ConvNeXt-Small | 49.47 M | 448x448 | 0.8583 | 0.2992 |
| DenseNet-121 | 6.97 M | 320x320 | 0.8488 | 0.2812 |

Peak AUROC and peak AUPRC are each the maximum over epochs and can come from
different epochs. The 0.8583 AUROC is epoch 7 of the 448x448 ConvNeXt run; its
AUPRC at that epoch is 0.2948. The 0.2992 figure is that run's peak AUPRC.

These are **not** numbers on the official NIH test list. CheXNet reports 0.841
mean AUROC on the official test list, while these numbers come from a random
patient split, so the comparison is only a rough reference.

Per-class metrics can be regenerated with `python -m src.evaluate`; the
historical artifacts did not save them.

## Dataset

NIH ChestX-ray14: 112,120 frontal radiographs from 30,805 patients. The code
splits on unique patient IDs, not images. The official image-level split is not
used because it contains patient overlap.

Class balance, co-occurrence, and view-position demographics are in
[docs/eda.md](docs/eda.md). Labels are mined from radiology reports and are
noisy.

## Explanations

![Confusion matrices](docs/figs/confusion_matrices.png)

The committed confusion-matrix figure was generated at a fixed 0.50 threshold
on the historical validation split. The API uses tuned thresholds when a run
contains `optimal_thresholds.json`.

| Finding | Overlay |
|---|---|
| Effusion (model score 84.1%) | ![Effusion](docs/figs/00012841_000_Effusion.png) |
| Infiltration (model score 63.2%) | ![Infiltration](docs/figs/00000181_011_Infiltration.png) |

The heatmap is generated for one selected class at a time. The UI lists all 14
predictions, marks which ones are above their threshold, and lets you select a
below-threshold finding to inspect its explanation.

## Run it

### 1. Python environment

```bash
git clone https://github.com/abderrahmenex86/Multi-Label-Chest-Pathology-XAI
cd Multi-Label-Chest-Pathology-XAI
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Use a CUDA build of PyTorch if you plan to train. The minimum versions in
`requirements.txt` install the default wheel for your platform; see the PyTorch
installation selector for a specific CUDA version.

### 2. Dataset

The downloader uses the Kaggle CLI. Install it (included in
`requirements.txt`) and configure `~/.kaggle/kaggle.json` first.

```bash
python -m src.train --download
```

That downloads the NIH archives, flattens the image folders into
`data/images/`, and writes `data/Data_Entry_2017.csv`. Expect roughly 42 GB of
free disk space.

### 3. Train

```bash
# Fast check: 20 steps on one batch, then exit.
python -m src.train --sanity --sanity-steps 20

# Full run. The current default split is 70/10/10 by patient.
python -m src.train --architecture convnext_small --batch 64 --epochs 50 \
  --width 448 --height 448
```

Useful options:

- `--val-fraction 0.1 --test-fraction 0.1` (default) or
  `--val-fraction 0.2 --test-fraction 0.0` for the historical 80/20 behavior.
- `--resume artifacts/<run>/last.pth` to continue a run.
- `--cache` loads the training images into RAM. This is roughly 23-26 GB at
  320x320 and 45-50 GB at 448x448 depending on the split; use it only on a
  machine with enough memory.
- `--no-compile` disables `torch.compile`.
- `--no-pretrained` skips the ImageNet download for offline or smoke-test runs.

Each run writes `hyperparameters.json`, `split.json`, `architecture.txt`,
`augmentation.png`, `history.json`, `best.pth`, `last.pth`, and, after training,
`test_metrics.json` to `artifacts/<run>/`.

### 4. Backend

```bash
python -m backend.main --host 127.0.0.1 --port 8000
```

The backend picks the newest complete run in `artifacts/`, then falls back to
the bundled `checkpoints/densenet121`. To force a run:

```bash
python -m backend.main --run checkpoints/densenet121
```

Environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `CHEST_XAI_RUN` | unset | Run directory to serve |
| `CHEST_XAI_ARTIFACTS` | `artifacts` | Directory containing runs |
| `CHEST_XAI_CORS_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` | Comma-separated allowlist |
| `CHEST_XAI_MAX_UPLOAD_BYTES` | `10485760` | Upload limit in bytes |

The API exposes `GET /health` and `POST /api/v1/predict`. Interactive docs are
at <http://127.0.0.1:8000/docs>.

### 5. Frontend

```bash
cd frontend
cp .env.example .env
npm install
npm run dev
```

Open <http://localhost:5173>. Set `VITE_API_URL` if the backend is not on
`http://localhost:8000`.

## Reproduce metrics and figures

```bash
# 1. Tune per-class thresholds on the validation split.
python -m src.find --run artifacts/<run>

# 2. Per-class AUROC/AUPRC and confusion counts on the test split.
python -m src.evaluate --run artifacts/<run> --split test

# 3. Confusion-matrix grid using the tuned thresholds.
python -m src.conf_matrices --run artifacts/<run> --split test

# 4. Training-curve and experiment comparison figures.
python -m src.compare
python -m src.plot --run artifacts/<run>
```

`src.find` writes `optimal_thresholds.json` and `thresholds_meta.json`.
`src.evaluate` writes `per_class_metrics_<split>.json`. See
[docs/reproduce.md](docs/reproduce.md) for a step-by-step version.

## Project layout

```text
backend/            FastAPI service
  engine.py         run loading and prediction wrapper
  main.py           HTTP routes and upload validation
checkpoints/        bundled DenseNet-121 demo checkpoint
docs/               EDA, figures, reproduction notes
frontend/           React 19 + Vite + Tailwind v4 viewer
src/
  dataset.py        NIH ChestX-ray14 dataset
  download.py       Kaggle download and extraction
  eda.py            dataset figures
  engine.py         training/evaluation loops
  evaluate.py       per-class metrics
  explain.py        HiResCAM and normalization
  find.py           Youden's J threshold tuning
  infer.py          CLI overlay generation
  labels.py         single label list
  model.py          torchvision backbones
  predictor.py      shared inference and CAM
  runs.py           run discovery and validation
  splits.py         patient-level splitting
  train.py          training entry point
  utils.py          seeding, logging, AMP, plotting
tests/              Python tests
```

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest
cd frontend && npm run lint && npm run typecheck && npm run build
```

The Python tests cover label parsing, patient-disjoint splits, the HiResCAM
formula, run resolution, API validation, and the bundled checkpoint.

## Limits

- Trained and evaluated only on NIH ChestX-ray14; other scanners, acquisition
  protocols, and populations are untested.
- AUPRC stays below 0.30, so rare-class precision is weak even with tuned
  thresholds.
- The historical runs used a single seed and no held-out test set. The current
  code adds a test split, but the committed numbers predate it.
- No calibration, subgroup/fairness analysis, or external validation is
  included.
- Heatmaps are qualitative. They have not been validated against radiologist
  bounding boxes or deletion/insertion metrics.
- A research prototype, not a diagnostic tool.

See [MODEL_CARD.md](MODEL_CARD.md) for intended use, out-of-scope use, and
ethical notes.

## Stack

Python · PyTorch · torchvision · FastAPI · React 19 · Tailwind v4

## References

- Wang et al., "ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks
  on Weakly-Supervised Classification and Localization of Common Thorax
  Diseases," CVPR 2017.
- Rajpurkar et al., "CheXNet: Radiologist-Level Pneumonia Detection on Chest
  X-Rays with Deep Learning," arXiv:1711.05225, 2017.
- Draelos and Carin, "Use HiResCAM instead of Grad-CAM for faithful
  explanations of convolutional neural networks," arXiv:2011.08891, 2020.
