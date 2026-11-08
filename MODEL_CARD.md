# Model card: ChestXray14 multi-label classifier

## Overview

A 14-class multi-label classifier for NIH ChestX-ray14 radiographs, built with
torchvision backbones and explained with HiResCAM. The repository also includes
a FastAPI inference service and a React viewer.

## Intended use

- Research and education in multi-label chest X-ray classification.
- Demonstrating gradient-based explanation methods.
- Portfolio and reproducibility work.

## Out-of-scope use

- Clinical diagnosis, triage, screening, or treatment decisions.
- Any use with real patient data without local ethics and privacy review.
- Populations, scanners, or acquisition protocols outside NIH ChestX-ray14.
- Claims that a heatmap is a lesion segmentation or a causal explanation.

This project is not a medical device and has not been cleared or approved by any
regulator.

## Training data

- Dataset: NIH ChestX-ray14 (Wang et al., 2017).
- 112,120 frontal chest radiographs from 30,805 patients.
- Labels are mined from radiology reports and are noisy.
- The code defaults to a patient-level 70/10/10 split. Historical artifacts in
  this repository were trained with an 80/20 patient split.
- Patient IDs are disjoint across splits. The official image-level NIH split is
  not used.

## Model

| Model | Parameters | Resolution | Historical peak val AUROC | Historical peak val AUPRC |
|---|---:|---:|---:|---:|
| ConvNeXt-Small | 49.47 M | 448x448 | 0.8583 | 0.2992 |
| DenseNet-121 | 6.97 M | 320x320 | 0.8488 | 0.2812 |

The bundled DenseNet-121 checkpoint (`checkpoints/densenet121`) comes from the
historical 80/20 run. New runs default to 70/10/10 so a held-out test split
exists.

A fresh ConvNeXt-Base run on the 70/10/10 split reached **0.8568 test AUROC /
0.3061 test AUPRC** after 11 epochs (best validation AUROC 0.8602 at epoch 7).
The test set contains 11,195 images from 3,080 held-out patients and was not
used for training or threshold tuning.

The training pipeline also supports ConvNeXt-Tiny, ConvNeXt-Base, and
ConvNeXt-Large. On a 24 GB RTX 3090, ConvNeXt-Base at 384x384 with batch 32 and
two gradient-accumulation steps is the practical larger-model option; it uses
native bf16 autocast.

## Preprocessing

- Load RGB, apply EXIF orientation, resize directly to the configured square
  resolution.
- Scale to `[0, 1]` and normalize with ImageNet mean/std.
- Training augmentation: random horizontal flip and +/-7 degree rotation.
- Inference uses the same resize and normalization without augmentation.

## Evaluation

- Macro AUROC and AUPRC are reported during training.
- Per-class thresholds are tuned with Youden's J on the validation split only.
- `python -m src.evaluate` computes per-class AUROC, AUPRC, sensitivity,
  specificity, and precision on the validation or test split.
- The historical results did not have a held-out test split. Do not compare
  them directly with CheXNet's official-test numbers.
- No bootstrap confidence intervals, multi-seed variance, calibration analysis,
  or subgroup analysis is bundled yet.

## Explainability

- HiResCAM: `ReLU(sum over channels(gradient * activation))`.
- The heatmap is computed for a single selected class per request.
- The method has not been validated against radiologist bounding boxes or
  deletion/insertion metrics in this repository. Treat the overlays as
  qualitative debugging aids.

## Limitations and risks

- Label noise from report mining.
- Low AUPRC for rare classes, especially Hernia.
- Single-seed historical runs; the ConvNeXt-vs-DenseNet gap is small and may not
  be statistically meaningful.
- No external validation results, calibration, or fairness analysis across sex,
  age, or view position are bundled yet. The repository includes external
  evaluation scripts for SIIM pneumothorax, RSNA pneumonia, and pediatric
  pneumonia, but those numbers must be generated with the datasets present.
- Heatmaps can reflect dataset artifacts or anatomy that is correlated with a
  label rather than the lesion itself.
- The public API has no authentication, rate limiting, or clinical privacy
  controls. Do not deploy it with patient data.

## Ethical and legal notes

- NIH ChestX-ray14 has its own usage terms; cite the dataset paper.
- The backend processes uploads in memory and does not intentionally persist
  them. Add explicit retention and privacy controls before any real deployment.

## References

- Wang, X. et al. "ChestX-ray8: Hospital-scale Chest X-ray Database and
  Benchmarks on Weakly-Supervised Classification and Localization of Common
  Thorax Diseases." CVPR 2017.
- Rajpurkar, P. et al. "CheXNet: Radiologist-Level Pneumonia Detection on Chest
  X-Rays with Deep Learning." arXiv:1711.05225, 2017.
- Draelos, R. L. and Carin, L. "Use HiResCAM instead of Grad-CAM for faithful
  explanations of convolutional neural networks." arXiv:2011.08891, 2020.
