# Checkpoints

## `checkpoints/densenet121`

A DenseNet-121 model from run
`2026_07_31_14_31_densenet121_lr0.0001_b128`, included so a fresh clone can
start the API without downloading or retraining a model.

- Architecture: DenseNet-121
- Input resolution: 320x320
- Parameters: 6,968,206
- Training split: historical patient-level 80/20 split (`seed=42`)
- Peak validation AUROC: 0.8488
- Peak validation AUPRC: 0.2812
- Decision thresholds: none bundled, so the API falls back to 0.5 per class

The checkpoint is a raw PyTorch state dict. Load it with:

```bash
python -m backend.main --run checkpoints/densenet121
```

This model is a research prototype trained on NIH ChestX-ray14. It is not a
medical device and must not be used for clinical decisions.
