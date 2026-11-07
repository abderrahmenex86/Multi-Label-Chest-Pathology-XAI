"""Evaluate a run on the validation or held-out test split with per-class metrics."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchmetrics.classification import MultilabelAUROC, MultilabelAveragePrecision

from .dataset import ChestDataset
from .labels import LABELS, NUM_LABELS, add_label_columns
from .model import ChestMultiLabelModel
from .runs import read_hyperparameters, resolve_run
from .splits import patient_split
from .utils import autocast_context, log, seed_everything


def clean_number(value: float) -> float | None:
    return None if math.isnan(value) or math.isinf(value) else float(value)


def load_thresholds(run_dir: Path) -> dict[str, float]:
    path = run_dir / "optimal_thresholds.json"
    if path.is_file():
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            return {label: float(payload.get(label, 0.5)) for label in LABELS}
    return {label: 0.5 for label in LABELS}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=None)
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--metadata", default="data/Data_Entry_2017.csv")
    parser.add_argument("--directory", default="data/images")
    parser.add_argument("--split", choices=("test", "val"), default="test")
    parser.add_argument("--batch", type=int, default=64)
    return parser


def main() -> int:
    args = build_parser().parse_args()

    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        run_dir = resolve_run(args.run, args.artifacts)
    except FileNotFoundError as error:
        log("error", str(error))
        return 1

    hyperparameters = read_hyperparameters(run_dir)
    dataframe = pd.read_csv(args.metadata)
    dataframe = add_label_columns(dataframe)

    split = patient_split(
        dataframe,
        seed=int(hyperparameters.get("seed", 42)),
        val_fraction=float(hyperparameters.get("val_fraction", 0.2)),
        test_fraction=float(hyperparameters.get("test_fraction", 0.0)),
    )
    metadata = split.test if args.split == "test" else split.validation
    if len(metadata) == 0:
        log("error", f"The {args.split} split is empty. Use --split val or retrain with a test fraction.")
        return 1

    dataset = ChestDataset(
        metadata,
        args.directory,
        int(hyperparameters["width"]),
        int(hyperparameters["height"]),
        augment=False,
        cache=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=4,
        pin_memory=device.type == "cuda",
    )

    model = ChestMultiLabelModel(
        NUM_LABELS,
        hyperparameters["architecture"],
        pretrained=False,
    ).to(device)
    model.load_state_dict(
        torch.load(run_dir / "best.pth", map_location=device, weights_only=True),
        strict=True,
    )
    model.eval()

    auroc = MultilabelAUROC(num_labels=NUM_LABELS, average=None).to(device)
    auprc = MultilabelAveragePrecision(num_labels=NUM_LABELS, average=None).to(device)

    outputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    log("evaluate", f"Running {args.split} inference for {run_dir.name}...")
    with torch.no_grad():
        for inputs, batch_targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            batch_targets = batch_targets.to(device, non_blocking=True)
            with autocast_context(device):
                logits = model(inputs)
            auroc.update(logits, batch_targets.long())
            auprc.update(logits, batch_targets.long())
            outputs.append(torch.sigmoid(logits).float().cpu().numpy())
            targets.append(batch_targets.float().cpu().numpy())

    probabilities = np.concatenate(outputs, axis=0)
    actual = np.concatenate(targets, axis=0)
    predictions_path = run_dir / f"predictions_{args.split}.npz"
    np.savez_compressed(
        predictions_path,
        probabilities=probabilities.astype(np.float32),
        targets=actual.astype(np.float32),
        labels=np.asarray(LABELS),
    )

    auroc_values = auroc.compute().detach().cpu().tolist()
    auprc_values = auprc.compute().detach().cpu().tolist()
    thresholds = load_thresholds(run_dir)

    per_class: dict[str, dict[str, float | int | None]] = {}
    for index, label in enumerate(LABELS):
        predicted = (probabilities[:, index] >= thresholds[label]).astype(float)
        class_actual = actual[:, index]

        tp = int(((predicted == 1) & (class_actual == 1)).sum())
        fp = int(((predicted == 1) & (class_actual == 0)).sum())
        tn = int(((predicted == 0) & (class_actual == 0)).sum())
        fn = int(((predicted == 0) & (class_actual == 1)).sum())

        sensitivity = tp / (tp + fn) if (tp + fn) else None
        specificity = tn / (tn + fp) if (tn + fp) else None
        precision = tp / (tp + fp) if (tp + fp) else None
        positive_count = int(class_actual.sum())
        negative_count = len(class_actual) - positive_count

        per_class[label] = {
            "auroc": clean_number(auroc_values[index]) if positive_count and negative_count else None,
            "auprc": clean_number(auprc_values[index]) if positive_count else None,
            "threshold": thresholds[label],
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "precision": precision,
        }

    valid_aurocs = [row["auroc"] for row in per_class.values() if row["auroc"] is not None]
    valid_auprcs = [row["auprc"] for row in per_class.values() if row["auprc"] is not None]
    payload = {
        "run": run_dir.name,
        "split": args.split,
        "images": int(len(metadata)),
        "macro": {
            "auroc": clean_number(float(np.mean(valid_aurocs))) if valid_aurocs else None,
            "auprc": clean_number(float(np.mean(valid_auprcs))) if valid_auprcs else None,
        },
        "per_class": per_class,
    }

    output_path = run_dir / f"per_class_metrics_{args.split}.json"
    output_path.write_text(json.dumps(payload, indent=4) + "\n")

    log("save", f"Per-class metrics saved to {output_path}")
    log("save", f"Raw predictions saved to {predictions_path}")
    for label in LABELS:
        row = per_class[label]
        auroc_text = "n/a" if row["auroc"] is None else f"{row['auroc']:.4f}"
        auprc_text = "n/a" if row["auprc"] is None else f"{row['auprc']:.4f}"
        log("metric", f"{label:<20} AUROC={auroc_text} AUPRC={auprc_text}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
