"""Tune per-class decision thresholds with Youden's J on the validation split."""

from __future__ import annotations

import argparse
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .dataset import ChestDataset
from .labels import LABELS, NUM_LABELS, add_label_columns
from .model import ChestMultiLabelModel
from .runs import read_hyperparameters, resolve_run
from .splits import patient_split
from .utils import autocast_context, log, seed_everything


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=None)
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--metadata", default="data/Data_Entry_2017.csv")
    parser.add_argument("--directory", default="data/images")
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
    if int(hyperparameters.get("classes", NUM_LABELS)) != NUM_LABELS:
        log("error", "The run class count does not match the label list.")
        return 1

    dataframe = pd.read_csv(args.metadata)
    dataframe = add_label_columns(dataframe)

    split = patient_split(
        dataframe,
        seed=int(hyperparameters.get("seed", 42)),
        val_fraction=float(hyperparameters.get("val_fraction", 0.2)),
        test_fraction=float(hyperparameters.get("test_fraction", 0.0)),
    )
    if len(split.validation) == 0:
        log("error", "Validation split is empty; cannot tune thresholds.")
        return 1

    dataset = ChestDataset(
        split.validation,
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

    outputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    log("thresholds", f"Collecting validation predictions from {run_dir.name}...")
    with torch.no_grad():
        for inputs, batch_targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            with autocast_context(device):
                logits = model(inputs)
                probabilities = torch.sigmoid(logits)
            outputs.append(probabilities.float().cpu().numpy())
            targets.append(batch_targets.float().cpu().numpy())

    probabilities = np.concatenate(outputs, axis=0)
    actual = np.concatenate(targets, axis=0)
    candidates = np.linspace(0.01, 0.99, 99)
    optimal_thresholds: dict[str, float] = {}

    for index, name in enumerate(LABELS):
        class_actual = actual[:, index]
        class_scores = probabilities[:, index]
        positives = int(class_actual.sum())

        # Youden's J is undefined without both positive and negative examples.
        if positives == 0 or positives == len(class_actual):
            optimal_thresholds[name] = 0.5
            continue

        best_j = -1.0
        best_threshold = 0.5

        for threshold in candidates:
            predicted = (class_scores >= threshold).astype(float)
            tp = ((predicted == 1) & (class_actual == 1)).sum()
            fp = ((predicted == 1) & (class_actual == 0)).sum()
            tn = ((predicted == 0) & (class_actual == 0)).sum()
            fn = ((predicted == 0) & (class_actual == 1)).sum()

            sensitivity = tp / (tp + fn + 1e-8)
            specificity = tn / (tn + fp + 1e-8)
            j_score = sensitivity + specificity - 1.0

            if j_score > best_j:
                best_j = j_score
                best_threshold = float(threshold)

        optimal_thresholds[name] = round(best_threshold, 3)

    thresholds_path = run_dir / "optimal_thresholds.json"
    thresholds_path.write_text(json.dumps(optimal_thresholds, indent=4) + "\n")

    meta_path = run_dir / "thresholds_meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "split": "validation",
                "seed": int(hyperparameters.get("seed", 42)),
                "val_fraction": float(hyperparameters.get("val_fraction", 0.2)),
                "test_fraction": float(hyperparameters.get("test_fraction", 0.0)),
                "validation_images": int(len(split.validation)),
            },
            indent=4,
        )
        + "\n"
    )

    figure, axes = plt.subplots(figsize=(12, 6), dpi=300)
    names = list(optimal_thresholds.keys())
    values = list(optimal_thresholds.values())
    bars = axes.bar(names, values, color="#0052cc")
    axes.set_title("Optimal Per-Class Decision Thresholds (Youden J Statistic)", fontsize=13, fontweight="bold")
    axes.set_ylabel("Optimal Threshold (0.0 - 1.0)", fontsize=11)
    axes.set_ylim(0, 1.0)
    plt.xticks(rotation=45, ha="right")

    for bar in bars:
        axes.annotate(
            f"{bar.get_height():.2f}",
            (bar.get_x() + bar.get_width() / 2.0, bar.get_height()),
            ha="center",
            va="center",
            xytext=(0, 5),
            textcoords="offset points",
            fontsize=8,
        )

    plt.tight_layout()
    plot_path = run_dir / "optimal_thresholds.png"
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close(figure)

    log("save", f"Optimal thresholds saved to {thresholds_path}")
    log("save", f"Threshold plot saved to {plot_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
