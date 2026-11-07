"""Generate per-class confusion matrices on a validation or test split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn
import torch
from torch.utils.data import DataLoader

from .dataset import ChestDataset
from .labels import LABELS, NUM_LABELS, add_label_columns
from .model import ChestMultiLabelModel
from .runs import read_hyperparameters, resolve_run
from .splits import patient_split
from .utils import autocast_context, log, seed_everything


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
    parser.add_argument("--threshold", type=float, default=None, help="Override per-class thresholds.")
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

    outputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    log("evaluate", f"Running {args.split} inference for confusion matrices...")
    with torch.no_grad():
        for inputs, batch_targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            with autocast_context(device):
                probabilities = torch.sigmoid(model(inputs))
            outputs.append(probabilities.float().cpu().numpy())
            targets.append(batch_targets.float().cpu().numpy())

    probabilities = np.concatenate(outputs, axis=0)
    actual = np.concatenate(targets, axis=0)

    if args.threshold is not None:
        thresholds = {label: float(args.threshold) for label in LABELS}
        threshold_label = f"{args.threshold:.2f}"
    else:
        thresholds = load_thresholds(run_dir)
        threshold_label = "per-class tuned thresholds"

    figure, axes = plt.subplots(4, 4, figsize=(16, 14), dpi=300)
    flat_axes = axes.flatten()

    for index, name in enumerate(LABELS):
        predicted = (probabilities[:, index] >= thresholds[name]).astype(float)
        class_actual = actual[:, index]

        tp = int(((predicted == 1) & (class_actual == 1)).sum())
        fp = int(((predicted == 1) & (class_actual == 0)).sum())
        tn = int(((predicted == 0) & (class_actual == 0)).sum())
        fn = int(((predicted == 0) & (class_actual == 1)).sum())

        matrix = np.array([[tn, fp], [fn, tp]])
        total = len(class_actual)

        annotations = np.array(
            [
                [f"TN: {tn}\n({tn / total * 100:.1f}%)", f"FP: {fp}\n({fp / total * 100:.1f}%)"],
                [f"FN: {fn}\n({fn / total * 100:.1f}%)", f"TP: {tp}\n({tp / total * 100:.1f}%)"],
            ]
        )

        seaborn.heatmap(
            matrix,
            annot=annotations,
            fmt="",
            cmap="Blues",
            cbar=False,
            ax=flat_axes[index],
            xticklabels=["Neg (0)", "Pos (1)"],
            yticklabels=["Neg (0)", "Pos (1)"],
        )

        flat_axes[index].set_title(f"{name}\nthreshold={thresholds[name]:.2f}", fontsize=10, fontweight="bold")
        flat_axes[index].set_xlabel("Predicted", fontsize=8)
        flat_axes[index].set_ylabel("Actual", fontsize=8)

    for index in range(len(LABELS), len(flat_axes)):
        figure.delaxes(flat_axes[index])

    plt.suptitle(
        f"Multi-Label Confusion Matrices ({args.split} split, {threshold_label})",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()

    output_path = run_dir / f"confusion_matrices_{args.split}.png"
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(figure)

    log("save", f"Confusion matrices saved to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
