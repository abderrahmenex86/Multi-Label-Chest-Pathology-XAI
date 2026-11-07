import argparse
import json
import os
import sys

import matplotlib.pyplot as plt

from .utils import log

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--output", default="docs/figs")
    args = parser.parse_args()

    if not os.path.exists(args.artifacts):
        log("error", f"Artifacts directory {args.artifacts} missing.")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    directories = [
        os.path.join(args.artifacts, item)
        for item in os.listdir(args.artifacts)
        if os.path.isdir(os.path.join(args.artifacts, item))
    ]

    if not directories:
        log("error", "No run directories found in artifacts.")
        sys.exit(1)

    directories.sort()

    runs_data = []

    for run_path in directories:
        history_file = os.path.join(run_path, "history.json")
        config_file = os.path.join(run_path, "hyperparameters.json")

        if not os.path.exists(history_file) or not os.path.exists(config_file):
            continue

        with open(history_file, "r") as handle:
            history = json.load(handle)

        with open(config_file, "r") as handle:
            config = json.load(handle)

        folder = os.path.basename(run_path)
        peak_auroc = max(history["val_auroc"]) if history["val_auroc"] else 0.0
        peak_auprc = max(history["val_auprc"]) if history["val_auprc"] else 0.0

        runs_data.append(
            {
                "folder": folder,
                "architecture": config.get("architecture", "densenet121"),
                "history": history,
                "peak_auroc": peak_auroc,
                "peak_auprc": peak_auprc,
            }
        )

    if not runs_data:
        log("error", "No valid run histories found to compare.")
        sys.exit(1)

    figure, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=300)

    for run in runs_data:
        epochs = range(1, len(run["history"]["val_auroc"]) + 1)
        label = run["folder"][:22]

        axes[0].plot(epochs, run["history"]["val_auroc"], label=label, linewidth=1.5)
        axes[1].plot(epochs, run["history"]["val_auprc"], label=label, linewidth=1.5)
        axes[2].plot(epochs, run["history"]["val_loss"], label=label, linewidth=1.5)

    axes[0].set_title("Validation AUROC Comparison", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epochs", fontsize=10)
    axes[0].set_ylabel("AUROC", fontsize=10)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].set_title("Validation AUPRC Comparison", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epochs", fontsize=10)
    axes[1].set_ylabel("AUPRC", fontsize=10)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    axes[2].set_title("Validation Loss Comparison", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Epochs", fontsize=10)
    axes[2].set_ylabel("Loss", fontsize=10)
    axes[2].grid(True, linestyle="--", alpha=0.5)

    axes[0].legend(fontsize=6, loc="lower right")

    plt.tight_layout()
    curves_output = os.path.join(args.output, "model_comparison_curves.png")
    plt.savefig(curves_output)
    plt.close()

    runs_data.sort(key=lambda item: item["peak_auroc"], reverse=True)

    figure, axes = plt.subplots(figsize=(12, 7), dpi=300)
    names = [run["folder"] for run in runs_data]
    scores = [run["peak_auroc"] for run in runs_data]
    colors = ["#0052cc" if "convnext" in run["architecture"] else "#0084ff" for run in runs_data]

    bars = axes.barh(names, scores, color=colors)
    axes.set_title("Ranked Experiment Comparison (Peak Validation AUROC)", fontsize=13, fontweight="bold")
    axes.set_xlabel("Peak AUROC Score", fontsize=11)
    axes.set_xlim(0.60, 0.90)
    axes.invert_yaxis()

    for bar in bars:
        axes.annotate(
            f"{bar.get_width():.4f}",
            (bar.get_width(), bar.get_y() + bar.get_height() / 2.0),
            ha="left",
            va="center",
            xytext=(5, 0),
            textcoords="offset points",
            fontsize=8,
            fontweight="bold",
        )

    plt.tight_layout()
    bar_output = os.path.join(args.output, "model_comparison_bar.png")
    plt.savefig(bar_output)
    plt.close()

    log("save", f"Model comparison curves saved to {curves_output}")
    log("save", f"Ranked model comparison bar chart saved to {bar_output}")
