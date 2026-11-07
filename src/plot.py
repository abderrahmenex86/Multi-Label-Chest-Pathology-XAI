import argparse
import json
import os
import sys

import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, default=None)
    parser.add_argument("--artifacts", default="artifacts")
    args = parser.parse_args()

    target = args.run
    if target is None:
        directories = [
            os.path.join(args.artifacts, item)
            for item in os.listdir(args.artifacts)
            if os.path.isdir(os.path.join(args.artifacts, item))
        ]
        if not directories:
            print("No run directories found in artifacts.")
            sys.exit(1)
        directories.sort(key=lambda item: os.path.getmtime(item), reverse=True)
        target = directories[0]

    source = os.path.join(target, "history.json")
    if not os.path.exists(source):
        print(f"history.json not found in {target}")
        sys.exit(1)

    with open(source, "r") as handle:
        history = json.load(handle)

    epochs = range(1, len(history["train_loss"]) + 1)

    figure, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    axes[0].plot(epochs, history["train_loss"], label="Train Loss", color="#1f77b4", linewidth=2)
    axes[0].plot(epochs, history["val_loss"], label="Val Loss", color="#ff7f0e", linewidth=2)
    axes[0].set_title("Loss Curves Over Epochs")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    axes[1].plot(epochs, history["val_auroc"], label="Val AUROC", color="#2ca02c", linewidth=2)
    axes[1].plot(epochs, history["val_auprc"], label="Val AUPRC", color="#d62728", linewidth=2)
    axes[1].set_title("Validation Metrics Over Epochs")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Score")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

    plt.tight_layout()
    output = os.path.join(target, "metrics.png")
    plt.savefig(output)
    plt.close()

    print(f"Saved metrics plot to {output}")
