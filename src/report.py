"""Generate plots and a summary for a completed training run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .labels import LABELS
from .runs import resolve_run
from .utils import log

trapezoid = getattr(np, "trapezoid", None) or np.trapz


def load_json(path: Path):
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def plot_history(history: dict, output: Path) -> None:
    if not history:
        return

    figure, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=250)
    epochs = range(1, len(history.get("train_loss", [])) + 1)

    if history.get("train_loss"):
        axes[0, 0].plot(epochs, history["train_loss"], label="Train loss", linewidth=2)
        axes[0, 0].plot(epochs, history.get("val_loss", []), label="Validation loss", linewidth=2)
        axes[0, 0].set_title("Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.25)

    if history.get("val_auroc"):
        axes[0, 1].plot(epochs, history["val_auroc"], label="AUROC", linewidth=2)
        axes[0, 1].plot(epochs, history.get("val_auprc", []), label="AUPRC", linewidth=2)
        axes[0, 1].set_title("Validation metrics")
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.25)

    if history.get("lr"):
        axes[1, 0].plot(epochs, history["lr"], color="#167a4b", linewidth=2)
        axes[1, 0].set_title("Learning rate")
        axes[1, 0].set_yscale("log")
        axes[1, 0].grid(alpha=0.25)

    if history.get("epoch_seconds"):
        axes[1, 1].plot(epochs, history["epoch_seconds"], color="#8a5a2b", linewidth=2)
        axes[1, 1].set_title("Epoch wall time")
        axes[1, 1].set_ylabel("seconds")
        axes[1, 1].grid(alpha=0.25)

    for axis in axes.flat:
        axis.set_xlabel("Epoch")
    figure.suptitle("Training history", fontsize=15, fontweight="bold")
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def plot_per_class(metrics: dict, output: Path) -> None:
    per_class = metrics.get("per_class") if metrics else None
    if not per_class:
        return

    names = [label for label in LABELS if label in per_class]
    aurocs = [per_class[name].get("auroc") for name in names]
    auprcs = [per_class[name].get("auprc") for name in names]
    positions = np.arange(len(names))

    figure, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=250)
    axes[0].barh(positions, [value or 0.0 for value in aurocs], color="#167a4b")
    axes[0].set_yticks(positions)
    axes[0].set_yticklabels(names)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 1)
    axes[0].set_title("Per-class AUROC")
    axes[0].grid(axis="x", alpha=0.25)

    axes[1].barh(positions, [value or 0.0 for value in auprcs], color="#8a5a2b")
    axes[1].set_yticks(positions)
    axes[1].set_yticklabels(names)
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 1)
    axes[1].set_title("Per-class AUPRC")
    axes[1].grid(axis="x", alpha=0.25)

    figure.suptitle(f"Per-class metrics ({metrics.get('split', 'test')} split)", fontsize=15, fontweight="bold")
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def roc_points(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    order = np.argsort(scores)[::-1]
    ordered = labels[order]
    true_positive = np.cumsum(ordered == 1)
    false_positive = np.cumsum(ordered == 0)
    positive_total = max(int(true_positive[-1]), 1)
    negative_total = max(int(false_positive[-1]), 1)
    tpr = np.r_[0.0, true_positive / positive_total]
    fpr = np.r_[0.0, false_positive / negative_total]
    return fpr, tpr, float(trapezoid(tpr, fpr))


def pr_points(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    order = np.argsort(scores)[::-1]
    ordered = labels[order]
    true_positive = np.cumsum(ordered == 1)
    false_positive = np.cumsum(ordered == 0)
    positive_total = max(int(true_positive[-1]), 1)
    precision = true_positive / np.maximum(true_positive + false_positive, 1)
    recall = true_positive / positive_total
    ap = float(trapezoid(precision, recall))
    return recall, precision, ap


def plot_curves(predictions_path: Path, output_dir: Path) -> None:
    if not predictions_path.is_file():
        return

    with np.load(predictions_path) as payload:
        probabilities = payload["probabilities"]
        targets = payload["targets"]
        labels = payload["labels"]

    roc_figure, roc_axes = plt.subplots(4, 4, figsize=(16, 14), dpi=250)
    pr_figure, pr_axes = plt.subplots(4, 4, figsize=(16, 14), dpi=250)
    roc_flat = roc_axes.flatten()
    pr_flat = pr_axes.flatten()

    for index, label in enumerate(labels):
        scores = probabilities[:, index]
        actual = targets[:, index].astype(np.int64)
        positives = int(actual.sum())
        negatives = len(actual) - positives
        roc_axis = roc_flat[index]
        pr_axis = pr_flat[index]

        if positives == 0 or negatives == 0:
            roc_axis.set_title(f"{label}\nundefined")
            roc_axis.axis("off")
            pr_axis.set_title(f"{label}\nundefined")
            pr_axis.axis("off")
            continue

        fpr, tpr, auc_value = roc_points(scores, actual)
        recall, precision, ap_value = pr_points(scores, actual)

        roc_axis.plot(fpr, tpr, color="#167a4b", linewidth=2)
        roc_axis.plot([0, 1], [0, 1], color="#aab5ad", linestyle="--", linewidth=1)
        roc_axis.set_title(f"{label}\nAUC {auc_value:.3f}", fontsize=9)
        roc_axis.set_xlabel("FPR", fontsize=8)
        roc_axis.set_ylabel("TPR", fontsize=8)

        pr_axis.plot(recall, precision, color="#8a5a2b", linewidth=2)
        pr_axis.set_title(f"{label}\nAP {ap_value:.3f}", fontsize=9)
        pr_axis.set_xlabel("Recall", fontsize=8)
        pr_axis.set_ylabel("Precision", fontsize=8)

    for index in range(len(labels), len(roc_flat)):
        roc_figure.delaxes(roc_flat[index])
        pr_figure.delaxes(pr_flat[index])

    roc_figure.suptitle("One-vs-rest ROC curves", fontsize=15, fontweight="bold")
    roc_figure.tight_layout()
    roc_figure.savefig(output_dir / "roc_curves.png", bbox_inches="tight")
    plt.close(roc_figure)

    pr_figure.suptitle("One-vs-rest precision-recall curves", fontsize=15, fontweight="bold")
    pr_figure.tight_layout()
    pr_figure.savefig(output_dir / "pr_curves.png", bbox_inches="tight")
    plt.close(pr_figure)


def plot_comparison(artifacts: Path, current_run: Path, output: Path) -> None:
    histories = []
    for history_path in sorted(artifacts.glob("*/history.json")):
        history = load_json(history_path)
        if history and history.get("val_auroc"):
            histories.append((history_path.parent.name, history))

    if not histories:
        return

    figure, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=250)
    for name, history in histories:
        epochs = range(1, len(history["val_auroc"]) + 1)
        alpha = 1.0 if name == current_run.name else 0.45
        linewidth = 2.4 if name == current_run.name else 1.4
        axes[0].plot(
            epochs,
            history["val_auroc"],
            label=name[:32],
            alpha=alpha,
            linewidth=linewidth,
        )
        if history.get("val_auprc"):
            axes[1].plot(
                epochs,
                history["val_auprc"],
                label=name[:32],
                alpha=alpha,
                linewidth=linewidth,
            )

    axes[0].set_title("Validation AUROC across runs")
    axes[1].set_title("Validation AUPRC across runs")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=6, loc="lower right")
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def write_summary(run_dir: Path, output_dir: Path, history: dict) -> None:
    run_info = load_json(run_dir / "run_info.json") or {}
    hyperparameters = load_json(run_dir / "hyperparameters.json") or {}
    best_meta = load_json(run_dir / "best_meta.json") or {}
    test_metrics = load_json(run_dir / "test_metrics.json")
    per_class = load_json(run_dir / "per_class_metrics_test.json")
    if per_class is None:
        per_class = load_json(run_dir / "per_class_metrics_val.json")
    thresholds = load_json(run_dir / "optimal_thresholds.json") or {}
    external = {}
    for path in sorted(run_dir.glob("external_*.json")):
        external[path.stem] = load_json(path)

    summary = {
        "run": run_dir.name,
        "run_info": run_info,
        "hyperparameters": hyperparameters,
        "best_meta": best_meta,
        "test_metrics": test_metrics,
        "per_class_metrics": per_class,
        "thresholds": thresholds,
        "external": external,
        "history_epochs": len(history.get("val_auroc", [])),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=4) + "\n")

    lines = [
        f"# Run summary: {run_dir.name}",
        "",
        f"- Architecture: `{hyperparameters.get('architecture', 'unknown')}`",
        f"- Resolution: `{hyperparameters.get('width', '?')}x{hyperparameters.get('height', '?')}`",
        f"- Git commit: `{run_info.get('git_commit', 'unknown')}`",
        f"- Started: {run_info.get('started_at', 'unknown')}",
        f"- Finished: {run_info.get('finished_at', 'unknown')}",
        f"- Duration: {run_info.get('duration_seconds', 'unknown')} seconds",
        f"- Epochs completed: {run_info.get('epochs_completed', len(history.get('val_auroc', [])))}",
        f"- Best epoch: {run_info.get('best_epoch', 'unknown')}",
        f"- Best validation AUROC: {run_info.get('best_auroc', 'unknown')}",
        f"- Early stopped: {run_info.get('early_stopped', 'unknown')}",
    ]
    if test_metrics:
        lines.extend(
            [
                f"- Test AUROC: {test_metrics.get('auroc'):.4f}",
                f"- Test AUPRC: {test_metrics.get('auprc'):.4f}",
                f"- Test loss: {test_metrics.get('loss'):.4f}",
            ]
        )

    if per_class and per_class.get("per_class"):
        lines.extend(["", "## Per-class metrics", "", "| Label | AUROC | AUPRC | Threshold |", "|---|---:|---:|---:|"])
        for label in LABELS:
            row = per_class["per_class"].get(label, {})
            auroc = row.get("auroc")
            auprc = row.get("auprc")
            threshold = thresholds.get(label, row.get("threshold"))
            lines.append(
                f"| {label} | {auroc if auroc is not None else 'n/a'} | "
                f"{auprc if auprc is not None else 'n/a'} | "
                f"{threshold if threshold is not None else 'n/a'} |"
            )

    if external:
        lines.extend(["", "## External evaluation", "", "| Dataset | Class | AUROC | AUPRC |", "|---|---|---:|---:|"])
        for name, metrics in external.items():
            if not metrics:
                continue
            lines.append(
                f"| {metrics.get('dataset', name)} | {metrics.get('target_class', '?')} | "
                f"{metrics.get('auroc', 'n/a')} | {metrics.get('auprc', 'n/a')} |"
            )

    (output_dir / "summary.md").write_text("\n".join(lines) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=None)
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--output", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        run_dir = resolve_run(args.run, args.artifacts)
    except FileNotFoundError as error:
        log("error", str(error))
        return 1

    output_dir = Path(args.output) if args.output else run_dir / "report"
    output_dir.mkdir(parents=True, exist_ok=True)
    history = load_json(run_dir / "history.json") or {}

    plot_history(history, output_dir / "training_history.png")
    per_class = load_json(run_dir / "per_class_metrics_test.json")
    if per_class is None:
        per_class = load_json(run_dir / "per_class_metrics_val.json")
    plot_per_class(per_class or {}, output_dir / "per_class_metrics.png")

    predictions_test = run_dir / "predictions_test.npz"
    predictions_val = run_dir / "predictions_val.npz"
    plot_curves(
        predictions_test if predictions_test.is_file() else predictions_val,
        output_dir,
    )

    plot_comparison(Path(args.artifacts), run_dir, output_dir / "comparison_curves.png")
    write_summary(run_dir, output_dir, history)

    log("save", f"Report written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
