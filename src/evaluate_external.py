"""Evaluate an NIH-trained run on external chest X-ray datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision

from .external import ExternalDataset, class_count, load_external_manifest
from .labels import LABELS
from .predictor import ChestPredictor
from .runs import resolve_run
from .utils import log


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=None)
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--dataset", choices=("siim", "rsna", "pediatric"), required=True)
    parser.add_argument("--data-dir", required=True, help="Dataset root directory.")
    parser.add_argument(
        "--split",
        default="test",
        help="Pediatric split (train/test). Ignored by SIIM and RSNA.",
    )
    parser.add_argument("--target-class", choices=LABELS, default=None)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=0, help="Evaluate only the first N images.")
    parser.add_argument("--crop-borders", action="store_true")
    parser.add_argument("--output", default=None)
    return parser


def safe_metric(metric) -> float | None:
    try:
        value = float(metric.compute().item())
    except Exception:
        return None
    if np.isnan(value) or np.isinf(value):
        return None
    return value


def main() -> int:
    args = build_parser().parse_args()
    if args.batch <= 0:
        raise SystemExit("--batch must be positive.")
    if args.workers < 0:
        raise SystemExit("--workers must be non-negative.")
    if args.limit < 0:
        raise SystemExit("--limit must be non-negative.")
    if not 0.0 <= args.threshold <= 1.0:
        raise SystemExit("--threshold must be between 0 and 1.")

    try:
        run_dir = resolve_run(args.run, args.artifacts)
    except FileNotFoundError as error:
        log("error", str(error))
        return 1

    manifest = load_external_manifest(args.dataset, args.data_dir, split=args.split)
    items = manifest.items[: args.limit] if args.limit > 0 else manifest.items
    if not items:
        log("error", f"No labeled images found for {manifest.name}.")
        return 1

    target_class = args.target_class or manifest.target_class
    if target_class not in LABELS:
        log("error", f"Unknown target class: {target_class}")
        return 1
    target = LABELS.index(target_class)

    predictor = ChestPredictor(run_dir)
    dataset = ExternalDataset(
        items,
        predictor.transform,
        predictor.height,
        predictor.width,
        crop_borders=args.crop_borders,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=predictor.device.type == "cuda",
        persistent_workers=args.workers > 0,
    )

    scores: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    invalid = 0

    log(
        "external",
        f"Evaluating {manifest.name} ({manifest.source}) for {target_class} "
        f"using {len(items)} images...",
    )

    for images, batch_labels, valid in loader:
        valid_mask = valid.bool()
        invalid += int((~valid_mask).sum().item())
        if not valid_mask.any():
            continue

        images = images[valid_mask]
        batch_labels = batch_labels[valid_mask]
        probabilities = predictor.predict_tensor(images)
        scores.append(probabilities[:, target].numpy())
        labels.append(batch_labels.numpy())

    if not scores:
        log("error", "No valid external images could be loaded.")
        return 1

    score_values = np.concatenate(scores)
    label_values = np.concatenate(labels).astype(np.int64)
    if label_values.min() == label_values.max():
        log("error", "External split contains only one class; AUROC is undefined.")
        return 1

    auroc = BinaryAUROC().update(torch.from_numpy(score_values), torch.from_numpy(label_values))
    auprc = BinaryAveragePrecision().update(
        torch.from_numpy(score_values), torch.from_numpy(label_values)
    )
    predictions = (score_values >= args.threshold).astype(np.int64)
    tp = int(((predictions == 1) & (label_values == 1)).sum())
    fp = int(((predictions == 1) & (label_values == 0)).sum())
    tn = int(((predictions == 0) & (label_values == 0)).sum())
    fn = int(((predictions == 0) & (label_values == 1)).sum())

    metrics = {
        "run": run_dir.name,
        "dataset": manifest.name,
        "source": manifest.source,
        "independent_of_nih": manifest.independent,
        "target_class": target_class,
        "threshold": args.threshold,
        "images_requested": len(items),
        "images_evaluated": int(len(label_values)),
        "images_invalid": invalid,
        "missing_images": manifest.missing_images,
        "class_counts": class_count(items),
        "auroc": safe_metric(auroc),
        "auprc": safe_metric(auprc),
        "accuracy": float((tp + tn) / max(tp + fp + tn + fn, 1)),
        "sensitivity": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
        "precision": float(tp / max(tp + fp, 1)),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }

    output_path = (
        Path(args.output)
        if args.output
        else run_dir / f"external_{manifest.name}_{target_class}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=4) + "\n")

    log(
        "external",
        f"{manifest.name}: AUROC={metrics['auroc']} AUPRC={metrics['auprc']} "
        f"sensitivity={metrics['sensitivity']:.3f} specificity={metrics['specificity']:.3f}",
    )
    log("save", f"External metrics saved to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
