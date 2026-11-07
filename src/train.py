"""Train a multi-label chest pathology classifier."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
import torchinfo
from torch.utils.data import DataLoader

from .dataset import ChestDataset
from .download import execute_download
from .engine import evaluate, train_epoch
from .labels import LABELS, NUM_LABELS, add_label_columns, get_image_column
from .model import SUPPORTED_ARCHITECTURES, ChestMultiLabelModel
from .splits import patient_split
from .utils import get_git_commit, log, plot_augmentation_steps, seed_everything


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", default="data/Data_Entry_2017.csv")
    parser.add_argument("--directory", default="data/images")
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--architecture", default="densenet121", choices=SUPPORTED_ARCHITECTURES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument(
        "--accumulate",
        type=int,
        default=1,
        help="Micro-batches per optimizer step; effective batch is batch * accumulate.",
    )
    parser.add_argument("--rate", type=float, default=1e-4)
    parser.add_argument("--decay", type=float, default=1e-4)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--classes", type=int, default=NUM_LABELS)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--prefetch", type=int, default=2)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--disable-pin", action="store_true")
    parser.add_argument("--disable-persistent", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true", help="Skip ImageNet weights (offline/smoke tests).")
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--cache", action="store_true")
    parser.add_argument(
        "--cache-workers",
        type=int,
        default=8,
        help="Threads used to fill the RAM cache.",
    )
    parser.add_argument("--sanity", action="store_true")
    parser.add_argument("--sanity-steps", type=int, default=20)
    parser.add_argument("--resume", default=None, help="Path to a last.pth checkpoint.")
    parser.add_argument("--download", action="store_true")
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Download and flatten the dataset, then exit.",
    )
    return parser


def build_loader(dataset, args, shuffle: bool, drop_last: bool = False) -> DataLoader:
    workers = args.workers
    prefetch = args.prefetch if workers > 0 else None
    persistent = (not args.disable_persistent) if workers > 0 else False
    pin = (not args.disable_pin) and torch.cuda.is_available()

    return DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=shuffle,
        num_workers=workers,
        persistent_workers=persistent,
        prefetch_factor=prefetch,
        pin_memory=pin,
        drop_last=drop_last,
    )


def main() -> int:
    args = build_parser().parse_args()

    if args.classes != NUM_LABELS:
        raise SystemExit(f"--classes must be {NUM_LABELS}; the label list is fixed.")
    if args.epochs <= 0:
        raise SystemExit("--epochs must be positive.")
    if args.warmup < 0 or args.warmup >= args.epochs:
        raise SystemExit("--warmup must be >= 0 and smaller than --epochs.")
    if args.sanity_steps <= 0:
        raise SystemExit("--sanity-steps must be positive.")
    if args.accumulate < 1:
        raise SystemExit("--accumulate must be at least 1.")
    if args.cache_workers < 1:
        raise SystemExit("--cache-workers must be at least 1.")

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.download or args.download_only:
        execute_download(args.metadata, args.directory)

    if args.download_only:
        return 0

    if not os.path.exists(args.metadata):
        log("error", f"Metadata missing at {args.metadata}. Use --download flag.")
        return 1

    log("system", f"Executing on {device} using {args.architecture} at {args.width}x{args.height}")
    log(
        "system",
        f"Effective batch: {args.batch * args.accumulate} "
        f"({args.batch} x {args.accumulate} accumulation steps)",
    )
    if device.type == "cuda":
        log("system", f"bf16 autocast supported: {torch.cuda.is_bf16_supported()}")

    dataframe = pd.read_csv(args.metadata)
    dataframe = add_label_columns(dataframe)

    split = patient_split(
        dataframe,
        seed=args.seed,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
    )
    log(
        "split",
        f"Images train/val/test: {split.image_counts} | patients: {split.patient_counts}",
    )
    if len(split.train) == 0 or len(split.validation) == 0:
        log("error", "The train or validation split is empty; adjust the fractions or seed.")
        return 1

    train_dataset = ChestDataset(
        split.train,
        args.directory,
        args.width,
        args.height,
        augment=True,
        cache=args.cache,
        cache_workers=args.cache_workers,
    )
    val_dataset = ChestDataset(
        split.validation, args.directory, args.width, args.height, augment=False, cache=False
    )
    test_dataset = (
        ChestDataset(split.test, args.directory, args.width, args.height, augment=False, cache=False)
        if len(split.test)
        else None
    )

    if args.cache:
        estimated_gb = len(train_dataset) * 3 * args.width * args.height / 1e9
        log("warning", f"RAM cache for the training split is approximately {estimated_gb:.1f} GB.")

    train_loader = build_loader(train_dataset, args, shuffle=True, drop_last=len(train_dataset) >= args.batch)
    val_loader = build_loader(val_dataset, args, shuffle=False)
    test_loader = build_loader(test_dataset, args, shuffle=False) if test_dataset is not None else None

    positives = split.train[list(LABELS)].sum().values
    negatives = len(split.train) - positives
    raw_weights = torch.tensor(negatives / (positives + 1e-5), dtype=torch.float32)
    weights = torch.clamp(raw_weights, max=10.0).to(device)

    model = ChestMultiLabelModel(
        args.classes,
        args.architecture,
        freeze=args.freeze,
        pretrained=not args.no_pretrained,
    ).to(device)

    resume_checkpoint = None
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_file():
            log("error", f"Resume checkpoint not found at {resume_path}.")
            return 1
        resume_checkpoint = torch.load(resume_path, map_location=device, weights_only=True)
        model.load_state_dict(resume_checkpoint["model"], strict=True)
        log("system", f"Resumed model weights from {resume_path}")

    if device.type == "cuda" and hasattr(torch, "compile") and not args.no_compile:
        try:
            model = torch.compile(model)
        except Exception as error:  # pragma: no cover - backend-specific
            log("warning", f"torch.compile failed, continuing eagerly: {error}")

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = None
    if device.type == "cuda":
        try:
            optimizer = torch.optim.AdamW(
                trainable_parameters,
                lr=args.rate,
                weight_decay=args.decay,
                fused=True,
            )
        except (TypeError, RuntimeError) as error:
            log("warning", f"Fused AdamW unavailable, using the standard implementation: {error}")
    if optimizer is None:
        optimizer = torch.optim.AdamW(trainable_parameters, lr=args.rate, weight_decay=args.decay)

    if args.warmup > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            total_iters=args.warmup,
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, args.epochs - args.warmup),
            eta_min=1e-6,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[args.warmup],
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, args.epochs),
            eta_min=1e-6,
        )

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=weights).to(device)

    history = {"train_loss": [], "val_loss": [], "val_auroc": [], "val_auprc": []}
    best = 0.0
    stagnant = 0
    start_epoch = 1

    if resume_checkpoint is not None:
        if "optimizer" in resume_checkpoint:
            optimizer.load_state_dict(resume_checkpoint["optimizer"])
        if "scheduler" in resume_checkpoint:
            scheduler.load_state_dict(resume_checkpoint["scheduler"])
        best = float(resume_checkpoint.get("best", 0.0))
        stagnant = int(resume_checkpoint.get("stagnant", 0))
        start_epoch = int(resume_checkpoint.get("epoch", 0)) + 1
        saved_history = resume_checkpoint.get("history")
        if isinstance(saved_history, dict):
            history.update({key: list(value) for key, value in saved_history.items() if key in history})

    if args.sanity:
        log("sanity", f"Running {args.sanity_steps} steps on a single batch.")
        batch = next(iter(train_loader))
        loader = [batch]
        losses: list[float] = []

        for step in range(1, args.sanity_steps + 1):
            loss = train_epoch(
                model,
                loader,
                optimizer,
                criterion,
                device,
                args.clip,
                args.accumulate,
            )
            losses.append(loss)
            if step == 1 or step % 5 == 0:
                log("sanity", f"Step {step} | Loss: {loss:.4f}")

        if losses[-1] >= losses[0]:
            log("error", f"Sanity check failed: loss did not decrease ({losses[0]:.4f} -> {losses[-1]:.4f}).")
            return 1

        log("sanity", f"Loss decreased from {losses[0]:.4f} to {losses[-1]:.4f}.")
        return 0

    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    run_name = (
        f"{timestamp}_{args.architecture}_lr{args.rate}_b{args.batch}"
        f"_a{args.accumulate}_r{args.width}x{args.height}_s{args.seed}"
    )
    run_dir = Path(args.artifacts) / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    metadata = {
        **vars(args),
        "git_commit": get_git_commit(),
        "split_counts": split.image_counts,
        "split_patient_counts": split.patient_counts,
    }
    (run_dir / "hyperparameters.json").write_text(json.dumps(metadata, indent=4) + "\n")
    (run_dir / "split.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "val_fraction": args.val_fraction,
                "test_fraction": args.test_fraction,
                "train_patients": split.train_patients,
                "validation_patients": split.validation_patients,
                "test_patients": split.test_patients,
            },
            indent=4,
        )
        + "\n"
    )

    log("system", f"Artifacts will be saved to {run_dir}")

    sample_image = os.path.join(
        args.directory,
        split.train.iloc[0][get_image_column(split.train.columns)],
    )
    if os.path.exists(sample_image):
        figure_path = run_dir / "augmentation.png"
        try:
            plot_augmentation_steps(sample_image, figure_path, args.width, args.height, rotation=7.0)
            log("system", f"Augmentation visualization saved to {figure_path}")
        except Exception as error:
            log("warning", f"Could not create augmentation visualization: {error}")

    raw_model = getattr(model, "_orig_mod", model)
    dummy = torch.randn(1, 3, args.height, args.width).to(device)
    stats = torchinfo.summary(raw_model, input_data=dummy, verbose=0)
    (run_dir / "architecture.txt").write_text(str(stats) + "\n")

    for epoch in range(start_epoch, args.epochs + 1):
        log("train", f"Starting Epoch {epoch}")

        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            args.clip,
            args.accumulate,
        )
        scheduler.step()

        val_metrics = evaluate(model, val_loader, criterion, device, args.classes)

        log(
            "eval",
            f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['loss']:.4f} | "
            f"AUROC: {val_metrics['auroc']:.4f} | AUPRC: {val_metrics['auprc']:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e}",
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_auroc"].append(val_metrics["auroc"])
        history["val_auprc"].append(val_metrics["auprc"])

        (run_dir / "history.json").write_text(json.dumps(history, indent=4) + "\n")

        improved = val_metrics["auroc"] > best
        if improved:
            best = val_metrics["auroc"]
            stagnant = 0
        else:
            stagnant += 1

        raw_state = getattr(model, "_orig_mod", model).state_dict()
        checkpoint = {
            "epoch": epoch,
            "model": raw_state,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best": best,
            "stagnant": stagnant,
            "history": history,
        }
        torch.save(checkpoint, run_dir / "last.pth")

        if improved:
            torch.save(raw_state, run_dir / "best.pth")
            (run_dir / "best_meta.json").write_text(
                json.dumps(
                    {
                        "epoch": epoch,
                        "val_auroc": val_metrics["auroc"],
                        "val_auprc": val_metrics["auprc"],
                        "architecture": args.architecture,
                        "width": args.width,
                        "height": args.height,
                        "git_commit": get_git_commit(),
                    },
                    indent=4,
                )
                + "\n"
            )
            log("save", f"New best AUROC {best:.4f} saved to {run_dir}")

        if stagnant >= args.patience:
            log("stop", f"Early stopping triggered at epoch {epoch}. Best AUROC: {best:.4f}")
            break

    if test_loader is not None:
        best_path = run_dir / "best.pth"
        if best_path.is_file():
            raw_model = getattr(model, "_orig_mod", model)
            raw_model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True), strict=True)
            test_metrics = evaluate(raw_model, test_loader, criterion, device, args.classes)
            (run_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=4) + "\n")
            log(
                "test",
                f"Test AUROC: {test_metrics['auroc']:.4f} | "
                f"Test AUPRC: {test_metrics['auprc']:.4f} | "
                f"Test Loss: {test_metrics['loss']:.4f}",
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
