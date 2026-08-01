import argparse
import json
import os
import sys
from datetime import datetime

import numpy
import pandas
import torch
import torchinfo
from torch.utils.data import DataLoader

from dataset import ChestDataset
from download import execute_download
from engine import evaluate, train_epoch
from model import DenseNetMultiLabel
from utils import log, plot_augmentation_steps, seed_everything

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default="data/Data_Entry_2017.csv")
    parser.add_argument("--directory", default="data/images")
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--architecture", default="densenet121")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--rate", type=float, default=1e-4)
    parser.add_argument("--decay", type=float, default=1e-4)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--classes", type=int, default=14)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--prefetch", type=int, default=2)
    parser.add_argument("--disable-pin", action="store_true")
    parser.add_argument("--disable-persistent", action="store_true")
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--sanity", action="store_true")
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.download:
        execute_download(args.metadata, args.directory)

    if not os.path.exists(args.metadata):
        log("error", f"Metadata missing at {args.metadata}. Use --download flag.")
        sys.exit(1)

    log("system", f"Executing on {device} using {args.architecture} at {args.width}x{args.height}")

    dataframe = pandas.read_csv(args.metadata)

    columns = [
        "Atelectasis",
        "Cardiomegaly",
        "Effusion",
        "Infiltration",
        "Mass",
        "Nodule",
        "Pneumonia",
        "Pneumothorax",
        "Consolidation",
        "Edema",
        "Emphysema",
        "Fibrosis",
        "Pleural_Thickening",
        "Hernia",
    ]

    if "Finding Labels" in dataframe.columns:
        for col in columns:
            if col not in dataframe.columns:
                target = col.replace("_", " ")
                dataframe[col] = (
                    dataframe["Finding Labels"].str.contains(col, regex=False)
                    | dataframe["Finding Labels"].str.contains(target, regex=False)
                ).astype(float)

    identifier = "Patient ID" if "Patient ID" in dataframe.columns else "Patient_ID"

    unique = dataframe[identifier].unique()
    numpy.random.shuffle(unique)

    split = int(len(unique) * 0.80)
    train_patients = unique[:split]

    train_metadata = dataframe[dataframe[identifier].isin(train_patients)]
    val_metadata = dataframe[~dataframe[identifier].isin(train_patients)]

    sample_image = os.path.join(
        args.directory,
        (
            train_metadata.iloc[0]["Image Index"]
            if "Image Index" in train_metadata.columns
            else train_metadata.iloc[0]["Image_Index"]
        ),
    )
    if os.path.exists(sample_image):
        figure_path = os.path.join("docs", "figs", "pre_train_augmentation.png")
        plot_augmentation_steps(sample_image, figure_path, args.width, args.height)
        log("system", f"Augmentation visualization saved to {figure_path}")

    train_dataset = ChestDataset(
        train_metadata, args.directory, args.width, args.height, augment=True, cache=args.cache
    )
    val_dataset = ChestDataset(val_metadata, args.directory, args.width, args.height, augment=False, cache=args.cache)

    workers = args.workers
    prefetch = args.prefetch if workers > 0 else None
    persistent = (not args.disable_persistent) if workers > 0 else False
    pin = not args.disable_pin

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin,
        persistent_workers=persistent,
        prefetch_factor=prefetch,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin,
        persistent_workers=persistent,
        prefetch_factor=prefetch,
    )

    positives = train_metadata[columns].sum().values
    negatives = len(train_metadata) - positives
    raw_weights = torch.tensor(negatives / (positives + 1e-5), dtype=torch.float32)
    weights = torch.clamp(raw_weights, max=10.0).to(device)

    model = DenseNetMultiLabel(args.classes, args.architecture, freeze=args.freeze).to(device)

    if device.type == "cuda" and hasattr(torch, "compile"):
        model = torch.compile(model)

    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    fused_support = device.type == "cuda" and "fused" in torch.optim.AdamW.__init__.__code__.co_varnames

    if fused_support:
        optimizer = torch.optim.AdamW(trainable_parameters, lr=args.rate, weight_decay=args.decay, fused=True)
    else:
        optimizer = torch.optim.AdamW(trainable_parameters, lr=args.rate, weight_decay=args.decay)

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=args.warmup)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - args.warmup, eta_min=1e-6
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[args.warmup]
    )

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=weights).to(device)

    if args.sanity:
        log("sanity", "Initiating single-batch overfit loop")
        batch = next(iter(train_loader))
        loader = [batch]
        step = 0

        while True:
            step += 1
            loss = train_epoch(model, loader, optimizer, criterion, device, args.clip)
            if step == 1 or step % 10 == 0:
                log("sanity", f"Step {step} | Loss: {loss:.4f}")

        sys.exit(0)

    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
    run_name = f"{timestamp}_{args.architecture}_lr{args.rate}_b{args.batch}"
    run_dir = os.path.join(args.artifacts, run_name)
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "hyperparameters.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    log("system", f"Artifacts will be saved to {run_dir}")

    dummy = torch.randn(1, 3, args.height, args.width).to(device)
    stats = torchinfo.summary(getattr(model, "_orig_mod", model), input_data=dummy, verbose=0)
    with open(os.path.join(run_dir, "architecture.txt"), "w") as f:
        f.write(str(stats))

    history = {"train_loss": [], "val_loss": [], "val_auroc": [], "val_auprc": []}
    best = 0.0
    stagnant = 0

    for epoch in range(1, args.epochs + 1):
        log("train", f"Starting Epoch {epoch}")

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, args.clip)
        scheduler.step()

        val_metrics = evaluate(model, val_loader, criterion, device, args.classes)

        log(
            "eval",
            f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['loss']:.4f} | AUROC: {val_metrics['auroc']:.4f} | AUPRC: {val_metrics['auprc']:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}",
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_auroc"].append(val_metrics["auroc"])
        history["val_auprc"].append(val_metrics["auprc"])

        with open(os.path.join(run_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=4)

        raw_state = getattr(model, "_orig_mod", model).state_dict()
        state = {"epoch": epoch, "model": raw_state, "optimizer": optimizer.state_dict(), "best": best}
        torch.save(state, os.path.join(run_dir, "last.pth"))

        if val_metrics["auroc"] > best:
            best = val_metrics["auroc"]
            stagnant = 0
            torch.save(raw_state, os.path.join(run_dir, "best.pth"))
            log("save", f"New best AUROC {best:.4f} saved to {run_dir}")
        else:
            stagnant += 1

        if stagnant >= args.patience:
            log("stop", f"Early stopping triggered at epoch {epoch}. Best AUROC: {best:.4f}")
            break
