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
from utils import log, seed_everything

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default="data/Data_Entry_2017.csv")
    parser.add_argument("--directory", default="data/images")
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--architecture", default="densenet121")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--rate", type=float, default=1e-4)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--classes", type=int, default=14)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=224)
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

    log("system", f"Executing on {device} using {args.architecture}")

    dataframe = pandas.read_csv(args.metadata)
    identifier = "Patient ID" if "Patient ID" in dataframe.columns else "Patient_ID"

    unique = dataframe[identifier].unique()
    numpy.random.shuffle(unique)

    split = int(len(unique) * 0.80)
    train_patients = unique[:split]

    train_metadata = dataframe[dataframe[identifier].isin(train_patients)]
    val_metadata = dataframe[~dataframe[identifier].isin(train_patients)]

    train_dataset = ChestDataset(train_metadata, args.directory, args.width, args.height, augment=True)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True, drop_last=True
    )

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

    positives = train_metadata[columns].sum().values
    negatives = len(train_metadata) - positives
    weights = torch.tensor(negatives / (positives + 1e-5), dtype=torch.float32).to(device)

    model = DenseNetMultiLabel(args.classes, args.architecture).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.rate)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=weights).to(device)

    if args.sanity:
        log("sanity", "Initiating continuous single-batch overfit loop")
        batch = next(iter(train_loader))
        loader = [batch]

        while True:
            loss = train_epoch(model, loader, optimizer, criterion, device, args.clip)
            log("sanity", f"Loss: {loss:.4f}")

        sys.exit(0)

    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
    run_name = f"{timestamp}_{args.architecture}_lr{args.rate}_b{args.batch}"
    run_dir = os.path.join(args.artifacts, run_name)
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "hyperparameters.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    log("system", f"Artifacts will be saved to {run_dir}")

    val_dataset = ChestDataset(val_metadata, args.directory, args.width, args.height, augment=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True)

    dummy = torch.randn(1, 3, args.height, args.width).to(device)
    stats = torchinfo.summary(model, input_data=dummy, verbose=0)
    with open(os.path.join(run_dir, "architecture.txt"), "w") as f:
        f.write(str(stats))

    history = {"train_loss": [], "val_loss": [], "val_auroc": [], "val_auprc": []}
    best = 0.0

    for epoch in range(1, args.epochs + 1):
        log("train", f"Starting Epoch {epoch}")

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, args.clip)
        val_metrics = evaluate(model, val_loader, criterion, device, args.classes)

        log(
            "eval",
            f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['loss']:.4f} | AUROC: {val_metrics['auroc']:.4f} | AUPRC: {val_metrics['auprc']:.4f}",
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_auroc"].append(val_metrics["auroc"])
        history["val_auprc"].append(val_metrics["auprc"])

        with open(os.path.join(run_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=4)

        state = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "best": best}
        torch.save(state, os.path.join(run_dir, "last.pth"))

        if val_metrics["auroc"] > best:
            best = val_metrics["auroc"]
            torch.save(model.state_dict(), os.path.join(run_dir, "best.pth"))
            log("save", f"New best AUROC {best:.4f} saved to {run_dir}")
