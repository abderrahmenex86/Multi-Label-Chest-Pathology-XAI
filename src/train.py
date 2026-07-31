import argparse
import os

import numpy
import pandas
import torch
from torch.utils.data import DataLoader

from dataset import ChestDataset
from engine import evaluate, train_epoch
from model import DenseNetMultiLabel
from utils import log, seed_everything

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--directory", required=True)
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
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.artifacts, exist_ok=True)

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
    val_dataset = ChestDataset(val_metadata, args.directory, args.width, args.height, augment=False)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True)

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

    best = 0.0

    for epoch in range(1, args.epochs + 1):
        log("train", f"Starting Epoch {epoch}")

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, args.clip)
        val_metrics = evaluate(model, val_loader, criterion, device, args.classes)

        log(
            "eval",
            f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['loss']:.4f} | AUROC: {val_metrics['auroc']:.4f} | AUPRC: {val_metrics['auprc']:.4f}",
        )

        if val_metrics["auroc"] > best:
            best = val_metrics["auroc"]
            path = os.path.join(args.artifacts, "best.pth")
            torch.save(model.state_dict(), path)
            log("save", f"New best AUROC {best:.4f} saved to {path}")
