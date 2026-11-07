"""Training and evaluation loops."""

from __future__ import annotations

import torch
from torchmetrics.classification import MultilabelAUROC, MultilabelAveragePrecision
from tqdm import tqdm

from .utils import autocast_context


def train_epoch(model, loader, optimizer, criterion, device, clip: float) -> float:
    """Run one training epoch and return the mean batch loss."""
    model.train()
    total = 0.0

    progress = tqdm(loader, desc="Training", leave=False, disable=len(loader) <= 1)

    for inputs, targets in progress:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast_context(device):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        total += loss.item()
        if len(loader) > 1:
            progress.set_postfix({"loss": f"{loss.item():.4f}"})

    return total / max(len(loader), 1)


def evaluate(model, loader, criterion, device, classes: int) -> dict[str, float]:
    """Compute macro AUROC/AUPRC and mean loss on a loader."""
    if len(loader) == 0:
        raise ValueError("Cannot evaluate an empty data loader.")

    model.eval()
    total = 0.0

    auroc = MultilabelAUROC(num_labels=classes, average="macro").to(device)
    auprc = MultilabelAveragePrecision(num_labels=classes, average="macro").to(device)

    progress = tqdm(loader, desc="Evaluating", leave=False, disable=len(loader) <= 1)

    with torch.no_grad():
        for inputs, targets in progress:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with autocast_context(device):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            total += loss.item()
            auroc.update(outputs, targets.long())
            auprc.update(outputs, targets.long())

    results = {
        "loss": total / max(len(loader), 1),
        "auroc": float(auroc.compute().item()),
        "auprc": float(auprc.compute().item()),
    }

    auroc.reset()
    auprc.reset()

    return results
