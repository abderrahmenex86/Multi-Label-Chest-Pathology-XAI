"""Training and evaluation loops."""

from __future__ import annotations

import torch
from torchmetrics.classification import MultilabelAUROC, MultilabelAveragePrecision
from tqdm import tqdm

from .utils import autocast_context


def train_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    clip: float,
    accumulate: int = 1,
) -> float:
    """Run one training epoch and return the mean batch loss.

    ``accumulate`` groups micro-batches before each optimizer step. The loss is
    divided by the group size so the gradient is an average, not a sum.
    """
    if accumulate < 1:
        raise ValueError("accumulate must be at least 1.")

    model.train()
    total = 0.0
    optimizer.zero_grad(set_to_none=True)
    total_batches = len(loader)

    progress = tqdm(loader, desc="Training", leave=False, disable=total_batches <= 1)

    for index, (inputs, targets) in enumerate(progress, start=1):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with autocast_context(device):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        remaining = total_batches - index + 1
        group_size = min(accumulate, remaining)
        (loss / group_size).backward()
        total += loss.item()

        if index % accumulate == 0 or index == total_batches:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if total_batches > 1:
            progress.set_postfix({"loss": f"{loss.item():.4f}"})

    return total / max(total_batches, 1)


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
