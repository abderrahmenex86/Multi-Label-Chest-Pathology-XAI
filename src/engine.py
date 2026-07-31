import torch
from torchmetrics.classification import MultilabelAUROC, MultilabelAveragePrecision
from tqdm import tqdm


def train_epoch(model, loader, optimizer, criterion, device, clip):
    model.train()
    total = 0.0

    progress = tqdm(loader, desc="Training", leave=False, disable=len(loader) <= 1)

    for inputs, targets in progress:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        total += loss.item()
        if len(loader) > 1:
            progress.set_postfix({"loss": f"{loss.item():.4f}"})

    return total / len(loader)


def evaluate(model, loader, criterion, device, classes):
    model.eval()
    total = 0.0

    auroc = MultilabelAUROC(num_labels=classes, average="macro").to(device)
    auprc = MultilabelAveragePrecision(num_labels=classes, average="macro").to(device)

    progress = tqdm(loader, desc="Evaluating", leave=False, disable=len(loader) <= 1)

    with torch.no_grad():
        for inputs, targets in progress:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            total += loss.item()
            auroc.update(outputs, targets.long())
            auprc.update(outputs, targets.long())

    results = {"loss": total / len(loader), "auroc": auroc.compute().item(), "auprc": auprc.compute().item()}

    auroc.reset()
    auprc.reset()

    return results
