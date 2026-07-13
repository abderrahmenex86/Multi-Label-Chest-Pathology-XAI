import json
import os

import torch
from tqdm.auto import tqdm

from helpers import log_message


def train_epoch(model, loader, criterion, optimizer, device, log_file, is_train):
    model.train() if is_train else model.eval()
    total_loss = 0.0

    progress_bar = tqdm(loader, desc="Train" if is_train else "Val", leave=False)

    for images, targets in progress_bar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=is_train):
            with torch.set_grad_enabled(is_train):
                outputs = model(images)
                loss = criterion(outputs, targets)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    return total_loss / len(loader)


def train(model, train_loader, val_loader, criterion, optimizer, scheduler, config, device, out_dir):
    log_file = os.path.join(out_dir, "run.log")
    history_file = os.path.join(out_dir, "model_history.json")
    best_model_path = os.path.join(out_dir, "best_model.pth")
    last_model_path = os.path.join(out_dir, "last_checkpoint.pth")

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    patience_counter = 0
    start_epoch = 0

    if os.path.exists(last_model_path):
        checkpoint = torch.load(last_model_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        history = checkpoint["history"]
        best_val = checkpoint["best_val"]
        patience_counter = checkpoint["patience"]
        start_epoch = checkpoint["epoch"] + 1
        log_message(f"Resumed from epoch {start_epoch}", "INFO", log_file)

    for epoch in range(start_epoch, config["epochs"]):
        log_message(f"Starting Epoch {epoch}", "EPOCH", log_file)

        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, log_file, True)
        val_loss = train_epoch(model, val_loader, criterion, optimizer, device, log_file, False)

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        with open(history_file, "w") as file_handle:
            json.dump(history, file_handle, indent=4)

        log_message(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}", "METRIC", log_file)

        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "history": history,
            "best_val": best_val,
            "patience": patience_counter,
        }
        torch.save(state, last_model_path)

        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            log_message("New best model serialized.", "SAVE", log_file)
        else:
            patience_counter += 1

        if patience_counter >= config["patience"]:
            log_message(f"Early stopping triggered at epoch {epoch}.", "STOP", log_file)
            break
