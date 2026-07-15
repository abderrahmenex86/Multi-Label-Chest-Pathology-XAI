import json
import os
import time
from datetime import datetime

import pandas as pd
import torch
import torch.nn as nn
import torchinfo
from torch.utils.data import DataLoader
from torchmetrics.classification import MultilabelAUROC, MultilabelAveragePrecision
from torchvision import transforms
from tqdm.auto import tqdm

from src.dataset import NIHChestDataset
from src.models import DenseNetMultiLabel
from src.utils import log_message, plot_pre_training_batch, set_determinism


def train_epoch(model, dataloader, optimizer, loss_fn, device, clip_threshold):
    model.train()
    total_loss = 0.0
    progress = tqdm(dataloader, desc="Training", leave=False)

    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(images)
            loss = loss_fn(logits, labels)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip_threshold)
        optimizer.step()

        total_loss += loss.item()
        progress.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / len(dataloader)


def evaluate(model, dataloader, loss_fn, device, auroc_macro, auprc_macro, auroc_none, auprc_none):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(images)
                loss = loss_fn(logits, labels)

            total_loss += loss.item()
            auroc_macro.update(logits, labels.long())
            auprc_macro.update(logits, labels.long())
            auroc_none.update(logits, labels.long())
            auprc_none.update(logits, labels.long())

    results = {
        "loss": total_loss / len(dataloader),
        "auroc_macro": auroc_macro.compute().item(),
        "auprc_macro": auprc_macro.compute().item(),
        "auroc_per_class": auroc_none.compute().tolist(),
        "auprc_per_class": auprc_none.compute().tolist(),
    }

    auroc_macro.reset()
    auprc_macro.reset()
    auroc_none.reset()
    auprc_none.reset()

    return results


def execute_training(config):
    if not os.path.exists(config.get("data_csv")):
        raise FileNotFoundError(f"Sanity Check: Metadata CSV missing at {config.get('data_csv')}")
    if not os.path.exists(config.get("dataset_directory")):
        raise FileNotFoundError(f"Sanity Check: Image directory missing at {config.get('dataset_directory')}")

    set_determinism(config.get("random_seed"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resume_dir = config.get("run_dir")
    is_resuming = resume_dir and os.path.exists(os.path.join(resume_dir, "last_checkpoint.pth"))

    if is_resuming:
        run_dir = resume_dir
        log_file = os.path.join(run_dir, "run.log")
        log_message("system", f"Resuming execution from directory: {run_dir}", log_file)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join("artifacts", f"{timestamp}_{config.get('architecture')}")
        if not config.get("overfit"):
            os.makedirs(run_dir, exist_ok=True)
            log_file = os.path.join(run_dir, "run.log")
            with open(os.path.join(run_dir, "hyperparameters.json"), "w") as f:
                json.dump(config, f, indent=4)
        else:
            log_file = None
            log_message("system", "OVERFIT MODE ENGAGED. No artifacts will be saved.")

    metadata = pd.read_csv(config.get("data_csv"))
    metadata = metadata[
        metadata["Image_Index"].apply(lambda x: os.path.exists(os.path.join(config.get("dataset_directory"), x)))
    ]

    if len(metadata) == 0:
        raise ValueError("Sanity Check: Metadata CSV is empty or images are missing.")

    patient_identifier = "Patient ID" if "Patient ID" in metadata.columns else "Patient_ID"
    unique_patients = metadata[patient_identifier].unique()
    np.random.shuffle(unique_patients)

    train_patients = unique_patients[: int(len(unique_patients) * 0.80)]
    train_metadata = metadata[metadata[patient_identifier].isin(train_patients)]
    val_metadata = metadata[~metadata[patient_identifier].isin(train_patients)]

    pathology_columns = [
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

    pos_counts = train_metadata[pathology_columns].sum().values
    neg_counts = len(train_metadata) - pos_counts
    pos_weight = torch.tensor(neg_counts / (pos_counts + 1e-5), dtype=torch.float32).to(device)

    train_transform = transforms.Compose(
        [
            transforms.Resize((config.get("image_height"), config.get("image_width"))),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((config.get("image_height"), config.get("image_width"))),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = NIHChestDataset(train_metadata, config.get("dataset_directory"), train_transform)
    val_dataset = NIHChestDataset(val_metadata, config.get("dataset_directory"), val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get("batch_size"),
        shuffle=True,
        num_workers=config.get("num_workers"),
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.get("batch_size"), shuffle=False, num_workers=2, pin_memory=True
    )

    if not is_resuming and not config.get("overfit"):
        images, labels = next(iter(train_loader))
        plot_pre_training_batch(
            images, labels, os.path.join("docs", "figs", f"pre_train_{os.path.basename(run_dir)}.png")
        )

    raw_model = DenseNetMultiLabel(architecture=config.get("architecture"), out_classes=config.get("out_classes")).to(
        device
    )
    model = torch.compile(raw_model) if not config.get("overfit") else raw_model

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    if not config.get("overfit"):
        dummy_input = torch.randn(
            1, config.get("in_channels"), config.get("image_height"), config.get("image_width")
        ).to(device)
        model_stats = torchinfo.summary(getattr(model, "module", model), input_data=dummy_input, verbose=0)
        with open(os.path.join(run_dir, "architecture.txt"), "w") as f:
            f.write(str(model_stats))

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.get("learning_rate"), weight_decay=config.get("weight_decay")
    )
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=config.get("warmup_epochs"))
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.get("epochs") - config.get("warmup_epochs"), eta_min=1e-7
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[config.get("warmup_epochs")]
    )

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)

    auroc_macro = MultilabelAUROC(num_labels=config.get("out_classes"), average="macro").to(device)
    auprc_macro = MultilabelAveragePrecision(num_labels=config.get("out_classes"), average="macro").to(device)
    auroc_none = MultilabelAUROC(num_labels=config.get("out_classes"), average="none").to(device)
    auprc_none = MultilabelAveragePrecision(num_labels=config.get("out_classes"), average="none").to(device)

    if config.get("overfit"):
        single_batch = next(iter(train_loader))
        overfit_loader = [single_batch]
        log_message("overfit", "Starting continuous single-batch overfit loop.")
        while True:
            train_loss = train_epoch(model, overfit_loader, optimizer, loss_fn, device, config.get("clip_threshold"))
            log_message("overfit", f"Loss: {train_loss:.4f}")

    start_epoch = 1
    best_auroc = 0.0
    stagnant_epochs = 0
    history = {"train_loss": [], "val_loss": [], "val_auroc": [], "val_auprc": []}

    if is_resuming:
        checkpoint = torch.load(os.path.join(run_dir, "last_checkpoint.pth"), map_location=device)
        getattr(model, "module", model).load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        best_auroc = checkpoint["best_auroc"]
        stagnant_epochs = checkpoint["stagnant_epochs"]
        history = checkpoint["history"]

    for epoch in range(start_epoch, config.get("epochs") + 1):
        epoch_start = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device, config.get("clip_threshold"))
        scheduler.step()
        val_metrics = evaluate(model, val_loader, loss_fn, device, auroc_macro, auprc_macro, auroc_none, auprc_none)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_auroc"].append(val_metrics["auroc_macro"])
        history["val_auprc"].append(val_metrics["auprc_macro"])

        with open(os.path.join(run_dir, "model_history.json"), "w") as f:
            json.dump(history, f, indent=4)

        duration = time.time() - epoch_start
        log_message(
            "train",
            f"Epoch {epoch} | Loss: {train_loss:.4f} | Val AUROC: {val_metrics['auroc_macro']:.4f} | Val AUPRC: {val_metrics['auprc_macro']:.4f} | LR: {scheduler.get_last_lr()[0]:.2e} | Time: {duration:.1f}s",
            log_file,
        )

        checkpoint_state = {
            "epoch": epoch,
            "model": getattr(model, "module", model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_auroc": best_auroc,
            "stagnant_epochs": stagnant_epochs,
            "history": history,
        }
        torch.save(checkpoint_state, os.path.join(run_dir, "last_checkpoint.pth"))

        if val_metrics["auroc_macro"] > best_auroc:
            best_auroc = val_metrics["auroc_macro"]
            stagnant_epochs = 0
            torch.save(checkpoint_state["model"], os.path.join(run_dir, "best_model.pth"))

            per_class_results = {
                "classes": pathology_columns,
                "auroc": val_metrics["auroc_per_class"],
                "auprc": val_metrics["auprc_per_class"],
            }
            with open(os.path.join(run_dir, "per_class_metrics.json"), "w") as f:
                json.dump(per_class_results, f, indent=4)

            log_message("save", f"New best AUROC: {best_auroc:.4f} serialized.", log_file)
        else:
            stagnant_epochs += 1

        if stagnant_epochs >= config.get("patience"):
            log_message("stop", f"Early stopping triggered at epoch {epoch}.", log_file)
            break
