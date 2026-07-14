import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import torch
from tqdm.auto import tqdm


def log_message(category, message):
    timestamp = datetime.now().strftime("%m/%d - %H:%M")
    tqdm.write(f"[{category.upper()}] [{timestamp}] {message}")


def plot_pre_training_batch(images, labels, save_path):
    batch_size = min(images.shape[0], 4)
    fig, axes = plt.subplots(1, batch_size, figsize=(4 * batch_size, 4))
    if batch_size == 1:
        axes = [axes]

    for i in range(batch_size):
        image_np = images[i].cpu().numpy().transpose(1, 2, 0)
        image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min() + 1e-8)

        active_labels = torch.where(labels[i] == 1.0)[0].tolist()
        label_str = f"Classes: {active_labels}" if active_labels else "No Findings"

        axes[i].imshow(image_np)
        axes[i].set_title(label_str, fontsize=8)
        axes[i].axis("off")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_comparative_metrics(artifacts_dir, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    run_directories = [
        os.path.join(artifacts_dir, d)
        for d in os.listdir(artifacts_dir)
        if os.path.isdir(os.path.join(artifacts_dir, d))
    ]

    for run_path in sorted(run_directories):
        history_path = os.path.join(run_path, "model_history.json")
        if not os.path.exists(history_path):
            continue

        with open(history_path, "r") as f:
            history = json.load(f)

        folder_name = os.path.basename(run_path)
        epochs = range(1, len(history["train_loss"]) + 1)

        axes[0].plot(epochs, history["train_loss"], label=folder_name, linewidth=2)
        axes[1].plot(epochs, history["val_loss"], label=folder_name, linewidth=2)

    axes[0].set_title("Training BCE Loss")
    axes[0].set_xlabel("Epochs")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    axes[1].set_title("Validation BCE Loss")
    axes[1].set_xlabel("Epochs")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
