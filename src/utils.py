"""Small shared utilities: seeding, logging, AMP, and plotting."""

from __future__ import annotations

import os
import random
import subprocess
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy
import torch
from tqdm import tqdm


def seed_everything(seed: int) -> None:
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def autocast_context(device: torch.device):
    """Use bf16 autocast on CUDA and a no-op context on CPU."""
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def get_git_commit() -> str | None:
    """Return the current short commit hash, or None outside a git checkout."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def log(category: str, message: str) -> None:
    timestamp = datetime.now().strftime("%m/%d - %H:%M")
    tqdm.write(f"[{category.upper()}] [{timestamp}] {message}")


def plot_augmentation_steps(
    image_path: str | Path,
    output_path: str | Path,
    width: int,
    height: int,
    rotation: float = 7.0,
) -> None:
    """Plot the actual training preprocessing/augmentation steps.

    This intentionally shows resize, horizontal flip, and rotation only. The
    earlier version also displayed a contrast-jitter step that the training
    transform never applied.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    original = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(original, (width, height), interpolation=cv2.INTER_LINEAR)
    flipped = cv2.flip(resized, 1)

    center = (width // 2, height // 2)
    matrix = cv2.getRotationMatrix2D(center, rotation, 1.0)
    rotated = cv2.warpAffine(flipped, matrix, (width, height))

    figure, axes = plt.subplots(1, 4, figsize=(16, 4), dpi=300)
    stages = [
        ("Original", original),
        (f"1. Resize ({width}x{height})", resized),
        ("2. Horizontal Flip", flipped),
        (f"3. Rotation (degrees={rotation:g})", rotated),
    ]

    for index, (title, image) in enumerate(stages):
        axes[index].imshow(image)
        axes[index].set_title(title, fontsize=9)
        axes[index].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def ensure_directory(path: str | Path) -> Path:
    """Create a directory and return it as a Path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def env_flag(name: str, default: bool = False) -> bool:
    """Read a boolean environment variable."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}
