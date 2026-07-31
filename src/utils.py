import os
import random
from datetime import datetime

import cv2
import matplotlib.pyplot as plt
import numpy
import torch
from tqdm import tqdm


def seed_everything(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def log(category, message):
    timestamp = datetime.now().strftime("%m/%d - %H:%M")
    tqdm.write(f"[{category.upper()}] [{timestamp}] {message}")


def plot_augmentation_steps(image_path, output_path, width, height):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    original = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    resized = cv2.resize(original, (width, height), interpolation=cv2.INTER_LINEAR)

    flipped = cv2.flip(resized, 1)

    center = (width // 2, height // 2)
    matrix = cv2.getRotationMatrix2D(center, 12, 1.0)
    rotated = cv2.warpAffine(flipped, matrix, (width, height))

    jittered = cv2.convertScaleAbs(rotated, alpha=1.2, beta=15)

    figure, axes = plt.subplots(1, 5, figsize=(18, 4), dpi=300)

    stages = [
        ("Original", original),
        ("1. Resize (320x320)", resized),
        ("2. Horizontal Flip", flipped),
        ("3. Affine Rotation", rotated),
        ("4. Contrast Jitter", jittered),
    ]

    for index, (title, image) in enumerate(stages):
        axes[index].imshow(image)
        axes[index].set_title(title, fontsize=9)
        axes[index].axis("off")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
