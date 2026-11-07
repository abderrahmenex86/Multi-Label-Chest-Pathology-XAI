from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.dataset import ChestDataset
from src.labels import LABELS


def test_parallel_cache_fills_expected_tensor(tmp_path: Path) -> None:
    rows = []
    for index in range(4):
        array = (np.random.default_rng(index).random((64, 64)) * 255).astype("uint8")
        name = f"scan_{index}.png"
        Image.fromarray(array).convert("RGB").save(tmp_path / name)
        row = {"Image Index": name}
        row.update({label: int(label == LABELS[index]) for label in LABELS})
        rows.append(row)

    dataset = ChestDataset(
        pd.DataFrame(rows),
        tmp_path,
        width=32,
        height=32,
        augment=False,
        cache=True,
        cache_workers=4,
    )

    assert dataset.images is not None
    assert dataset.images.shape == (4, 3, 32, 32)
    assert dataset.images.dtype == torch.uint8

    inputs, targets = dataset[0]
    assert inputs.shape == (3, 32, 32)
    assert targets.shape == (len(LABELS),)
    assert targets[LABELS.index("Atelectasis")] == 1.0
