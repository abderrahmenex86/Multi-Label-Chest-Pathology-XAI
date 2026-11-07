"""Manifests and datasets for external chest X-ray evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .imaging import read_image
from .labels import LABELS


@dataclass(frozen=True)
class ExternalManifest:
    name: str
    items: list[tuple[Path, int]]
    target_class: str
    independent: bool
    source: str
    missing_images: int = 0


class ExternalDataset(Dataset):
    """Load and transform external images, returning a validity flag per item."""

    def __init__(self, items, transform, height: int, width: int, crop_borders: bool = False):
        self.items = items
        self.transform = transform
        self.height = height
        self.width = width
        self.crop_borders = crop_borders

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        path, label = self.items[index]
        try:
            image = read_image(path, crop_borders=self.crop_borders)
            tensor = self.transform(image)
            return tensor, torch.tensor(label, dtype=torch.float32), torch.tensor(True)
        except Exception:
            # The evaluator filters invalid items out after collation.
            return (
                torch.zeros(3, self.height, self.width, dtype=torch.float32),
                torch.tensor(label, dtype=torch.float32),
                torch.tensor(False),
            )


def _image_index(root: Path, suffixes: tuple[str, ...]) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in suffixes:
            index[path.name] = path
            index[path.stem] = path
    return index


def load_siim_manifest(data_dir: str | Path) -> ExternalManifest:
    """SIIM-ACR pneumothorax: image-level labels from train-rle.csv."""
    root = Path(data_dir)
    labels_path = root / "train-rle.csv"
    image_root = root / "dicom-images-train"
    if not labels_path.is_file() or not image_root.is_dir():
        raise FileNotFoundError(
            f"Expected {labels_path} and {image_root}. Download the SIIM-ACR train data first."
        )

    frame = pd.read_csv(labels_path)
    required = {"ImageId", "EncodedPixels"}
    if not required.issubset(frame.columns):
        raise ValueError(f"SIIM labels must contain {sorted(required)}.")

    labels = frame.groupby("ImageId")["EncodedPixels"].apply(
        lambda values: int(
            any(
                not pd.isna(value) and str(value).strip() != "-1"
                for value in values
            )
        )
    )
    index = _image_index(image_root, (".dcm", ".dicom"))
    items = []
    missing = 0
    for image_id, label in labels.items():
        path = index.get(str(image_id)) or index.get(Path(str(image_id)).name)
        if path is None:
            missing += 1
            continue
        items.append((path, int(label)))

    return ExternalManifest(
        name="siim",
        items=items,
        target_class="Pneumothorax",
        independent=True,
        source="SIIM-ACR Pneumothorax Segmentation",
        missing_images=missing,
    )


def load_rsna_manifest(data_dir: str | Path) -> ExternalManifest:
    """RSNA pneumonia: image-level target from stage_2_train_labels.csv."""
    root = Path(data_dir)
    labels_path = root / "stage_2_train_labels.csv"
    image_root = root / "stage_2_train_images"
    if not labels_path.is_file() or not image_root.is_dir():
        raise FileNotFoundError(
            f"Expected {labels_path} and {image_root}. Download the RSNA train data first."
        )

    frame = pd.read_csv(labels_path)
    required = {"patientId", "Target"}
    if not required.issubset(frame.columns):
        raise ValueError(f"RSNA labels must contain {sorted(required)}.")

    labels = frame.groupby("patientId")["Target"].max()
    index = _image_index(image_root, (".dcm", ".dicom"))
    items = []
    missing = 0
    for patient_id, label in labels.items():
        path = index.get(str(patient_id)) or index.get(Path(str(patient_id)).name)
        if path is None:
            missing += 1
            continue
        items.append((path, int(label)))

    return ExternalManifest(
        name="rsna",
        items=items,
        target_class="Pneumonia",
        independent=False,
        source="RSNA Pneumonia Detection Challenge (images overlap NIH ChestX-ray14)",
        missing_images=missing,
    )


def load_pediatric_manifest(data_dir: str | Path, split: str = "test") -> ExternalManifest:
    """Pediatric pneumonia: labels come from the NORMAL/PNEUMONIA directory names."""
    root = Path(data_dir) / split
    if not root.is_dir():
        root = Path(data_dir)

    items = []
    for class_name, label in (("NORMAL", 0), ("PNEUMONIA", 1)):
        class_dir = root / class_name
        if not class_dir.is_dir():
            continue
        for path in class_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                items.append((path, label))

    if not items:
        raise FileNotFoundError(
            f"No NORMAL/PNEUMONIA images found under {root}. "
            "Expected the Kaggle pediatric chest X-ray directory layout."
        )

    return ExternalManifest(
        name="pediatric",
        items=items,
        target_class="Pneumonia",
        independent=True,
        source="Chest X-Ray Images (Pneumonia), Guangzhou pediatric cohort",
    )


def load_external_manifest(dataset: str, data_dir: str | Path, split: str = "test") -> ExternalManifest:
    dataset = dataset.lower()
    if dataset == "siim":
        return load_siim_manifest(data_dir)
    if dataset == "rsna":
        return load_rsna_manifest(data_dir)
    if dataset == "pediatric":
        return load_pediatric_manifest(data_dir, split=split)
    raise ValueError(f"Unknown external dataset: {dataset}")


def target_index(manifest: ExternalManifest) -> int:
    if manifest.target_class not in LABELS:
        raise ValueError(f"{manifest.target_class} is not one of the model labels.")
    return LABELS.index(manifest.target_class)


def class_count(items: list[tuple[Path, int]]) -> dict[str, int]:
    labels = np.asarray([label for _, label in items], dtype=np.int64)
    return {
        "positive": int((labels == 1).sum()),
        "negative": int((labels == 0).sum()),
    }
