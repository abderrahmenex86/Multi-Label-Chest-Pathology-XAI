from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.external import (
    ExternalDataset,
    class_count,
    load_external_manifest,
    load_pediatric_manifest,
    load_rsna_manifest,
    load_siim_manifest,
)
from src.preprocess import build_transform


def test_siim_manifest_reads_image_level_labels(tmp_path: Path) -> None:
    image_root = tmp_path / "dicom-images-train" / "nested"
    image_root.mkdir(parents=True)
    (image_root / "positive.dcm").write_bytes(b"placeholder")
    (image_root / "negative.dcm").write_bytes(b"placeholder")
    pd.DataFrame(
        {
            "ImageId": ["positive.dcm", "negative.dcm"],
            "EncodedPixels": ["1 2 3", "-1"],
        }
    ).to_csv(tmp_path / "train-rle.csv", index=False)

    manifest = load_siim_manifest(tmp_path)

    assert manifest.name == "siim"
    assert manifest.target_class == "Pneumothorax"
    assert len(manifest.items) == 2
    assert class_count(manifest.items) == {"positive": 1, "negative": 1}
    assert manifest.missing_images == 0


def test_rsna_manifest_aggregates_duplicate_patient_rows(tmp_path: Path) -> None:
    image_root = tmp_path / "stage_2_train_images"
    image_root.mkdir()
    (image_root / "patient-a.dcm").write_bytes(b"placeholder")
    (image_root / "patient-b.dcm").write_bytes(b"placeholder")
    pd.DataFrame(
        {
            "patientId": ["patient-a", "patient-a", "patient-b"],
            "Target": [0, 1, 0],
        }
    ).to_csv(tmp_path / "stage_2_train_labels.csv", index=False)

    manifest = load_rsna_manifest(tmp_path)

    assert manifest.name == "rsna"
    assert manifest.independent is False
    assert len(manifest.items) == 2
    assert class_count(manifest.items) == {"positive": 1, "negative": 1}


def test_pediatric_manifest_uses_directory_labels(tmp_path: Path) -> None:
    normal = tmp_path / "test" / "NORMAL"
    pneumonia = tmp_path / "test" / "PNEUMONIA"
    normal.mkdir(parents=True)
    pneumonia.mkdir(parents=True)
    (normal / "normal.png").write_bytes(b"placeholder")
    (pneumonia / "pneumonia.png").write_bytes(b"placeholder")

    manifest = load_pediatric_manifest(tmp_path, split="test")

    assert manifest.name == "pediatric"
    assert manifest.target_class == "Pneumonia"
    assert class_count(manifest.items) == {"positive": 1, "negative": 1}


def test_external_dispatch(tmp_path: Path) -> None:
    normal = tmp_path / "test" / "NORMAL"
    normal.mkdir(parents=True)
    (normal / "normal.png").write_bytes(b"placeholder")

    manifest = load_external_manifest("pediatric", tmp_path, split="test")

    assert manifest.name == "pediatric"


def test_external_dataset_marks_valid_and_invalid_images(tmp_path: Path) -> None:
    image_path = tmp_path / "scan.png"
    array = (np.random.default_rng(0).random((64, 64)) * 255).astype("uint8")
    Image.fromarray(array).convert("RGB").save(image_path)

    dataset = ExternalDataset(
        [(image_path, 1), (tmp_path / "missing.png", 0)],
        build_transform(32, 32),
        height=32,
        width=32,
    )

    tensor, label, valid = dataset[0]
    assert tensor.shape == (3, 32, 32)
    assert label.item() == 1.0
    assert valid.item() is True

    _, _, invalid = dataset[1]
    assert invalid.item() is False
