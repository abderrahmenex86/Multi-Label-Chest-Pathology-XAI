"""Shared label definitions and metadata helpers for NIH ChestX-ray14."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

LABELS = (
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
)

NUM_LABELS = len(LABELS)


def get_image_column(columns: Iterable[str]) -> str:
    """Return the image filename column used by the metadata file."""
    available = set(columns)
    for candidate in ("Image Index", "Image_Index"):
        if candidate in available:
            return candidate
    raise KeyError("Metadata must contain an 'Image Index' or 'Image_Index' column.")


def get_patient_column(columns: Iterable[str]) -> str:
    """Return the patient identifier column used by the metadata file."""
    available = set(columns)
    for candidate in ("Patient ID", "Patient_ID"):
        if candidate in available:
            return candidate
    raise KeyError("Metadata must contain a 'Patient ID' or 'Patient_ID' column.")


def add_label_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Add the 14 binary label columns in-place and return the dataframe.

    The NIH CSV stores labels as a pipe-separated ``Finding Labels`` string.
    ``Pleural_Thickening`` uses an underscore in the model but a space in the
    CSV, so both spellings are checked. If the columns are already present the
    dataframe is returned unchanged.
    """
    if "Finding Labels" not in dataframe.columns:
        missing = [label for label in LABELS if label not in dataframe.columns]
        if missing:
            raise ValueError(f"Metadata is missing label columns: {', '.join(missing)}")
        return dataframe

    findings = dataframe["Finding Labels"].fillna("")
    for label in LABELS:
        if label in dataframe.columns:
            continue
        spaced = label.replace("_", " ")
        dataframe[label] = (
            findings.str.contains(label, regex=False)
            | findings.str.contains(spaced, regex=False)
        ).astype("float32")

    return dataframe
