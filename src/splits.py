"""Deterministic patient-level train/validation/test splitting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .labels import get_patient_column


@dataclass(frozen=True)
class DataSplit:
    """A patient-disjoint split of the metadata dataframe."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    train_patients: list[int]
    validation_patients: list[int]
    test_patients: list[int]

    @property
    def image_counts(self) -> dict[str, int]:
        return {
            "train": len(self.train),
            "validation": len(self.validation),
            "test": len(self.test),
        }

    @property
    def patient_counts(self) -> dict[str, int]:
        return {
            "train": len(self.train_patients),
            "validation": len(self.validation_patients),
            "test": len(self.test_patients),
        }


def patient_split(
    dataframe: pd.DataFrame,
    seed: int = 42,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
) -> DataSplit:
    """Split metadata by unique patient ID.

    ``RandomState(seed)`` is used instead of ``default_rng`` so that
    ``val_fraction=0.2, test_fraction=0.0`` reproduces the original 80/20
    patient split from the historical artifacts.
    """
    if val_fraction < 0 or test_fraction < 0:
        raise ValueError("Validation and test fractions must be non-negative.")
    if val_fraction + test_fraction <= 0:
        raise ValueError("At least one of validation or test fraction must be positive.")
    if val_fraction + test_fraction >= 1:
        raise ValueError("Validation and test fractions must leave a training split.")

    patient_column = get_patient_column(dataframe.columns)
    unique = dataframe[patient_column].dropna().unique()

    rng = np.random.RandomState(seed)
    rng.shuffle(unique)

    total = len(unique)
    test_count = int(round(total * test_fraction))
    validation_count = int(round(total * val_fraction))
    train_count = total - validation_count - test_count

    # Keep the historical ordering: train first, then validation, then test.
    train_patients = unique[:train_count]
    validation_patients = unique[train_count : train_count + validation_count]
    test_patients = unique[train_count + validation_count :]

    train = dataframe[dataframe[patient_column].isin(train_patients)]
    validation = dataframe[dataframe[patient_column].isin(validation_patients)]
    test = dataframe[dataframe[patient_column].isin(test_patients)]

    train_ids = {int(value) for value in train_patients}
    validation_ids = {int(value) for value in validation_patients}
    test_ids = {int(value) for value in test_patients}
    if train_ids & validation_ids or train_ids & test_ids or validation_ids & test_ids:
        raise AssertionError("Patient IDs overlap between splits.")

    return DataSplit(
        train=train,
        validation=validation,
        test=test,
        train_patients=[int(value) for value in train_patients],
        validation_patients=[int(value) for value in validation_patients],
        test_patients=[int(value) for value in test_patients],
    )
