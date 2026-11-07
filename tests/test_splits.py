from __future__ import annotations

import numpy as np
import pandas as pd

from src.splits import patient_split


def make_metadata(patient_count: int = 100) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Patient ID": list(range(patient_count)),
            "Image Index": [f"image_{index}.png" for index in range(patient_count)],
        }
    )


def test_split_is_patient_disjoint_and_deterministic() -> None:
    dataframe = make_metadata()

    first = patient_split(dataframe, seed=42, val_fraction=0.1, test_fraction=0.1)
    second = patient_split(dataframe, seed=42, val_fraction=0.1, test_fraction=0.1)

    assert first.train_patients == second.train_patients
    assert first.validation_patients == second.validation_patients
    assert first.test_patients == second.test_patients

    train = set(first.train_patients)
    validation = set(first.validation_patients)
    test = set(first.test_patients)

    assert not train & validation
    assert not train & test
    assert not validation & test
    assert len(train) + len(validation) + len(test) == len(dataframe)


def test_legacy_80_20_split_matches_original_random_state() -> None:
    dataframe = make_metadata(100)
    unique = dataframe["Patient ID"].unique().copy()
    expected = np.random.RandomState(42)
    expected.shuffle(unique)

    split = patient_split(dataframe, seed=42, val_fraction=0.2, test_fraction=0.0)

    assert split.train_patients == [int(value) for value in unique[:80]]
    assert split.validation_patients == [int(value) for value in unique[80:]]
    assert split.test_patients == []


def test_zero_test_fraction_produces_empty_test_split() -> None:
    split = patient_split(make_metadata(), val_fraction=0.2, test_fraction=0.0)

    assert len(split.test) == 0
    assert len(split.test_patients) == 0


def test_invalid_fractions_are_rejected() -> None:
    dataframe = make_metadata(10)

    for validation, test in ((0.0, 0.0), (0.6, 0.5), (-0.1, 0.1)):
        try:
            patient_split(dataframe, val_fraction=validation, test_fraction=test)
        except ValueError:
            continue
        raise AssertionError(f"Expected ValueError for fractions {validation}, {test}")
