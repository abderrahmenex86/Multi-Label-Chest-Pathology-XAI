from __future__ import annotations

import pandas as pd

from src.labels import LABELS, add_label_columns


def test_add_label_columns_parses_pipe_separated_findings() -> None:
    dataframe = pd.DataFrame(
        {
            "Finding Labels": [
                "Effusion|Infiltration",
                "Pleural Thickening",
                "No Finding",
            ]
        }
    )

    result = add_label_columns(dataframe.copy())

    assert result.loc[0, "Effusion"] == 1.0
    assert result.loc[0, "Infiltration"] == 1.0
    assert result.loc[1, "Pleural_Thickening"] == 1.0
    assert result.loc[2, list(LABELS)].sum() == 0.0


def test_add_label_columns_keeps_existing_columns() -> None:
    dataframe = pd.DataFrame({"Finding Labels": ["Effusion"], "Effusion": [1.0]})

    result = add_label_columns(dataframe.copy())

    assert result.loc[0, "Effusion"] == 1.0


def test_add_label_columns_rejects_missing_metadata() -> None:
    dataframe = pd.DataFrame({"Other": [1]})

    try:
        add_label_columns(dataframe)
    except ValueError as error:
        assert "label columns" in str(error)
    else:
        raise AssertionError("Expected a ValueError for missing label columns.")
