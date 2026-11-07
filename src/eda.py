"""Exploratory data analysis for NIH ChestX-ray14."""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy
import pandas
import seaborn

from .labels import LABELS, add_label_columns
from .utils import log


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", default="data/Data_Entry_2017.csv")
    parser.add_argument("--artifacts", default="docs/figs")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if not os.path.exists(args.metadata):
        log("error", f"Metadata file missing at {args.metadata}")
        return 1

    os.makedirs(args.artifacts, exist_ok=True)
    dataframe = pandas.read_csv(args.metadata)
    dataframe = add_label_columns(dataframe)
    columns = list(LABELS)

    counts = dataframe[columns].sum().sort_values(ascending=False)

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    figure, axes = plt.subplots(figsize=(12, 6), dpi=300)
    bars = counts.plot(kind="bar", color="#0052cc", ax=axes)
    axes.set_title("Pathology Class Frequency Distribution", fontsize=14, fontweight="bold")
    axes.set_ylabel("Positive Scans Count", fontsize=12)
    axes.set_xlabel("Pathology Class", fontsize=12)
    plt.xticks(rotation=45, ha="right")

    for patch in bars.patches:
        axes.annotate(
            f"{int(patch.get_height()):,}",
            (patch.get_x() + patch.get_width() / 2.0, patch.get_height()),
            ha="center",
            va="center",
            xytext=(0, 5),
            textcoords="offset points",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(os.path.join(args.artifacts, "eda_class_distribution.png"), bbox_inches="tight")
    plt.close(figure)

    matrix = numpy.zeros((len(columns), len(columns)), dtype=int)
    for row_index, column_a in enumerate(columns):
        for column_index, column_b in enumerate(columns):
            if row_index == column_index:
                if "Finding Labels" in dataframe.columns:
                    spaced = column_a.replace("_", " ")
                    solitary = (dataframe["Finding Labels"] == column_a) | (
                        dataframe["Finding Labels"] == spaced
                    )
                    matrix[row_index, column_index] = solitary.sum()
                else:
                    matrix[row_index, column_index] = dataframe[column_a].sum()
            else:
                matrix[row_index, column_index] = (
                    (dataframe[column_a] == 1.0) & (dataframe[column_b] == 1.0)
                ).sum()

    figure, axes = plt.subplots(figsize=(12, 10), dpi=300)
    seaborn.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="YlGnBu",
        xticklabels=columns,
        yticklabels=columns,
        ax=axes,
        cbar_kws={"label": "Co-occurrence Count"},
    )
    axes.set_title(
        "Pathology Co-Occurrence Matrix\n(Diagonal = Diagnosed Solitary/Alone)",
        fontsize=13,
        fontweight="bold",
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(args.artifacts, "eda_co_occurrence_heatmap.png"), bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=300)

    identifier = "Patient ID" if "Patient ID" in dataframe.columns else "Patient_ID"
    scans_per_patient = dataframe[identifier].value_counts()
    axes[0].hist(scans_per_patient, bins=30, color="#0052cc", edgecolor="black", alpha=0.8)
    axes[0].set_title("Scans per Patient Distribution", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Number of Scans")
    axes[0].set_ylabel("Patient Count")
    axes[0].set_yscale("log")

    if "Patient Gender" in dataframe.columns and "Patient Age" in dataframe.columns:
        cleaned_age = dataframe[(dataframe["Patient Age"] > 0) & (dataframe["Patient Age"] < 100)]
        seaborn.histplot(
            data=cleaned_age,
            x="Patient Age",
            hue="Patient Gender",
            multiple="stack",
            ax=axes[1],
            bins=20,
            palette={"M": "#0052cc", "F": "#ffc300"},
        )
        axes[1].set_title("Patient Demographics: Age & Gender", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Age (Years)")
        axes[1].set_ylabel("Scan Count")

    if "View Position" in dataframe.columns:
        view_counts = dataframe["View Position"].value_counts()
        view_counts.plot(
            kind="pie",
            ax=axes[2],
            autopct="%1.1f%%",
            colors=["#0052cc", "#ffc300"],
            startangle=90,
        )
        axes[2].set_title("Scan View Position (PA vs AP)", fontsize=12, fontweight="bold")
        axes[2].set_ylabel("")

    plt.tight_layout()
    plt.savefig(os.path.join(args.artifacts, "eda_demographics_and_views.png"), bbox_inches="tight")
    plt.close(figure)

    log("eda", f"EDA figures saved successfully to {args.artifacts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
