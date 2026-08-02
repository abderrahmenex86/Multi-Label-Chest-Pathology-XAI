import argparse
import os
import pandas
import numpy
import matplotlib.pyplot as plt
import seaborn

from utils import log

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default="data/Data_Entry_2017.csv")
    parser.add_argument("--artifacts", default="docs/figs")
    args = parser.parse_args()

    if not os.path.exists(args.metadata):
        log("error", f"Metadata file missing at {args.metadata}")
        exit(1)

    os.makedirs(args.artifacts, exist_ok=True)
    dataframe = pandas.read_csv(args.metadata)

    columns = [
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
    ]

    if "Finding Labels" in dataframe.columns:
        for col in columns:
            if col not in dataframe.columns:
                target = col.replace("_", " ")
                dataframe[col] = (
                    dataframe["Finding Labels"].str.contains(col, regex=False)
                    | dataframe["Finding Labels"].str.contains(target, regex=False)
                ).astype(float)

    counts = dataframe[columns].sum().sort_values(ascending=False)

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    figure, axes = plt.subplots(figsize=(12, 6), dpi=300)
    bars = counts.plot(kind="bar", color="#0052cc", ax=axes)
    axes.set_title("Pathology Class Frequency Distribution", fontsize=14, fontweight="bold")
    axes.set_ylabel("Positive Scans Count", fontsize=12)
    axes.set_xlabel("Pathology Class", fontsize=12)
    plt.xticks(rotation=45, ha="right")

    for p in bars.patches:
        axes.annotate(
            f"{int(p.get_height()):,}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="center",
            xytext=(0, 5),
            textcoords="offset points",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(os.path.join(args.artifacts, "eda_class_distribution.png"))
    plt.close()

    matrix = numpy.zeros((len(columns), len(columns)), dtype=int)
    for i, col_a in enumerate(columns):
        for j, col_b in enumerate(columns):
            if i == j:
                target_str = col_a.replace("_", " ")
                solitary = (dataframe["Finding Labels"] == col_a) | (dataframe["Finding Labels"] == target_str)
                matrix[i, j] = solitary.sum()
            else:
                matrix[i, j] = ((dataframe[col_a] == 1.0) & (dataframe[col_b] == 1.0)).sum()

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
        "Pathology Co-Occurrence Matrix\n(Diagonal = Diagnosed Solitary/Alone)", fontsize=13, fontweight="bold"
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(args.artifacts, "eda_co_occurrence_heatmap.png"))
    plt.close()

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
        view_counts.plot(kind="pie", ax=axes[2], autopct="%1.1f%%", colors=["#0052cc", "#ffc300"], startangle=90)
        axes[2].set_title("Scan View Position (PA vs AP)", fontsize=12, fontweight="bold")
        axes[2].set_ylabel("")

    plt.tight_layout()
    plt.savefig(os.path.join(args.artifacts, "eda_demographics_and_views.png"))
    plt.close()

    log("eda", f"EDA figures saved successfully to {args.artifacts}")
