import glob
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms


def execute_eda(data_csv, image_dir, out_dir):
    if not os.path.exists(data_csv):
        raise FileNotFoundError(f"Sanity Check: Metadata CSV missing at {data_csv}")
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Sanity Check: Image directory missing at {image_dir}")

    os.makedirs(out_dir, exist_ok=True)
    metadata = pd.read_csv(data_csv)
    total_images = len(metadata)

    if total_images == 0:
        raise ValueError("Sanity Check: Metadata CSV is empty.")

    pathology_columns = [
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

    class_counts = metadata[pathology_columns].sum().sort_values(ascending=False)
    plt.figure(figsize=(12, 6), dpi=300)
    class_counts.plot(kind="bar", color="#4B5563")
    plt.title(f"Class Distribution | Total Images: {total_images}")
    plt.ylabel("Positive Count")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "eda_class_distribution.png"))
    plt.close()

    metadata["Combination"] = metadata[pathology_columns].astype(int).astype(str).agg("".join, axis=1)
    combination_counts = metadata["Combination"].value_counts().head(15)
    plt.figure(figsize=(12, 6), dpi=300)
    combination_counts.plot(kind="bar", color="#1A56DB")
    plt.title("Top 15 Label Combinations (Binary Strings)")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "eda_label_combinations.png"))
    plt.close()

    patient_identifier = "Patient ID" if "Patient ID" in metadata.columns else "Patient_ID"
    patient_counts = metadata[patient_identifier].value_counts()
    plt.figure(figsize=(10, 6), dpi=300)
    plt.hist(patient_counts, bins=range(1, patient_counts.max() + 2), align="left", color="#DC2626", alpha=0.8)
    plt.title("Images per Patient Distribution")
    plt.xlabel("Number of Scans")
    plt.ylabel("Number of Patients")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "eda_patient_distribution.png"))
    plt.close()

    all_files = glob.glob(os.path.join(image_dir, "*.*"))
    if all_files:
        sample_files = np.random.choice(all_files, size=min(200, len(all_files)), replace=False)
        pixel_values = []
        for path in sample_files:
            img_array = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img_array is not None:
                pixel_values.extend(img_array.flatten())

        plt.figure(figsize=(10, 6), dpi=300)
        plt.hist(pixel_values, bins=256, range=(0, 255), color="#10B981", alpha=0.7)
        plt.title("Global Pixel Intensity Distribution (200 Image Sample)")
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "eda_pixel_distribution.png"))
        plt.close()

    sample_image = os.path.join(image_dir, metadata.iloc[0]["Image_Index"])
    if os.path.exists(sample_image):
        transform_list = [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.RandomRotation(degrees=45),
        ]

        current_state = Image.open(sample_image).convert("RGB")
        fig, axes = plt.subplots(1, len(transform_list) + 1, figsize=(18, 4), dpi=300)
        axes[0].imshow(np.array(current_state))
        axes[0].set_title("Original")
        axes[0].axis("off")

        for index, operation in enumerate(transform_list):
            current_state = operation(current_state)
            axes[index + 1].imshow(np.array(current_state))
            axes[index + 1].set_title(operation.__class__.__name__, fontsize=8)
            axes[index + 1].axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "eda_transform_steps.png"))
        plt.close()
