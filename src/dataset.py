import os

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class NIHChestDataset(Dataset):
    def __init__(self, metadata, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.records = metadata.to_dict("records")
        self.pathology_columns = [
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

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        filename = record["Image_Index"]
        path = os.path.join(self.image_dir, filename)

        image_array = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)

        labels = [float(record[col]) for col in self.pathology_columns]
        target = torch.tensor(labels, dtype=torch.float32)

        if self.transform:
            image_tensor = self.transform(image_array)
        else:
            image_tensor = torch.from_numpy(image_array).float().permute(2, 0, 1) / 255.0

        return image_tensor, target


def build_transforms():
    train_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return train_transform, val_transform


def build_loaders(train_df, val_df, image_dir, batch_size, num_workers, pin_memory):
    train_transform, val_transform = build_transforms()

    train_dataset = NIHChestDataset(train_df, image_dir, train_transform)
    val_dataset = NIHChestDataset(val_df, image_dir, val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader
