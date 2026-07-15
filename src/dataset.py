import os

import torch
from PIL import Image
from torch.utils.data import Dataset


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

    def __getitem__(self, idx):
        record = self.records[idx]
        path = os.path.join(self.image_dir, record["Image_Index"])

        image = Image.open(path).convert("RGB")
        labels = torch.tensor([float(record[col]) for col in self.pathology_columns], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, labels
