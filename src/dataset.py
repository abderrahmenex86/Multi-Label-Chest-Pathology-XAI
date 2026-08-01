import os

import cv2
import pandas
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from tqdm import tqdm

cv2.setNumThreads(0)


class ChestDataset(Dataset):
    def __init__(self, metadata, directory, width, height, augment, cache=False):
        self.directory = directory
        self.width = width
        self.height = height
        self.augment = augment
        self.records = metadata.to_dict("records")
        self.columns = [
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

        if augment:
            self.transform = v2.Compose(
                [
                    v2.RandomHorizontalFlip(p=0.5),
                    v2.RandomRotation(degrees=7),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        else:
            self.transform = v2.Compose(
                [
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )

        self.cache = cache
        if self.cache:
            self.images = torch.empty((len(self.records), 3, height, width), dtype=torch.uint8)
            progress = tqdm(enumerate(self.records), total=len(self.records), desc="Caching images to RAM", leave=False)
            for index, record in progress:
                filename = record["Image Index"] if "Image Index" in record else record["Image_Index"]
                path = os.path.join(self.directory, filename)
                bgr = cv2.imread(path, cv2.IMREAD_COLOR)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                resized = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)
                self.images[index] = torch.from_numpy(resized).permute(2, 0, 1)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]

        if self.cache:
            image_tensor = self.images[index]
        else:
            filename = record["Image Index"] if "Image Index" in record else record["Image_Index"]
            path = os.path.join(self.directory, filename)
            bgr = cv2.imread(path, cv2.IMREAD_COLOR)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
            image_tensor = torch.from_numpy(resized).permute(2, 0, 1)

        inputs = self.transform(image_tensor)
        targets = torch.tensor([float(record[col]) for col in self.columns], dtype=torch.float32)

        return inputs, targets
