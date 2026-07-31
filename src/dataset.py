import os

import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms

cv2.setNumThreads(0)


class ChestDataset(Dataset):
    def __init__(self, metadata, directory, width, height, augment):
        self.directory = directory
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
            self.transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize((height, width)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(degrees=10),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize((height, width)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        filename = record["Image Index"] if "Image Index" in record else record["Image_Index"]
        path = os.path.join(self.directory, filename)

        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        inputs = self.transform(rgb)
        targets = torch.tensor([float(record[col]) for col in self.columns], dtype=torch.float32)

        return inputs, targets
