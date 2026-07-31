import os

import cv2
import numpy
import pandas
import torch
from torch.utils.data import Dataset
from torchvision import transforms

cv2.setNumThreads(0)


class ChestDataset(Dataset):
    def __init__(self, metadata, directory, width, height, augment):
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
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        filename = record["Image Index"] if "Image Index" in record else record["Image_Index"]
        path = os.path.join(self.directory, filename)

        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.width, self.height), interpolation=cv2.INTER_LINEAR)

        if self.augment:
            if numpy.random.rand() > 0.5:
                resized = cv2.flip(resized, 1)

            angle = numpy.random.uniform(-10, 10)
            center = (self.width // 2, self.height // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            resized = cv2.warpAffine(resized, matrix, (self.width, self.height))

        tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        inputs = self.normalize(tensor)
        targets = torch.tensor([float(record[col]) for col in self.columns], dtype=torch.float32)

        return inputs, targets
