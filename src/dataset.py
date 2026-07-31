import os

import cv2
import numpy
import pandas
import torch
from torch.utils.data import Dataset
from torchvision import transforms

cv2.setNumThreads(0)


class ChestDataset(Dataset):
    def __init__(self, metadata, directory, width, height, augment, cache=True):
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
            rgb = image_tensor.permute(1, 2, 0).numpy()
        else:
            filename = record["Image Index"] if "Image Index" in record else record["Image_Index"]
            path = os.path.join(self.directory, filename)
            bgr = cv2.imread(path, cv2.IMREAD_COLOR)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (self.width, self.height), interpolation=cv2.INTER_LINEAR)

        if self.augment:
            if numpy.random.rand() > 0.5:
                rgb = cv2.flip(rgb, 1)

            angle = numpy.random.uniform(-15, 15)
            center = (self.width // 2, self.height // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rgb = cv2.warpAffine(rgb, matrix, (self.width, self.height))

            alpha = numpy.random.uniform(0.8, 1.2)
            beta = numpy.random.uniform(-15, 15)
            rgb = cv2.convertScaleAbs(rgb, alpha=alpha, beta=beta)

        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        inputs = self.normalize(tensor)
        targets = torch.tensor([float(record[col]) for col in self.columns], dtype=torch.float32)

        return inputs, targets
