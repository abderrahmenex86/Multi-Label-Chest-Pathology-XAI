"""NIH ChestX-ray14 dataset."""

from __future__ import annotations

import os

import cv2
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from tqdm import tqdm

from .labels import LABELS, get_image_column
from .preprocess import IMAGENET_MEAN, IMAGENET_STD

cv2.setNumThreads(0)


class ChestDataset(Dataset):
    """Load chest X-rays and their multi-label targets."""

    def __init__(self, metadata, directory, width, height, augment, cache=False):
        self.directory = directory
        self.width = width
        self.height = height
        self.augment = augment
        self.records = metadata.to_dict("records")
        self.image_column = get_image_column(metadata.columns)
        self.columns = list(LABELS)

        if augment:
            self.transform = v2.Compose(
                [
                    v2.RandomHorizontalFlip(p=0.5),
                    v2.RandomRotation(degrees=7),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ]
            )
        else:
            self.transform = v2.Compose(
                [
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ]
            )

        self.cache = cache
        self.images = None
        if self.cache:
            self.images = torch.empty((len(self.records), 3, height, width), dtype=torch.uint8)
            progress = tqdm(
                enumerate(self.records),
                total=len(self.records),
                desc="Caching images to RAM",
                leave=False,
            )
            for index, record in progress:
                self.images[index] = torch.from_numpy(self._load(record)).permute(2, 0, 1)

    def _load(self, record):
        filename = record[self.image_column]
        path = os.path.join(self.directory, filename)
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(
                f"Could not read image '{path}'. Check --directory and the metadata."
            )
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return cv2.resize(rgb, (self.width, self.height), interpolation=cv2.INTER_LINEAR)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]

        if self.cache and self.images is not None:
            image_tensor = self.images[index]
        else:
            image_tensor = torch.from_numpy(self._load(record)).permute(2, 0, 1)

        inputs = self.transform(image_tensor)
        targets = torch.tensor([float(record[column]) for column in self.columns], dtype=torch.float32)

        return inputs, targets
