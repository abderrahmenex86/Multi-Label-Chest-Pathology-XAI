"""Shared inference preprocessing for uploaded images."""

from __future__ import annotations

import io
from pathlib import Path
from typing import BinaryIO

from PIL import Image, ImageOps
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Keep a bounded image size so a crafted upload cannot exhaust server memory.
MAX_IMAGE_PIXELS = 50_000_000
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS


def build_transform(height: int, width: int) -> transforms.Compose:
    """Build the same ImageNet-normalized transform used at training time."""
    return transforms.Compose(
        [
            transforms.Resize((height, width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def to_rgb(image: Image.Image) -> Image.Image:
    """Apply EXIF orientation and convert to RGB, with a pixel-count guard."""
    if image.width * image.height > MAX_IMAGE_PIXELS:
        raise ValueError(
            f"Image is too large ({image.width}x{image.height}); "
            f"the maximum is {MAX_IMAGE_PIXELS:,} pixels."
        )

    image = ImageOps.exif_transpose(image)
    if image.width * image.height > MAX_IMAGE_PIXELS:
        raise ValueError(
            f"Image is too large after EXIF orientation ({image.width}x{image.height}); "
            f"the maximum is {MAX_IMAGE_PIXELS:,} pixels."
        )
    return image.convert("RGB")


def open_image(source: str | Path | BinaryIO | bytes | bytearray) -> Image.Image:
    """Open an image from a path, file object, or raw bytes."""
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    with Image.open(source) as image:
        return to_rgb(image)
