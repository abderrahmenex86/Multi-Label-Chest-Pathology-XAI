"""Image loading and cleaning for external chest X-ray datasets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .preprocess import to_rgb

try:  # pydicom is only needed for DICOM external datasets.
    import pydicom
except ImportError:  # pragma: no cover - exercised only in minimal installs
    pydicom = None

MIN_IMAGE_SIZE = 64
MIN_STD = 1.0


def _first_value(value):
    if isinstance(value, (str, bytes)) or not hasattr(value, "__iter__"):
        return value
    values = list(value)
    return values[0] if values else value


def _dicom_to_image(path: Path) -> Image.Image:
    if pydicom is None:
        raise RuntimeError("pydicom is required to read DICOM files.")

    dataset = pydicom.dcmread(str(path), force=True)
    array = dataset.pixel_array.astype(np.float32)

    if array.ndim == 2:
        slope = float(getattr(dataset, "RescaleSlope", 1.0))
        intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
        array = array * slope + intercept

        window_center = getattr(dataset, "WindowCenter", None)
        window_width = getattr(dataset, "WindowWidth", None)
        if window_center is not None and window_width is not None:
            center = float(_first_value(window_center))
            width = float(_first_value(window_width))
            low = center - width / 2.0
            high = center + width / 2.0
        else:
            low, high = np.percentile(array, [1.0, 99.0])

        array = np.clip(array, low, high)
        array = (array - low) / max(high - low, 1e-6)
        array = np.clip(array, 0.0, 1.0)

        if str(getattr(dataset, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
            array = 1.0 - array

        image = Image.fromarray(np.uint8(array * 255.0)).convert("RGB")
    elif array.ndim == 3:
        if array.shape[2] in (3, 4):
            image = Image.fromarray(np.uint8(np.clip(array, 0, 255))).convert("RGB")
        elif array.shape[0] in (3, 4):
            image = Image.fromarray(
                np.uint8(np.clip(np.transpose(array, (1, 2, 0)), 0, 255))
            ).convert("RGB")
        else:
            raise ValueError(f"Unsupported DICOM pixel shape: {array.shape}.")
    else:
        raise ValueError(f"Unsupported DICOM pixel shape: {array.shape}.")

    return image


def crop_black_borders(image: Image.Image, margin: int = 4) -> Image.Image:
    """Crop uniform black borders without touching the anatomy."""
    grayscale = np.asarray(image.convert("L"))
    mask = grayscale > 8
    coordinates = np.argwhere(mask)
    if coordinates.size == 0:
        return image

    top, left = coordinates.min(axis=0)
    bottom, right = coordinates.max(axis=0) + 1
    top = max(0, int(top) - margin)
    left = max(0, int(left) - margin)
    bottom = min(image.height, int(bottom) + margin)
    right = min(image.width, int(right) + margin)

    if bottom - top < MIN_IMAGE_SIZE or right - left < MIN_IMAGE_SIZE:
        return image
    return image.crop((left, top, right, bottom))


def validate_image(image: Image.Image) -> None:
    """Reject images that are too small or effectively blank."""
    if image.width < MIN_IMAGE_SIZE or image.height < MIN_IMAGE_SIZE:
        raise ValueError(f"Image is too small: {image.width}x{image.height}.")
    array = np.asarray(image.convert("L"), dtype=np.float32)
    if float(array.std()) < MIN_STD:
        raise ValueError("Image is effectively blank.")


def read_image(path: str | Path, crop_borders: bool = False) -> Image.Image:
    """Read a PNG/JPEG/DICOM image, apply orientation, and optionally crop borders."""
    image_path = Path(path)
    suffix = image_path.suffix.lower()

    if suffix in {".dcm", ".dicom"}:
        image = _dicom_to_image(image_path)
    else:
        with Image.open(image_path) as handle:
            image = to_rgb(handle)

    if crop_borders:
        image = crop_black_borders(image)

    validate_image(image)
    return image
