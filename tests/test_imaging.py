from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.imaging import crop_black_borders, read_image


def test_read_png_and_crop_black_borders(tmp_path: Path) -> None:
    array = np.zeros((100, 100), dtype=np.uint8)
    array[15:85, 15:85] = 128
    path = tmp_path / "scan.png"
    Image.fromarray(array).save(path)

    image = read_image(path, crop_borders=True)

    assert image.mode == "RGB"
    assert 70 <= image.width <= 80
    assert 70 <= image.height <= 80


def test_crop_black_borders_is_a_noop_without_borders() -> None:
    array = np.full((80, 80), 128, dtype=np.uint8)
    image = Image.fromarray(array).convert("RGB")

    cropped = crop_black_borders(image)

    assert cropped.size == image.size


def test_blank_image_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "blank.png"
    Image.new("RGB", (128, 128), "black").save(path)

    with pytest.raises(ValueError, match="blank"):
        read_image(path)


def test_read_dicom_grayscale(tmp_path: Path) -> None:
    pytest.importorskip("pydicom")
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

    path = tmp_path / "scan.dcm"
    array = np.tile(np.arange(64, dtype=np.uint8), (64, 1))
    metadata = FileMetaDataset()
    metadata.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    metadata.MediaStorageSOPInstanceUID = generate_uid()
    metadata.TransferSyntaxUID = ExplicitVRLittleEndian

    dataset = FileDataset(str(path), {}, file_meta=metadata, preamble=b"\0" * 128)
    dataset.SOPClassUID = metadata.MediaStorageSOPClassUID
    dataset.SOPInstanceUID = metadata.MediaStorageSOPInstanceUID
    dataset.StudyInstanceUID = generate_uid()
    dataset.SeriesInstanceUID = generate_uid()
    dataset.Modality = "CR"
    dataset.PatientName = "Test"
    dataset.PatientID = "1"
    dataset.Rows, dataset.Columns = array.shape
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.SamplesPerPixel = 1
    dataset.BitsAllocated = 8
    dataset.BitsStored = 8
    dataset.HighBit = 7
    dataset.PixelRepresentation = 0
    dataset.PixelData = array.tobytes()
    dataset.save_as(str(path), enforce_file_format=True)

    image = read_image(path)

    assert image.mode == "RGB"
    assert image.size == (64, 64)
