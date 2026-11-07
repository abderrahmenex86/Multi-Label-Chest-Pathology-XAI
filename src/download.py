"""Download and flatten the NIH ChestX-ray14 dataset via the Kaggle CLI."""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
import zipfile

from .utils import log


def safe_extract(archive: str, directory: str) -> None:
    """Extract a zip archive while rejecting path-traversal entries."""
    base = os.path.abspath(directory)
    with zipfile.ZipFile(archive, "r") as handle:
        for member in handle.infolist():
            target = os.path.abspath(os.path.join(directory, member.filename))
            if os.path.commonpath([base, target]) != base:
                raise ValueError(f"Archive entry escapes the target directory: {member.filename}")
        handle.extractall(directory)


def execute_download(metadata: str, directory: str) -> None:
    os.makedirs(directory, exist_ok=True)
    os.makedirs(os.path.dirname(metadata) or ".", exist_ok=True)

    if os.path.exists(metadata) and len(glob.glob(os.path.join(directory, "*.png"))) > 1000:
        log("download", "Dataset exists. Skipping download.")
        return

    dataset = "nih-chest-xrays/data"
    archive = os.path.join(directory, "data.zip")

    log("download", f"Initiating download for {dataset}")
    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", dataset, "-p", directory],
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        log(
            "error",
            "Download failed. Install the Kaggle CLI and configure ~/.kaggle/kaggle.json "
            f"before using --download. Details: {error}",
        )
        raise

    if not os.path.exists(archive):
        candidates = glob.glob(os.path.join(directory, "*.zip"))
        if len(candidates) == 1:
            archive = candidates[0]
        else:
            raise FileNotFoundError(f"Expected {archive} after the Kaggle download.")

    log("download", "Extracting archive")
    safe_extract(archive, directory)

    extracted = os.path.join(directory, "Data_Entry_2017.csv")
    if os.path.exists(extracted):
        shutil.move(extracted, metadata)

    nested = glob.glob(os.path.join(directory, "images_*", "images"))
    for folder in nested:
        for image_file in glob.glob(os.path.join(folder, "*.png")):
            destination = os.path.join(directory, os.path.basename(image_file))
            shutil.move(image_file, destination)
        shutil.rmtree(os.path.dirname(folder))

    if os.path.exists(archive):
        os.remove(archive)

    log("download", "Dataset structure flattened successfully")
