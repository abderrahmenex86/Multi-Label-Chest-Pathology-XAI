import os
import glob
import shutil
import zipfile
import subprocess
from utils import log


def execute_download(metadata, directory):
    os.makedirs(directory, exist_ok=True)
    os.makedirs(os.path.dirname(metadata) or ".", exist_ok=True)

    if os.path.exists(metadata) and len(glob.glob(os.path.join(directory, "*.png"))) > 1000:
        log("download", "Dataset exists. Skipping download.")
        return

    dataset = "nih-chest-xrays/data"
    archive = os.path.join(directory, "data.zip")

    log("download", f"Initiating download for {dataset}")
    try:
        subprocess.run(["kaggle", "datasets", "download", "-d", dataset, "-p", directory], check=True)
    except Exception as error:
        log("error", f"Download failed. Check Kaggle CLI authentication: {error}")
        raise error

    log("download", "Extracting archive")
    with zipfile.ZipFile(archive, "r") as handle:
        handle.extractall(directory)

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
