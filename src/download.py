import glob
import os
import shutil
import subprocess
import zipfile

from src.utils import log_message


def execute_download(data_csv, dataset_directory):
    os.makedirs(dataset_directory, exist_ok=True)
    os.makedirs(os.path.dirname(data_csv), exist_ok=True)

    if os.path.exists(data_csv) and len(glob.glob(os.path.join(dataset_directory, "*.png"))) > 1000:
        log_message("download", "Dataset already exists. Skipping.")
        return

    kaggle_dataset = "nih-chest-xrays/data"
    archive_path = os.path.join(dataset_directory, "data.zip")

    log_message("download", f"Initiating download for {kaggle_dataset}...")
    try:
        subprocess.run(["kaggle", "datasets", "download", "-d", kaggle_dataset, "-p", dataset_directory], check=True)
    except Exception as exception_instance:
        log_message("error", f"Download failed. Ensure kaggle CLI is authenticated: {exception_instance}")
        raise exception_instance

    log_message("download", "Extracting compressed dataset...")
    with zipfile.ZipFile(archive_path, "r") as archive_handle:
        archive_handle.extractall(dataset_directory)

    log_message("download", "Mapping files and flattening directory structure...")
    extracted_csv = os.path.join(dataset_directory, "Data_Entry_2017.csv")
    if os.path.exists(extracted_csv):
        shutil.move(extracted_csv, data_csv)

    nested_image_directories = glob.glob(os.path.join(dataset_directory, "images_*", "images"))
    for directory in nested_image_directories:
        for image_path in glob.glob(os.path.join(directory, "*.png")):
            destination = os.path.join(dataset_directory, os.path.basename(image_path))
            shutil.move(image_path, destination)
        shutil.rmtree(os.path.dirname(directory))

    if os.path.exists(archive_path):
        os.remove(archive_path)

    log_message("download", "NIH ChestX-ray14 successfully mapped to local environment.")
