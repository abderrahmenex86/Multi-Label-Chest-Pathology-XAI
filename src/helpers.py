import os
import json
import datetime
from tqdm.auto import tqdm
from torchinfo import summary


def log_message(message, log_type, log_file=None):
    stamp = datetime.datetime.now().strftime("%m/%d - %H:%M")
    formatted = f"[{stamp}] [{log_type}] {message}"
    tqdm.write(formatted)

    if log_file:
        with open(log_file, "a") as file_handle:
            file_handle.write(formatted + "\n")


def serialize_config(config, out_dir):
    path = os.path.join(out_dir, "hyperparameters.json")
    with open(path, "w") as file_handle:
        json.dump(config, file_handle, indent=4)


def serialize_architecture(model, input_size, out_dir):
    path = os.path.join(out_dir, "architecture.txt")
    stats = summary(model, input_size=input_size, verbose=0)
    with open(path, "w") as file_handle:
        file_handle.write(str(stats))
