import json
import os

import matplotlib.pyplot as plt


def plot_comparative_metrics(artifacts_dir, output_path):
    plt.figure(figsize=(10, 6), dpi=300)

    for run_folder in os.listdir(artifacts_dir):
        run_path = os.path.join(artifacts_dir, run_folder)
        history_file = os.path.join(run_path, "model_history.json")

        if os.path.exists(history_file):
            with open(history_file, "r") as file_handle:
                data = json.load(file_handle)

            epochs = range(len(data["val_loss"]))
            plt.plot(epochs, data["val_loss"], label=f"{run_folder} Val Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
