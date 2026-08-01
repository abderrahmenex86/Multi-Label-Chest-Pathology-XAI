import argparse
import json
import os

import cv2
import numpy
import pandas
import torch
from PIL import Image
from torchvision import transforms

from model import DenseNetMultiLabel
from utils import log, seed_everything

activations = None
gradients = None


def save_activation(module, inputs, output):
    global activations
    activations = output


def save_gradient(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, default=None)
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--metadata", default="data/Data_Entry_2017.csv")
    parser.add_argument("--directory", default="data/images")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--target", type=str, default=None)
    args = parser.parse_args()

    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    target_run = args.run
    if target_run is None:
        directories = [
            os.path.join(args.artifacts, item)
            for item in os.listdir(args.artifacts)
            if os.path.isdir(os.path.join(args.artifacts, item))
        ]
        if not directories:
            log("error", "No run directories found in artifacts.")
            exit(1)
        directories.sort(key=lambda item: os.path.getmtime(item), reverse=True)
        target_run = directories[0]

    config_path = os.path.join(target_run, "hyperparameters.json")
    weights_path = os.path.join(target_run, "best.pth")

    if not os.path.exists(config_path) or not os.path.exists(weights_path):
        log("error", f"Missing configuration or weights in {target_run}")
        exit(1)

    with open(config_path, "r") as handle:
        hyperparameters = json.load(handle)

    labels = [
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

    model = DenseNetMultiLabel(hyperparameters["classes"], hyperparameters["architecture"]).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    model.backbone.features.register_forward_hook(save_activation)
    model.backbone.features.register_full_backward_hook(save_gradient)

    output_dir = os.path.join(target_run, "predictions")
    os.makedirs(output_dir, exist_ok=True)

    dataframe = pandas.read_csv(args.metadata)
    if "Finding Labels" in dataframe.columns:
        for col in labels:
            if col not in dataframe.columns:
                match_label = col.replace("_", " ")
                dataframe[col] = (
                    dataframe["Finding Labels"].str.contains(col, regex=False)
                    | dataframe["Finding Labels"].str.contains(match_label, regex=False)
                ).astype(float)

    positive_samples = dataframe[dataframe[labels].sum(axis=1) > 0].sample(
        n=min(args.count, len(dataframe)), random_state=42
    )

    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((hyperparameters["height"], hyperparameters["width"])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    log("infer", f"Generating Grad-CAM overlays for {len(positive_samples)} sample images...")

    for index, row in positive_samples.iterrows():
        filename = row["Image Index"] if "Image Index" in row else row["Image_Index"]
        source = os.path.join(args.directory, filename)

        if not os.path.exists(source):
            continue

        raw_bgr = cv2.imread(source, cv2.IMREAD_COLOR)
        height, width, _ = raw_bgr.shape
        rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)

        inputs = transform(rgb).unsqueeze(0).to(device)

        model.zero_grad()
        logits = model(inputs)
        probabilities = torch.sigmoid(logits).squeeze(0)

        if args.target and args.target in labels:
            target_index = labels.index(args.target)
        else:
            target_index = torch.argmax(probabilities).item()

        target_class = labels[target_index]
        score = probabilities[target_index].item()

        logits[0, target_index].backward()

        weights = torch.mean(gradients, dim=[2, 3], keepdim=True)
        cam = torch.sum(weights * activations, dim=1).squeeze().detach().cpu().numpy()
        cam = numpy.maximum(cam, 0)

        max_value = numpy.max(cam)
        if max_value > 0:
            cam = cam / max_value

        resized_cam = cv2.resize(cam, (width, height), interpolation=cv2.INTER_LINEAR)
        scaled_cam = numpy.uint8(255 * resized_cam)
        colored = cv2.applyColorMap(scaled_cam, cv2.COLORMAP_JET)

        overlay = cv2.addWeighted(raw_bgr, 0.6, colored, 0.4, 0)

        header = numpy.zeros((40, width * 2, 3), dtype=numpy.uint8)
        text = f"Pred: {target_class} ({score*100:.1f}%)"
        cv2.putText(header, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        combined = numpy.hstack((raw_bgr, overlay))
        final_output = numpy.vstack((header, combined))

        output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_{target_class}.png")
        cv2.imwrite(output_path, final_output)

    log("infer", f"Grad-CAM evaluation maps saved to {output_dir}")
