import base64
import io
import json
import os
import sys

import cv2
import numpy
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from model import DenseNetMultiLabel
from utils import seed_everything

seed_everything(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

activations = None
gradients = None


def save_activation(module, inputs, output):
    global activations
    activations = output


def save_gradient(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0]


def get_latest_run(artifacts="artifacts"):
    if not os.path.exists(artifacts):
        return None
    directories = [
        os.path.join(artifacts, item) for item in os.listdir(artifacts) if os.path.isdir(os.path.join(artifacts, item))
    ]
    if not directories:
        return None
    directories.sort(key=lambda item: os.path.getmtime(item), reverse=True)
    return directories[0]


target_run = get_latest_run()

optimal_thresholds = {}

if target_run:
    with open(os.path.join(target_run, "hyperparameters.json"), "r") as handle:
        hyperparameters = json.load(handle)

    thresholds_path = os.path.join(target_run, "optimal_thresholds.json")
    if os.path.exists(thresholds_path):
        with open(thresholds_path, "r") as handle:
            optimal_thresholds = json.load(handle)

    model = DenseNetMultiLabel(hyperparameters["classes"], hyperparameters["architecture"]).to(device)
    model.load_state_dict(torch.load(os.path.join(target_run, "best.pth"), map_location=device), strict=False)
    model.eval()

    model.target_layer.register_forward_hook(save_activation)
    model.target_layer.register_full_backward_hook(save_gradient)
else:
    model = None
    hyperparameters = {"height": 320, "width": 320}

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


def predict(buffer, target):
    image = Image.open(io.BytesIO(buffer)).convert("RGB")

    transform = transforms.Compose(
        [
            transforms.Resize((hyperparameters["height"], hyperparameters["width"])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    inputs = transform(image).unsqueeze(0).to(device)

    model.zero_grad()
    logits = model(inputs)
    probabilities = torch.sigmoid(logits).squeeze(0)

    results = [
        {"name": name, "probability": float(prob), "threshold": optimal_thresholds.get(name, 0.20)}
        for name, prob in zip(labels, probabilities)
    ]
    results.sort(key=lambda item: item["probability"], reverse=True)

    if target is None or target == "" or target not in labels:
        target_index = labels.index(results[0]["name"])
    else:
        target_index = labels.index(target)

    logits[0, target_index].backward()

    rectified_activations = F.relu(activations)
    cam = torch.sum(F.relu(gradients * rectified_activations), dim=1).squeeze().detach().cpu().numpy()

    max_val = numpy.max(cam)
    if max_val > 0:
        cam = cam / max_val

    cam[cam < 0.20] = 0.0

    resized_cam = cv2.resize(cam, (image.width, image.height), interpolation=cv2.INTER_LINEAR)
    scaled_cam = numpy.uint8(255 * resized_cam)
    colored = cv2.applyColorMap(scaled_cam, cv2.COLORMAP_JET)

    bgra = cv2.cvtColor(colored, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = scaled_cam

    _, encoded = cv2.imencode(".png", bgra)
    payload = base64.b64encode(encoded).decode("utf-8")

    return {"predictions": results, "heatmap": payload}
