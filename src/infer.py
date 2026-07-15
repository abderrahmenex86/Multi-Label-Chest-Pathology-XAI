import glob
import json
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.models import DenseNetMultiLabel, XAIHookManager
from src.utils import log_message


def generate_xai_maps(cam_tensor, target_height, target_width):
    cam_np = cam_tensor.detach().cpu().numpy()
    cam_np = cam_np - np.min(cam_np)
    cam_max = np.max(cam_np)
    if cam_max != 0:
        cam_np = cam_np / cam_max
    cam_np = cv2.resize(cam_np, (target_width, target_height))
    cam_np = np.uint8(255 * cam_np)
    return cv2.applyColorMap(cam_np, cv2.COLORMAP_JET)


def build_side_by_side(raw_image_path, cam_heatmap, target_height, target_width):
    original_bgr = cv2.resize(cv2.imread(raw_image_path), (target_width, target_height))
    overlay = cv2.addWeighted(original_bgr, 0.5, cam_heatmap, 0.5, 0)
    return np.concatenate((original_bgr, overlay), axis=1)


def execute_inference(inference_dir, run_dir, target_class=0):
    if not run_dir or not os.path.exists(run_dir):
        raise FileNotFoundError(f"Sanity Check: run_dir '{run_dir}' does not exist.")
    if not os.path.exists(inference_dir):
        raise FileNotFoundError(f"Sanity Check: inference_dir '{inference_dir}' does not exist.")

    hyperparameters_path = os.path.join(run_dir, "hyperparameters.json")
    best_model_path = os.path.join(run_dir, "best_model.pth")

    if not os.path.exists(hyperparameters_path) or not os.path.exists(best_model_path):
        raise FileNotFoundError("Sanity Check: Artifacts missing in run_dir.")

    image_paths = sorted(glob.glob(os.path.join(inference_dir, "*.*")))
    if not image_paths:
        log_message("infer", f"No images found in {inference_dir}. Exiting.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(hyperparameters_path, "r") as f:
        config = json.load(f)

    base_model = DenseNetMultiLabel(architecture=config.get("architecture"), out_classes=config.get("out_classes"))
    base_model.load_state_dict(torch.load(best_model_path, map_location=device))
    base_model.to(device)
    base_model.eval()

    xai_model = XAIHookManager(base_model)

    transform = transforms.Compose(
        [
            transforms.Resize((config.get("image_height"), config.get("image_width"))),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    out_dir = os.path.join(run_dir, "predictions")
    os.makedirs(out_dir, exist_ok=True)

    log_message("infer", f"Starting XAI inference on {len(image_paths)} images for Class {target_class}.")

    for path in image_paths:
        raw_image = Image.open(path).convert("RGB")
        image_tensor = transform(raw_image).unsqueeze(0).to(device)

        xai_model.zero_grad()
        logits = xai_model(image_tensor)
        score = logits[0, target_class]
        score.backward()

        activations = xai_model.activations[0]
        gradients = xai_model.gradients[0]

        weights_gradcam = torch.mean(gradients, dim=(1, 2), keepdim=True)
        cam_gradcam = F.relu(torch.sum(weights_gradcam * activations, dim=0))

        alpha_num = gradients.pow(2)
        alpha_denom = 2.0 * gradients.pow(2) + torch.sum(activations * gradients.pow(3), dim=(1, 2), keepdim=True)
        alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))
        alphas = alpha_num / alpha_denom
        weights_gradcam_plus = torch.sum(alphas * F.relu(gradients), dim=(1, 2), keepdim=True)
        cam_gradcam_plus = F.relu(torch.sum(weights_gradcam_plus * activations, dim=0))

        cam_hires = torch.sum(gradients * activations, dim=0)

        out_name = os.path.basename(path).split(".")[0]
        h, w = config.get("image_height"), config.get("image_width")

        gradcam_map = generate_xai_maps(cam_gradcam, h, w)
        gradcam_plus_map = generate_xai_maps(cam_gradcam_plus, h, w)
        hires_map = generate_xai_maps(cam_hires, h, w)

        cv2.imwrite(os.path.join(out_dir, f"{out_name}_gradcam.png"), build_side_by_side(path, gradcam_map, h, w))
        cv2.imwrite(
            os.path.join(out_dir, f"{out_name}_gradcam_plus.png"), build_side_by_side(path, gradcam_plus_map, h, w)
        )
        cv2.imwrite(os.path.join(out_dir, f"{out_name}_hirescam.png"), build_side_by_side(path, hires_map, h, w))

    log_message("infer", f"XAI Inference saved to {out_dir}")
