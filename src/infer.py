import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from models import DenseNetMultiLabel, XAIHookManager


def generate_cams(model_path, image_tensor, target_class, out_dir, device):
    base_model = DenseNetMultiLabel(num_classes=14)
    base_model.load_state_dict(torch.load(model_path, map_location=device))
    base_model.eval()
    base_model.to(device)

    xai_model = XAIHookManager(base_model)
    images = image_tensor.unsqueeze(0).to(device)

    xai_model.zero_grad()
    logits = xai_model(images)
    score = logits[0, target_class]
    score.backward()

    activations = xai_model.activations[0]
    gradients = xai_model.gradients[0]

    weights_gradcam = torch.mean(gradients, dim=(1, 2), keepdim=True)
    cam_gradcam = torch.sum(weights_gradcam * activations, dim=0)
    cam_gradcam = F.relu(cam_gradcam)

    alpha_num = gradients.pow(2)
    alpha_denom = 2.0 * gradients.pow(2) + torch.sum(activations * gradients.pow(3), dim=(1, 2), keepdim=True)
    alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))
    alphas = alpha_num / alpha_denom
    weights_gradcam_plus = torch.sum(alphas * F.relu(gradients), dim=(1, 2), keepdim=True)
    cam_gradcam_plus = torch.sum(weights_gradcam_plus * activations, dim=0)
    cam_gradcam_plus = F.relu(cam_gradcam_plus)

    cam_hires = torch.sum(gradients * activations, dim=0)

    def process_cam(cam):
        cam = cam.detach().cpu().numpy()
        cam = cam - np.min(cam)
        cam_max = np.max(cam)
        if cam_max != 0:
            cam = cam / cam_max
        cam = cv2.resize(cam, (224, 224))
        cam = np.uint8(255 * cam)
        return cv2.applyColorMap(cam, cv2.COLORMAP_JET)

    cams = {
        "gradcam": process_cam(cam_gradcam),
        "gradcam_plus": process_cam(cam_gradcam_plus),
        "hirescam": process_cam(cam_hires),
    }

    for name, cam_img in cams.items():
        cv2.imwrite(os.path.join(out_dir, f"{name}_{target_class}.png"), cam_img)
