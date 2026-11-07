"""HiResCAM and CAM post-processing shared by the API and CLI tools."""

from __future__ import annotations

import numpy as np
import torch


def hirescam(gradients: torch.Tensor, activations: torch.Tensor) -> torch.Tensor:
    """Compute HiResCAM as ReLU(sum_channels(gradient * activation)).

    This follows the published HiResCAM formulation: the element-wise product
    is summed over channels first, then rectified. Applying ReLU to each
    element before the sum changes which channels contribute and is a
    different explanation method.

    Accepts either ``[C, H, W]`` or ``[1, C, H, W]`` tensors and returns
    ``[H, W]``.
    """
    if gradients.shape != activations.shape:
        raise ValueError(f"Gradient shape {tuple(gradients.shape)} does not match activation shape {tuple(activations.shape)}.")
    if gradients.dim() == 3:
        gradients = gradients.unsqueeze(0)
        activations = activations.unsqueeze(0)
    if gradients.dim() != 4:
        raise ValueError("Expected gradients and activations with 3 or 4 dimensions.")
    cam = torch.relu((gradients * activations).sum(dim=1))
    return cam.squeeze(0)


def normalize_cam(cam: torch.Tensor, cutoff: float = 0.0) -> np.ndarray:
    """Normalize a CAM to ``[0, 1]`` and optionally zero values below cutoff.

    The default cutoff is zero: the 20% cutoff used by the earlier prototype
    was an undocumented post-processing choice, not part of HiResCAM.
    """
    if cam.dim() == 3 and cam.shape[0] == 1:
        cam = cam.squeeze(0)
    if cam.dim() != 2:
        raise ValueError("Expected a CAM with shape [H, W] or [1, H, W].")

    cam = cam.detach().float().cpu().clamp(min=0.0)
    maximum = float(cam.max().item()) if cam.numel() else 0.0
    if maximum > 0:
        cam = cam / maximum

    array = cam.numpy()
    if cutoff > 0:
        array = array.copy()
        array[array < cutoff] = 0.0
    return np.clip(array, 0.0, 1.0)
