from __future__ import annotations

import numpy as np
import torch

from src.explain import hirescam, normalize_cam


def test_hirescam_sums_before_rectifying() -> None:
    gradients = torch.tensor([[[[1.0, -1.0]]]])
    activations = torch.tensor([[[[1.0, 1.0]]]])

    cam = hirescam(gradients, activations)

    # Per-element ReLU would return [[1, 1]]; published HiResCAM returns [[1, 0]].
    assert torch.equal(cam, torch.tensor([[1.0, 0.0]]))


def test_hirescam_accepts_batchless_inputs() -> None:
    gradients = torch.ones(2, 3, 4)
    activations = torch.ones(2, 3, 4)

    cam = hirescam(gradients, activations)

    assert cam.shape == (3, 4)
    assert torch.equal(cam, torch.full((3, 4), 2.0))


def test_hirescam_rejects_mismatched_shapes() -> None:
    try:
        hirescam(torch.ones(1, 2, 3, 4), torch.ones(1, 2, 3, 5))
    except ValueError:
        return
    raise AssertionError("Expected a ValueError for mismatched shapes.")


def test_normalize_cam_scales_and_cuts_off() -> None:
    cam = torch.tensor([[0.0, 0.5], [1.0, 2.0]])

    normalized = normalize_cam(cam)
    assert np.allclose(normalized, np.array([[0.0, 0.25], [0.5, 1.0]], dtype=np.float32))

    cut = normalize_cam(cam, cutoff=0.4)
    assert np.allclose(cut, np.array([[0.0, 0.0], [0.5, 1.0]], dtype=np.float32))
