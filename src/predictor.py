"""Thread-safe inference and HiResCAM generation."""

from __future__ import annotations

import base64
import json
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from .explain import hirescam, normalize_cam
from .labels import LABELS, NUM_LABELS
from .model import ChestMultiLabelModel
from .preprocess import MAX_IMAGE_PIXELS, build_transform, to_rgb
from .runs import read_hyperparameters

DEFAULT_THRESHOLD = 0.5


class ChestPredictor:
    """Load one run and answer prediction requests without shared CAM state.

    A lock serializes the forward/backward pair because the model and the
    forward hook are shared by all threads. The FastAPI endpoint calls this
    object from a threadpool, so concurrent requests are safe but queued.
    """

    def __init__(self, run_dir: str | Path, device: str | torch.device | None = None):
        self.run_dir = Path(run_dir)
        self.hyperparameters = read_hyperparameters(self.run_dir)

        classes = int(self.hyperparameters.get("classes", NUM_LABELS))
        if classes != NUM_LABELS:
            raise ValueError(f"Run expects {classes} classes but this code defines {NUM_LABELS}.")

        self.architecture = str(self.hyperparameters["architecture"])
        self.height = int(self.hyperparameters["height"])
        self.width = int(self.hyperparameters["width"])

        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # The checkpoint supplies all weights; do not download ImageNet weights.
        self.model = ChestMultiLabelModel(
            NUM_LABELS,
            self.architecture,
            pretrained=False,
        ).to(self.device)
        state = torch.load(
            self.run_dir / "best.pth",
            map_location=self.device,
            weights_only=True,
        )
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

        self.transform = build_transform(self.height, self.width)
        self.thresholds = {label: DEFAULT_THRESHOLD for label in LABELS}

        thresholds_path = self.run_dir / "optimal_thresholds.json"
        if thresholds_path.is_file():
            with thresholds_path.open("r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            for label in LABELS:
                value = loaded.get(label)
                if isinstance(value, (int, float)) and 0.0 <= float(value) <= 1.0:
                    self.thresholds[label] = float(value)

        self._lock = threading.Lock()
        self._warmup()

    def _warmup(self) -> None:
        """Allocate CUDA kernels before the first user request."""
        with torch.inference_mode():
            dummy = torch.zeros(1, 3, self.height, self.width, device=self.device)
            self.model(dummy)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def predict(self, image: Image.Image, target: str | None = None) -> dict[str, Any]:
        """Return sorted probabilities, the selected target, and an RGBA heatmap."""
        with self._lock:
            image = to_rgb(image)
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise ValueError("Image exceeds the maximum supported pixel count.")

            inputs = self.transform(image).unsqueeze(0).to(self.device)

            activation: torch.Tensor | None = None

            def capture_activation(module, module_inputs, output):
                nonlocal activation
                activation = output

            handle = self.model.target_layer.register_forward_hook(capture_activation)
            try:
                logits = self.model(inputs)
                if activation is None:
                    raise RuntimeError("The target layer did not produce an activation.")

                probabilities = torch.sigmoid(logits).squeeze(0).detach().float().cpu()
                results = [
                    {
                        "name": name,
                        "probability": float(probabilities[index]),
                        "threshold": float(self.thresholds[name]),
                    }
                    for index, name in enumerate(LABELS)
                ]
                results.sort(key=lambda item: item["probability"], reverse=True)

                if target is None or target == "" or target not in LABELS:
                    target_name = results[0]["name"]
                else:
                    target_name = target

                target_index = LABELS.index(target_name)
                gradients = torch.autograd.grad(
                    logits[0, target_index],
                    activation,
                    retain_graph=False,
                    create_graph=False,
                )[0]
            finally:
                handle.remove()

            cam = hirescam(gradients, activation)
            cam = normalize_cam(cam)
            resized = cv2.resize(cam, (image.width, image.height), interpolation=cv2.INTER_LINEAR)
            scaled = np.uint8(np.clip(resized, 0.0, 1.0) * 255.0)
            colored = cv2.applyColorMap(scaled, cv2.COLORMAP_JET)
            bgra = cv2.cvtColor(colored, cv2.COLOR_BGR2BGRA)
            bgra[:, :, 3] = scaled

            encoded_ok, encoded = cv2.imencode(".png", bgra)
            if not encoded_ok:
                raise RuntimeError("Could not encode the heatmap as PNG.")

            payload = base64.b64encode(encoded.tobytes()).decode("ascii")

            for item in results:
                item["positive"] = item["probability"] >= item["threshold"]

            return {
                "predictions": results,
                "heatmap": payload,
                "target": target_name,
                "run": self.run_dir.name,
                "architecture": self.architecture,
            }
