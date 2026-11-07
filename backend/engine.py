"""Backend inference service."""

from __future__ import annotations

import io
import logging
from typing import Any

from PIL import Image

from src.predictor import ChestPredictor
from src.runs import resolve_run

logger = logging.getLogger("chest_xai.backend")

predictor: ChestPredictor | None = None
target_run: str | None = None
architecture: str | None = None


def configure(run: str | None = None, artifacts: str = "artifacts") -> None:
    """Load the requested run, or the newest complete run, or the bundled demo.

    A missing model does not crash the process: the health endpoint reports a
    degraded state and prediction requests return 503 with a clear message.
    """
    global predictor, target_run, architecture

    try:
        run_path = resolve_run(run, artifacts, fallback="checkpoints/densenet121")
    except FileNotFoundError as error:
        predictor = None
        target_run = None
        architecture = None
        logger.error("Model could not be loaded: %s", error)
        return

    try:
        loaded = ChestPredictor(run_path)
    except Exception as error:  # pragma: no cover - defensive startup logging
        predictor = None
        target_run = str(run_path)
        architecture = None
        logger.exception("Failed to initialize predictor from %s: %s", run_path, error)
        return

    predictor = loaded
    target_run = str(run_path)
    architecture = loaded.architecture
    logger.info(
        "Loaded run=%s architecture=%s resolution=%sx%s device=%s",
        run_path.name,
        loaded.architecture,
        loaded.width,
        loaded.height,
        loaded.device,
    )


def predict(buffer: bytes, target: str | None = None) -> dict[str, Any]:
    """Run inference on raw image bytes."""
    if predictor is None:
        raise RuntimeError("No model is loaded. Check /health and the server logs.")

    image = Image.open(io.BytesIO(buffer))
    return predictor.predict(image, target)


def model_loaded() -> bool:
    return predictor is not None


def health_payload() -> dict[str, Any]:
    loaded = model_loaded()
    return {
        "status": "ok" if loaded else "degraded",
        "model_loaded": loaded,
        "target_run": target_run,
        "architecture": architecture,
    }
