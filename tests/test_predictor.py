from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from src.labels import LABELS
from src.predictor import ChestPredictor

CHECKPOINT = Path("checkpoints/densenet121")


@pytest.mark.skipif(not (CHECKPOINT / "best.pth").is_file(), reason="Bundled checkpoint is unavailable.")
def test_bundled_checkpoint_produces_predictions_and_heatmap() -> None:
    predictor = ChestPredictor(CHECKPOINT)
    result = predictor.predict(Image.new("RGB", (64, 64), "white"), target="Effusion")

    assert result["target"] == "Effusion"
    assert len(result["predictions"]) == len(LABELS)
    assert {item["name"] for item in result["predictions"]} == set(LABELS)
    assert isinstance(result["heatmap"], str)
    assert result["heatmap"]
    assert result["run"] == "densenet121"
