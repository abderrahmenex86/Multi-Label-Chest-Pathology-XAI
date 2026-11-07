"""Generate HiResCAM overlay images for validation/test samples."""

from __future__ import annotations

import argparse
import base64
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from .labels import LABELS, add_label_columns, get_image_column
from .predictor import ChestPredictor
from .preprocess import to_rgb
from .runs import resolve_run
from .utils import log, seed_everything


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=None)
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--metadata", default="data/Data_Entry_2017.csv")
    parser.add_argument("--directory", default="data/images")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--target", default=None)
    return parser


def decode_heatmap(payload: str) -> np.ndarray:
    encoded = np.frombuffer(base64.b64decode(payload), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError("Could not decode the heatmap payload.")
    return image


def composite_overlay(bgr: np.ndarray, rgba: np.ndarray) -> np.ndarray:
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError("Expected an RGBA heatmap.")
    alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
    return (
        bgr.astype(np.float32) * (1.0 - alpha)
        + rgba[:, :, :3].astype(np.float32) * alpha
    ).astype(np.uint8)


def main() -> int:
    args = build_parser().parse_args()

    if args.target is not None and args.target not in LABELS:
        raise SystemExit(f"Unknown target '{args.target}'. Choose from: {', '.join(LABELS)}.")

    seed_everything(42)

    try:
        run_dir = resolve_run(args.run, args.artifacts, fallback="checkpoints/densenet121")
    except FileNotFoundError as error:
        log("error", str(error))
        return 1

    predictor = ChestPredictor(run_dir)

    output_dir = Path(run_dir) / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataframe = pd.read_csv(args.metadata)
    dataframe = add_label_columns(dataframe)

    positive_pool = dataframe[dataframe[list(LABELS)].sum(axis=1) > 0]
    sample_count = min(args.count, len(positive_pool))
    samples = positive_pool.sample(n=sample_count, random_state=42)

    image_column = get_image_column(dataframe.columns)
    log("infer", f"Generating {len(samples)} HiResCAM overlays from {run_dir.name}...")

    for _, row in samples.iterrows():
        source = os.path.join(args.directory, row[image_column])
        if not os.path.exists(source):
            log("warning", f"Skipping missing image: {source}")
            continue

        with Image.open(source) as handle:
            image = to_rgb(handle)

        result = predictor.predict(image, args.target)
        target_name = result["target"]
        score = next(
            item["probability"] for item in result["predictions"] if item["name"] == target_name
        )

        rgb = np.array(image)
        raw_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        rgba = decode_heatmap(result["heatmap"])
        overlay = composite_overlay(raw_bgr, rgba)

        width = raw_bgr.shape[1]
        header = np.zeros((40, width * 2, 3), dtype=np.uint8)
        text = f"Pred: {target_name} ({score * 100:.1f}%) [HiResCAM]"
        cv2.putText(header, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        combined = np.hstack((raw_bgr, overlay))
        final_output = np.vstack((header, combined))

        output_path = output_dir / f"{Path(row[image_column]).stem}_{target_name}.png"
        cv2.imwrite(str(output_path), final_output)

    log("save", f"HiResCAM overlays saved to {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
