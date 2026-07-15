import argparse

from src.eda import execute_eda
from src.engine import execute_training
from src.infer import execute_inference
from src.profile import discover_batch_size
from src.utils import plot_comparative_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Ex86 NIH Chest X-Ray Architecture")

    parser.add_argument("--train", action="store_true")
    parser.add_argument("--overfit", action="store_true")
    parser.add_argument("--infer", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--eda", action="store_true")
    parser.add_argument("--tools", action="store_true")

    parser.add_argument("--architecture", type=str, default="densenet121", choices=["densenet121", "densenet161"])
    parser.add_argument("--data_csv", type=str, default="dataset/metadata.csv")
    parser.add_argument("--dataset_directory", type=str, default="dataset/images")
    parser.add_argument("--run_dir", type=str, default=None)
    parser.add_argument("--inference_dir", type=str, default="dataset/inference")

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--clip_threshold", type=float, default=1.0)

    parser.add_argument("--image_height", type=int, default=224)
    parser.add_argument("--image_width", type=int, default=224)
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--out_classes", type=int, default=14)
    parser.add_argument("--random_seed", type=int, default=42)

    return vars(parser.parse_args())


def main():
    config = parse_args()

    if config.get("eda"):
        execute_eda(config.get("data_csv"), config.get("dataset_directory"), "docs/figs")

    if config.get("profile"):
        discover_batch_size(config)

    if config.get("train") or config.get("overfit"):
        execute_training(config)

    if config.get("infer"):
        execute_inference(config.get("inference_dir"), config.get("run_dir"))

    if config.get("tools"):
        plot_comparative_metrics("artifacts", "docs/figs/comparative_metrics_evaluation.png")


if __name__ == "__main__":
    main()
