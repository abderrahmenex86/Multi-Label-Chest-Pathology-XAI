import argparse
import datetime
import os

import pandas as pd
import torch
from torchvision.io import read_image

from src.dataset import build_loaders, build_transforms
from src.factory import build_criterion, build_model, build_optimizer, build_scheduler
from src.helpers import log_message, serialize_architecture, serialize_config
from src.infer import generate_cams
from src.profile import discover_batch_size
from src.trainer import train
from tools import plot_comparative_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--infer", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--tools", action="store_true")

    parser.add_argument("--data_csv", default="dataset/metadata.csv")
    parser.add_argument("--image_dir", default="dataset/images")
    parser.add_argument("--infer_image", default="")
    parser.add_argument("--infer_model", default="")
    parser.add_argument("--infer_target", default=0, type=int)

    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--warmup_epochs", default=5, type=int)
    parser.add_argument("--patience", default=10, type=int)
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--weight_decay", default=1e-5, type=float)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--pin_memory", action="store_true")

    args = parser.parse_args()
    config = vars(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.profile:
        safe_batch = discover_batch_size(device, 224)
        print(f"Optimal OOM-Safe Batch Size: {safe_batch}")
        return

    if args.train:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join("artifacts", timestamp)
        os.makedirs(run_dir, exist_ok=True)

        serialize_config(config, run_dir)

        metadata = pd.read_csv(args.data_csv)
        train_df = metadata.sample(frac=0.8, random_state=42)
        val_df = metadata.drop(train_df.index)

        train_loader, val_loader = build_loaders(
            train_df, val_df, args.image_dir, args.batch_size, args.num_workers, args.pin_memory
        )

        model = build_model(14).to(device)
        serialize_architecture(model, (1, 3, 224, 224), run_dir)

        optimizer = build_optimizer(model, args.learning_rate, args.weight_decay)
        scheduler = build_scheduler(optimizer, args.warmup_epochs, args.epochs)
        criterion = build_criterion()

        train(model, train_loader, val_loader, criterion, optimizer, scheduler, config, device, run_dir)

    if args.infer:
        _, val_transform = build_transforms()
        image = read_image(args.infer_image)
        image_tensor = val_transform(image)

        out_dir = "docs/figs"
        os.makedirs(out_dir, exist_ok=True)

        generate_cams(args.infer_model, image_tensor, args.infer_target, out_dir, device)

    if args.tools:
        os.makedirs("docs/figs", exist_ok=True)
        plot_comparative_metrics("artifacts", "docs/figs/comparative_metrics_evaluation.png")


if __name__ == "__main__":
    main()
