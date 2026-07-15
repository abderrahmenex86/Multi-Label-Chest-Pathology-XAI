import gc

import torch
import torchinfo

from src.models import DenseNetMultiLabel
from src.utils import log_message


def discover_batch_size(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        log_message("profile", "Profiling requires CUDA. Skipping.")
        return

    log_message("profile", "Initiating batch size limits profiling...")

    model = DenseNetMultiLabel(architecture=config.get("architecture"), out_classes=config.get("out_classes")).to(
        device
    )
    dummy_input = torch.randn(
        1, config.get("in_channels"), config.get("image_height"), config.get("image_width"), device=device
    )
    log_message(
        "profile", f"Model Architecture Summary:\n{str(torchinfo.summary(model, input_data=dummy_input, verbose=0))}"
    )

    del model, dummy_input
    torch.cuda.empty_cache()
    gc.collect()

    current_batch_size = 2
    max_safe_batch_size = 2

    while True:
        try:
            model = DenseNetMultiLabel(
                architecture=config.get("architecture"), out_classes=config.get("out_classes")
            ).to(device)
            if torch.cuda.device_count() > 1:
                model = torch.nn.DataParallel(model)

            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            loss_fn = torch.nn.BCEWithLogitsLoss().to(device)

            dummy_images = torch.randn(
                current_batch_size,
                config.get("in_channels"),
                config.get("image_height"),
                config.get("image_width"),
                device=device,
            )
            dummy_labels = torch.empty(current_batch_size, config.get("out_classes"), device=device).random_(2)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(dummy_images)
                loss = loss_fn(logits, dummy_labels)

            loss.backward()
            optimizer.step()

            max_safe_batch_size = current_batch_size
            current_batch_size += 2

            del model, optimizer, loss_fn, dummy_images, dummy_labels, logits, loss
            torch.cuda.empty_cache()
            gc.collect()

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                gc.collect()
                break
            raise e

    log_message("profile", f"Absolute Max Batch Size: {max_safe_batch_size}")
    log_message("profile", f"Recommended Safe Batch Size: {max(2, max_safe_batch_size - 2)}")
