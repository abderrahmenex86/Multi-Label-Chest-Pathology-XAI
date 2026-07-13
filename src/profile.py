import torch
from models import DenseNetMultiLabel


def discover_batch_size(device, resolution):
    model = DenseNetMultiLabel(num_classes=14).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()

    batch_size = 2
    max_safe = 2

    while True:
        try:
            images = torch.randn(batch_size, 3, resolution, resolution, device=device)
            targets = torch.empty(batch_size, 14, device=device).random_(2)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(images)
                loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            max_safe = batch_size
            batch_size += 2
            torch.cuda.empty_cache()

        except RuntimeError as memory_exception:
            if "out of memory" in str(memory_exception).lower():
                torch.cuda.empty_cache()
                return max_safe
            raise memory_exception
