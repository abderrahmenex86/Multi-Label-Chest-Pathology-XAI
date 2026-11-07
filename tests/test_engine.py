from __future__ import annotations

import pytest
import torch

from src.engine import train_epoch


def test_train_epoch_supports_gradient_accumulation() -> None:
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = torch.nn.BCEWithLogitsLoss()
    batches = [
        (torch.randn(2, 4), torch.rand(2, 2)),
        (torch.randn(2, 4), torch.rand(2, 2)),
        (torch.randn(2, 4), torch.rand(2, 2)),
    ]

    before = model.weight.detach().clone()
    loss = train_epoch(
        model,
        batches,
        optimizer,
        criterion,
        torch.device("cpu"),
        clip=1.0,
        accumulate=2,
    )

    assert isinstance(loss, float)
    assert not torch.equal(before, model.weight.detach())


def test_train_epoch_rejects_invalid_accumulation() -> None:
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = torch.nn.BCEWithLogitsLoss()

    with pytest.raises(ValueError, match="accumulate"):
        train_epoch(
            model,
            [(torch.randn(2, 4), torch.rand(2, 2))],
            optimizer,
            criterion,
            torch.device("cpu"),
            clip=1.0,
            accumulate=0,
        )
