import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from models import DenseNetMultiLabel


def build_model(num_classes):
    return DenseNetMultiLabel(num_classes=num_classes)


def build_optimizer(model, learning_rate, weight_decay):
    return AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


def build_scheduler(optimizer, warmup_epochs, total_epochs):
    warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=(total_epochs - warmup_epochs))
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])


def build_criterion():
    return torch.nn.BCEWithLogitsLoss()
