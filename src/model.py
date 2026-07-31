import torch
from torchvision.models import (
    DenseNet121_Weights,
    DenseNet161_Weights,
    densenet121,
    densenet161,
)


class DenseNetMultiLabel(torch.nn.Module):
    def __init__(self, classes, architecture="densenet121"):
        super().__init__()

        if architecture == "densenet161":
            self.backbone = densenet161(weights=DenseNet161_Weights.DEFAULT)
        else:
            self.backbone = densenet121(weights=DenseNet121_Weights.DEFAULT)

        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = torch.nn.Linear(in_features, classes)

    def forward(self, inputs):
        return self.backbone(inputs)
