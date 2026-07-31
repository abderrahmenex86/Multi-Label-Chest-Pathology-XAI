import torch
from torchvision.models import densenet121, DenseNet121_Weights


class DenseNetMultiLabel(torch.nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.backbone = densenet121(weights=DenseNet121_Weights.DEFAULT)
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = torch.nn.Linear(in_features, classes)

    def forward(self, inputs):
        return self.backbone(inputs)
