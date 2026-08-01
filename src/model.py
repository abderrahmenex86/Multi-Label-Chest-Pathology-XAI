import torch
from torchvision.models import (
    DenseNet121_Weights,
    DenseNet161_Weights,
    EfficientNet_B0_Weights,
    MobileNet_V3_Small_Weights,
    ResNet18_Weights,
    densenet121,
    densenet161,
    efficientnet_b0,
    mobilenet_v3_small,
    resnet18,
)


class DenseNetMultiLabel(torch.nn.Module):
    def __init__(self, classes, architecture="densenet121", freeze=False):
        super().__init__()
        self.architecture = architecture

        if architecture == "densenet161":
            self.backbone = densenet161(weights=DenseNet161_Weights.DEFAULT)
            in_features = self.backbone.classifier.in_features
            self.backbone.classifier = torch.nn.Linear(in_features, classes)
            self.target_layer = self.backbone.features
        elif architecture == "resnet18":
            self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = torch.nn.Linear(in_features, classes)
            self.target_layer = self.backbone.layer4
        elif architecture == "efficientnet_b0":
            self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier[1] = torch.nn.Linear(in_features, classes)
            self.target_layer = self.backbone.features
        elif architecture == "mobilenet_v3_small":
            self.backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
            in_features = self.backbone.classifier[3].in_features
            self.backbone.classifier[3] = torch.nn.Linear(in_features, classes)
            self.target_layer = self.backbone.features
        else:
            self.backbone = densenet121(weights=DenseNet121_Weights.DEFAULT)
            in_features = self.backbone.classifier.in_features
            self.backbone.classifier = torch.nn.Linear(in_features, classes)
            self.target_layer = self.backbone.features

        if freeze:
            for parameter in self.target_layer.parameters():
                parameter.requires_grad = False

    def forward(self, inputs):
        return self.backbone(inputs)
