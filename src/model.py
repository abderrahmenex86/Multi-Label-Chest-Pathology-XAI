import torch
from torchvision.models import (
    ConvNeXt_Small_Weights,
    DenseNet121_Weights,
    DenseNet161_Weights,
    EfficientNet_B0_Weights,
    EfficientNet_B2_Weights,
    MobileNet_V3_Small_Weights,
    ResNet18_Weights,
    ResNet50_Weights,
    convnext_small,
    densenet121,
    densenet161,
    efficientnet_b0,
    efficientnet_b2,
    mobilenet_v3_small,
    resnet18,
    resnet50,
)


class DenseNetMultiLabel(torch.nn.Module):
    def __init__(self, classes, architecture="densenet121", freeze=False):
        super().__init__()
        self.architecture = architecture

        if architecture == "densenet161":
            self.backbone = densenet161(weights=DenseNet161_Weights.DEFAULT)
            in_features = self.backbone.classifier.in_features
            self.backbone.classifier = torch.nn.Linear(in_features, classes)
        elif architecture == "resnet50":
            self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = torch.nn.Linear(in_features, classes)
        elif architecture == "efficientnet_b2":
            self.backbone = efficientnet_b2(weights=EfficientNet_B2_Weights.DEFAULT)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier[1] = torch.nn.Linear(in_features, classes)
        elif architecture == "convnext_small":
            self.backbone = convnext_small(weights=ConvNeXt_Small_Weights.DEFAULT)
            in_features = self.backbone.classifier[2].in_features
            self.backbone.classifier[2] = torch.nn.Linear(in_features, classes)
        elif architecture == "resnet18":
            self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = torch.nn.Linear(in_features, classes)
        elif architecture == "efficientnet_b0":
            self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier[1] = torch.nn.Linear(in_features, classes)
        elif architecture == "mobilenet_v3_small":
            self.backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
            in_features = self.backbone.classifier[3].in_features
            self.backbone.classifier[3] = torch.nn.Linear(in_features, classes)
        else:
            self.backbone = densenet121(weights=DenseNet121_Weights.DEFAULT)
            in_features = self.backbone.classifier.in_features
            self.backbone.classifier = torch.nn.Linear(in_features, classes)

        if freeze:
            for parameter in self.target_layer.parameters():
                parameter.requires_grad = False

    @property
    def target_layer(self):
        if "resnet" in self.architecture:
            return self.backbone.layer4
        elif "convnext" in self.architecture or "efficientnet" in self.architecture or "densenet" in self.architecture:
            return self.backbone.features
        return self.backbone.features

    def forward(self, inputs):
        if "densenet" in self.architecture:
            features = self.backbone.features(inputs)
            out = torch.relu(features)
            out = torch.nn.functional.adaptive_avg_pool2d(out, (1, 1))
            out = torch.flatten(out, 1)
            return self.backbone.classifier(out)
        return self.backbone(inputs)
