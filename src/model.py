"""Multi-label chest pathology model built on torchvision backbones."""

from __future__ import annotations

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

from .labels import NUM_LABELS

SUPPORTED_ARCHITECTURES = (
    "densenet121",
    "densenet161",
    "convnext_small",
    "efficientnet_b0",
    "efficientnet_b2",
    "mobilenet_v3_small",
    "resnet18",
    "resnet50",
)


class ChestMultiLabelModel(torch.nn.Module):
    """A torchvision backbone with a 14-way sigmoid classification head."""

    def __init__(
        self,
        classes: int = NUM_LABELS,
        architecture: str = "densenet121",
        freeze: bool = False,
        pretrained: bool = True,
    ):
        super().__init__()
        architecture = architecture.lower()
        if architecture not in SUPPORTED_ARCHITECTURES:
            raise ValueError(
                f"Unsupported architecture '{architecture}'. "
                f"Choose one of: {', '.join(SUPPORTED_ARCHITECTURES)}."
            )

        self.architecture = architecture
        weights = DenseNet121_Weights.DEFAULT if pretrained else None

        if architecture == "densenet161":
            self.backbone = densenet161(weights=DenseNet161_Weights.DEFAULT if pretrained else None)
            in_features = self.backbone.classifier.in_features
            self.backbone.classifier = torch.nn.Linear(in_features, classes)
        elif architecture == "resnet50":
            self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = torch.nn.Linear(in_features, classes)
        elif architecture == "efficientnet_b2":
            self.backbone = efficientnet_b2(weights=EfficientNet_B2_Weights.DEFAULT if pretrained else None)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier[1] = torch.nn.Linear(in_features, classes)
        elif architecture == "convnext_small":
            self.backbone = convnext_small(weights=ConvNeXt_Small_Weights.DEFAULT if pretrained else None)
            in_features = self.backbone.classifier[2].in_features
            self.backbone.classifier[2] = torch.nn.Linear(in_features, classes)
        elif architecture == "resnet18":
            self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = torch.nn.Linear(in_features, classes)
        elif architecture == "efficientnet_b0":
            self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT if pretrained else None)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier[1] = torch.nn.Linear(in_features, classes)
        elif architecture == "mobilenet_v3_small":
            self.backbone = mobilenet_v3_small(
                weights=MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            )
            in_features = self.backbone.classifier[3].in_features
            self.backbone.classifier[3] = torch.nn.Linear(in_features, classes)
        else:
            self.backbone = densenet121(weights=weights)
            in_features = self.backbone.classifier.in_features
            self.backbone.classifier = torch.nn.Linear(in_features, classes)

        if freeze:
            # Freeze the feature extractor but keep the classifier trainable.
            for name, parameter in self.backbone.named_parameters():
                if not (name.startswith("classifier") or name.startswith("fc")):
                    parameter.requires_grad = False

    @property
    def target_layer(self) -> torch.nn.Module:
        """The last convolutional feature map used for HiResCAM."""
        if self.architecture.startswith("resnet"):
            return self.backbone.layer4
        return self.backbone.features

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.architecture.startswith("densenet"):
            features = self.backbone.features(inputs)
            out = torch.relu(features)
            out = torch.nn.functional.adaptive_avg_pool2d(out, (1, 1))
            out = torch.flatten(out, 1)
            return self.backbone.classifier(out)
        return self.backbone(inputs)


# Backwards-compatible name for older scripts and cached state dicts.
DenseNetMultiLabel = ChestMultiLabelModel
