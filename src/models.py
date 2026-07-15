import torch
import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights, densenet161, DenseNet161_Weights


class DenseNetMultiLabel(nn.Module):
    def __init__(self, architecture="densenet121", out_classes=14):
        super().__init__()
        if architecture == "densenet161":
            self.backbone = densenet161(weights=DenseNet161_Weights.DEFAULT)
        else:
            self.backbone = densenet121(weights=DenseNet121_Weights.DEFAULT)

        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Linear(in_features, out_classes)

    def forward(self, x):
        return self.backbone(x)


class XAIHookManager(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.activations = None
        self.gradients = None

        target_layer = getattr(self.model, "module", self.model).backbone.features
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def forward(self, images):
        return self.model(images)
