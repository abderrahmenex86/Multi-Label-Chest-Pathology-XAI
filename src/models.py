import torch
import torch.nn as nn
from torchvision.models import DenseNet121_Weights, densenet121


class DenseNetMultiLabel(nn.Module):
    def __init__(self, num_classes=14):
        super().__init__()
        self.backbone = densenet121(weights=DenseNet121_Weights.DEFAULT)

        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Linear(in_features, num_classes)

    def forward(self, images):
        return self.backbone(images)


class XAIHookManager(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.activations = None
        self.gradients = None

        target_layer = self.model.backbone.features.denseblock4.denselayer16.conv2

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def forward(self, images):
        return self.model(images)
