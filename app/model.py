from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class CoordinateModel(nn.Module):
    """Baseline image-to-human-coordinate model.

    Output: normalized x, normalized y, log uncertainty x, log uncertainty y.
    Training requires real independent human annotations.
    """

    def __init__(self) -> None:
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        self.head = nn.Sequential(
            nn.Linear(features, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 4)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))
