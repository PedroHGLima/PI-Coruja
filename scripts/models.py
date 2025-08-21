import torch
import torch.nn as nn
from torchvision import models

class CorujaResNet(nn.Module):
    def __init__(self, unfreeze_head: bool = False):
        super().__init__()
        self.base = models.resnet18(weights='IMAGENET1K_V1')
        for param in self.base.parameters():
            param.requires_grad = False
        if unfreeze_head:
            for name, param in self.base.named_parameters():
                if "layer4" in name or "fc" in name:
                    param.requires_grad = True
        num_ftrs = self.base.fc.in_features
        self.base.fc = nn.Linear(num_ftrs, 1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x)
