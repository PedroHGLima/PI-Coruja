import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms

# humanos, animais, veículos
classes = {
    (0): 'humanos',
    (1, 2, 3) : 'veiculos',
    (15, 16): 'animais'
}

class CorujaResNet(nn.Module):
    def __init__(self, unfreeze_head: bool = False):
        super().__init__()
        self.base = models.resnet50(weights='IMAGENET1K_V1')
        for param in self.base.parameters():
            param.requires_grad = unfreeze_head
        num_ftrs = self.base.fc.in_features
        self.base.fc = nn.Linear(num_ftrs, 3)
        
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.base(x))
   
transforms_map = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(512, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}
