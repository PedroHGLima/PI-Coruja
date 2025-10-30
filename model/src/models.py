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
    """
    ResNet50 para classificação multi-label de 3 classes.
    
    Saída: Tensor de shape [batch_size, 3] com valores em [-1, 1] (tanh)
    - Índice 0: human (pessoa)
    - Índice 1: animal (gato, cachorro)
    - Índice 2: vehicle (carro, moto, ônibus)
    """
    def __init__(self, unfreeze_head: bool = False):
        super().__init__()
        self.base = models.resnet50(weights='IMAGENET1K_V1')
        for param in self.base.parameters():
            param.requires_grad = unfreeze_head
        num_ftrs = self.base.fc.in_features
        # 3 neurônios de saída para multi-label
        self.base.fc = nn.Linear(num_ftrs, 3)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Tensor de entrada [batch_size, 3, H, W]
        
        Returns:
            Tensor [batch_size, 3] com valores em [-1, 1] (tanh)
        """
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
