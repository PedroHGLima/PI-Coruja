#!/usr/bin/env python3
"""
Script para treinar uma CNN classificadora binária (humano vs. não-humano)
usando Transfer Learning com PyTorch e um modelo ResNet18 pré-treinado.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms

from pathlib import Path
import time
import copy
from tqdm import tqdm

DATA_DIR = Path("../data") / "dataset_10k_train"
MODEL_SAVE_PATH = Path("../models") / "coruja_classifier_best.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
NUM_EPOCHS = 15
LEARNING_RATE = 0.001
VAL_SPLIT = 0.2

MODEL_SAVE_PATH.parent.mkdir(exist_ok=True)
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

full_dataset = datasets.ImageFolder(DATA_DIR)

val_size = int(len(full_dataset) * VAL_SPLIT)
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_dataset.dataset.transform = data_transforms['train']
val_dataset.dataset.transform = data_transforms['val']

dataloaders = {
    'train': DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True),
    'val': DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
}
dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}
class_names = full_dataset.classes

print(f"Dispositivo de treinamento: {DEVICE}")
print(f"Classes encontradas: {class_names}")
print(f"Tamanho do dataset: {len(full_dataset)} imagens")
print(f"Dividido em {dataset_sizes['train']} para treino e {dataset_sizes['val']} para validação.")
model = models.resnet18(weights='IMAGENET1K_V1')

for param in model.parameters():
    param.requires_grad = False

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 1)

model = model.to(DEVICE)

def train_model(model, criterion, optimizer, num_epochs=10):
    start_time = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Época {epoch + 1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dataloaders[phase], desc=f"{phase.capitalize()}..."):
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE).float().view(-1, 1)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    preds = (outputs > 0.0).float()

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase.capitalize()} | Perda: {epoch_loss:.4f} | Acurácia: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, MODEL_SAVE_PATH)
                print(f"Melhor modelo salvo em '{MODEL_SAVE_PATH}' com acurácia de {best_acc:.4f}")

    time_elapsed = time.time() - start_time
    print(f'\nTreinamento concluído em {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Melhor acurácia na validação: {best_acc:.4f}')

    model.load_state_dict(best_model_wts)
    return model


if __name__ == '__main__':
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)
    trained_model = train_model(model, criterion, optimizer, num_epochs=NUM_EPOCHS)
