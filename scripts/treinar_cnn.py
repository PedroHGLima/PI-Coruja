#!/usr/bin/env python3
"""
Script para treinar uma CNN classificadora binária (humano vs. não-humano)
usando Transfer Learning com PyTorch e um modelo ResNet18 pré-treinado.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
#!/usr/bin/env python3
"""Treinamento de classificador binário (humano / não-humano) com ResNet18.

Principais alterações:
- Registro de parâmetros e métricas com MLflow (incluindo AUC).
- Estrutura modular (carregamento de dados, treino, avaliação).
- Pequenas otimizações: pin_memory apenas quando CUDA, cálculo de AUC na validação.
"""

import argparse
import time
import copy
from pathlib import Path

import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser(description="Treina uma CNN binária com ResNet18 e registra no MLflow")
    p.add_argument("--data-dir", default="../data/dataset_10k_train")
    p.add_argument("--models-dir", default="../models")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--experiment", type=str, default="coruja_experiment")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--unfreeze-head", action="store_true", help="Descongelar últimas camadas para fine-tune")
    p.add_argument("--kfolds", type=int, default=5, help="Número de folds para validação cruzada (default=5)")
    return p.parse_args()


def get_transforms():
    return {
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
        ])
    }


def get_image_paths_and_labels(data_dir):
    from pathlib import Path
    import os
    # ImageFolder: subpastas por classe
    data_dir = Path(data_dir)
    classes = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    img_paths = []
    labels = []
    for idx, cls in enumerate(classes):
        for img_file in (data_dir / cls).glob("*.jpg"):
            img_paths.append(str(img_file))
            labels.append(idx)
    return img_paths, labels, classes


def build_model(device, unfreeze_head=False):
    model = models.resnet18(weights='IMAGENET1K_V1')
    for param in model.parameters():
        param.requires_grad = False

    if unfreeze_head:
        # Ex: descongela a camada final e a penúltima bloco
        for name, param in model.named_parameters():
            if "layer4" in name or "fc" in name:
                param.requires_grad = True

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)
    model = model.to(device)
    return model


def evaluate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Eval", leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device).float().view(-1, 1)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            probs = torch.sigmoid(outputs).cpu().numpy().ravel()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.cpu().numpy().ravel().tolist())

            running_loss += loss.item() * inputs.size(0)

    avg_loss = running_loss / len(dataloader.dataset)

    # Acurácia (threshold 0.5)
    preds = [1 if p > 0.5 else 0 for p in all_probs]
    acc = accuracy_score(all_labels, preds)

    # AUC (tratamento se classe única)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc = float('nan')

    return avg_loss, acc, auc


def plot_and_log_roc_curve(labels, probs, output_path):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    try:
        fpr, tpr, _ = roc_curve(labels, probs)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('Curva ROC - Validação Final')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        return True
    except Exception:
        return False

from sklearn.model_selection import StratifiedKFold
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def train_kfold(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_paths, labels, class_names = get_image_paths_and_labels(args.data_dir)
    img_paths = np.array(img_paths)
    labels = np.array(labels)
    skf = StratifiedKFold(n_splits=args.kfolds, shuffle=True, random_state=42)

    mlflow.set_tracking_uri(str(Path(args.data_dir).parent / "mlruns"))
    mlflow.set_experiment(args.experiment)
    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_params({
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'lr': args.lr,
            'device': device.type,
            'num_workers': args.num_workers,
            'unfreeze_head': args.unfreeze_head,
            'kfolds': args.kfolds,
        })
        mean_fpr = np.linspace(0, 1, 100)
        tprs = []
        aucs = []
        best_auc = 0.0
        best_wts = None
        best_model_idx = -1
        best_fpr = None
        best_tpr = None
        best_fold_probs = None
        best_fold_labels = None
        for fold, (train_idx, val_idx) in enumerate(skf.split(img_paths, labels)):
            print(f"Fold {fold+1}/{args.kfolds}")
            # Preparar datasets
            train_imgs = img_paths[train_idx]
            train_lbls = labels[train_idx]
            val_imgs = img_paths[val_idx]
            val_lbls = labels[val_idx]
            # Custom dataset
            class SimpleDataset(torch.utils.data.Dataset):
                def __init__(self, img_paths, labels, transform):
                    self.img_paths = img_paths
                    self.labels = labels
                    self.transform = transform
                def __len__(self):
                    return len(self.img_paths)
                def __getitem__(self, idx):
                    from PIL import Image
                    img = Image.open(self.img_paths[idx]).convert('RGB')
                    img = self.transform(img)
                    label = self.labels[idx]
                    return img, label
            transforms_map = get_transforms()
            train_ds = SimpleDataset(train_imgs, train_lbls, transforms_map['train'])
            val_ds = SimpleDataset(val_imgs, val_lbls, transforms_map['val'])
            pin_memory = True if device.type == 'cuda' else False
            train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin_memory)
            val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory)
            # Modelo
            model = build_model(device, unfreeze_head=args.unfreeze_head)
            params_to_optimize = [p for p in model.parameters() if p.requires_grad]
            optimizer = optim.Adam(params_to_optimize, lr=args.lr)
            criterion = nn.BCEWithLogitsLoss()
            # Treinamento
            for epoch in range(args.epochs):
                model.train()
                for inputs, labels_batch in train_loader:
                    inputs = inputs.to(device)
                    labels_batch = labels_batch.to(device).float().view(-1, 1)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels_batch)
                    loss.backward()
                    optimizer.step()
            # Avaliação
            model.eval()
            val_probs = []
            val_true = []
            with torch.no_grad():
                for inputs, labels_batch in val_loader:
                    inputs = inputs.to(device)
                    labels_batch = labels_batch.to(device).float().view(-1, 1)
                    outputs = model(inputs)
                    probs = torch.sigmoid(outputs).cpu().numpy().ravel()
                    val_probs.extend(probs.tolist())
                    val_true.extend(labels_batch.cpu().numpy().ravel().tolist())
            try:
                fpr, tpr, _ = roc_curve(val_true, val_probs)
                fold_auc = auc(fpr, tpr)
                interp_tpr = np.interp(mean_fpr, fpr, tpr)
                interp_tpr[0] = 0.0
                tprs.append(interp_tpr)
                aucs.append(fold_auc)
                print(f"Fold {fold+1} AUC: {fold_auc:.4f}")
                # Salvar modelo se for o melhor
                if fold_auc > best_auc:
                    best_auc = fold_auc
                    best_wts = copy.deepcopy(model.state_dict())
                    best_model_idx = fold
                    best_fpr = fpr
                    best_tpr = tpr
                    best_fold_probs = val_probs
                    best_fold_labels = val_true
            except Exception:
                print(f"Fold {fold+1} não pôde calcular ROC/AUC.")
        # ROC média e std
        mean_tpr = np.mean(tprs, axis=0)
        std_tpr = np.std(tprs, axis=0)
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        # Plot ROC
        plt.figure(figsize=(10, 8))
        plt.plot(mean_fpr, mean_tpr, color='blue', lw=2, label=f'Média ROC (AUC = {mean_auc:.3f} ± {std_auc:.3f})')
        plt.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr, color='blue', alpha=0.2, label='Desvio padrão')
        # ROC do melhor modelo
        if best_fpr is not None and best_tpr is not None:
            plt.plot(best_fpr, best_tpr, color='red', lw=2, linestyle='--', label=f'Melhor modelo ROC (AUC = {best_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle=':')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Taxa de Falsos Positivos (FPR)')
        plt.ylabel('Taxa de Verdadeiros Positivos (TPR)')
        plt.title('Curva ROC Média (K-Fold) e Melhor Modelo')
        plt.legend(loc="lower right", fontsize=12)
        plt.grid(True)
        roc_path = Path(args.models_dir) / "roc_kfold.png"
        plt.savefig(roc_path)
        plt.close()
        mlflow.log_artifact(str(roc_path))
        # Salvar melhor modelo
        if best_wts is not None:
            best_model_path = Path(args.models_dir) / "coruja_classifier_best.pth"
            torch.save(best_wts, best_model_path)
            mlflow.log_artifact(str(best_model_path))
        mlflow.log_metric('best_val_auc', best_auc)
        mlflow.log_metric('mean_val_auc', mean_auc)
        mlflow.log_metric('std_val_auc', std_auc)
    return best_auc, mean_auc, std_auc


def main():
    args = parse_args()
    train_kfold(args)


if __name__ == '__main__':
    main()
