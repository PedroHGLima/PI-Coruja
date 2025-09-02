import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc, accuracy_score
import mlflow
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from models import CorujaResNet, transforms_map
from datasets import SimpleDataset, get_image_paths_and_labels
from sklearn.model_selection import StratifiedKFold

class CorujaTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transforms_map = transforms_map
        self.img_paths, self.labels, self.class_names = get_image_paths_and_labels(args.data_dir)
        self.img_paths = np.array(self.img_paths)
        self.labels = np.array(self.labels)
        self.skf = StratifiedKFold(n_splits=args.kfolds, shuffle=True, random_state=42)
        self.mean_fpr = np.linspace(0, 1, 100)

    def get_dataloaders(self, train_idx, val_idx):
        train_imgs = self.img_paths[train_idx]
        train_lbls = self.labels[train_idx]
        val_imgs = self.img_paths[val_idx]
        val_lbls = self.labels[val_idx]
        train_ds = SimpleDataset(train_imgs, train_lbls, self.transforms_map['train'])
        val_ds = SimpleDataset(val_imgs, val_lbls, self.transforms_map['val'])
        pin_memory = True if self.device.type == 'cuda' else False
        nw = self.args.num_workers if self.args.num_workers is not None else 0
        train_loader = DataLoader(train_ds, batch_size=self.args.batch_size, shuffle=True, num_workers=nw, pin_memory=pin_memory)
        val_loader = DataLoader(val_ds, batch_size=self.args.batch_size, shuffle=False, num_workers=nw, pin_memory=pin_memory)
        return train_loader, val_loader

    def train_epoch(self, model, train_loader, optimizer, criterion):
        model.train()
        for inputs, labels_batch in train_loader:
            inputs = inputs.to(self.device)
            labels_batch = labels_batch.to(self.device).float().view(-1, 1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()

    def evaluate_epoch(self, model, val_loader):
        model.eval()
        val_probs = []
        val_true = []
        with torch.no_grad():
            for inputs, labels_batch in val_loader:
                inputs = inputs.to(self.device)
                labels_batch = labels_batch.to(self.device).float().view(-1, 1)
                outputs = model(inputs)
                probs = torch.sigmoid(outputs).cpu().numpy().ravel()
                val_probs.extend(probs.tolist())
                val_true.extend(labels_batch.cpu().numpy().ravel().tolist())
        val_preds = [1 if p > 0.5 else 0 for p in val_probs]
        val_acc = accuracy_score(val_true, val_preds)
        return val_acc, val_preds, val_probs, val_true

    def check_early_stopping(self, val_acc_history):
        if len(val_acc_history) > 1:
            if val_acc_history[-1] < max(val_acc_history[:-1]) + self.args.early_stop_delta:
                return True
        return False

    def train_kfold(self):
        mlflow.set_experiment(self.args.experiment)
        with mlflow.start_run(run_name=self.args.run_name, log_system_metrics=True):
            mlflow.log_params({
                'batch_size': self.args.batch_size,
                'epochs': self.args.epochs,
                'lr': self.args.lr,
                'device': self.device.type,
                'num_workers': self.args.num_workers,
                'unfreeze_head': self.args.unfreeze_head,
                'kfolds': self.args.kfolds,
            })
            tprs = []
            aucs = []
            best_auc = 0.0
            best_auc_fold = 0.0
            best_wts = None
            best_model_idx = -1
            best_fpr = None
            best_tpr = None
            best_fold_probs = None
            best_fold_labels = None
            for fold, (train_idx, val_idx) in enumerate(tqdm(list(self.skf.split(self.img_paths, self.labels)))):
                val_acc_history = []
                stop_counter = 0
                train_loader, val_loader = self.get_dataloaders(train_idx, val_idx)
                model = CorujaResNet(unfreeze_head=self.args.unfreeze_head).to(self.device)
                params_to_optimize = [p for p in model.parameters() if p.requires_grad]
                optimizer = optim.Adam(params_to_optimize, lr=self.args.lr)
                criterion = nn.BCEWithLogitsLoss()
                for epoch in tqdm(range(self.args.epochs), desc=f"Fold {fold+1}", leave=False):
                    start_time = time.time()
                    self.train_epoch(model, train_loader, optimizer, criterion)
                    val_acc, val_preds, val_probs, val_true = self.evaluate_epoch(model, val_loader)
                    val_acc_history.append(val_acc)
                    # Calcular loss
                    val_loss = nn.BCEWithLogitsLoss()(torch.tensor(val_probs).unsqueeze(1), torch.tensor(val_true).unsqueeze(1))
                    # Calcular AUC
                    try:
                        val_auc = auc(*roc_curve(val_true, val_probs)[:2])
                    except Exception:
                        val_auc = float('nan')
                    mlflow.log_metric(f"fold{fold+1}_val_acc", val_acc, step=epoch)
                    mlflow.log_metric(f"fold{fold+1}_val_loss", val_loss.item(), step=epoch)
                    mlflow.log_metric(f"fold{fold+1}_val_auc", val_auc, step=epoch)
                    if val_auc > best_auc_fold:
                        best_auc_fold = val_auc
                        best_wts_fold = copy.deepcopy(model.state_dict())
                    if self.check_early_stopping(val_acc_history):
                        stop_counter += 1
                    else:
                        stop_counter = 0
                    if stop_counter >= self.args.early_stop_patience:
                        # print(f"Fold {fold+1}: Parando treinamento por estagnação de acurácia (val_acc={val_acc:.4f})")
                        break
                model.load_state_dict(best_wts_fold)
                # Avaliação final do fold
                try:
                    fpr, tpr, _ = roc_curve(val_true, val_probs)
                    fold_auc = auc(fpr, tpr)
                    interp_tpr = np.interp(self.mean_fpr, fpr, tpr)
                    interp_tpr[0] = 0.0
                    tprs.append(interp_tpr)
                    aucs.append(fold_auc)
                    # print(f"Fold {fold+1} AUC: {fold_auc:.4f}")
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
            plt.figure(figsize=(10, 8))
            plt.plot(self.mean_fpr, mean_tpr, color='blue', lw=2, label=f'Média ROC (AUC = {mean_auc:.3f} ± {std_auc:.3f})')
            plt.fill_between(self.mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr, color='blue', alpha=0.2, label='Desvio padrão')
            if best_fpr is not None and best_tpr is not None:
                plt.plot(best_fpr, best_tpr, color='red', lw=2, linestyle='--', label=f'Melhor modelo ROC (AUC = {best_auc:.3f})')
                # Encontrar FPR quando TPR ~ 0.98
                tpr_array = np.array(best_tpr)
                fpr_array = np.array(best_fpr)
                idx_98 = np.argmin(np.abs(tpr_array - 0.98))
                fpr_at_98 = fpr_array[idx_98]
                tpr_at_98 = tpr_array[idx_98]
                plt.scatter(fpr_at_98, tpr_at_98, color='black', zorder=5, label=f'FPR={fpr_at_98:.3f} @ TPR=0.98')
                plt.annotate(f'FPR={fpr_at_98:.3f}', (fpr_at_98, tpr_at_98), textcoords="offset points", xytext=(10,-20), ha='center', color='black', fontsize=10, arrowprops=dict(arrowstyle='->', color='black'))
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle=':')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Taxa de Falsos Positivos (FPR)')
            plt.ylabel('Taxa de Verdadeiros Positivos (TPR)')
            plt.title('Curva ROC Média (K-Fold) e Melhor Modelo')
            plt.legend(loc="lower right", fontsize=12)
            plt.grid(True)
            roc_path = Path(self.args.models_dir) / "roc_kfold.png"
            plt.savefig(roc_path)
            plt.close()
            mlflow.log_artifact(str(roc_path))
            if best_wts is not None:
                best_model_path = Path(self.args.models_dir) / (self.args.run_name+".pt" if self.args.run_name else "coruja_classifier_best.pt")
                torch.save(model, str(best_model_path))
                mlflow.log_artifact(str(best_model_path))
            mlflow.log_metric('best_val_auc', best_auc)
            mlflow.log_metric('mean_val_auc', mean_auc)
            mlflow.log_metric('std_val_auc', std_auc)
        return best_auc, mean_auc, std_auc
