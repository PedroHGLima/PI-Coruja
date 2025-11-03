import copy
import logging
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import CorujaResNet, transforms_map
from datasets import SimpleDataset, get_image_paths_and_labels


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")


class CorujaTrainer:
    """
    Classe responsável por treinar e avaliar uma CNN com k-fold cross-validation.
    Mantém saída em tanh e rótulos {-1, 1}.
    """

    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Detectar se stdout é terminal → controlar tqdm
        self.use_tqdm = sys.stdout.isatty()

        # Carregar dados e configurar folds
        self.img_paths, self.labels, self.class_names = get_image_paths_and_labels(args.data_dir)
        self.img_paths = np.array(self.img_paths)
        self.labels = np.array(self.labels)
        self.skf = StratifiedKFold(n_splits=args.kfolds, shuffle=True, random_state=42)
        self.mean_fpr = np.linspace(0, 1, 100)

        # Caminho para salvar melhor modelo global (modelo completo .pt)
        model_name = f"{args.run_name}.pt" if args.run_name else "coruja_classifier_best.pt"
        self.model_path = Path(self.args.models_dir) / model_name
        self.model_path.parent.mkdir(parents=True, exist_ok=True)

        # Validar parâmetro start_at
        if hasattr(args, "start_at") and args.start_at > args.kfolds:
            raise ValueError(f"start_at ({args.start_at}) > kfolds ({args.kfolds})")

        # Logs iniciais
        logging.info("Training parameters:")
        for k, v in vars(args).items():
            logging.info(f"  {k}: {v}")
        logging.info(f"Modelos serão salvos em: {self.model_path}")
        logging.info(f"Device: {self.device}")
        if self.device.type == "cuda":
            logging.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # ------------------------
    # Utils
    # ------------------------
    def _make_loader(self, imgs, labels, split: str, shuffle=False):
        """Cria um DataLoader para o split especificado"""
        ds = SimpleDataset(imgs, labels, transforms_map[split])
        nw = self.args.num_workers or 0
        pin = self.device.type == "cuda"
        return DataLoader(ds, batch_size=self.args.batch_size,
                          shuffle=shuffle, num_workers=nw, pin_memory=pin)

    def get_dataloaders(self, train_idx, val_idx):
        """Cria DataLoaders de treino e validação"""
        return (
            self._make_loader(self.img_paths[train_idx], self.labels[train_idx], "train", shuffle=True),
            self._make_loader(self.img_paths[val_idx], self.labels[val_idx], "val")
        )

    def get_val_dataloader(self, val_idx):
        """Cria apenas o DataLoader de validação"""
        return self._make_loader(self.img_paths[val_idx], self.labels[val_idx], "val")

    # ------------------------
    # Treino e avaliação
    # ------------------------
    def train_epoch(self, model, train_loader, optimizer, criterion):
        """Treina o modelo por uma época"""
        model.train()
        for inputs, batch_labels in train_loader:
            inputs = inputs.to(self.device)
            batch_labels = batch_labels.to(self.device).float().view(-1, 1)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(self.device.type == "cuda")):
                outputs = model(inputs)
                loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

    def evaluate_epoch(self, model, val_loader):
        """Avalia modelo em todo o conjunto de validação"""
        model.eval()
        all_probs, all_true = [], []
        with torch.no_grad():
            for inputs, batch_labels in val_loader:
                inputs = inputs.to(self.device)
                batch_labels = batch_labels.to(self.device).float().view(-1, 1)
                outputs = model(inputs)

                # Saída do modelo já está em [-1, 1] devido ao tanh
                scores = outputs.squeeze().cpu().numpy()
                all_probs.extend(scores.tolist())
                all_true.extend(batch_labels.cpu().numpy().ravel().tolist())

        preds = [1 if p >= 0.0 else -1 for p in all_probs]
        acc = accuracy_score(all_true, preds)
        return acc, preds, all_probs, all_true

    def check_early_stopping(self, val_acc_history):
        """Verifica critério de early stopping baseado em acurácia"""
        if len(val_acc_history) > 1:
            if val_acc_history[-1] < max(val_acc_history[:-1]) + self.args.early_stop_delta:
                return True
        return False

    # ------------------------
    # Carregar folds anteriores
    # ------------------------
    def carregar_anteriores(self, start_fold: int):
        """
        Carrega modelos state_dict dos folds anteriores e calcula métricas.
        """
        aucs, tprs = [], []
        best_auc, best_model, best_fpr, best_tpr = float('-inf'), None, None, None

        splits = list(self.skf.split(self.img_paths, self.labels))
        for fold in range(start_fold):
            fold_model_path = Path(self.args.models_dir) / f"fold_{fold+1}" / "model.pth"
            if not fold_model_path.exists():
                logging.warning(f"Fold {fold+1}: modelo não encontrado. Pulando.")
                continue
            try:
                state = torch.load(str(fold_model_path), map_location=self.device)
                model = CorujaResNet(unfreeze_head=self.args.unfreeze_head).to(self.device)
                model.load_state_dict(state)
                model.eval()

                _, val_idx = splits[fold]
                val_loader = self.get_val_dataloader(val_idx)
                val_acc, _, val_probs, val_true = self.evaluate_epoch(model, val_loader)

                fpr, tpr, _ = roc_curve(val_true, val_probs, pos_label=1)
                fold_auc = auc(fpr, tpr)
                aucs.append(fold_auc)
                tprs.append(np.interp(self.mean_fpr, fpr, tpr))
                tprs[-1][0] = 0.0

                logging.info(f"Fold {fold+1} carregado: AUC = {fold_auc:.4f}, ACC = {val_acc:.4f}")
                if fold_auc > best_auc:
                    best_auc, best_model, best_fpr, best_tpr = fold_auc, model, fpr, tpr
            except Exception as e:
                logging.warning(f"Erro ao carregar fold {fold+1}: {e}")

        return aucs, tprs, best_auc, best_model, best_fpr, best_tpr

    # ------------------------
    # Treino por fold
    # ------------------------
    def train_fold(self, train_idx, val_idx, fold):
        """Treina um fold específico"""
        val_acc_history, stop_counter = [], 0
        train_loader, val_loader = self.get_dataloaders(train_idx, val_idx)

        model = CorujaResNet(unfreeze_head=self.args.unfreeze_head).to(self.device)
        params_to_optimize = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.Adam(params_to_optimize, lr=self.args.lr)
        criterion = nn.MSELoss()

        best_auc_fold = float("-inf")
        best_wts_fold = copy.deepcopy(model.state_dict())

        for epoch in tqdm(range(self.args.epochs),
                          desc=f"Fold {fold+1}",
                          leave=False,
                          disable=not self.use_tqdm):
            self.train_epoch(model, train_loader, optimizer, criterion)
            val_acc, _, val_probs, val_true = self.evaluate_epoch(model, val_loader)
            val_acc_history.append(val_acc)

            val_loss = criterion(torch.tensor(val_probs).unsqueeze(1),
                                 torch.tensor(val_true).unsqueeze(1))
            try:
                fpr_arr, tpr_arr, _ = roc_curve(val_true, val_probs, pos_label=1)
                val_auc = auc(fpr_arr, tpr_arr)
            except Exception:
                val_auc = float("nan")

            mlflow.log_metrics({
                f"fold{fold+1}_val_acc": val_acc,
                f"fold{fold+1}_val_loss": val_loss.item(),
                f"fold{fold+1}_val_auc": val_auc,
            }, step=epoch)
            if not self.use_tqdm:
                logging.info(f"Fold {fold+1} Epoch {epoch+1}/{self.args.epochs} - Val Acc: {val_acc:.4f}, Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")

            if val_auc > best_auc_fold:
                best_auc_fold = val_auc
                best_wts_fold = copy.deepcopy(model.state_dict())

            stop_counter = stop_counter + 1 if self.check_early_stopping(val_acc_history) else 0
            if stop_counter >= self.args.early_stop_patience:
                break

        model.load_state_dict(best_wts_fold)

        try:
            _, _, val_probs, val_true = self.evaluate_epoch(model, val_loader)
            fpr, tpr, _ = roc_curve(val_true, val_probs, pos_label=1)
            fold_auc = auc(fpr, tpr)
        except Exception:
            logging.warning(f"Fold {fold+1} não pôde calcular ROC/AUC.")
            fold_auc, fpr, tpr = float("nan"), None, None

        # Salvar apenas state_dict (.pth) por fold
        fold_dir = Path(self.args.models_dir) / f"fold_{fold+1}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        fold_model_path = fold_dir / "model.pth"
        torch.save(model.state_dict(), fold_model_path)
        mlflow.log_artifact(str(fold_model_path))

        return model, fold_auc, fpr, tpr

    # ------------------------
    # Treino K-Fold
    # ------------------------
    def train_kfold(self):
        """Treina o modelo com validação cruzada k-fold"""
        mlflow.set_experiment(self.args.experiment)
        with mlflow.start_run(run_name=self.args.run_name, log_system_metrics=True):
            mlflow.log_params(vars(self.args))
            mlflow.set_tags({
                "model": "CorujaResNet",
                "data_dir": str(self.args.data_dir),
            })

            aucs, tprs = [], []
            best_auc, best_model, best_fpr, best_tpr = float("-inf"), None, None, None
            splits_list = list(self.skf.split(self.img_paths, self.labels))
            start_fold = getattr(self.args, "start_at", 1) - 1

            if start_fold > 0:
                loaded_aucs, loaded_tprs, loaded_best_auc, loaded_best_model, loaded_best_fpr, loaded_best_tpr = self.carregar_anteriores(start_fold)
                aucs.extend(loaded_aucs)
                tprs.extend(loaded_tprs)
                if loaded_best_model is not None and loaded_best_auc > best_auc:
                    best_auc, best_model, best_fpr, best_tpr = loaded_best_auc, loaded_best_model, loaded_best_fpr, loaded_best_tpr

            for fold in range(start_fold, self.args.kfolds):
                train_idx, val_idx = splits_list[fold]
                logging.info(f"\nTreinando fold {fold+1}/{self.args.kfolds}...")
                fold_model, fold_auc, fold_fpr, fold_tpr = self.train_fold(train_idx, val_idx, fold)

                if fold_auc > best_auc:
                    best_auc, best_model, best_fpr, best_tpr = fold_auc, fold_model, fold_fpr, fold_tpr

                aucs.append(fold_auc)
                if fold_fpr is not None and fold_tpr is not None:
                    tprs.append(np.interp(self.mean_fpr, fold_fpr, fold_tpr))
                    tprs[-1][0] = 0.0

            mean_tpr, std_tpr = np.mean(tprs, axis=0), np.std(tprs, axis=0) if tprs else (np.zeros_like(self.mean_fpr), np.zeros_like(self.mean_fpr))
            mean_auc, std_auc = np.mean(aucs), np.std(aucs) if aucs else (float("nan"), float("nan"))

            plt.figure(figsize=(10, 8))
            plt.plot(self.mean_fpr, mean_tpr, color="blue", lw=2,
                     label=f"Média ROC (AUC = {mean_auc:.3f} ± {std_auc:.3f})")
            plt.fill_between(self.mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr,
                             color="blue", alpha=0.2, label="Desvio padrão")

            if best_fpr is not None and best_tpr is not None:
                plt.plot(best_fpr, best_tpr, color="red", lw=2, linestyle="--",
                         label=f"Melhor modelo ROC (AUC = {best_auc:.3f})")
                target_tpr = 0.98
                idx = np.argmin(np.abs(best_tpr - target_tpr))
                plt.plot(best_fpr[idx], best_tpr[idx], 'go')  # ponto verde
                plt.text(best_fpr[idx], best_tpr[idx],
                         f"({best_fpr[idx]:.2%}, {best_tpr[idx]:.2%})",
                         fontsize=10, verticalalignment='bottom', horizontalalignment='right')

            plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle=":")
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])                
            plt.xlabel("Taxa de Falsos Positivos (FPR)")
            plt.ylabel("Taxa de Verdadeiros Positivos (TPR)")
            plt.title("Curva ROC Média (K-Fold) e Melhor Modelo")
            plt.legend(loc="lower right", fontsize=12)
            plt.grid(True)

            roc_path = Path(self.args.models_dir) / "roc_kfold.png"
            plt.savefig(roc_path)
            plt.close()
            mlflow.log_artifact(str(roc_path))

            if best_model is not None:
                # Melhor modelo global salvo como objeto completo (.pt)
                torch.save(best_model, str(self.model_path))
                mlflow.log_artifact(str(self.model_path))

            metrics_path = Path(self.args.models_dir) / "fold_metrics.json"
            with open(metrics_path, "w") as f:
                json.dump({"aucs": aucs}, f)
            mlflow.log_artifact(str(metrics_path))

            mlflow.log_metrics({
                "best_val_auc": best_auc,
                "mean_val_auc": mean_auc,
                "std_val_auc": std_auc,
            })

        return best_auc, mean_auc, std_auc
