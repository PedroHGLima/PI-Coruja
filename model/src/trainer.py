import copy
import logging
import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_curve, auc, accuracy_score, hamming_loss, f1_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import CorujaResNet, transforms_map
from datasets import SimpleDataset, get_image_paths_and_labels

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")


def set_seed(seed: int = 42):
    """
    Configura todas as seeds para garantir reprodutibilidade.
    
    Args:
        seed: Valor da seed (default=42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Para multi-GPU
    
    # Configurações adicionais para garantir determinismo
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Para operações do DataLoader
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)


class CorujaTrainer:
    """
    Classe responsável por treinar e avaliar uma CNN com k-fold cross-validation.
    Suporta classificação multi-label para 3 classes: human, animal, vehicle.
    """

    def __init__(self, args):
        self.args = args
        
        # Configurar seed para reprodutibilidade
        seed = getattr(args, 'seed', 42)
        set_seed(seed)
        logging.info(f"Seed configurada: {seed}")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Detectar se stdout é terminal → controlar tqdm
        self.use_tqdm = sys.stdout.isatty()

        # Carregar dados e configurar folds
        self.img_paths, self.labels, self.class_names = get_image_paths_and_labels(args.data_dir)
        self.img_paths = np.array(self.img_paths)
        self.labels = np.array(self.labels)
        
        # Para StratifiedKFold em multi-label, usamos a classe mais prevalente ou soma
        # Vamos usar uma estratégia simples: criar uma string única para cada combinação
        stratify_labels = [''.join(map(str, label)) for label in self.labels]
        self.skf = StratifiedKFold(n_splits=args.kfolds, shuffle=True, random_state=seed)
        self.stratify_labels = np.array(stratify_labels)
        
        self.mean_fpr = np.linspace(0, 1, 100)
        self.num_classes = len(self.class_names)  # 3 classes

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
        
        if not MLFLOW_AVAILABLE:
            logging.warning("MLflow não está disponível. Treinamento continuará sem logging no MLflow.")

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
            self._make_loader(self.img_paths[train_idx], self.labels[train_idx].tolist(), "train", shuffle=True),
            self._make_loader(self.img_paths[val_idx], self.labels[val_idx].tolist(), "val")
        )

    def get_val_dataloader(self, val_idx):
        """Cria apenas o DataLoader de validação"""
        return self._make_loader(self.img_paths[val_idx], self.labels[val_idx].tolist(), "val")

    # ------------------------
    # Treino e avaliação
    # ------------------------
    def train_epoch(self, model, train_loader, optimizer, criterion):
        """Treina o modelo por uma época (multi-label)"""
        model.train()
        for inputs, batch_labels in train_loader:
            inputs = inputs.to(self.device)
            batch_labels = batch_labels.to(self.device).float()  # Shape: [batch_size, 3], valores em [0, 1]

            optimizer.zero_grad()
            outputs = model(inputs)  # Shape: [batch_size, 3], valores em [0, 1] (sigmoid)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

    def evaluate_epoch(self, model, val_loader):
        """
        Avalia modelo em todo o conjunto de validação (multi-label).
        
        Returns:
            tuple: (hamming_accuracy, all_preds, all_probs, all_true)
        """
        model.eval()
        all_probs = []
        all_true = []
        
        with torch.no_grad():
            for inputs, batch_labels in val_loader:
                inputs = inputs.to(self.device)
                batch_labels = batch_labels.to(self.device).float()  # Shape: [batch_size, 3], valores em [0, 1]
                outputs = model(inputs)  # Shape: [batch_size, 3], valores em [-1, 1] (tanh)
                
                # Converter tanh [-1, 1] para probabilidades [0, 1]
                # tanh: -1 = ausente, +1 = presente
                probs = (outputs + 1) / 2  # Mapeia [-1, 1] -> [0, 1]
                
                all_probs.extend(probs.cpu().numpy())
                all_true.extend(batch_labels.cpu().numpy())
        
        all_probs = np.array(all_probs)  # Shape: [n_samples, 3]
        all_true = np.array(all_true)    # Shape: [n_samples, 3]
        
        # Threshold 0.5 para decisão binária
        all_preds = (all_probs >= 0.5).astype(int)
        
        # Hamming accuracy (1 - hamming_loss)
        hamming_acc = 1 - hamming_loss(all_true, all_preds)
        
        return hamming_acc, all_preds, all_probs, all_true

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
        Carrega modelos state_dict dos folds anteriores e calcula métricas (multi-label).
        """
        all_f1_macros = []
        all_aucs_per_class = {class_name: [] for class_name in self.class_names}
        best_f1_macro = float('-inf')
        best_model = None

        splits = list(self.skf.split(self.img_paths, self.stratify_labels))
        
        for fold in range(start_fold):
            fold_model_path = Path(self.args.models_dir) / f"fold_{fold+1}" / "model.pth"
            fold_metrics_path = Path(self.args.models_dir) / f"fold_{fold+1}" / "metrics.json"
            
            if not fold_model_path.exists():
                logging.warning(f"Fold {fold+1}: modelo não encontrado. Pulando.")
                continue
            
            try:
                # Carrega modelo
                state = torch.load(str(fold_model_path), map_location=self.device)
                model = CorujaResNet(unfreeze_head=self.args.unfreeze_head).to(self.device)
                model.load_state_dict(state)
                model.eval()

                # Carrega métricas salvas se existirem
                if fold_metrics_path.exists():
                    with open(fold_metrics_path, 'r') as f:
                        fold_metrics = json.load(f)
                    
                    f1_macro = fold_metrics.get("f1_macro", 0)
                    aucs_per_class = fold_metrics.get("aucs_per_class", {})
                else:
                    # Recalcula métricas
                    _, val_idx = splits[fold]
                    val_loader = self.get_val_dataloader(val_idx)
                    _, val_preds, val_probs, val_true = self.evaluate_epoch(model, val_loader)
                    
                    f1_macro = f1_score(val_true, val_preds, average='macro', zero_division=0)
                    aucs_per_class = {}
                    for i, class_name in enumerate(self.class_names):
                        try:
                            fpr, tpr, _ = roc_curve(val_true[:, i], val_probs[:, i])
                            aucs_per_class[class_name] = auc(fpr, tpr)
                        except:
                            aucs_per_class[class_name] = float("nan")
                
                all_f1_macros.append(f1_macro)
                for class_name in self.class_names:
                    all_aucs_per_class[class_name].append(aucs_per_class.get(class_name, float("nan")))

                logging.info(f"Fold {fold+1} carregado: F1 Macro = {f1_macro:.4f}")
                
                if f1_macro > best_f1_macro:
                    best_f1_macro = f1_macro
                    best_model = model
                    
            except Exception as e:
                logging.warning(f"Erro ao carregar fold {fold+1}: {e}")

        return all_f1_macros, all_aucs_per_class, best_f1_macro, best_model

    # ------------------------
    # Treino por fold
    # ------------------------
    def train_fold(self, train_idx, val_idx, fold):
        """Treina um fold específico (multi-label)"""
        val_acc_history, stop_counter = [], 0
        train_loader, val_loader = self.get_dataloaders(train_idx, val_idx)

        model = CorujaResNet(unfreeze_head=self.args.unfreeze_head).to(self.device)
        params_to_optimize = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.Adam(params_to_optimize, lr=self.args.lr)
        
        # Multi-label com sigmoid: BCELoss é apropriado para saída em [0, 1]
        criterion = nn.BCELoss()

        best_hamming_acc = float("-inf")
        best_f1_macro = float("-inf")
        best_wts_fold = copy.deepcopy(model.state_dict())

        for epoch in tqdm(range(self.args.epochs),
                          desc=f"Fold {fold+1}",
                          leave=False,
                          disable=not self.use_tqdm):
            self.train_epoch(model, train_loader, optimizer, criterion)
            val_hamming_acc, val_preds, val_probs, val_true = self.evaluate_epoch(model, val_loader)
            val_acc_history.append(val_hamming_acc)

            # Calcula F1 macro para multi-label
            val_f1_macro = f1_score(val_true, val_preds, average='macro', zero_division=0)
            val_f1_micro = f1_score(val_true, val_preds, average='micro', zero_division=0)
            
            # Calcula F1 por classe
            f1_per_class = f1_score(val_true, val_preds, average=None, zero_division=0)

            if MLFLOW_AVAILABLE:
                mlflow.log_metrics({
                    f"fold{fold+1}_val_hamming_acc": float(val_hamming_acc),
                    f"fold{fold+1}_val_f1_macro": float(val_f1_macro),
                    f"fold{fold+1}_val_f1_micro": float(val_f1_micro),
                    f"fold{fold+1}_val_f1_human": float(f1_per_class[0]) if len(f1_per_class) > 0 else 0.0,
                    f"fold{fold+1}_val_f1_animal": float(f1_per_class[1]) if len(f1_per_class) > 1 else 0.0,
                    f"fold{fold+1}_val_f1_vehicle": float(f1_per_class[2]) if len(f1_per_class) > 2 else 0.0,
                }, step=epoch)
            
            if not self.use_tqdm:
                logging.info(
                    f"Fold {fold+1} Epoch {epoch+1}/{self.args.epochs} - "
                    f"Hamming Acc: {val_hamming_acc:.4f}, F1 Macro: {val_f1_macro:.4f}"
                )

            # Salva melhor modelo baseado no F1 macro
            if val_f1_macro > best_f1_macro:
                best_f1_macro = val_f1_macro
                best_hamming_acc = val_hamming_acc
                best_wts_fold = copy.deepcopy(model.state_dict())

            stop_counter = stop_counter + 1 if self.check_early_stopping(val_acc_history) else 0
            if stop_counter >= self.args.early_stop_patience:
                break

        model.load_state_dict(best_wts_fold)

        # Avaliação final do fold
        _, val_preds, val_probs, val_true = self.evaluate_epoch(model, val_loader)
        final_f1_macro = f1_score(val_true, val_preds, average='macro', zero_division=0)
        
        # Calcula ROC AUC por classe (One-vs-Rest)
        aucs_per_class = {}
        fprs_per_class = {}
        tprs_per_class = {}
        
        for i, class_name in enumerate(self.class_names):
            try:
                fpr, tpr, _ = roc_curve(val_true[:, i], val_probs[:, i])
                class_auc = auc(fpr, tpr)
                aucs_per_class[class_name] = class_auc
                fprs_per_class[class_name] = fpr
                tprs_per_class[class_name] = tpr
            except Exception as e:
                logging.warning(f"Fold {fold+1} não pôde calcular ROC/AUC para {class_name}: {e}")
                aucs_per_class[class_name] = float("nan")

        # Salvar apenas state_dict (.pth) por fold
        fold_dir = Path(self.args.models_dir) / f"fold_{fold+1}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        fold_model_path = fold_dir / "model.pth"
        torch.save(model.state_dict(), fold_model_path)
        if MLFLOW_AVAILABLE:
            mlflow.log_artifact(str(fold_model_path))
        
        # Salva métricas do fold
        fold_metrics = {
            "f1_macro": float(final_f1_macro),
            "hamming_acc": float(best_hamming_acc),
            "aucs_per_class": {k: float(v) for k, v in aucs_per_class.items()}
        }
        fold_metrics_path = fold_dir / "metrics.json"
        with open(fold_metrics_path, 'w') as f:
            json.dump(fold_metrics, f, indent=2)
        if MLFLOW_AVAILABLE:
            mlflow.log_artifact(str(fold_metrics_path))

        return model, aucs_per_class, fprs_per_class, tprs_per_class, final_f1_macro

    # ------------------------
    # Treino K-Fold
    # ------------------------
    def train_kfold(self):
        """Treina o modelo com validação cruzada k-fold (multi-label)"""
        if MLFLOW_AVAILABLE:
            mlflow.set_experiment(self.args.experiment)
            mlflow.start_run(run_name=self.args.run_name, log_system_metrics=True)
            mlflow.log_params(vars(self.args))
            mlflow.set_tags({
                "model": "CorujaResNet",
                "data_dir": str(self.args.data_dir),
                "num_classes": self.num_classes,
                "classes": ', '.join(self.class_names)
            })
        
        try:

            all_f1_macros = []
            all_aucs_per_class = {class_name: [] for class_name in self.class_names}
            best_f1_macro = float("-inf")
            best_model = None
            
            splits_list = list(self.skf.split(self.img_paths, self.stratify_labels))
            start_fold = getattr(self.args, "start_at", 1) - 1

            # Carrega folds anteriores se necessário
            if start_fold > 0:
                loaded_f1s, loaded_aucs, loaded_best_f1, loaded_best_model = self.carregar_anteriores(start_fold)
                all_f1_macros.extend(loaded_f1s)
                for class_name in self.class_names:
                    all_aucs_per_class[class_name].extend(loaded_aucs[class_name])
                if loaded_best_model is not None and loaded_best_f1 > best_f1_macro:
                    best_f1_macro = loaded_best_f1
                    best_model = loaded_best_model

            # Treina os folds
            for fold in range(start_fold, self.args.kfolds):
                train_idx, val_idx = splits_list[fold]
                logging.info(f"\nTreinando fold {fold+1}/{self.args.kfolds}...")
                
                fold_model, aucs_per_class, fprs_per_class, tprs_per_class, f1_macro = \
                    self.train_fold(train_idx, val_idx, fold)

                all_f1_macros.append(f1_macro)
                for class_name in self.class_names:
                    all_aucs_per_class[class_name].append(aucs_per_class.get(class_name, float("nan")))

                if f1_macro > best_f1_macro:
                    best_f1_macro = f1_macro
                    best_model = fold_model

            # Calcula estatísticas finais
            mean_f1_macro = np.nanmean(all_f1_macros) if all_f1_macros else float("nan")
            std_f1_macro = np.nanstd(all_f1_macros) if all_f1_macros else float("nan")
            
            mean_aucs_per_class = {}
            std_aucs_per_class = {}
            for class_name in self.class_names:
                aucs = all_aucs_per_class[class_name]
                mean_aucs_per_class[class_name] = np.nanmean(aucs) if aucs else float("nan")
                std_aucs_per_class[class_name] = np.nanstd(aucs) if aucs else float("nan")

            # Plota curvas ROC de todas as classes no mesmo gráfico
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            
            # Coleta todas as curvas ROC de todos os folds
            all_tprs_per_class = {class_name: [] for class_name in self.class_names}
            
            # Re-calcular curvas ROC de todos os folds para plotagem
            for fold in range(self.args.kfolds):
                train_idx, val_idx = splits_list[fold]
                fold_model_path = Path(self.args.models_dir) / f"fold_{fold+1}" / "model.pth"
                
                if fold_model_path.exists():
                    try:
                        state = torch.load(str(fold_model_path), map_location=self.device)
                        model = CorujaResNet(unfreeze_head=self.args.unfreeze_head).to(self.device)
                        model.load_state_dict(state)
                        model.eval()
                        
                        val_loader = self.get_val_dataloader(val_idx)
                        _, _, val_probs, val_true = self.evaluate_epoch(model, val_loader)
                        
                        for i, class_name in enumerate(self.class_names):
                            try:
                                fpr, tpr, _ = roc_curve(val_true[:, i], val_probs[:, i])
                                # Interpolar TPR nos FPRs médios
                                interp_tpr = np.interp(self.mean_fpr, fpr, tpr)
                                interp_tpr[0] = 0.0
                                all_tprs_per_class[class_name].append(interp_tpr)
                            except:
                                pass
                    except Exception as e:
                        logging.warning(f"Erro ao recarregar fold {fold+1} para plotagem: {e}")
            
            # Cores para cada classe
            colors = {'human': '#1f77b4', 'animal': '#ff7f0e', 'vehicle': '#2ca02c'}
            
            for idx, class_name in enumerate(self.class_names):
                mean_auc = mean_aucs_per_class[class_name]
                std_auc = std_aucs_per_class[class_name]
                color = colors.get(class_name, f'C{idx}')
                
                # Plota curvas individuais dos folds (opcional, comentado para não poluir)
                tprs = all_tprs_per_class[class_name]
                # for i, tpr in enumerate(tprs):
                #     ax.plot(self.mean_fpr, tpr, lw=1, alpha=0.15, color=color)
                
                # Plota curva ROC média
                if tprs:
                    mean_tpr = np.mean(tprs, axis=0)
                    mean_tpr[-1] = 1.0
                    ax.plot(self.mean_fpr, mean_tpr, color=color, lw=2.5,
                           label=f'{class_name.capitalize()} (AUC = {mean_auc:.3f} ± {std_auc:.3f})')
                    
                    # Adiciona área de desvio padrão
                    std_tpr = np.std(tprs, axis=0)
                    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
                    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
                    ax.fill_between(self.mean_fpr, tprs_lower, tprs_upper, 
                                   color=color, alpha=0.15)
            
            # Plota linha diagonal de referência
            ax.plot([0, 1], [0, 1], color="gray", lw=2, linestyle="--", label="Aleatório")
            
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel("Taxa de Falsos Positivos (FPR)", fontsize=12)
            ax.set_ylabel("Taxa de Verdadeiros Positivos (TPR)", fontsize=12)
            ax.set_title("Curvas ROC - Classificação Multi-label", fontsize=14, fontweight='bold')
            ax.legend(loc="lower right", fontsize=10)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            roc_path = Path(self.args.models_dir) / "roc_kfold_multiclass.png"
            plt.savefig(roc_path, dpi=150)
            plt.close()
            if MLFLOW_AVAILABLE:
                mlflow.log_artifact(str(roc_path))

            # Salva melhor modelo
            if best_model is not None:
                torch.save(best_model, str(self.model_path))
                if MLFLOW_AVAILABLE:
                    mlflow.log_artifact(str(self.model_path))
                logging.info(f"Melhor modelo salvo em: {self.model_path}")

            # Salva métricas consolidadas
            metrics_summary = {
                "f1_macros": [float(x) for x in all_f1_macros],
                "mean_f1_macro": float(mean_f1_macro),
                "std_f1_macro": float(std_f1_macro),
                "best_f1_macro": float(best_f1_macro),
                "aucs_per_class": {
                    class_name: {
                        "values": [float(x) for x in all_aucs_per_class[class_name]],
                        "mean": float(mean_aucs_per_class[class_name]),
                        "std": float(std_aucs_per_class[class_name])
                    }
                    for class_name in self.class_names
                }
            }
            
            metrics_path = Path(self.args.models_dir) / "kfold_metrics_summary.json"
            with open(metrics_path, "w") as f:
                json.dump(metrics_summary, f, indent=2)
            if MLFLOW_AVAILABLE:
                mlflow.log_artifact(str(metrics_path))

            # Log métricas finais no MLflow
            if MLFLOW_AVAILABLE:
                mlflow.log_metrics({
                    "best_f1_macro": float(best_f1_macro),
                    "mean_f1_macro": float(mean_f1_macro),
                    "std_f1_macro": float(std_f1_macro),
                    **{f"mean_auc_{class_name}": float(mean_aucs_per_class[class_name]) 
                       for class_name in self.class_names}
                })

            logging.info(f"\n{'='*60}")
            logging.info(f"Treinamento K-Fold concluído!")
            logging.info(f"{'='*60}")
            logging.info(f"Melhor F1 Macro: {best_f1_macro:.4f}")
            logging.info(f"F1 Macro médio: {mean_f1_macro:.4f} ± {std_f1_macro:.4f}")
            logging.info(f"\nAUC por classe:")
            for class_name in self.class_names:
                logging.info(f"  {class_name}: {mean_aucs_per_class[class_name]:.4f} ± {std_aucs_per_class[class_name]:.4f}")

        finally:
            if MLFLOW_AVAILABLE:
                mlflow.end_run()

        return best_f1_macro, mean_f1_macro, std_f1_macro
