import copy
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
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.transforms_map = transforms_map
        self.img_paths, self.labels, self.class_names = get_image_paths_and_labels(
            args.data_dir)
        self.img_paths = np.array(self.img_paths)
        self.labels = np.array(self.labels)
        self.skf = StratifiedKFold(
            n_splits=args.kfolds, shuffle=True, random_state=42)
        self.mean_fpr = np.linspace(0, 1, 100)

        self.model_path = Path(self.args.models_dir) / (self.args.run_name +
                                                        ".pt" if self.args.run_name else "coruja_classifier_best.pt")

        # Validar start_at
        if hasattr(args, 'start_at') and args.start_at > args.kfolds:
            raise ValueError(
                f"start_at ({args.start_at}) não pode ser maior que kfolds ({args.kfolds})")

        # print parameters
        print("Training parameters:")
        print(vars(args))
        print("Salvando modelo em: ", str(self.model_path))
        print(f"Using device: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    def get_dataloaders(self, train_idx, val_idx):
        train_imgs = self.img_paths[train_idx]
        train_lbls = self.labels[train_idx]
        val_imgs = self.img_paths[val_idx]
        val_lbls = self.labels[val_idx]
        train_ds = SimpleDataset(
            train_imgs, train_lbls, self.transforms_map['train'])
        val_ds = SimpleDataset(val_imgs, val_lbls, self.transforms_map['val'])
        pin_memory = True if self.device.type == 'cuda' else False
        nw = self.args.num_workers if self.args.num_workers is not None else 0
        train_loader = DataLoader(train_ds, batch_size=self.args.batch_size,
                                  shuffle=True, num_workers=nw, pin_memory=pin_memory)
        val_loader = DataLoader(val_ds, batch_size=self.args.batch_size,
                                shuffle=False, num_workers=nw, pin_memory=pin_memory)
        return train_loader, val_loader

    def get_val_dataloader(self, val_idx):
        """Cria apenas o dataloader de validação"""
        val_imgs = self.img_paths[val_idx]
        val_lbls = self.labels[val_idx]
        val_ds = SimpleDataset(val_imgs, val_lbls, self.transforms_map['val'])
        pin_memory = True if self.device.type == 'cuda' else False
        nw = self.args.num_workers if self.args.num_workers is not None else 0
        val_loader = DataLoader(val_ds, batch_size=self.args.batch_size,
                                shuffle=False, num_workers=nw, pin_memory=pin_memory)
        return val_loader

    def train_epoch(self, model, train_loader, optimizer, criterion):
        model.train()
        for inputs, labels_batch in train_loader:
            inputs = inputs.to(self.device)
            # Labels já em {-1, 1}
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
                # Saída do modelo já está em [-1, 1] devido ao tanh
                scores = outputs.cpu().numpy().ravel().tolist()
                val_probs.extend(scores)
                val_true.extend(labels_batch.cpu().numpy().ravel().tolist())
        # Converter para rótulos binários usando threshold 0
        val_preds = [1 if p >= 0.0 else -1 for p in val_probs]
        val_acc = accuracy_score(val_true, val_preds)
        return val_acc, val_preds, val_probs, val_true

    def check_early_stopping(self, val_acc_history):
        if len(val_acc_history) > 1:
            if val_acc_history[-1] < max(val_acc_history[:-1]) + self.args.early_stop_delta:
                return True
        return False

    def load_existing_fold_metrics(self, fold):
        """
        Carrega um modelo de fold existente e calcula suas métricas (TPR, FPR, AUC)
        """
        fold_model_path = self.model_path.with_name(f"model_fold{fold+1}.pt")
        if not fold_model_path.exists():
            return None, None, None

        try:
            print(f"Carregando modelo existente: {fold_model_path}")
            # Usar weights_only=False para permitir carregamento de modelos personalizados
            model = torch.load(
                fold_model_path, map_location=self.device, weights_only=False)
            model.eval()

            # Obter os índices de validação para este fold
            splits = list(self.skf.split(self.img_paths, self.labels))
            if fold >= len(splits):
                print(
                    f"Fold {fold+1} não existe nos splits atuais (total: {len(splits)})")
                return None, None, None

            _, val_idx = splits[fold]

            if len(val_idx) == 0:
                print(f"Fold {fold+1} não tem dados de validação")
                return None, None, None

            # Criar apenas o dataloader de validação
            val_loader = self.get_val_dataloader(val_idx)

            # Verificar se o dataloader tem dados
            if len(val_loader.dataset) == 0:
                print(f"Fold {fold+1}: Dataset de validação vazio")
                return None, None, None

            # Avaliar o modelo
            val_acc, val_preds, val_probs, val_true = self.evaluate_epoch(
                model, val_loader)

            # Verificar se temos dados suficientes para ROC
            if len(val_true) < 2:
                print(f"Fold {fold+1}: Dados insuficientes para calcular ROC")
                return None, None, None

            # Calcular ROC/AUC
            fpr, tpr, _ = roc_curve(val_true, val_probs, pos_label=1)
            fold_auc = auc(fpr, tpr)

            print(
                f"Fold {fold+1} carregado: AUC = {fold_auc:.4f}, ACC = {val_acc:.4f}")
            return fold_auc, fpr, tpr

        except Exception as e:
            print(f"Erro ao carregar modelo do fold {fold+1}: {e}")
            return None, None, None

    def train_fold(self, train_idx, val_idx, fold):
        val_acc_history = []
        stop_counter = 0
        train_loader, val_loader = self.get_dataloaders(train_idx, val_idx)
        model = CorujaResNet(
            unfreeze_head=self.args.unfreeze_head).to(self.device)
        params_to_optimize = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.Adam(params_to_optimize, lr=self.args.lr)
        # Como os alvos estão em {-1,1} e a saída em tanh, usamos MSELoss
        criterion = nn.MSELoss()
        # Resetar controle do melhor dentro do fold
        best_auc_fold = float('-inf')
        best_wts_fold = copy.deepcopy(model.state_dict())
        for epoch in tqdm(range(self.args.epochs), desc=f"Fold {fold+1}", leave=False):
            self.train_epoch(model, train_loader, optimizer, criterion)
            val_acc, val_preds, val_probs, val_true = self.evaluate_epoch(
                model, val_loader)
            val_acc_history.append(val_acc)
            # Calcular loss de validação em espaço [-1,1]
            val_loss = nn.MSELoss()(torch.tensor(val_probs).unsqueeze(1),
                                    torch.tensor(val_true).unsqueeze(1))
            # Calcular AUC em labels {-1,1} com pos_label=1
            try:
                fpr_arr, tpr_arr, _ = roc_curve(
                    val_true, val_probs, pos_label=1)
                val_auc = auc(fpr_arr, tpr_arr)
            except Exception:
                val_auc = float('nan')
            mlflow.log_metric(f"fold{fold+1}_val_acc", val_acc, step=epoch)
            mlflow.log_metric(f"fold{fold+1}_val_loss",
                              val_loss.item(), step=epoch)
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
        # Avaliação final do fold usando os melhores pesos do fold
        try:
            _, _, val_probs, val_true = self.evaluate_epoch(model, val_loader)
            fpr, tpr, _ = roc_curve(val_true, val_probs, pos_label=1)
            fold_auc = auc(fpr, tpr)
        except Exception:
            print(f"Fold {fold+1} não pôde calcular ROC/AUC.")
        fold_name = self.model_path.with_name(f"model_fold{fold+1}.pt")
        torch.save(model, fold_name)
        mlflow.log_artifact(str(fold_name))

        return model, fold_auc, fpr, tpr

    def carregar_anteriores(self, start_fold: int):
        aucs = []
        tprs = []
        best_auc = float('-inf')
        best_model = None
        best_fpr = None
        best_tpr = None

        for fold in range(start_fold):
            fold_auc, fold_fpr, fold_tpr = self.load_existing_fold_metrics(
                fold)
            if fold_auc is not None:
                aucs.append(fold_auc)
                tprs.append(np.interp(self.mean_fpr, fold_fpr, fold_tpr))
                tprs[-1][0] = 0.0

                # Verificar se é o melhor modelo até agora
                if fold_auc > best_auc:
                    best_auc = fold_auc
                    # Carregar o modelo para ser o melhor
                    fold_model_path = self.model_path.with_name(
                        f"model_fold{fold+1}.pt")
                    try:
                        best_model = torch.load(
                            fold_model_path, map_location=self.device, weights_only=False)
                        best_fpr = fold_fpr
                        best_tpr = fold_tpr
                    except Exception as e:
                        print(
                            f"Erro ao carregar melhor modelo do fold {fold+1}: {e}")
            else:
                print(
                    f"Aviso: Não foi possível carregar o fold {fold+1}. Continuando...")

        print(
            f"Carregados {len(aucs)} folds anteriores. Melhor AUC até agora: {best_auc:.4f}")
        return aucs, tprs, best_auc, best_model, best_fpr, best_tpr

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
                'start_at': getattr(self.args, 'start_at', 1),
            })
            start_fold = getattr(self.args, 'start_at', 1) - 1
            if start_fold > 0:
                aucs, tprs, best_auc, best_model, best_fpr, best_tpr = self.carregar_anteriores(
                    start_fold)
            else:
                tprs = []
                aucs = []
                best_auc = float('-inf')
                best_model = None
                best_fpr = None
                best_tpr = None

            # Treinar folds restantes
            splits_list = list(self.skf.split(self.img_paths, self.labels))
            for fold in range(start_fold, self.args.kfolds):
                train_idx, val_idx = splits_list[fold]
                print(f"\nTreinando fold {fold+1}/{self.args.kfolds}...")
                fold_model, fold_auc, fold_fpr, fold_tpr = self.train_fold(
                    train_idx, val_idx, fold)

                if fold_auc > best_auc:
                    best_auc = fold_auc
                    best_model = fold_model
                    best_fpr = fold_fpr
                    best_tpr = fold_tpr

                aucs.append(fold_auc)
                tprs.append(np.interp(self.mean_fpr, fold_fpr, fold_tpr))
                tprs[-1][0] = 0.0

            # ROC média e std
            mean_tpr = np.mean(tprs, axis=0)
            std_tpr = np.std(tprs, axis=0)
            mean_auc = np.mean(aucs)
            std_auc = np.std(aucs)
            plt.figure(figsize=(10, 8))
            plt.plot(self.mean_fpr, mean_tpr, color='blue', lw=2,
                     label=f'Média ROC (AUC = {mean_auc:.3f} ± {std_auc:.3f})')
            plt.fill_between(self.mean_fpr, mean_tpr - std_tpr, mean_tpr +
                             std_tpr, color='blue', alpha=0.2, label='Desvio padrão')
            if best_fpr is not None and best_tpr is not None:
                plt.plot(best_fpr, best_tpr, color='red', lw=2, linestyle='--',
                         label=f'Melhor modelo ROC (AUC = {best_auc:.3f})')
                # Encontrar FPR quando TPR ~ 0.98
                tpr_array = np.array(best_tpr)
                fpr_array = np.array(best_fpr)
                idx_98 = np.argmin(np.abs(tpr_array - 0.98))
                fpr_at_98 = fpr_array[idx_98]
                tpr_at_98 = tpr_array[idx_98]
                plt.scatter(fpr_at_98, tpr_at_98, color='black',
                            zorder=5, label=f'FPR={fpr_at_98:.3f} @ TPR=0.98')
                plt.annotate(f'FPR={fpr_at_98:.3f}', (fpr_at_98, tpr_at_98), textcoords="offset points", xytext=(
                    10, -20), ha='center', color='black', fontsize=10, arrowprops=dict(arrowstyle='->', color='black'))
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
            if best_model is not None:
                # Garantir que salvamos os melhores pesos globais
                torch.save(best_model, str(self.model_path))
                mlflow.log_artifact(str(self.model_path))
            mlflow.log_metric('best_val_auc', best_auc)
            mlflow.log_metric('mean_val_auc', mean_auc)
            mlflow.log_metric('std_val_auc', std_auc)
        return best_auc, mean_auc, std_auc
