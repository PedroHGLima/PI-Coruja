#!/usr/bin/env python3
"""
Script para avaliar e comparar o desempenho de múltiplos modelos YOLO
usando validação cruzada estratificada (K-Fold com K=5).
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm

# --- Configurações e Constantes ---
MODELS_DIR = Path("../models")
DATASET_DIR = Path("../data/dataset_10k")
MODEL_NAMES = ['yolov8n.pt', 'yolo11n.pt', 'yolo12n.pt']
ID_CLASSE_PESSOA = 0
N_SPLITS = 5  # Número de folds para a validação cruzada

# ... (a função load_models permanece a mesma) ...
def load_models(names: list[str], models_path: Path) -> dict:
    models = {}
    for name in names:
        model_file = models_path / name
        if not model_file.exists():
            print(f"  Aviso: Modelo '{name}' não encontrado. A biblioteca tentará baixá-lo.")
        try:
            print(f"Carregando {name}...")
            models[name.replace('.pt', '')] = YOLO(model_file)
        except Exception as e:
            print(f"  Erro ao carregar {name}: {e}")
    print("Modelos carregados.\n")
    return models

def evaluate_models_cv(models: dict, base_dir: Path) -> dict:
    """
    Executa a avaliação com validação cruzada estratificada (5-fold).
    """
    human_images = list((base_dir / "human").glob("*.jpg"))
    no_human_images = list((base_dir / "no_human").glob("*.jpg"))
    
    # Prepara os dados X (caminhos das imagens) e y (rótulos)
    X = np.array(human_images + no_human_images)
    y = np.array([1] * len(human_images) + [0] * len(no_human_images))
    
    results_data = {}

    # Itera sobre cada modelo
    for model_name, model in models.items():
        tprs_per_fold = []
        aucs_per_fold = []
        time_taken = []
        mean_fpr = np.linspace(0, 1, 100) # Eixo X comum para todas as curvas
        
        # Inicializa o K-Fold Estratificado
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

        # Itera sobre cada fold
        for fold, (train_idx, test_idx) in enumerate(tqdm(skf.split(X, y), total=N_SPLITS, desc=f"Folds {model_name}")):
            X_test, y_test = X[test_idx], y[test_idx]
            y_scores_fold = []

            # Realiza a predição apenas no conjunto de teste do fold atual
            start = time.time()
            for img_path in tqdm(X_test, leave=False, desc=f"Fold {fold+1}"):
                max_confidence = 0.0
                preds = model(img_path, verbose=False)
                for box in preds[0].boxes:
                    if int(box.cls[0]) == ID_CLASSE_PESSOA:
                        max_confidence = max(max_confidence, float(box.conf[0]))
                y_scores_fold.append(max_confidence)

            # Calcula a curva ROC para este fold
            fpr, tpr, _ = roc_curve(y_test, y_scores_fold)
            aucs_per_fold.append(auc(fpr, tpr))
            
            # Interpola o TPR para o eixo FPR comum
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs_per_fold.append(interp_tpr)
            time_taken.append((time.time() - start)/len(X_test))

        # Calcula as métricas médias e de desvio padrão entre os folds
        mean_tpr = np.mean(tprs_per_fold, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = np.mean(aucs_per_fold)
        std_auc = np.std(aucs_per_fold)
        std_tpr = np.std(tprs_per_fold, axis=0)
        
        results_data[model_name] = {
            'mean_fpr': mean_fpr,
            'mean_tpr': mean_tpr,
            'std_tpr': std_tpr,
            'mean_auc': mean_auc,
            'std_auc': std_auc,
            'time_taken': np.mean(time_taken)
        }
        
    print("\nAvaliação concluída.\n")
    return results_data

def plot_roc_curves_cv(results: dict, output_filename: str = "roc_curves_cv.png"):
    """Gera o gráfico da Curva ROC média a partir dos resultados da validação cruzada."""
    plt.figure(figsize=(12, 10))

    for model_name, data in results.items():
        # Plota a curva ROC média
        plt.plot(data['mean_fpr'], data['mean_tpr'], lw=2, 
                 label=f"{model_name} (AUC = {data['mean_auc']:.3f} ± {data['std_auc']:.3f}), (1/T = {1/(data['time_taken']):.2f}Hz)")

        # Plota a área de desvio padrão (incerteza)
        tprs_upper = np.minimum(data['mean_tpr'] + data['std_tpr'], 1)
        tprs_lower = np.maximum(data['mean_tpr'] - data['std_tpr'], 0)
        plt.fill_between(data['mean_fpr'], tprs_lower, tprs_upper, alpha=0.2)

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Aleatório (AUC = 0.500)')
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos (FPR)', fontsize=14)
    plt.ylabel('Taxa de Verdadeiros Positivos (TPR)', fontsize=14)
    plt.title('Curva ROC Média com Validação Cruzada (5-Folds)', fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True)
    
    plt.savefig(output_filename)
    plt.close()

def main():
    """Função principal que orquestra a execução do script."""
    if not DATASET_DIR.is_dir():
        print(f"Erro: Diretório do dataset '{DATASET_DIR}' não encontrado.")
        return
        
    models = load_models(MODEL_NAMES, MODELS_DIR)
    results = evaluate_models_cv(models, DATASET_DIR)
    plot_roc_curves_cv(results, output_filename="../data/plots/roc_curves_cv.png")

if __name__ == "__main__":
    main()
