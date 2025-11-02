import time
from ultralytics import YOLO
from datasets import get_image_paths_and_labels
from tqdm import tqdm
import torch
import numpy as np
from sklearn.metrics import roc_curve, auc, f1_score, hamming_loss
import matplotlib.pyplot as plt
import argparse
from PIL import Image

from teste_especifico import carregar_modelo
from models import transforms_map

def get_args():
    parser = argparse.ArgumentParser(
        description="Avaliar modelos Coruja e YOLO em um dataset multi-label.")
    parser.add_argument("--input", "-i", type=str, required=False,
                        default="../data/dataset_10k/", 
                        help="Caminho para o diretório do dataset (com labels.json e images/).")
    parser.add_argument("--model", "-m", type=str, required=False,
                        default="../models/best_model.pt", 
                        help="Caminho para o modelo Coruja (.pt ou .pth).")
    parser.add_argument(
        "--reference", "-r", "--references",
        nargs="+",
        type=str,
        required=False,
        default=["../models/yolov8n.pt"],
        help="Um ou mais caminhos para modelos YOLO de referência (ex.: yolov8n.pt yolov11n.pt)."
    )
    parser.add_argument("--num_imgs", "-n", type=int, default=5_000,
                        help="Número máximo de imagens a serem avaliadas (0 = todas).")
    parser.add_argument(
        "--output", "-o", "--outputs",
        nargs="+",
        type=str,
        default=["roc_curve_multiclass.png"],
        help="Um ou mais caminhos para salvar a curva ROC (ex.: roc.png roc.pdf roc.svg)."
    )
    parser.add_argument("--device", "-d", type=str, default="cuda" if torch.cuda.is_available()
                        else "cpu", help="Dispositivo para avaliação (cuda ou cpu).")
    args = parser.parse_args()
    return args


def classificar_imagem(img_path: str, model: torch.nn.Module, device: torch.device) -> np.ndarray:
    """
    Classifica uma imagem usando o modelo Coruja multi-label.
    
    Returns:
        np.ndarray: Array [3] com probabilidades para [human, animal, vehicle]
    """
    img = Image.open(img_path).convert("RGB")
    input_tensor = transforms_map['val'](img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Output: [3] com valores sigmoid [0, 1]
        output = model(input_tensor).detach().cpu().numpy()
    
    return output


def prepare_dataset(input_path: str, n_imgs: int) -> tuple[list[str], np.ndarray]:
    """
    Prepara dataset para avaliação multi-label.
    
    Returns:
        tuple: (img_paths, labels)
            - img_paths: lista de caminhos das imagens
            - labels: array [n_images, 3] com labels binários para [human, animal, vehicle]
    """
    img_paths, labels, classes = get_image_paths_and_labels(input_path)
    
    # Limita o número de imagens se necessário
    if n_imgs > 0 and len(img_paths) > n_imgs:
        indices = np.random.choice(len(img_paths), n_imgs, replace=False)
        img_paths = [img_paths[i] for i in indices]
        labels = [labels[i] for i in indices]
    
    return img_paths, np.array(labels)


def ref_classificar(img_path: str, model: YOLO) -> np.ndarray:
    """
    Classifica uma imagem usando YOLO para comparação multi-label.
    
    Mapeia classes YOLO COCO para as 3 classes:
    - human: person (id 0)
    - animal: cat (id 15), dog (id 16)
    - vehicle: car (id 2), motorcycle (id 3), bus (id 5)
    
    Returns:
        np.ndarray: Array [3] com probabilidades para [human, animal, vehicle]
    """
    # Mapeamento COCO id -> nossa classe
    coco_to_class = {
        0: 0,   # person -> human
        15: 1,  # cat -> animal
        16: 1,  # dog -> animal
        2: 2,   # car -> vehicle
        3: 2,   # motorcycle -> vehicle
        5: 2    # bus -> vehicle
    }
    
    probs = np.zeros(3, dtype=np.float32)
    
    results = model.predict(img_path, verbose=False)
    r = results[0]
    
    if r.boxes is not None and len(r.boxes) > 0:
        cls_data = r.boxes.cls
        conf_data = r.boxes.conf
        
        # Converte para numpy se for tensor
        if hasattr(cls_data, 'cpu'):
            cls = cls_data.detach().cpu().numpy()
        else:
            cls = np.array(cls_data)
            
        if hasattr(conf_data, 'cpu'):
            conf = conf_data.detach().cpu().numpy()
        else:
            conf = np.array(conf_data)
        
        # Para cada classe de interesse, pega a máxima confiança
        for coco_id, our_class_idx in coco_to_class.items():
            mask = (cls == coco_id)
            if mask.any():
                probs[our_class_idx] = max(probs[our_class_idx], conf[mask].max())
    
    return probs


def main():
    args = get_args()
    input_path = args.input
    model_path = args.model
    reference_paths = args.reference
    device = torch.device(args.device)
    output_paths = args.output
    n_imgs = args.num_imgs

    print(
        f"Avaliando dataset {input_path} com Coruja {model_path} e referências {', '.join(reference_paths)}")
    print(f"Número máximo de imagens: {n_imgs}")
    print(f"Arquivos de saída da ROC: {', '.join(output_paths)}")
    print(f"Usando dispositivo: {device}")

    img_paths, labels = prepare_dataset(input_path, n_imgs)
    print(f"Total de imagens: {len(img_paths)}")
    print(f"Distribuição de labels:")
    print(f"  - Human: {labels[:, 0].sum()}")
    print(f"  - Animal: {labels[:, 1].sum()}")
    print(f"  - Vehicle: {labels[:, 2].sum()}")

    class_names = ['Human', 'Animal', 'Vehicle']
    
    # Preparar estruturas para resultados por classe
    # rocs[class_idx] = lista de tuplas (model_name, fpr, tpr, auc, freq_mean, freq_std)
    rocs_per_class = [[] for _ in range(3)]
    
    # Inferência Coruja
    coruja = carregar_modelo(model_path, device)
    coruja_outputs = np.zeros((len(img_paths), 3), dtype=np.float32)  # [n_images, 3]
    coruja_tempo = np.zeros(len(img_paths), dtype=float)

    for i, img in enumerate(tqdm(img_paths, desc="Inferência Coruja")):
        start = time.time()
        output = classificar_imagem(img, coruja, device)
        coruja_outputs[i] = output
        coruja_tempo[i] = time.time() - start

    # Calcula métricas multi-label para Coruja
    coruja_preds = (coruja_outputs >= 0.5).astype(int)
    coruja_hamming_acc = 1 - hamming_loss(labels, coruja_preds)
    coruja_f1_macro = f1_score(labels, coruja_preds, average='macro', zero_division=0)
    
    print(f"\nCoruja - Métricas Multi-Label:")
    print(f"  Hamming Accuracy: {coruja_hamming_acc:.4f}")
    print(f"  F1 Macro: {coruja_f1_macro:.4f}")
    
    # Curvas ROC por classe para Coruja
    coruja_freq = 1.0 / np.clip(coruja_tempo, 0.01, None)
    freq_mean = float(np.mean(coruja_freq))
    freq_std = float(np.std(coruja_freq))
    
    for class_idx in range(3):
        try:
            fpr, tpr, _ = roc_curve(labels[:, class_idx], coruja_outputs[:, class_idx])
            auc_val = auc(fpr, tpr)
            rocs_per_class[class_idx].append(
                ("Coruja", fpr, tpr, auc_val, freq_mean, freq_std)
            )
            print(f"  AUC {class_names[class_idx]}: {auc_val:.4f}")
        except Exception as e:
            print(f"  Erro ao calcular ROC para {class_names[class_idx]}: {e}")

    # Inferência dos YOLOs
    for ref_path in reference_paths:
        try:
            yolo = YOLO(ref_path)
        except FileNotFoundError as e:
            print(f"\nErro ao carregar YOLO {ref_path}: {e}")
            continue
            
        yolo_name = ref_path.split("/")[-1].split(".")[0]
        yolo_outputs = np.zeros((len(img_paths), 3), dtype=np.float32)
        yolo_tempo = np.zeros(len(img_paths), dtype=float)
        
        for i, img in enumerate(tqdm(img_paths, desc=f"YOLO {yolo_name}")):
            start = time.time()
            yolo_outputs[i] = ref_classificar(img, yolo)
            yolo_tempo[i] = time.time() - start
        
        # Métricas multi-label para YOLO
        yolo_preds = (yolo_outputs >= 0.5).astype(int)
        yolo_hamming_acc = 1 - hamming_loss(labels, yolo_preds)
        yolo_f1_macro = f1_score(labels, yolo_preds, average='macro', zero_division=0)
        
        print(f"\n{yolo_name} - Métricas Multi-Label:")
        print(f"  Hamming Accuracy: {yolo_hamming_acc:.4f}")
        print(f"  F1 Macro: {yolo_f1_macro:.4f}")
        
        # Frequência
        yolo_freq = 1.0 / np.clip(yolo_tempo, 0.01, None)
        freq_mean = float(np.mean(yolo_freq))
        freq_std = float(np.std(yolo_freq))
        
        # Curvas ROC por classe para YOLO
        for class_idx in range(3):
            try:
                fpr, tpr, _ = roc_curve(labels[:, class_idx], yolo_outputs[:, class_idx])
                auc_val = auc(fpr, tpr)
                rocs_per_class[class_idx].append(
                    (yolo_name, fpr, tpr, auc_val, freq_mean, freq_std)
                )
                print(f"  AUC {class_names[class_idx]}: {auc_val:.4f}")
            except Exception as e:
                print(f"  Erro ao calcular ROC para {class_names[class_idx]}: {e}")

    # Plota ROC curves - uma por classe
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for class_idx in range(3):
        ax = axes[class_idx]
        
        for model_name, fpr, tpr, auc_val, f_mean, f_std in rocs_per_class[class_idx]:
            ax.plot(
                fpr, tpr, 
                label=f"{model_name} (AUC={auc_val:.3f}, f={f_mean:.0f}±{f_std:.0f}Hz)"
            )
        
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)  # linha diagonal
        ax.set_xlabel('False Positive Rate', fontsize=10)
        ax.set_ylabel('True Positive Rate', fontsize=10)
        ax.set_title(f'ROC Curve - {class_names[class_idx]}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
    
    plt.tight_layout()
    
    for out_path in output_paths:
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"\nROC salva em: {out_path}")
    plt.close(fig)
    
    # Sumário final
    print("\n" + "="*60)
    print("RESUMO - AUC por Classe")
    print("="*60)
    for class_idx in range(3):
        print(f"\n{class_names[class_idx]}:")
        for model_name, _, _, auc_val, f_mean, f_std in rocs_per_class[class_idx]:
            print(f"  {model_name:15s}: AUC={auc_val:.4f} | f={f_mean:.1f}±{f_std:.1f} Hz")
    print("="*60)


if __name__ == "__main__":
    main()
