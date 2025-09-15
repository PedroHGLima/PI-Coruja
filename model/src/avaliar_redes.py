import time
from ultralytics import YOLO
from teste_especifico import preparar_imagens, carregar_modelo
from datasets import get_image_paths_and_labels
from tqdm import tqdm
import torch
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import argparse


def get_args():
    parser = argparse.ArgumentParser(
        description="Avaliar modelos Coruja e YOLO em um dataset balanceado.")
    parser.add_argument("--input", "-i", type=str, required=False,
                        default="../data/dataset_10k/", help="Caminho para a imagem de entrada.")
    parser.add_argument("--model", "-m", type=str, required=False,
                        default="../models/tanh.pt", help="Caminho para o modelo Coruja.")
    parser.add_argument("--reference", "-r", type=str, required=False,
                        default="../models/yolov8n.pt", help="Caminho para o modelo de referência.")
    parser.add_argument("--num_imgs", "-n", type=int, default=5_000,
                        help="Número máximo de imagens a serem avaliadas (balanceado entre classes).")
    parser.add_argument("--sections", "-s", type=int, default=3,
                        help="Número de seções para dividir a imagem.")
    parser.add_argument(
        "--output", "-o", "--outputs",
        nargs="+",
        type=str,
        default=["roc_curve.png"],
        help="Um ou mais caminhos para salvar a curva ROC (ex.: roc.png roc.pdf roc.svg)."
    )
    parser.add_argument("--device", "-d", type=str, default="cuda" if torch.cuda.is_available()
                        else "cpu", help="Dispositivo para avaliação (cuda ou cpu).")
    args = parser.parse_args()
    return args


def classificar_imagem(img_path: str, model: torch.nn.Module, device: torch.device, sections: int = 2) -> torch.Tensor:
    input_tensor = preparar_imagens(img_path, sections).to(device)
    with torch.no_grad():
        output = model(input_tensor).detach().cpu()
    return output


def prepare_dataset(input_path: str, n_imgs: int) -> tuple[list[str], list[int]]:
    img_paths, labels, classes = get_image_paths_and_labels(input_path)

    human_imgs = [img for img, label in zip(img_paths, labels) if label == 1]
    no_human_imgs = [img for img, label in zip(
        img_paths, labels) if label == -1]
    min_count = min(len(human_imgs), len(no_human_imgs), n_imgs)
    img_paths = human_imgs[:min_count] + no_human_imgs[:min_count]
    labels = [1]*min_count + [-1]*min_count

    return img_paths, labels

def ref_classificar(img_path: str, model: YOLO) -> float:
    results = model.predict(img_path, verbose=False)
    r = results[0]
    if r.boxes is not None and len(r.boxes) > 0:
        cls = r.boxes.cls.detach().cpu().numpy()
        conf = r.boxes.conf.detach().cpu().numpy()
        person_conf = conf[cls == 0]  # id 0 é 'person' no COCO
        return float(person_conf.max()) if person_conf.size > 0 else 0.0
    return 0.0


def main():
    args = get_args()
    input_path = args.input
    model_path = args.model
    reference_path = args.reference
    sections = args.sections
    device = torch.device(args.device)
    output_paths = args.output
    n_imgs = args.num_imgs

    print(
        f"Avaliando a imagem {input_path} o modelo {model_path} e referência {reference_path}")
    print(f"Número máximo de imagens por classe: {n_imgs}")
    print(f"Número de seções por imagem: {sections}")
    print(f"Arquivos de saída da ROC: {', '.join(output_paths)}")
    print(f"Usando dispositivo: {device}")

    img_paths, labels = prepare_dataset(input_path, n_imgs)

    coruja = carregar_modelo(model_path, device)
    yolo = YOLO(reference_path)

    coruja_outputs = np.zeros(len(img_paths), dtype=np.float32)
    yolo_outputs = np.zeros(len(img_paths), dtype=np.float32)
    
    coruja_tempo = np.zeros(len(img_paths), dtype=float)
    yolo_tempo = np.zeros(len(img_paths), dtype=float)

    for i, img in enumerate(tqdm(img_paths)):
        # --- Modelo Coruja ---
        start = time.time()
        output = classificar_imagem(
            img, coruja, device, sections).detach().cpu().numpy()
        coruja_outputs[i] = (np.max(output) + 1) / \
            2  # mapear tanh [-1,1] -> [0,1]
        coruja_tempo[i] = time.time() - start

        # --- YOLO ---
        start = time.time()
        yolo_outputs[i] = ref_classificar(img, yolo)
        yolo_tempo[i] = time.time() - start

    fpr_coruja, tpr_coruja, _ = roc_curve(
        [1 if l == 1 else 0 for l in labels], coruja_outputs)
    fpr_yolo, tpr_yolo, _ = roc_curve(
        [1 if l == 1 else 0 for l in labels], yolo_outputs)
    auc_coruja = auc(fpr_coruja, tpr_coruja)
    auc_yolo = auc(fpr_yolo, tpr_yolo)

    print(f"AUC Coruja: {auc_coruja:.4f}")
    print(f"AUC YOLO: {auc_yolo:.4f}")

    fig, ax = plt.subplots()
    ax.plot(fpr_coruja, tpr_coruja, label=f'Coruja ($1/T = {1/(np.mean(coruja_tempo)):.4f} \\pm {1/(np.std(coruja_tempo)):.4f} Hz)$)')
    ax.plot(fpr_yolo, tpr_yolo, label=f'YOLO ($1/T = {1/(np.mean(yolo_tempo)):.4f} \\pm {1/(np.std(yolo_tempo)):.4f} Hz)$)')
    ax.plot([0, 1], [0, 1], 'k--')  # linha diagonal
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend()
    for out_path in output_paths:
        fig.savefig(out_path)
        print(f"ROC salva em: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
