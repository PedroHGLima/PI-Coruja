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
    parser.add_argument(
        "--reference", "-r", "--references",
        nargs="+",
        type=str,
        required=False,
        default=["../models/yolov8n.pt"],
        help="Um ou mais caminhos para modelos YOLO de referência (ex.: yolov8n.pt yolov11n.pt)."
    )
    parser.add_argument("--num_imgs", "-n", type=int, default=5_000,
                        help="Número máximo de imagens a serem avaliadas (balanceado entre classes).")
    parser.add_argument("--sections", "-s", type=int, default=1,
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
    reference_paths = args.reference
    sections = args.sections
    device = torch.device(args.device)
    output_paths = args.output
    n_imgs = args.num_imgs

    print(
        f"Avaliando dataset {input_path} com Coruja {model_path} e referências {', '.join(reference_paths)}")
    print(f"Número máximo de imagens por classe: {n_imgs}")
    print(f"Número de seções por imagem: {sections}")
    print(f"Arquivos de saída da ROC: {', '.join(output_paths)}")
    print(f"Usando dispositivo: {device}")

    img_paths, labels = prepare_dataset(input_path, n_imgs)

    coruja = carregar_modelo(model_path, device)
    # Preparar estruturas
    coruja_outputs = np.zeros(len(img_paths), dtype=np.float32)
    coruja_tempo = np.zeros(len(img_paths), dtype=float)
    yolo_outputs_map: dict[str, np.ndarray] = {}
    yolo_tempos_map: dict[str, np.ndarray] = {}

    for i, img in enumerate(tqdm(img_paths, desc="Inferência Coruja")):
        # --- Modelo Coruja ---
        start = time.time()
        output = classificar_imagem(
            img, coruja, device, sections).detach().cpu().numpy()
        coruja_outputs[i] = (np.max(output) + 1) / 2  # mapear tanh [-1,1] -> [0,1]
        coruja_tempo[i] = time.time() - start

    # Inferir para cada referência YOLO separadamente (evita carregar vários por imagem)
    rocs = []  # lista de tuplas (label, fpr, tpr, auc, freq_mean, freq_std)
    # Curva da Coruja
    fpr_coruja, tpr_coruja, _ = roc_curve(
        [1 if l == 1 else 0 for l in labels], coruja_outputs)
    auc_coruja = auc(fpr_coruja, tpr_coruja)
    # Frequência por amostra: f = 1/T (Hz). Evitar divisão por zero.
    coruja_freq = 1.0 / np.clip(coruja_tempo, 1e-12, None)
    rocs.append(("Coruja", fpr_coruja, tpr_coruja, auc_coruja,
                 float(np.mean(coruja_freq)), float(np.std(coruja_freq))))

    for ref_path in reference_paths:
        yolo = YOLO(ref_path)
        yolo_name = ref_path.split('/')[-1].split('.')[0]
        yolo_outputs = np.zeros(len(img_paths), dtype=np.float32)
        yolo_tempo = np.zeros(len(img_paths), dtype=float)
        for i, img in enumerate(tqdm(img_paths, desc=f"YOLO {ref_path}")):
            start = time.time()
            yolo_outputs[i] = ref_classificar(img, yolo)
            yolo_tempo[i] = time.time() - start
        yolo_outputs_map[ref_path] = yolo_outputs
        yolo_tempos_map[ref_path] = yolo_tempo
        fpr_y, tpr_y, _ = roc_curve(
            [1 if l == 1 else 0 for l in labels], yolo_outputs)
        auc_y = auc(fpr_y, tpr_y)
        yolo_freq = 1.0 / np.clip(yolo_tempo, 1e-12, None)
        rocs.append((yolo_name, fpr_y, tpr_y, auc_y, float(
            np.mean(yolo_freq)), float(np.std(yolo_freq))))

    print("Resultados AUC:")
    for label, _, _, auc_val, f_mean, f_std in rocs:
        print(
            f" - {label}: AUC={auc_val:.4f} | f = {f_mean:.1f} ± {f_std:.1f} Hz")

    fig, ax = plt.subplots()
    for label, fpr, tpr, auc_val, f_mean, f_std in rocs:
        ax.plot(
            fpr, tpr, label=f"{label} $(f = {f_mean:.1f} ± {f_std:.1f} Hz)$")
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
