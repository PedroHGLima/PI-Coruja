from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms
import json
import numpy as np

class SimpleDataset(torch.utils.data.Dataset):
    """Dataset para classificação multi-label"""
    def __init__(self, img_paths: list[str], labels: list[list[int]], transform: transforms.Compose):
        self.img_paths = img_paths
        self.labels = labels  # Agora é uma lista de listas/arrays [human, animal, vehicle]
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.img_paths)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img = Image.open(self.img_paths[idx]).convert('RGB')
        img = self.transform(img)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return img, label

def get_image_paths_and_labels(data_dir: str) -> tuple[list[str], list[list[int]], list[str]]:
    """
    Carrega imagens e labels multi-label do formato JSON.
    
    Args:
        data_dir: Diretório contendo 'labels.json' e pasta 'images/'
    
    Returns:
        tuple: (img_paths, labels, class_names)
            - img_paths: lista de caminhos das imagens
            - labels: lista de labels multi-label [[human, animal, vehicle], ...]
            - class_names: lista com os nomes das classes ['human', 'animal', 'vehicle']
    """
    data_dir = Path(data_dir)
    json_path = data_dir / "labels.json"
    images_dir = data_dir / "images"
    
    if not json_path.exists():
        raise FileNotFoundError(f"Arquivo labels.json não encontrado em {data_dir}")
    
    # Carrega o JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        dataset_info = json.load(f)
    
    class_names = dataset_info.get("classes", ["human", "animal", "vehicle"])
    images_data = dataset_info["images"]
    
    img_paths = []
    labels = []
    
    for img_info in images_data:
        img_path = images_dir / img_info["filename"]
        if img_path.exists():
            img_paths.append(str(img_path))
            # Cria vetor de labels [human, animal, vehicle]
            label_vector = [
                img_info.get("human", 0),
                img_info.get("animal", 0),
                img_info.get("vehicle", 0)
            ]
            labels.append(label_vector)
    
    return img_paths, labels, class_names


def get_image_paths_and_labels_legacy(data_dir: str) -> tuple[list[str], list[int], list[str]]:
    """
    LEGADO: Carrega imagens do formato antigo (pastas human/no_human).
    Mantido para compatibilidade com datasets antigos.
    """
    data_dir_path = Path(data_dir)
    # Garante mapeamento explícito: 'human' -> +1 (alerta), 'no_human' -> -1
    classes = []
    img_paths: list[str] = []
    labels: list[int] = []

    human_dir = data_dir_path / "human"
    no_human_dir = data_dir_path / "no_human"

    if human_dir.is_dir():
        classes.append("human")
        for img_file in human_dir.glob("*.jpg"):
            img_paths.append(str(img_file))
            labels.append(1)

    if no_human_dir.is_dir():
        classes.append("no_human")
        for img_file in no_human_dir.glob("*.jpg"):
            img_paths.append(str(img_file))
            labels.append(-1)

    return img_paths, labels, classes
