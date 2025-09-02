from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, img_paths: list[str], labels: list[int], transform: transforms.Compose):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform
    def __len__(self) -> int:
        return len(self.img_paths)
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img = Image.open(self.img_paths[idx]).convert('RGB')
        img = self.transform(img)
        label = self.labels[idx]
        return img, label

def get_image_paths_and_labels(data_dir: str) -> tuple[list[str], list[int], list[str]]:
    data_dir = Path(data_dir)
    classes = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    img_paths = []
    labels = []
    for idx, cls in enumerate(classes):
        for img_file in (data_dir / cls).glob("*.jpg"):
            img_paths.append(str(img_file))
            labels.append(idx)
    return img_paths, labels, classes
