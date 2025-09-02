import os
import argparse
import torch
from PIL import Image

from numpy import short
from models import CorujaResNet, transforms_map

def get_args():
    parser = argparse.ArgumentParser(description="Avaliar Modelos")
    parser.add_argument("--input", "-i", type=str, required=True, help="Caminho para a imagem de entrada")
    parser.add_argument("--model", "-m", type=str, required=True, help="Caminho para o modelo")
    args = parser.parse_args()
    return args

def carregar_modelo(model_path, device):
    ext = os.path.splitext(model_path)[1]
    if ext == '.pth':
        model = CorujaResNet()
        model.load_state_dict(torch.load(model_path, map_location=device))
    elif ext == '.pt':
        model = torch.load(model_path, map_location=device, weights_only=False)
    else:
        raise ValueError(f"Extensão de modelo não suportada: {ext}")
    model.to(device)
    model.eval()
    return model

def main(input_path:str, model_path:str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Avaliando a imagem {input_path} o modelo {model_path}")

    model = carregar_modelo(model_path, device)
    img = Image.open(input_path).convert("RGB")
    input_tensor = transforms_map['val'](img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.sigmoid(output).item()
    print(1-prob)

    return

if __name__ == "__main__":
    args = get_args()
    input_path = args.input
    model_path = args.model
    main(input_path, model_path)
