import os
import argparse
import torch
from PIL import Image

import numpy as np
from models import CorujaResNet, transforms_map

def get_args():
    parser = argparse.ArgumentParser(description="Avaliar Modelos")
    parser.add_argument("--input", "-i", type=str, required=True, help="Caminho para a imagem de entrada")
    parser.add_argument("--model", "-m", type=str, required=True, help="Caminho para o modelo")
    parser.add_argument("--sections", "-s", type=int, default=3, help="Número de seções para dividir a imagem")
    parser.add_argument("--debug", "-d", action='store_true', help="Ativar modo debug")
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

def preparar_imagens(img_path:str, sections:int, debug:bool):
    img = Image.open(img_path).convert("RGB")
    width, height = img.size
    tensors = []
    
    for crop in range(1, sections + 1):
        section_height = height // crop
        section_width = width // crop
        for w in range(crop):
            for h in range(crop):
                left = w * section_width
                upper = h * section_height
                right = (w + 1) * section_width if (w + 1) * section_width <= width else width
                lower = (h + 1) * section_height if (h + 1) * section_height <= height else height
                
                box = (left, upper, right, lower)
                img_section = img.crop(box)
                if debug:
                    img_section.save(f"debug_crop_{crop}_{w}_{h}.jpg")
                
                input_tensor = transforms_map['val'](img_section).unsqueeze(0)
                tensors.append(input_tensor)
                
    tensors = torch.cat(tensors, dim=0)
    
    return tensors
    

def main(input_path:str, model_path:str, sections:int, debug:bool):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Avaliando a imagem {input_path} o modelo {model_path}")

    model = carregar_modelo(model_path, device)
    input_tensor = preparar_imagens(input_path, sections, debug).to(device)
    with torch.no_grad():
        output = model(input_tensor).detach().cpu().numpy()
        i = 0
        for crop in range(1, sections + 1):
            for w in range(crop):
                for h in range(crop):
                    print(f"Seção {crop} ({w},{h}): {output[i]}")
                    i += 1

        print(np.max(output))

    return

if __name__ == "__main__":
    args = get_args()
    input_path = args.input
    model_path = args.model
    sections = args.sections
    debug = args.debug
    main(input_path, model_path, sections, debug)