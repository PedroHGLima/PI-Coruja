import argparse
import torch
from pathlib import Path
from torchvision import transforms as tv_transforms


def get_input_shape_from_transforms():
    """Extrai o input shape a partir de transforms_map['val'] em models.py."""
    try:
        from models import transforms_map
        val_transform = transforms_map.get('val')
        if val_transform:
            transforms_list = getattr(val_transform, 'transforms', [val_transform])
            for t in transforms_list:
                if isinstance(t, (tv_transforms.Resize, tv_transforms.CenterCrop)):
                    size = getattr(t, 'size', None)
                    if size:
                        if isinstance(size, int):
                            return (1, 3, size, size)
                        elif isinstance(size, (tuple, list)) and len(size) == 2:
                            return (1, 3, int(size[0]), int(size[1]))
    except Exception:
        pass
    return (1, 3, 512, 512)


def main():
    parser = argparse.ArgumentParser(description="Converte modelo .pt para .onnx")
    parser.add_argument("-p", "--pt", required=True, help="Caminho do arquivo .pt")
    parser.add_argument("-o", "--onnx", required=True, help="Caminho do arquivo .onnx de saída")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Carregar modelo
    print(f"Carregando modelo: {args.pt}")
    model = torch.load(args.pt, map_location=device, weights_only=False)
    model.eval()

    # Obter input shape
    input_shape = get_input_shape_from_transforms()
    print(f"Input shape: {input_shape}")

    # Criar tensor de exemplo
    dummy_input = torch.randn(input_shape, device=device)

    # Validar forward
    with torch.no_grad():
        _ = model(dummy_input)

    # Exportar para ONNX
    output_path = Path(args.onnx)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Exportando para ONNX: {output_path}")
    
    # Tentar exportador moderno primeiro (dynamo=True)
    try:
        print("Tentando exportador moderno (dynamo=True)...")
        torch.onnx.export(
            model,
            (dummy_input,),
            str(output_path),
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            dynamo=True,
            input_names=['input'],
            output_names=['output']
        )
        print(f"Modelo exportado com sucesso (exportador moderno): {output_path.resolve()}")
    except Exception as e:
        print(f"Exportador moderno falhou: {type(e).__name__}")
        print("Tentando exportador legacy...")
        
        # Fallback para exportador legacy
        torch.onnx.export(
            model,
            (dummy_input,),
            str(output_path),
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output']
        )
        print(f"Modelo exportado com sucesso (exportador legacy): {output_path.resolve()}")


if __name__ == "__main__":
    main()
