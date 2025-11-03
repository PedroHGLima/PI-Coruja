import fiftyone as fo
import fiftyone.zoo as foz
from pathlib import Path
from fiftyone import ViewField as F
import argparse

def parse_arguments():
    """
    Configura e analisa os argumentos da linha de comando.
    """
    parser = argparse.ArgumentParser(
        description="Baixa, filtra e exporta um subconjunto do dataset COCO-2017."
    )
    
    parser.add_argument(
        "--dataset-name", "-n",
        type=str,
        default="dataset_10k",
        help="Nome do diretório do dataset (padrão: dataset_10k)"
    )
    
    parser.add_argument(
        "--split", "-s",
        type=str,
        choices=["train", "validation", "test"],
        default="validation",
        help="Split do COCO para usar (padrão: validation)"
    )
    
    parser.add_argument(
        "--num-positivos", "-p",
        type=int,
        default=10000,
        help="Número de imagens com pessoas (padrão: 10000, use -1 para pegar todas)"
    )
    
    parser.add_argument(
        "--num-negativos", "-g",
        type=int,
        default=10000,
        help="Número de imagens sem pessoas (padrão: 10000, use -1 para pegar todas)"
    )
    
    return parser.parse_args()

def main():
    """
    Baixa, filtra e exporta um subconjunto do dataset COCO-2017.
    """
    args = parse_arguments()
    
    # Configurações baseadas nos argumentos
    dataset_dir = Path("../data") / args.dataset_name
    
    dataset = foz.load_zoo_dataset("coco-2017", split=args.split)
    dataset.persistent = True

    print(f"Filtrando imagens com a classe 'person'...")
    positive_view = dataset.filter_labels("ground_truth", F("label") == "person")
    
    if args.num_positivos == -1:
        positive_subset = positive_view
        print(f"Pegando TODAS as {len(positive_subset)} imagens com pessoas disponíveis.")
    else:
        positive_subset = positive_view.take(args.num_positivos)
        print(f"Pegando {args.num_positivos} imagens com pessoas.")

    print(f"Filtrando imagens SEM a classe 'person'...")
    negative_view = dataset.exclude(positive_view)
    
    if args.num_negativos == -1:
        negative_subset = negative_view
        print(f"Pegando TODAS as {len(negative_subset)} imagens sem pessoas disponíveis.")
    else:
        negative_subset = negative_view.take(args.num_negativos)
        print(f"Pegando {args.num_negativos} imagens sem pessoas.")
    
    # Exporta imagens com humanos
    path_positivos = dataset_dir / "human"
    path_positivos.mkdir(parents=True, exist_ok=True)
    print(f"Exportando {len(positive_subset)} imagens para '{path_positivos}'...")
    positive_subset.export(
        export_dir=str(path_positivos),
        dataset_type=fo.types.ImageDirectory()
    )

    # Exporta imagens sem humanos
    path_negativos = dataset_dir / "no_human"
    path_negativos.mkdir(parents=True, exist_ok=True)
    print(f"Exportando {len(negative_subset)} imagens para '{path_negativos}'...")
    negative_subset.export(
        export_dir=str(path_negativos),
        dataset_type=fo.types.ImageDirectory()
    )

if __name__ == "__main__":
    main()
