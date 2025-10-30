import fiftyone as fo
import fiftyone.zoo as foz
from pathlib import Path
from fiftyone import ViewField as F
import argparse

class_labels = {
    "human": ["person"],
    "animal": ["cat", "dog"],
    "vehicle": ["car", "motorcycle", "bus"]
}

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

    for classe in ["human", "animal", "vehicle"]:
        path_positivos = dataset_dir / classe
        path_positivos.mkdir(parents=True, exist_ok=True)
        
        positive_subset = dataset.filter_labels("ground_truth", F("label").is_in(class_labels[classe]))\
            .take(args.num_positivos if args.num_positivos != -1 else dataset.count)
        
        print(f"Exportando {len(positive_subset)} imagens para '{path_positivos}'...")
        positive_subset.export(
            export_dir=str(path_positivos),
            dataset_type=fo.types.ImageDirectory()
        )  
    
    # create a negative subset (images without any of the target classes) similarly to the creation of positive subsets
    negative_subset = dataset\
        .filter_labels("ground_truth", ~F("label").is_in(class_labels["human"]))\
        .filter_labels("ground_truth", ~F("label").is_in(class_labels["animal"]))\
        .filter_labels("ground_truth", ~F("label").is_in(class_labels["vehicle"]))\
        .take(args.num_negativos if args.num_negativos != -1 else dataset.count)
    
    path_negativos = dataset_dir / "negatives"
    path_negativos.mkdir(parents=True, exist_ok=True)
    print(f"Exportando {len(negative_subset)} imagens para '{path_negativos}'...")
    negative_subset.export(
        export_dir=str(path_negativos),
        dataset_type=fo.types.ImageDirectory()
    )

if __name__ == "__main__":
    main()
