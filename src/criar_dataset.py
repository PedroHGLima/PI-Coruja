import fiftyone as fo
import fiftyone.zoo as foz
from pathlib import Path
from fiftyone import ViewField as F

# --- Configurações ---
NUM_POSITIVOS = 10_000
NUM_NEGATIVOS = 10_000
DATASET_DIR = Path("../data/dataset_10k")

def main():
    """
    Baixa, filtra e exporta um subconjunto do dataset COCO-2017.
    """
    dataset = foz.load_zoo_dataset("coco-2017", split="validation")
    dataset.persistent = True

    print(f"Filtrando {NUM_POSITIVOS} imagens com a classe 'person'...")
    positive_view = dataset.filter_labels("ground_truth", F("label") == "person")
    positive_subset = positive_view.take(NUM_POSITIVOS)

    print(f"Filtrando {NUM_NEGATIVOS} imagens SEM a classe 'person'...")
    negative_view = dataset.exclude(positive_view)
    negative_subset = negative_view.take(NUM_NEGATIVOS)
    
    # Exporta imagens com humanos
    path_positivos = DATASET_DIR / "human"
    path_positivos.mkdir(parents=True, exist_ok=True)
    print(f"Exportando {len(positive_subset)} imagens para '{path_positivos}'...")
    positive_subset.export(
        export_dir=str(path_positivos),
        dataset_type=fo.types.ImageDirectory()
    )

    # Exporta imagens sem humanos
    path_negativos = DATASET_DIR / "no_human"
    path_negativos.mkdir(parents=True, exist_ok=True)
    print(f"Exportando {len(negative_subset)} imagens para '{path_negativos}'...")
    negative_subset.export(
        export_dir=str(path_negativos),
        dataset_type=fo.types.ImageDirectory()
    )

if __name__ == "__main__":
    main()
