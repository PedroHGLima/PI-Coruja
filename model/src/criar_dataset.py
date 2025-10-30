import fiftyone as fo
import fiftyone.zoo as foz
from pathlib import Path
from fiftyone import ViewField as F
from fiftyone.types import ImageDirectory
import argparse
import json
import shutil
from collections import defaultdict

class_labels = {
    "human": ["person"],
    "animal": ["cat", "dog"],
    "vehicle": ["car", "motorcycle", "bus"]
}

# Todas as classes COCO que queremos considerar
ALL_CLASSES = ["person", "cat", "dog", "car", "motorcycle", "bus"]

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
        help="Número de imagens sem nenhuma classe (padrão: 10000, use -1 para pegar todas)"
    )
    
    parser.add_argument(
        "--max-images", "-m",
        type=int,
        default=30000,
        help="Número máximo total de imagens no dataset (padrão: 30000)"
    )
    
    return parser.parse_args()

def get_image_labels(sample):
    """
    Extrai os labels de uma imagem do COCO e retorna um dicionário
    indicando quais das 3 meta-classes estão presentes.
    
    Returns:
        dict: {'human': 0/1, 'animal': 0/1, 'vehicle': 0/1}
    """
    labels = {"human": 0, "animal": 0, "vehicle": 0}
    
    if sample.ground_truth is None:
        return labels
    
    # Obtém todos os labels da imagem
    detected_labels = [det.label for det in sample.ground_truth.detections]
    
    # Verifica cada meta-classe
    for meta_class, coco_classes in class_labels.items():
        if any(label in detected_labels for label in coco_classes):
            labels[meta_class] = 1
    
    return labels


def should_include_image(labels, image_counter, num_positivos, class_list):
    """
    Verifica se uma imagem deve ser incluída baseado nos contadores atuais.
    
    Args:
        labels: Dict com labels da imagem
        image_counter: Lista com contadores por classe
        num_positivos: Número máximo de exemplos por classe (-1 = ilimitado)
        class_list: Lista com nomes das classes
    
    Returns:
        bool: True se a imagem deve ser incluída
    """
    for class_name in class_list:
        class_idx = class_list.index(class_name)
        if labels[class_name] == 1:
            if num_positivos == -1 or image_counter[class_idx] < num_positivos:
                return True
    return False


def update_class_counters(labels, image_counter, class_list):
    """
    Atualiza os contadores das classes presentes na imagem.
    
    Args:
        labels: Dict com labels da imagem
        image_counter: Lista com contadores por classe (modificada in-place)
        class_list: Lista com nomes das classes
    """
    for class_name in class_list:
        if labels[class_name] == 1:
            class_idx = class_list.index(class_name)
            image_counter[class_idx] += 1


def copy_image_and_save_metadata(sample, total_counter, images_dir, labels):
    """
    Copia a imagem para o diretório de destino e retorna os metadados.
    
    Args:
        sample: Sample do FiftyOne
        total_counter: Contador total de imagens
        images_dir: Path do diretório de destino
        labels: Dict com labels da imagem
    
    Returns:
        dict: Metadados da imagem para o JSON
    """
    image_path = Path(sample.filepath)
    new_image_name = f"{total_counter:06d}{image_path.suffix}"
    new_image_path = images_dir / new_image_name
    
    shutil.copy2(sample.filepath, new_image_path)
    
    return {
        "filename": new_image_name,
        "human": labels["human"],
        "animal": labels["animal"],
        "vehicle": labels["vehicle"]
    }


def collect_positive_images(dataset_filtered, args, images_dir, class_list):
    """
    Coleta imagens que contêm pelo menos uma das classes de interesse.
    
    Args:
        dataset_filtered: Dataset filtrado do FiftyOne
        args: Argumentos da linha de comando
        images_dir: Path do diretório de destino
        class_list: Lista com nomes das classes
    
    Returns:
        tuple: (images_data, image_counter, total_image_counter)
    """
    images_data = []
    image_counter = [0, 0, 0]  # [human, animal, vehicle]
    total_image_counter = 0
    
    print(f"Coletando até {args.num_positivos} imagens por classe...")
    
    for sample in dataset_filtered:
        if total_image_counter >= args.max_images:
            break
            
        labels = get_image_labels(sample)
        
        # Ignora se não tem nenhum label
        if sum(labels.values()) == 0:
            continue
        
        # Verifica se alguma classe ainda precisa de mais exemplos
        if not should_include_image(labels, image_counter, args.num_positivos, class_list):
            continue
        
        # Atualiza contadores das classes presentes
        update_class_counters(labels, image_counter, class_list)
        
        # Copia imagem e salva metadados
        image_data = copy_image_and_save_metadata(sample, total_image_counter, images_dir, labels)
        images_data.append(image_data)
        
        total_image_counter += 1
        
        if total_image_counter % 1000 == 0:
            print(f"  Processadas {total_image_counter} imagens...")
            print(f"    Contadores: human={image_counter[0]}, animal={image_counter[1]}, vehicle={image_counter[2]}")
    
    return images_data, image_counter, total_image_counter


def collect_negative_images(dataset, args, images_dir, images_data, total_image_counter):
    """
    Coleta imagens que não contêm nenhuma das classes de interesse.
    
    Args:
        dataset: Dataset completo do FiftyOne
        args: Argumentos da linha de comando
        images_dir: Path do diretório de destino
        images_data: Lista de imagens já coletadas (modificada in-place)
        total_image_counter: Contador total atual
    
    Returns:
        tuple: (total_image_counter atualizado, negative_counter)
    """
    negative_counter = 0
    
    if args.num_negativos <= 0:
        return total_image_counter, negative_counter
    
    print(f"\nColetando até {args.num_negativos} imagens negativas...")
    
    dataset_negatives = dataset.filter_labels(
        "ground_truth", 
        ~F("label").is_in(ALL_CLASSES)
    )
    
    print(f"Total de imagens negativas disponíveis: {len(dataset_negatives)}")
    
    for sample in dataset_negatives:
        if total_image_counter >= args.max_images:
            break
            
        if args.num_negativos != -1 and negative_counter >= args.num_negativos:
            break
        
        # Copia imagem
        image_path = Path(sample.filepath)
        new_image_name = f"{total_image_counter:06d}{image_path.suffix}"
        new_image_path = images_dir / new_image_name
        
        shutil.copy2(sample.filepath, new_image_path)
        
        images_data.append({
            "filename": new_image_name,
            "human": 0,
            "animal": 0,
            "vehicle": 0
        })
        
        total_image_counter += 1
        negative_counter += 1
        
        if negative_counter % 1000 == 0:
            print(f"  Processadas {negative_counter} imagens negativas...")
    
    return total_image_counter, negative_counter


def save_dataset_json(dataset_dir, args, images_data):
    """
    Salva o arquivo JSON com metadados e labels do dataset.
    
    Args:
        dataset_dir: Path do diretório do dataset
        args: Argumentos da linha de comando
        images_data: Lista de metadados das imagens
    
    Returns:
        Path: Caminho do arquivo JSON salvo
    """
    dataset_info = {
        "dataset_name": args.dataset_name,
        "split": args.split,
        "total_images": len(images_data),
        "classes": ["human", "animal", "vehicle"],
        "class_mapping": class_labels,
        "images": images_data
    }
    
    json_path = dataset_dir / "labels.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)
    
    return json_path


def print_dataset_statistics(images_data, dataset_dir, json_path, images_dir, image_counter):
    """
    Imprime estatísticas do dataset criado.
    
    Args:
        images_data: Lista de metadados das imagens
        dataset_dir: Path do diretório do dataset
        json_path: Path do arquivo JSON
        images_dir: Path do diretório de imagens
        image_counter: Lista com contadores por classe
    """
    # Calcula estatísticas
    total_humans = sum(1 for img in images_data if img['human'] == 1)
    total_animals = sum(1 for img in images_data if img['animal'] == 1)
    total_vehicles = sum(1 for img in images_data if img['vehicle'] == 1)
    total_negatives = sum(1 for img in images_data if img['human'] == 0 and img['animal'] == 0 and img['vehicle'] == 0)
    
    multi_label_images = [img for img in images_data if (img['human'] + img['animal'] + img['vehicle']) > 1]
    
    # Conta combinações
    combinations = defaultdict(int)
    for img in multi_label_images:
        combo = []
        if img['human'] == 1:
            combo.append('human')
        if img['animal'] == 1:
            combo.append('animal')
        if img['vehicle'] == 1:
            combo.append('vehicle')
        combinations['+'.join(combo)] += 1
    
    print(f"\n{'='*60}")
    print(f"Dataset criado com sucesso em: {dataset_dir}")
    print(f"{'='*60}")
    print(f"Total de imagens: {len(images_data)}")
    print(f"Imagens salvas em: {images_dir}")
    print(f"Labels salvos em: {json_path}")
    print(f"\nDistribuição de labels:")
    print(f"  - Humanos: {total_humans} imagens (contador: {image_counter[0]})")
    print(f"  - Animais: {total_animals} imagens (contador: {image_counter[1]})")
    print(f"  - Veículos: {total_vehicles} imagens (contador: {image_counter[2]})")
    print(f"  - Negativas (sem nenhum label): {total_negatives} imagens")
    print(f"\nImagens com múltiplos labels:")
    print(f"  - Total: {len(multi_label_images)} imagens")
    if len(multi_label_images) > 0:
        print(f"\nExemplos de combinações:")
        for combo, count in sorted(combinations.items(), key=lambda x: x[1], reverse=True):
            print(f"    {combo}: {count} imagens")


def main():
    """
    Baixa, filtra e exporta um subconjunto do dataset COCO-2017 com multi-label.
    Salva todas as imagens em uma única pasta e cria um JSON com os labels.
    """
    # Parse argumentos
    args = parse_arguments()
    
    # Configura diretórios
    dataset_dir = Path("../data") / args.dataset_name
    images_dir = dataset_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Carrega dataset COCO
    print(f"Carregando dataset COCO-2017 split '{args.split}'...")
    dataset = foz.load_zoo_dataset("coco-2017", split=args.split)
    dataset.persistent = True
    
    # Filtra imagens com classes de interesse
    dataset_filtered = dataset.filter_labels(
        "ground_truth", 
        F("label").is_in(ALL_CLASSES)
    )
    print(f"Total de imagens com classes de interesse: {len(dataset_filtered)}")
    
    # Lista de classes
    class_list = list(class_labels.keys())  # ['human', 'animal', 'vehicle']
    
    # Coleta imagens positivas (com labels)
    images_data, image_counter, total_image_counter = collect_positive_images(
        dataset_filtered, args, images_dir, class_list
    )
    
    print(f"\nTotal de imagens com labels: {len(images_data)}")
    print(f"Contadores finais por classe:")
    print(f"  - Human: {image_counter[0]} imagens")
    print(f"  - Animal: {image_counter[1]} imagens")
    print(f"  - Vehicle: {image_counter[2]} imagens")
    
    # Coleta imagens negativas (sem labels)
    total_image_counter, negative_counter = collect_negative_images(
        dataset, args, images_dir, images_data, total_image_counter
    )
    
    if negative_counter > 0:
        print(f"\nTotal de imagens negativas: {negative_counter}")
    
    # Salva JSON com metadados
    json_path = save_dataset_json(dataset_dir, args, images_data)
    
    # Imprime estatísticas finais
    print_dataset_statistics(images_data, dataset_dir, json_path, images_dir, image_counter)


if __name__ == "__main__":
    main()
