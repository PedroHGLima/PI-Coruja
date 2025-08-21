#!/usr/bin/env python3
"""
Script para treinar uma CNN classificadora binária (humano vs. não-humano)
usando Transfer Learning com PyTorch e um modelo ResNet18 pré-treinado.
"""

import argparse
from trainer import CorujaTrainer
import os

def main() -> None:
    parser = argparse.ArgumentParser(description="Treina uma CNN binária com ResNet18 e registra no MLflow")
    parser.add_argument("--data-dir", default="../data/dataset_10k_train")
    parser.add_argument("--models-dir", default="../models")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--experiment", type=str, default="coruja_experiment")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--unfreeze-head", action="store_true", help="Descongelar últimas camadas para fine-tune")
    parser.add_argument("--kfolds", type=int, default=5, help="Número de folds para validação cruzada (default=5)")
    parser.add_argument("--max-batches", type=int, default=100, help="Número máximo de batches por época (default=100)")
    parser.add_argument("--early-stop-patience", type=int, default=5, help="Número de épocas sem evolução mínima de acurácia para parar (default=5)")
    parser.add_argument("--early-stop-delta", type=float, default=0.01, help="Evolução mínima de acurácia para não parar (default=0.01 = 1%)")
    parser.add_argument("--debug", action="store_true", help="Modo debug: treinamento rápido com poucos dados e épocas")
    args = parser.parse_args()

    if args.debug:
        args.epochs = 5
        args.max_batches = 5
        args.kfolds = 2
        args.batch_size = 8
        debug_dataset = args.data_dir.replace('dataset_10k_train', 'dataset_10k_debug')
        args.data_dir = debug_dataset if os.path.exists(debug_dataset) else args.data_dir
        print("[DEBUG] Modo rápido ativado: epochs=5, max_batches=5, kfolds=2, batch_size=8, data_dir=", args.data_dir)

    trainer = CorujaTrainer(args)
    trainer.train_kfold()

if __name__ == '__main__':
    main()
