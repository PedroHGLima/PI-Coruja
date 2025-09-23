# Pipeline: Dataset → Treino → Avaliação

Este diretório contém os scripts para: (1) gerar o dataset, (2) treinar o classificador Coruja com validação cruzada e rastreamento no MLflow, e (3) avaliar o classificador comparando com modelos YOLO de referência via curvas ROC/AUC.

Arquivos principais:
- `criar_dataset.py`: baixa/filtra/exporta imagens em `human/` e `no_human`.
- `treinar_cnn.py`: treina o classificador (k-fold estratificado) e registra métricas no MLflow.
- `avaliar_redes.py`: compara o classificador Coruja com YOLOs (AUC e desempenho em Hz) e salva curvas ROC.
- `models.py`: define a arquitetura (`CorujaResNet`) e os `transforms_map` de treino/validação.

Diretórios esperados:
- `../data/<dataset>/human` e `../data/<dataset>/no_human` (imagens por classe)
- `../models` (modelos treinados, ex.: `model_fold1.pt`, ...)
- `../mlruns` (rastreamento local do MLflow)

## Pré‑requisitos

Recomendado usar virtualenv:

```bash
cd model
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 1) Gerar dataset

O script usa FiftyOne/COCO para exportar imagens balanceadas em duas pastas: `human/` (com pessoas) e `no_human/` (sem pessoas).

Comandos rápidos:

```bash
cd src
python3 criar_dataset.py

# Todos os positivos/negativos do split validation
python3 criar_dataset.py --dataset-name dataset_all --split validation -p -1 -g -1

# 5k/5k a partir do split train
python3 criar_dataset.py -n dataset_5k -s train -p 5000 -g 5000
```

Argumentos relevantes:
- `--dataset-name|-n`: nome do diretório em `../data` (default: `dataset_10k`).
- `--split|-s`: `train|validation|test` do COCO (default: `validation`).
- `--num-positivos|-p`: quantidade de imagens com pessoa; `-1` pega todas (default: `10000`).
- `--num-negativos|-g`: quantidade de imagens sem pessoa; `-1` pega todas (default: `10000`).

Saída esperada:
```
../data/<dataset_name>/
	human/
	no_human/
```

## 2) Treinar o classificador (k‑fold + MLflow)

O treino usa `StratifiedKFold` e rastreia métricas no MLflow (pasta `../mlruns`). Modelos por fold são salvos como `../models/model_fold<N>.pt` e registrados como artefatos.

Comandos rápidos:

```bash
cd src

# Padrões (dataset em ../data/dataset_10k_train)
python3 treinar_cnn.py

# 20 épocas, batch 32, LR 5e-4, nome da run
python3 treinar_cnn.py --epochs 20 --batch-size 32 --lr 0.0005 --run-name run_local_01

# Fine‑tune da cabeça (descongela layer4/fc)
python3 treinar_cnn.py --unfreeze-head --run-name ft_head

# Treino rápido (debug) com menos dados e 2 folds
python3 treinar_cnn.py --debug

# Começar a partir do fold 3 (aproveita folds anteriores salvos)
python3 treinar_cnn.py --start-at 3 --kfolds 5 --run-name resume_f3
```

O modelo padrão, v60, com maior avaliação até o momento foi treinado com `--unfreeze-head` e `--early-stop-patience=10` no dataset completo de treino (≈100k imagens balanceadas).
O comando para seu treino é:

```bash
python3 treinar_cnn.py --data-dir=../data/dataset_train_full/ --unfreeze-head --early-stop-patience=10 --run-name=v60
```

Argumentos principais:
- `--data-dir`: raiz com `human/` e `no_human` (default: `../data/dataset_10k_train`).
- `--models-dir`: destino dos `.pt` (default: `../models`).
- `--batch-size`, `--epochs`, `--lr`, `--num-workers`.
- `--experiment`, `--run-name`: nomes no MLflow.
- `--unfreeze-head`: fine‑tune das últimas camadas (layer4/fc) da ResNet.
- `--kfolds`: número de folds (default: `5`).
- `--early-stop-patience`, `--early-stop-delta`: parada antecipada por acurácia.
- `--start-at`: fold inicial (1‑based) para pular folds já feitos.
- `--debug`: reduz dados/épocas para iteração rápida.

Detalhes de implementação úteis:
- Arquitetura padrão em `models.py` (`CorujaResNet` sobre ResNet‑50) com saída `tanh` em `[-1, 1]`.
- Alvos em `{−1, 1}` e loss `MSELoss` coerente com a saída `tanh`.
- Métricas: ACC, AUC (ROC); melhor AUC por fold é acompanhado e salvo.
- `transforms_map` define augmentations de treino e pré‑processamento de validação.

MLflow UI local:

```bash
mlflow ui --port 5000
# Navegue em http://localhost:5000 e selecione o experimento (default: coruja_experiment)
```

## 3) Avaliar e comparar com YOLO (ROC/AUC)

Este script calcula a ROC/AUC do classificador Coruja e de um ou mais modelos YOLO (Ultralytics) sobre um dataset balanceado (amostra igual de `human` e `no_human`). Também mede o desempenho médio em Hz por amostra.

Comandos rápidos:

```bash
cd src

# Usando defaults (Coruja ../models/tanh.pt e YOLOv8n como referência)
python3 avaliar_redes.py

# Várias referências YOLO e múltiplas saídas de ROC
python3 avaliar_redes.py -i ../data/dataset_10k/ \
	-m ../models/v60.pt \
	-r ../models/yolov8n.pt ../models/yolo11n.pt \
	-n 10000 -s 1 \
	-o roc_curve.png roc_curve.pdf roc_curve.svg

# Forçar CPU
python3 avaliar_redes.py --device cpu
```

Argumentos principais:
- `--input|-i`: raiz do dataset de avaliação (com `human/` e `no_human`).
- `--model|-m`: caminho do modelo Coruja (`.pt`).
- `--reference|-r`: um ou mais pesos YOLO (ex.: `yolov8n.pt`, `yolo11n.pt`).
- `--num_imgs|-n`: máximo por classe (balanceado).
- `--sections|-s`: número de seções por imagem (pré‑processamento do classificador).
- `--output|-o`: um ou mais arquivos para salvar a ROC (`.png|.pdf|.svg`).
- `--device|-d`: `cuda` (se disponível) ou `cpu`.

Saídas:
- Curvas ROC salvas em cada arquivo de `--output`.
- Log no terminal com AUC de cada modelo e desempenho médio/σ em Hz.

## Alterar a arquitetura do modelo

Edite `models.py`:
- Classe `CorujaResNet`: por padrão usa `torchvision.models.resnet50` com pesos ImageNet e ativa `tanh` na saída (1 logit → binário).
- Parâmetro `unfreeze_head`: permite fine‑tune das camadas `layer4` e `fc`.
- `transforms_map`: ajuste augmentations, `Resize/Crop`, normalização, etc.

Exemplos de alterações:
- Trocar backbone: substituir `models.resnet50(...)` por `models.resnet18(...)` (ou outro modelo suportado).
- Aumentar capacidade: alterar `self.base.fc = nn.Linear(num_ftrs, 1)` para mais camadas (lembre de manter `tanh` se a loss continuar sendo `MSELoss` com alvos em `{−1,1}`).

## Dicas e solução de problemas
- Sem GPU? Use `--device cpu` na avaliação; o treino detecta automaticamente (`cuda` se disponível).
- Dataset próprio: garanta a estrutura `human/` e `no_human` em `--data-dir`.
- Treino lento? Ajuste `--batch-size`, `--num-workers` e considere `--unfreeze-head` apenas quando necessário.
- Avaliação demorada com múltiplos YOLOs: aumente `-n` gradualmente e/ou use `cuda`.
