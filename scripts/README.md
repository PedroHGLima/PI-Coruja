# Treinamento e rastreamento com MLflow (scripts)

Este README descreve como usar o script `treinar_cnn.py` e como rodar/visualizar as execuções no MLflow.
---

# Scripts: criar_dataset.py e treinar_cnn.py

Este README foi simplificado: o fluxo de MLflow é pensado para uso local (tracking em `mlruns/`).

Pré-requisitos (recomendado usar virtualenv)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# se mlflow não estiver no requirements
pip install mlflow
```

1) Gerar o dataset (ex.: baixa e exporta subconjunto do COCO usando `fiftyone`):

```bash
# usa os padrões definidos em scripts/criar_dataset.py
python3 scripts/criar_dataset.py

# exemplos com parâmetros:
python3 scripts/criar_dataset.py --dataset-name dataset_10k --split validation -p -1 -g -1
python3 scripts/criar_dataset.py -n dataset_5k -s train -p 5000 -g 5000
```

2) Treinar o classificador (usa MLflow local por padrão):

```bash
# executar com padrões
python3 scripts/treinar_cnn.py

# executar com parâmetros (ex.: 20 épocas, batch 32)
python3 scripts/treinar_cnn.py --epochs 20 --batch-size 32 --lr 0.0005 --run-name run_local_01
```

Observações:
- O script `treinar_cnn.py` registra parâmetros e métricas no MLflow local por padrão (pasta `mlruns/`).
- O melhor modelo (por AUC de validação) é salvo localmente em `--models-dir` (padrão `../models`) e também é registrado como artefato na run.


Como iniciar o MLflow UI localmente

```bash
mlflow ui --port 5000
# Abra no navegador: http://localhost:5000
```

Dicas rápidas
- Se o dataset estiver em outro lugar, use `--data-dir` em `treinar_cnn.py`.
- Use `-p -1 -g -1` em `criar_dataset.py` para exportar todas as imagens disponíveis.
- Execute `mlflow ui` em um terminal e rode o script em outro; a UI exibirá runs assim que registradas.

Na UI, selecione o experimento (`coruja_experiment` por padrão) e visualize as runs, métricas por época (train/val loss, acc, auc), parâmetros e artefatos.
