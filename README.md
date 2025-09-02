# PI-Coruja

Projeto de classificação binária de imagens (humano vs. não-humano) usando PyTorch, ResNet18 e MLflow.

## Passo a passo para começar

1. **Clone o repositório:**
   ```bash
   git clone https://github.com/PedroHGLima/PI-Coruja.git
   cd PI-Coruja
   ```
2. **Crie e ative o ambiente virtual:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```
5. **Execute o MLflow UI para visualizar experimentos:**
   ```bash
   mlflow ui --port 5000
   ```

## Estrutura do Projeto
- `scripts/`: scripts de treinamento, avaliação e definição de modelos
- `models/`: arquivos de pesos salvos (.pth ou .pt)
- `data/`: datasets de treino, validação e imagens para teste
- `notebooks/`: notebooks para inferência, visualização e análise

## Treinamento
O treinamento é feito via `scripts/treinar_cnn.py`, que utiliza validação cruzada (K-Fold), early stopping e logging automático de métricas e artefatos no MLflow.

Exemplo de uso:
```bash
python3 scripts/treinar_cnn.py --data-dir data/dataset_10k_train --models-dir models --epochs 100 --batch-size 64 --experiment coruja_experiment --run-name teste
```

<!-- ## Avaliação
O script `scripts/eval.py` permite avaliar o modelo em um dataset ou classificar imagens de um diretório arbitrário, mostrando predição e probabilidade. -->

## Inferência e Visualização
Notebooks em `notebooks/` permitem:
- Carregar modelos salvos (.pth ou .pt)
- Visualizar imagens, predição e certeza
- Testar robustez do modelo com imagens modificadas (baixa resolução, preto e branco, distorcidas)

## Como importar modelos
- Para modelos salvos como `state_dict` (.pth):
  ```python
  from models import CorujaResNet
  model = CorujaResNet()
  model.load_state_dict(torch.load('models/modelo.pth'))
  model.eval()
  ```
- Para modelos salvos inteiros (.pt):
  ```python
  model = torch.load('models/modelo.pt')
  model.eval()
  ```

## Requisitos
- Python 3.8+
- PyTorch
- torchvision
- tqdm
- scikit-learn
- matplotlib
- MLflow
- torchinfo (para resumo do modelo)

## Observações
- Os scripts e notebooks funcionam tanto em CPU quanto GPU.
- Para importar módulos locais em notebooks, adicione o diretório ao `sys.path`:
  ```python
  import sys
  sys.path.append('../scripts')
  from models import CorujaResNet
  ```
- O projeto está preparado para testes de robustez e visualização interativa.
