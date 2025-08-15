# Projeto Coruja: Detector de Humanos para Otimização de Sistemas de Vigilância

![Status](https://img.shields.io/badge/status-em%20desenvolvimento-yellow)
![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green)

Um projeto de Visão Computacional desenvolvido para a disciplina de **Projeto Integrado**, com o objetivo de reduzir falsos positivos em sistemas de câmeras de segurança através da detecção de presença humana.

---

## 📝 Índice

- [Projeto Coruja: Detector de Humanos para Otimização de Sistemas de Vigilância](#projeto-coruja-detector-de-humanos-para-otimização-de-sistemas-de-vigilância)
  - [📝 Índice](#-índice)
  - [📖 Sobre o Projeto](#-sobre-o-projeto)
  - [🎯 Funcionalidades](#-funcionalidades)
  - [🛠️ Tecnologias Utilizadas](#️-tecnologias-utilizadas)
  - [🚀 Começando](#-começando)
    - [Pré-requisitos](#pré-requisitos)
    - [Instalação](#instalação)
  - [💡 Como Usar](#-como-usar)

---

## 📖 Sobre o Projeto

Sistemas de vigilância equipados com sensores de movimento são notórios por gerar um alto volume de **falsos alarmes**, acionados por eventos irrelevantes como animais, sombras, mudanças de iluminação ou o vento balançando árvores. Este "ruído" acarreta dois problemas principais:

1.  **Desgaste e Custo de Armazenamento:** Gravações desnecessárias ocupam um espaço valioso, reduzindo drasticamente a vida útil de mídias de armazenamento e gerando custos de substituição.
2.  **Fadiga de Alerta:** Usuários que recebem notificações constantes e irrelevantes tendem a ignorar todos os alertas, diminuindo a eficácia do sistema de segurança.

O **Projeto Coruja** atua como uma camada de vigilância inteligente. Como uma coruja que observa atentamente e só age quando necessário, o sistema analisa cada disparo do sensor de movimento. Ele captura um quadro do evento e, usando um modelo de detecção de objetos (YOLOv8), verifica se há uma presença humana. O alerta só é considerado válido — e a gravação iniciada — se uma pessoa for detectada.

Essa validação inteligente visa filtrar o ruído e garantir que o sistema de armazenamento e a atenção do usuário sejam dedicados apenas a eventos verdadeiramente relevantes.

---

## 🎯 Funcionalidades

-   **Detecção Precisa:** Identifica a presença de humanos em imagens com alta acurácia usando o modelo YOLOv8.
-   **Filtragem de Alertas:** Funciona como um classificador binário ("humano" / "não-humano") para validar gatilhos de movimento.
-   **Redução de Falsos Positivos:** Ignora movimentos causados por animais, objetos e outros eventos não-humanos.
-   **Otimização de Armazenamento:** Aumenta a vida útil de mídias de armazenamento ao salvar apenas vídeos relevantes.

---

## 🛠️ Tecnologias Utilizadas

O núcleo deste projeto foi construído com as seguintes tecnologias:

-   **Python 3.9+**
-   **PyTorch**
-   **Ultralytics (YOLOv8)**
-   **OpenCV-Python**
-   **Pillow (PIL)**
-   **NumPy**

---

## 🚀 Começando

Siga os passos abaixo para configurar e executar o projeto em seu ambiente local.

### Pré-requisitos

-   Python 3.9 ou superior
-   PIP (gerenciador de pacotes do Python)
-   Git

### Instalação

1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/](https://github.com/)[SEU-USUARIO]/projeto-coruja.git
    cd projeto-coruja
    ```

2.  **Crie e ative um ambiente virtual** (altamente recomendado):
    ```bash
    # Cria o ambiente
    python -m venv venv

    # Ativa no Windows
    .\venv\Scripts\activate

    # Ativa no Linux/macOS
    source venv/bin/activate
    ```

3.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```

---

## 💡 Como Usar

Para analisar uma única imagem e verificar a presença de humanos, utilize o script principal na pasta `src`. O modelo YOLOv8 será baixado automaticamente na primeira execução.

```bash
python src/detector.py --imagem /caminho/completo/para/sua/imagem.jpg
