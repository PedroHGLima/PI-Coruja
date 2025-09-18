# Base com PyTorch + CUDA (troque para pytorch/pytorch:2.3.0-cpu se não usar GPU)
FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependências úteis para instalar pacotes via pip (ajuste conforme necessário)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copie o requirements.txt (ajuste o caminho se ele estiver fora deste diretório)
COPY requirements.txt .

# Instala as dependências do projeto
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Copia o restante do código
# COPY . .

# Comando padrão; ajuste para seu entrypoint (ex.: script de treino)
CMD ["/bin/bash"]
