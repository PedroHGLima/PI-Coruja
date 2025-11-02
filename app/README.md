# Projeto Coruja - Sistema Modular

## Estrutura do Código

A aplicação foi refatorada em módulos independentes para facilitar manutenção:

```
app/
├── main.py              # Orquestrador principal - inicializa e conecta os módulos
├── detector.py          # Lógica de detecção - carregamento e inferência do modelo
├── batch_manager.py     # Gerenciamento de batches - captura, processamento e gravação
├── ui.py               # Interface gráfica - Tkinter UI com controles
└── live.py             # [LEGADO] Versão monolítica anterior (manter para referência)
```

## Módulos

### 1. `detector.py` - ModeloDetector

**Responsabilidade:** Carregamento do modelo ONNX e execução de inferências

**Classe principal:** `ModeloDetector`

**Métodos:**
- `__init__(caminho_modelo, limiar)`: Inicializa e carrega o modelo ONNX
- `detectar_batch(frames)`: Processa lista de frames e retorna (bool_deteccao, confianca, resultados)
- `atualizar_limiar(novo_limiar)`: Atualiza o limiar de decisão dinamicamente

**Parâmetros do modelo:**
- Tamanho de entrada: 512x512
- Normalização: ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- Saída: sigmoid (0 a 1)
- Limiar padrão: 0.5

---

### 2. `batch_manager.py` - GerenciadorBatches

**Responsabilidade:** Captura de vídeo, gerenciamento de batches e concatenação/gravação

**Classe principal:** `GerenciadorBatches`

**Métodos:**
- `__init__(detector, duracao_batch, batch_size, ...)`: Configura o gerenciador
- `iniciar()`: Inicia thread de captura de vídeo
- `parar()`: Finaliza captura e salva vídeos pendentes
- `atualizar_parametros(duracao_batch, batch_size)`: Atualiza configurações em tempo real
- `obter_estatisticas()`: Retorna métricas da sessão

**Lógica de concatenação:**
1. Frames são coletados durante `duracao_batch` segundos
2. Ao completar um batch, frames são amostrados e enviados ao detector
3. Se detecção positiva:
   - Inicia nova gravação (se não houver uma ativa)
   - Concatena frames ao vídeo ativo
4. Se sem detecção:
   - Finaliza e salva vídeo ativo (se houver)

**Callbacks:**
- `frame_callback(frame, status)`: Envia frames para a UI de forma thread-safe

---

### 3. `ui.py` - CorujaApp

**Responsabilidade:** Interface gráfica Tkinter com controles e visualização

**Classe principal:** `CorujaApp`

**Métodos:**
- `__init__(root, gerenciador_batches, detector)`: Configura a interface
- `run()`: Inicia o loop principal do Tkinter
- `_toggle_gravacao()`: Controla início/parada da gravação
- `_processar_frames_ui()`: Atualiza canvas com frames (thread-safe via queue)

**Controles disponíveis:**
- **Duração do Batch:** 5-30 segundos (padrão: 10s)
- **Frames para Análise:** 5-30 frames (padrão: 10)
- **Limiar de Decisão:** 0.0 a 1.0 (padrão: 0.5)
- **Botões:** Iniciar/Parar Gravação, Sair

**Thread-safety:**
- Usa `queue.Queue` para receber frames do thread de captura
- Atualiza UI apenas no thread principal via `root.after()`

---

### 4. `main.py` - Orquestrador

**Responsabilidade:** Inicializar e conectar todos os módulos

**Fluxo de execução:**
1. Cria instância de `ModeloDetector` com modelo ONNX
2. Cria instância de `GerenciadorBatches` com o detector
3. Cria janela Tkinter e instância de `CorujaApp`
4. Executa loop principal da UI

**Configurações padrão:**
```python
MODELO_ONNX = "../model/models/v61.onnx"
PASTA_GRAVACOES = "CorujaRecordings"
DEFAULT_DURACAO_BATCH = 10.0
DEFAULT_BATCH_SIZE = 10
DEFAULT_LIMIAR = 0.5  # para modelos com saída sigmoid
FPS_GRAVACAO = 30
```

## Execução

### Pré-requisitos
```bash
cd /home/pedro/PI-Coruja/app
source app_env/bin/activate  # Ativar ambiente virtual
pip install opencv-python pillow  # Se ainda não instalados
```

### Executar aplicação
```bash
python main.py
```

## Arquitetura de Threads

```
Thread Principal (Tkinter)
├── CorujaApp.run()
└── CorujaApp._processar_frames_ui()  # Consome queue a cada 33ms

Thread de Captura (GerenciadorBatches)
├── _run_capture_loop()
│   ├── cap.read()                    # Captura frames
│   ├── batch_frames.append()         # Acumula no batch
│   ├── _processar_batch()            # Quando batch completo
│   └── frame_callback()              # Envia para UI via queue
```

## Fluxo de Dados

```
[Webcam] → [GerenciadorBatches] → [ModeloDetector] → [Decisão]
                  ↓                                        ↓
          [Queue de Frames]                         [Concatenação]
                  ↓                                        ↓
            [CorujaApp]                            [Salvar Vídeo]
```

## Vantagens da Modularização

1. **Separação de responsabilidades:** Cada módulo tem função única e bem definida
2. **Testabilidade:** Módulos podem ser testados independentemente
3. **Manutenibilidade:** Alterações em UI não afetam lógica de detecção
4. **Reutilização:** `ModeloDetector` pode ser usado em outros projetos
5. **Escalabilidade:** Fácil adicionar novos recursos (ex: múltiplas câmeras, streaming)

## Estatísticas Exibidas

Ao parar a gravação, o sistema mostra:
- Total de batches processados
- Batches com detecção de humanos
- Número de vídeos salvos
- Taxa de detecção (%)

## Observações

- Os avisos do Pylance sobre `__getitem__` e `sticky` são falsos positivos
- O código funciona corretamente apesar dos warnings
- Vídeos são salvos em formato `.avi` com codec XVID
- A concatenação é automática: batches consecutivos com detecção formam um único vídeo
