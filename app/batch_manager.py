"""
Módulo de gerenciamento de batches - Captura, processamento e gravação
Refatorado para arquitetura multi-thread (Produtor-Consumidor)
"""
import cv2
import numpy as np
import os
import time
import threading
import queue


class GerenciadorBatches:
    """Gerencia a captura de vídeo, processamento em batches e gravação
    usando threads separadas para captura, processamento e gravação (I/O).
    """
    
    def __init__(self, detector, duracao_batch=10.0, frequencia_amostragem=1.0, 
                 pasta_gravacoes="CorujaRecordings", fps_gravacao=30,
                 frame_callback=None):
        
        self.detector = detector
        self.duracao_batch = duracao_batch
        self.frequencia_amostragem = frequencia_amostragem
        self.pasta_gravacoes = pasta_gravacoes
        self.fps_gravacao = fps_gravacao
        self.frame_callback = frame_callback
        
        self._atualizar_batch_size()
        
        # Estado
        self.gravando = False
        self.running = False
        
        # Propriedades do vídeo (serão atualizadas pela thread de captura)
        self.frame_width = 640
        self.frame_height = 480
        
        # Estado da gravação (controlado pela thread de processamento)
        self.gravacao_ativa = False
        
        # Estatísticas
        self.total_batches = 0
        self.batches_salvos = 0
        self.videos_salvos = 0
        
        # Status
        self.status_atual = "Iniciando..."
        self.status_lock = threading.Lock()
        
        # --- Filas Thread-Safe ---
        # Thread 1 (Captura) -> Thread 2 (Processamento)
        self.fila_processamento = queue.Queue(maxsize=300)
        
        # Thread 2 (Processamento) -> Thread 3 (Gravação)
        self.fila_gravacao = queue.Queue()

        # --- Threads ---
        self.capture_thread = None
        self.processing_thread = None
        self.recording_thread = None
        
        # --- Lógica de Post-Roll (ADIÇÃO OBRIGATÓRIA) ---
        # Contador de batches sem detecção APÓS uma gravação ativa
        self.batches_sem_deteccao_pos = 0

        # Quantidade de batches para continuar gravando após a última detecção
        # (ex: 2 batches * 5s = 10 segundos de post-roll)
        self.max_batches_post_roll = 2
        
        os.makedirs(pasta_gravacoes, exist_ok=True)
    
    def iniciar(self):
        """Inicia as 3 threads de operação"""
        self.running = True
        
        # Thread 1: Apenas captura e distribui
        self.capture_thread = threading.Thread(target=self._loop_captura, daemon=True)
        
        # Thread 2: Monta batches e detecta
        self.processing_thread = threading.Thread(target=self._loop_processamento, daemon=True)
        
        # Thread 3: Grava no disco (I/O lento)
        self.recording_thread = threading.Thread(target=self._loop_gravacao, daemon=True)
        
        self.capture_thread.start()
        self.processing_thread.start()
        self.recording_thread.start()
        print("Gerenciador iniciado com 3 threads (Captura, Processamento, Gravação).")
    
    def parar(self):
        """Para todas as threads e finaliza gravações pendentes"""
        if not self.running:
            return
            
        print("Iniciando parada do Gerenciador de Batches...")
        self.running = False
        
        # Envia sinais de parada (None) para as threads baseadas em filas
        # (A thread de captura para por self.running = False)
        self.fila_processamento.put(None)
        self.fila_gravacao.put(None)
        
        # Aguardar threads
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)
        if self.recording_thread and self.recording_thread.is_alive():
            self.recording_thread.join(timeout=5.0) # Mais tempo para I/O de disco
        
        print("Gerenciador de batches encerrado.")
    
    def _atualizar_batch_size(self):
        self.batch_size = max(1, int(self.duracao_batch * self.frequencia_amostragem))
    
    def atualizar_parametros(self, duracao_batch=None, frequencia_amostragem=None):
        if duracao_batch is not None:
            self.duracao_batch = duracao_batch
        if frequencia_amostragem is not None:
            self.frequencia_amostragem = frequencia_amostragem
        self._atualizar_batch_size()
    
    # -----------------------------------------------------------------
    # THREAD 1: Captura (Rápido)
    # -----------------------------------------------------------------
    def _loop_captura(self):
        """Loop principal de captura (roda na Thread 1)"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Erro: Não foi possível abrir a webcam.")
            if self.frame_callback:
                erro_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(erro_frame, "ERRO: Camera nao encontrada", (50, 240),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                self.frame_callback(erro_frame, "Erro: Câmera não disponível")
            self.running = False
            return
            
        # Atualizar dimensões para a thread de gravação
        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Câmera iniciada ({self.frame_width}x{self.frame_height})")
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                print("Erro: Não foi possível ler o frame da câmera.")
                time.sleep(0.1)
                continue
            
            # 1. Enviar para UI (sempre)
            if self.frame_callback:
                with self.status_lock:
                    status = self.status_atual
                self.frame_callback(frame, status) # A UI já tem sua própria fila
            
            # 2. Enviar para processamento (se gravando)
            if self.gravando:
                try:
                    # 'block=False' descarta o frame se a fila de processamento estiver cheia
                    self.fila_processamento.put(frame.copy(), block=False)
                except queue.Full:
                    # Isso é bom! Evita que a RAM estoure se a detecção atrasar.
                    pass # print("Aviso: Fila de processamento cheia, descartando frame.")
            
            #time.sleep(0.01) # Pequeno sleep para não sobrecarregar a CPU
        
        cap.release()
        print("Thread de captura encerrada.")
    
    # -----------------------------------------------------------------
    # THREAD 2: Processamento/Detecção (CPU/AI)
    # -----------------------------------------------------------------
    def _loop_processamento(self):
        """Loop que monta batches e roda a detecção (roda na Thread 2)"""
        batch_frames = []
        batch_start_time = time.time()
        
        while True:
            try:
                # Espera por um frame (ou sinal de parada)
                frame = self.fila_processamento.get(timeout=0.5)
                if frame is None: # Sinal de parada
                    break
                
                if self.gravando:
                    batch_frames.append(frame)
                
                # Verificar se batch está completo
                tempo_decorrido = time.time() - batch_start_time
                if self.gravando and tempo_decorrido >= self.duracao_batch:
                    self._processar_deteccao(batch_frames)
                    batch_frames = [] # Limpa batch local
                    batch_start_time = time.time()

            except queue.Empty:
                # Timeout de 0.5s
                if not self.running:
                    break
                
                # Se não estiver gravando, limpa frames acumulados
                if not self.gravando and len(batch_frames) > 0:
                    batch_frames = []
                
                # Se o tempo estourou e temos frames, processa (mesmo se parou de gravar)
                tempo_decorrido = time.time() - batch_start_time
                if len(batch_frames) > 0 and (tempo_decorrido >= self.duracao_batch):
                    self._processar_deteccao(batch_frames)
                    batch_frames = []
                    batch_start_time = time.time()
                
                continue
        
        # Processar o que sobrou
        if len(batch_frames) > 0:
            self._processar_deteccao(batch_frames)

        print("Thread de processamento encerrada.")

    def _processar_deteccao(self, frames_batch):
        """Processa um batch: Inferência e lógica de Pre/Post-Roll"""
        if len(frames_batch) == 0:
            return
        
        self.total_batches += 1
        num_frames = len(frames_batch)
        
        # --- Amostragem (lógica do seu arquivo original) ---
        num_amostras = min(self.batch_size, num_frames)
        print(f"\n[Batch] Processando batch #{self.total_batches} com {num_frames} frames. Amostrando {num_amostras}...")
        
        indices = np.linspace(0, num_frames-1, num_amostras, dtype=int)
        frames_amostra = [frames_batch[i] for i in indices]
        
        # --- Fazer inferência ---
        deteccao, confianca, resultados = self.detector.detectar_batch(frames_amostra)
        
        # --- Lógica de Gravação (com Post-Roll) ---
        if deteccao:
            self.batches_salvos += 1
            print(f"[Batch] ✓ HUMANO DETECTADO! Confiança: {confianca:.3f}")
            
            # 1. Enviar o batch inteiro para a THREAD DE GRAVAÇÃO
            self.fila_gravacao.put(list(frames_batch)) # Envia uma cópia
            
            # 2. Iniciar gravação (se não estiver ativa)
            if not self.gravacao_ativa:
                self.gravacao_ativa = True
                print(f"[Gravação] Sinal de INÍCIO enviado para thread de gravação.")
            
            # 3. Resetar o contador de "post-roll"
            self.batches_sem_deteccao_pos = 0
            
            with self.status_lock:
                self.status_atual = f"Gravando... (Detecção) | Salvos: {self.videos_salvos} vídeos"
        
        else:
            # --- Sem detecção neste batch ---
            print(f"[Batch] Sem detecção (confiança máx: {np.max(resultados) if len(resultados) > 0 else 'N/A'})")
            
            if self.gravacao_ativa:
                # Uma gravação ESTÁ ativa. Verificar se é hora de parar.
                self.batches_sem_deteccao_pos += 1
                
                if self.batches_sem_deteccao_pos <= self.max_batches_post_roll:
                    # Ainda estamos no período de "post-roll"
                    print(f"[Gravação] Gravando 'post-roll' ({self.batches_sem_deteccao_pos}/{self.max_batches_post_roll})")
                    
                    # Salva o batch MESMO sem detecção
                    self.fila_gravacao.put(list(frames_batch)) 
                    
                    with self.status_lock:
                        self.status_atual = f"Gravando... (Post-Roll {self.batches_sem_deteccao_pos}) | Salvos: {self.videos_salvos} vídeos"
                else:
                    # Cooldown do post-roll acabou. Parar a gravação.
                    print(f"[Gravação] Sinal de FIM (post-roll concluído) enviado.")
                    self.fila_gravacao.put("STOP")
                    self.gravacao_ativa = False
                    self.batches_sem_deteccao_pos = 0 # Resetar contador
                    
                    with self.status_lock:
                        self.status_atual = f"Monitorando... Salvos: {self.videos_salvos} vídeos"
            else:
                # Não estava gravando e não detectou. Apenas monitorando.
                with self.status_lock:
                    self.status_atual = f"Monitorando... Salvos: {self.videos_salvos} vídeos"

    # -----------------------------------------------------------------
    # THREAD 3: Gravação em Disco (I/O Lento)
    # -----------------------------------------------------------------
    def _loop_gravacao(self):
        """Loop que salva vídeos no disco (roda na Thread 3)"""
        writer = None
        current_timestamp = None
        frames_escritos = 0
        
        while True:
            try:
                # Bloqueia até receber um batch (lista), um sinal ("STOP") ou (None)
                data = self.fila_gravacao.get() 
                
                if data is None: 
                    # Sinal de parada FINAL (de self.parar())
                    break
                
                if data == "STOP":
                    # Sinal para finalizar o vídeo ATUAL
                    if writer:
                        writer.release()
                        writer = None
                        self.videos_salvos += 1
                        duracao = frames_escritos / self.fps_gravacao
                        print(f"[Gravação] ✓ Vídeo salvo: {current_timestamp} ({duracao:.1f}s, {frames_escritos} frames)")
                        current_timestamp = None
                        frames_escritos = 0
                
                elif isinstance(data, list):
                    # É um batch de frames para gravar
                    
                    # Se não há um vídeo sendo gravado, cria um novo
                    if writer is None:
                        current_timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
                        nome_arquivo = os.path.join(
                            self.pasta_gravacoes, 
                            f"deteccao_{current_timestamp}.avi"
                        )
                        fourcc = cv2.VideoWriter_fourcc(*'XVID')
                        writer = cv2.VideoWriter(nome_arquivo, fourcc, self.fps_gravacao, 
                                               (self.frame_width, self.frame_height))
                        frames_escritos = 0
                        print(f"[Gravação] Iniciando novo vídeo: {nome_arquivo}")

                    # Escreve os frames no disco (operação lenta e bloqueante)
                    for frame in data:
                        writer.write(frame)
                    
                    frames_escritos += len(data)
                    # print(f"[Gravação] Batch de {len(data)} frames escrito no disco.")
            
            except Exception as e:
                print(f"Erro na thread de gravação: {e}")
        
        # Limpeza final se o app parar no meio de uma gravação
        if writer:
            writer.release()
            self.videos_salvos += 1
            duracao = frames_escritos / self.fps_gravacao
            print(f"[Gravação] ✓ Vídeo final (por parada) salvo: {current_timestamp} ({duracao:.1f}s)")

        print("Thread de gravação encerrada.")

    
    def obter_estatisticas(self):
        """Retorna estatísticas da sessão"""
        return {
            'total_batches': self.total_batches,
            'batches_salvos': self.batches_salvos,
            'videos_salvos': self.videos_salvos,
            'taxa_deteccao': (self.batches_salvos / max(1, self.total_batches) * 100)
        }
