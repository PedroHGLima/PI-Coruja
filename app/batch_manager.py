"""
Módulo de gerenciamento de batches - Captura, processamento e gravação
"""
import cv2
import numpy as np
import os
import time
import threading
import queue
from detector import ModeloDetector


class GerenciadorBatches:
    """Gerencia a captura de vídeo, processamento em batches e gravação"""
    
    def __init__(self, detector, duracao_batch=10.0, frequencia_amostragem=1.0, 
                 pasta_gravacoes="CorujaRecordings", fps_gravacao=30,
                 frame_callback=None):
        """
        Inicializa o gerenciador de batches
        
        Args:
            detector: Instância de ModeloDetector
            duracao_batch: Duração de cada batch em segundos
            frequencia_amostragem: Frequência de amostragem em Hz (frames/segundo)
            pasta_gravacoes: Pasta onde salvar os vídeos
            fps_gravacao: FPS dos vídeos salvos
            frame_callback: Função callback(frame, status) para enviar frames para UI
        """
        self.detector = detector
        self.duracao_batch = duracao_batch
        self.frequencia_amostragem = frequencia_amostragem
        self.pasta_gravacoes = pasta_gravacoes
        self.fps_gravacao = fps_gravacao
        self.frame_callback = frame_callback
        
        # Calcular batch_size baseado na frequência e duração
        self._atualizar_batch_size()
        
        # Estado de gravação
        self.gravando = False
        self.running = False
        
        # Batch atual
        self.batch_frames = []
        self.batch_start_time = None
        
        # Concatenação de vídeos
        self.gravacao_ativa = False
        self.frames_gravacao_ativa = []
        self.timestamp_gravacao = None
        
        # Estatísticas
        self.total_batches = 0
        self.batches_salvos = 0
        self.videos_salvos = 0
        
        # Status
        self.status_atual = "Iniciando..."
        self.status_lock = threading.Lock()
        
        # Thread de captura
        self.capture_thread = None
        
        # Criar pasta de gravações
        os.makedirs(pasta_gravacoes, exist_ok=True)
    
    def iniciar(self):
        """Inicia a thread de captura"""
        self.running = True
        self.capture_thread = threading.Thread(target=self._run_capture_loop, daemon=True)
        self.capture_thread.start()
    
    def parar(self):
        """Para a captura e finaliza gravações pendentes"""
        self.running = False
        
        # Finalizar gravação ativa se houver
        if self.gravacao_ativa and len(self.frames_gravacao_ativa) > 0:
            h, w = self.frames_gravacao_ativa[0].shape[:2]
            self._finalizar_gravacao(w, h)
        
        # Aguardar thread
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
        
        print("Gerenciador de batches encerrado.")
    
    def _atualizar_batch_size(self):
        """Calcula batch_size baseado em duracao_batch e frequencia_amostragem"""
        self.batch_size = max(1, int(self.duracao_batch * self.frequencia_amostragem))
    
    def atualizar_parametros(self, duracao_batch=None, frequencia_amostragem=None):
        """Atualiza parâmetros dinamicamente"""
        if duracao_batch is not None:
            self.duracao_batch = duracao_batch
        if frequencia_amostragem is not None:
            self.frequencia_amostragem = frequencia_amostragem
        
        # Recalcular batch_size após qualquer atualização
        self._atualizar_batch_size()
    
    def _run_capture_loop(self):
        """Loop principal de captura (roda em thread separada)"""
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
        
        # Propriedades do vídeo
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.batch_start_time = time.time()
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                print("Erro: Não foi possível ler o frame da câmera.")
                time.sleep(0.1)
                continue
            
            agora = time.time()
            
            # Adicionar frame ao batch se estiver gravando
            if self.gravando:
                self.batch_frames.append(frame.copy())
                
                # Verificar se batch está completo
                tempo_decorrido = agora - self.batch_start_time
                if tempo_decorrido >= self.duracao_batch:
                    self._processar_batch(frame_width, frame_height)
                    self.batch_start_time = agora
            
            # Enviar frame para UI
            if self.frame_callback:
                with self.status_lock:
                    status = self.status_atual
                self.frame_callback(frame, status)
            
            time.sleep(0.033)  # ~30 FPS
        
        cap.release()
        print("Thread de captura encerrada.")
    
    def _processar_batch(self, width, height):
        """Processa um batch: inferência e gravação"""
        if len(self.batch_frames) == 0:
            return
        
        self.total_batches += 1
        num_frames = len(self.batch_frames)
        print(f"\n[Batch] Processando batch #{self.total_batches} com {num_frames} frames...")
        
        # Usar batch_size já calculado (duracao_batch * frequencia_amostragem)
        num_amostras = min(self.batch_size, num_frames)  # Não pode ser maior que frames disponíveis
        
        print(f"[Batch] Amostrando {num_amostras} frames a {self.frequencia_amostragem:.2f} Hz...")
        
        # Amostrar frames para inferência
        indices = np.linspace(0, num_frames-1, num_amostras, dtype=int)
        frames_amostra = [self.batch_frames[i] for i in indices]
        
        # Fazer inferência
        deteccao, confianca, resultados = self.detector.detectar_batch(frames_amostra)
        
        if deteccao:
            self.batches_salvos += 1
            print(f"[Batch] ✓ HUMANO DETECTADO! Confiança: {confianca:.3f}")
            
            # Iniciar nova gravação se necessário
            if not self.gravacao_ativa:
                self.timestamp_gravacao = time.strftime("%Y-%m-%d_%H-%M-%S")
                self.gravacao_ativa = True
                self.frames_gravacao_ativa = []
                print(f"[Gravação] Iniciando novo vídeo: {self.timestamp_gravacao}")
            
            # Concatenar frames
            self.frames_gravacao_ativa.extend(self.batch_frames)
            duracao_atual = len(self.frames_gravacao_ativa) / self.fps_gravacao
            print(f"[Gravação] Batch concatenado. Duração total: {duracao_atual:.1f}s")
            
            with self.status_lock:
                self.status_atual = f"Gravando vídeo: {duracao_atual:.1f}s | Salvos: {self.videos_salvos} vídeos"
        else:
            print(f"[Batch] Sem detecção (confiança máx: {np.max(resultados) if len(resultados) > 0 else 'N/A'})")
            
            # Finalizar gravação ativa
            if self.gravacao_ativa:
                self._finalizar_gravacao(width, height)
            
            with self.status_lock:
                self.status_atual = f"Monitorando... Salvos: {self.videos_salvos} vídeos"
        
        # Limpar batch
        self.batch_frames = []
    
    def _finalizar_gravacao(self, width, height):
        """Finaliza e salva a gravação ativa"""
        if not self.gravacao_ativa or len(self.frames_gravacao_ativa) == 0:
            return
        
        nome_arquivo = os.path.join(
            self.pasta_gravacoes, 
            f"deteccao_{self.timestamp_gravacao}.avi"
        )
        
        fourcc = cv2.VideoWriter.fourcc(*'XVID')
        out = cv2.VideoWriter(nome_arquivo, fourcc, self.fps_gravacao, (width, height))
        
        for frame in self.frames_gravacao_ativa:
            out.write(frame)
        
        out.release()
        self.videos_salvos += 1
        
        duracao = len(self.frames_gravacao_ativa) / self.fps_gravacao
        print(f"[Gravação] ✓ Vídeo salvo: {nome_arquivo}")
        print(f"[Gravação] Duração: {duracao:.1f}s ({len(self.frames_gravacao_ativa)} frames)")
        
        # Reset
        self.gravacao_ativa = False
        self.frames_gravacao_ativa = []
        self.timestamp_gravacao = None
    
    def obter_estatisticas(self):
        """Retorna estatísticas da sessão"""
        return {
            'total_batches': self.total_batches,
            'batches_salvos': self.batches_salvos,
            'videos_salvos': self.videos_salvos,
            'taxa_deteccao': (self.batches_salvos / max(1, self.total_batches) * 100)
        }
