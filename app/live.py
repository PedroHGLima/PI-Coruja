import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import messagebox
import threading
import queue
import time

# --- Configurações da Aplicação ---
ARQUIVO_PERMISSAO = ".permissao_camera_concedida"
NOME_JANELA_CAMERA = 'Coruja AI - Detecção (ESC para sair)'

# --- Configurações da Estratégia de Inferência (Conforme discutido) ---
# Quantos frames coletar antes de processar em lote
FRAMES_PARA_LOTE = 10 
# A cada quantos segundos capturamos um frame para o lote (1.0 = 1 frame/seg)
SEGUNDOS_POR_AMOSTRA = 1.0 

# --- Configurações do Modelo (Baseado no seu models.py) ---
# !!! ATENÇÃO: COLOQUE O NOME DO SEU ARQUIVO .ONNX AQUI !!!
CAMINHO_MODELO_ONNX = "../model/models/teste.onnx"

# 1. Tamanho da entrada do modelo
TAMANHO_INPUT = 512

# 2. Normalização (do transforms['val'] do seu models.py)
# Estes valores emulam torchvision.transforms.ToTensor() e Normalize()
SCALE_FACTOR = 1.0 / 255.0 # (Converte 0-255 para 0-1)
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

# 3. Limiar de Decisão (do torch.tanh no seu models.py)
# Como a saída é tanh (-1 a +1), o limiar é 0.
LIMIAR_DECISAO = 0.0


def solicitar_permissao_camera_gui():
    """
    Exibe uma caixa de diálogo gráfica para solicitar permissão da câmera.
    (Função original do seu script)
    """
    if os.path.exists(ARQUIVO_PERMISSAO):
        return True
    root = tk.Tk()
    root.withdraw()
    resposta = messagebox.askyesno(
        "Solicitação de Acesso à Câmera",
        "Este programa precisa de acesso à sua webcam para análise em tempo real. Você permite?"
    )
    if resposta:
        with open(ARQUIVO_PERMISSAO, 'w') as f:
            pass
        messagebox.showinfo("Permissão Concedida", "Acessando a câmera...")
        return True
    else:
        messagebox.showerror("Permissão Negada", "O programa será encerrado.")
        return False

class RealTimeDetector:
    """
    Classe que gerencia a captura e a inferência em threads separadas.
    """
    def __init__(self, caminho_modelo):
        # 1. Estado Compartilhado entre Threads
        # Fila thread-safe para passar frames da Thread 1 (Captura) para a Thread 2 (Inferência)
        self.frame_queue = queue.Queue(maxsize=FRAMES_PARA_LOTE + 2) 
        # Variável para o resultado, com um Lock para evitar race conditions
        self.status_atual = "Iniciando..."
        self.status_lock = threading.Lock()
        # Flag para sinalizar o encerramento das threads
        self.running = True

        # 2. Carregar o Modelo ONNX
        print(f"Carregando modelo ONNX de: {caminho_modelo}")
        if not os.path.exists(caminho_modelo):
            messagebox.showerror("Erro de Modelo", f"Arquivo de modelo não encontrado: {caminho_modelo}")
            raise FileNotFoundError(caminho_modelo)
            
        self.net = cv2.dnn.readNetFromONNX(caminho_modelo)
        # Otimização: Define o backend para usar a CPU de forma eficiente
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        print("Modelo carregado com sucesso. Backend: OpenCV/CPU.")

        # 3. Preparar Threads
        # A Thread 2 (Inferência) será daemon, rodando em segundo plano
        self.inference_thread = threading.Thread(target=self.run_inference_worker, daemon=True)

    def start(self):
        """
        Inicia a thread de inferência e, em seguida, executa o loop de captura
        na thread principal (necessário para cv2.imshow).
        """
        print("Iniciando thread de inferência...")
        self.running = True
        self.inference_thread.start()
        
        print("Iniciando loop de captura na thread principal...")
        # O loop de captura bloqueia a thread principal até o usuário sair
        self.run_capture_loop()

    def stop(self):
        """
        Sinaliza para todas as threads pararem e limpa os recursos.
        """
        print("Parando threads...")
        self.running = False
        # Coloca um item 'None' na fila para desbloquear a thread de inferência
        # que pode estar esperando em self.frame_queue.get()
        self.frame_queue.put(None)
        
        # Espera a thread de inferência terminar (com um timeout)
        self.inference_thread.join(timeout=2.0)
        cv2.destroyAllWindows()
        print("Aplicação encerrada.")

    def run_inference_worker(self):
        """
        Esta função roda na Thread 2 (Worker).
        Ela coleta lotes de frames e executa a inferência.
        """
        print("Thread de Inferência (Worker) iniciada.")
        lote_frames = []

        while self.running:
            try:
                # 1. Coletar o lote (espera por frames)
                # .get() bloqueia a thread até um item estar disponível
                # Usamos um timeout para que a thread possa verificar self.running
                frame = self.frame_queue.get(timeout=SEGUNDOS_POR_AMOSTRA)
                
                if frame is None: # Sinal de parada vindo de self.stop()
                    break
                
                lote_frames.append(frame)
                self.frame_queue.task_done() # Sinaliza que o item foi processado

                # 2. Se o lote não está cheio, volta a esperar
                if len(lote_frames) < FRAMES_PARA_LOTE:
                    continue

                # --- O LOTE ESTÁ CHEIO (10 frames) ---
                print(f"[Inferência] Lote de {len(lote_frames)} frames recebido. Processando...")
                start_time = time.time()

                # 3. Pré-processamento: Criar o "blob" para o lote
                # Esta função lida com redimensionamento, troca BGR->RGB e normalização
                # OBS: OpenCV não suporta o parâmetro 'std' em blobFromImages; aplicamos Normalize manualmente.
                blob = cv2.dnn.blobFromImages(
                    lote_frames,
                    scalefactor=SCALE_FACTOR,
                    size=(TAMANHO_INPUT, TAMANHO_INPUT),
                    mean=(0.0, 0.0, 0.0), # não subtrair média aqui
                    swapRB=True, # OpenCV (BGR) -> PyTorch (RGB)
                    crop=False
                )
                mean = np.array(NORM_MEAN, dtype=np.float32).reshape(1, 3, 1, 1)
                std = np.array(NORM_STD, dtype=np.float32).reshape(1, 3, 1, 1)
                blob = (blob - mean) / std

                # 4. Executar Inferência
                self.net.setInput(blob)
                resultados:np.ndarray = self.net.forward()
                
                proc_time = time.time() - start_time
                print(f"[Inferência] Lote processado em {proc_time:.3f}s. (Taxa: {len(lote_frames)/proc_time:.1f} FPS)")

                # 5. Analisar Resultados
                # A saída é (tanh), então positivo (> 0.0) é "Humano"
                deteccao_encontrada = False
                for r in resultados:
                    if r[0] > LIMIAR_DECISAO: # r[0] é o valor da saída
                        deteccao_encontrada = True
                        print("[Inferência] Humano detectado com confiança {:.3f}".format(r[0]))
                        break # Se um frame no lote é positivo, o lote é positivo
                else:
                    print("[Inferência] Nenhum humano detectado no lote com confiança máxima {:.3f}".format(np.abs(np.min(resultados))))
                
                # 6. Atualizar Status (de forma thread-safe)
                with self.status_lock:
                    if deteccao_encontrada:
                        self.status_atual = "Humano Detectado (nos ultimos 10s)"
                    else:
                        self.status_atual = "Status: OK (sem deteccao)"

                # Limpa o lote para a próxima rodada
                lote_frames = []

            except queue.Empty:
                # O .get() deu timeout, o que é normal.
                # Apenas continua o loop e verifica self.running.
                continue
            except Exception as e:
                print(f"Erro na thread de inferência: {e}")
                time.sleep(1) # Evita spam de erro

        print("Thread de Inferência (Worker) encerrada.")

    def run_capture_loop(self):
        """
        Esta função roda na Thread 1 (Principal).
        Ela captura da webcam, exibe o vídeo e amostra frames para a Thread 2.
        """
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Erro Câmera", "Não foi possível abrir a webcam.")
            self.stop()
            return

        proxima_amostra_time = time.time()

        while self.running:
            # 1. Capturar e exibir vídeo (o mais rápido possível)
            ret, frame = cap.read()
            if not ret:
                print("Erro: Não foi possível ler o frame da câmera.")
                time.sleep(0.1)
                continue

            agora = time.time()
            
            # 2. Lógica de Amostragem (1 frame por segundo)
            if agora >= proxima_amostra_time:
                if not self.frame_queue.full():
                    # IMPORTANTE: Copiamos o frame.
                    # Se não copiarmos, a thread de inferência pode processar
                    # um frame que já foi sobrescrito pelo próximo cap.read()
                    frame_copia = frame.copy()
                    self.frame_queue.put(frame_copia)
                else:
                    # A thread de inferência está sobrecarregada
                    print("Aviso: Fila de inferência cheia. Pulando amostra.")
                
                # Agenda a próxima amostra
                proxima_amostra_time = agora + SEGUNDOS_POR_AMOSTRA

            # 3. Obter o status atual (thread-safe)
            with self.status_lock:
                status_para_exibir = self.status_atual
            
            # 4. Desenhar o status na tela
            cor = (0, 0, 255) if "Detectado" in status_para_exibir else (0, 255, 0)
            cv2.putText(
                frame, 
                status_para_exibir, 
                (10, 30), # Posição
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, # Tamanho da fonte
                cor, 
                2 # Espessura
            )
            
            # 5. Exibir o frame
            cv2.imshow(NOME_JANELA_CAMERA, frame)

            # 6. Verificar encerramento (ESC ou fechar janela)
            key = cv2.waitKey(1) & 0xFF
            if key == 27: # Tecla ESC
                break
            if cv2.getWindowProperty(NOME_JANELA_CAMERA, cv2.WND_PROP_VISIBLE) < 1:
                break
        
        # --- Loop encerrado ---
        self.stop()
        cap.release()

# --- Ponto de Entrada Principal ---
def main():
    if not solicitar_permissao_camera_gui():
        return # Usuário negou a permissão

    try:
        detector = RealTimeDetector(CAMINHO_MODELO_ONNX)
        detector.start() # Esta chamada bloqueia até o usuário sair
    except FileNotFoundError:
        print(f"Encerrando: O arquivo {CAMINHO_MODELO_ONNX} não foi encontrado.")
    except KeyboardInterrupt:
        print("\nInterrupção do usuário detectada.")
        detector.stop()
    except Exception as e:
        print(f"Erro fatal não tratado: {e}")
        # Tenta limpar se o detector foi instanciado
        if 'detector' in locals():
            detector.stop()

if __name__ == "__main__":
    main()
