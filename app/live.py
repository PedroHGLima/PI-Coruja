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

class RealTimeDetector():
    """
    Classe que gerencia a captura e a inferência em threads separadas.
    """
    def __init__(self, caminho_modelo, batch_size=FRAMES_PARA_LOTE, periodo_amostra=SEGUNDOS_POR_AMOSTRA, limiar=LIMIAR_DECISAO):
        self.periodo_amostra = periodo_amostra
        self.batch_size = batch_size
        self.threshold = limiar
        
        self.frame_queue = queue.Queue(maxsize=batch_size + 2) 

        self.status_atual = "Iniciando..."
        self.status_lock = threading.Lock()
        self.running = True

        print(f"Carregando modelo ONNX de: {caminho_modelo}")
        if not os.path.exists(caminho_modelo):
            messagebox.showerror("Erro de Modelo", f"Arquivo de modelo não encontrado: {caminho_modelo}")
            raise FileNotFoundError(caminho_modelo)
            
        self.net = cv2.dnn.readNetFromONNX(caminho_modelo)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        self.inference_thread = threading.Thread(target=self.run_inference_worker, daemon=True)

    def start(self):
        self.running = True
        self.inference_thread.start()
        
        self.run_capture_loop()

    def stop(self):
        self.running = False
        self.frame_queue.put(None)
        
        self.inference_thread.join(timeout=2.0)
        cv2.destroyAllWindows()
        print("Aplicação encerrada.")

    def run_inference_worker(self):
        lote_frames = []

        while self.running:
            try:
                frame = self.frame_queue.get(timeout=self.periodo_amostra)
                
                if frame is None:
                    break
                
                lote_frames.append(frame)
                self.frame_queue.task_done()

                if len(lote_frames) < self.batch_size:
                    continue

                print(f"[Inferência] Lote de {len(lote_frames)} frames recebido. Processando...")
                start_time = time.time()

                blob = cv2.dnn.blobFromImages(
                    lote_frames,
                    scalefactor=SCALE_FACTOR,
                    size=(TAMANHO_INPUT, TAMANHO_INPUT),
                    mean=(0.0, 0.0, 0.0),
                    swapRB=True,
                    crop=False
                )
                mean = np.array(NORM_MEAN, dtype=np.float32).reshape(1, 3, 1, 1)
                std = np.array(NORM_STD, dtype=np.float32).reshape(1, 3, 1, 1)
                blob = (blob - mean) / std

                self.net.setInput(blob)
                resultados:np.ndarray = self.net.forward()
                
                proc_time = time.time() - start_time
                print(f"[Inferência] Lote processado em {proc_time:.3f}s. (Taxa: {len(lote_frames)/proc_time:.1f} FPS)")

                deteccao_encontrada = False
                for r in resultados:
                    if r[0] > self.threshold:
                        deteccao_encontrada = True
                        print("[Inferência] Humano detectado com confiança {:.3f}".format(r[0]))
                        break # Se um frame no lote é positivo, o lote é positivo
                else:
                    print("[Inferência] Nenhum humano detectado no lote com confiança máxima {:.3f}".format(np.abs(np.min(resultados))))
                
                with self.status_lock:
                    if deteccao_encontrada:
                        self.status_atual = "Humano Detectado (nos ultimos 10s)"
                    else:
                        self.status_atual = "Status: OK (sem deteccao)"

                lote_frames = []

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Erro na thread de inferência: {e}")
                time.sleep(1)

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
            ret, frame = cap.read()
            if not ret:
                print("Erro: Não foi possível ler o frame da câmera.")
                time.sleep(0.1)
                continue

            agora = time.time()
            
            if agora >= proxima_amostra_time:
                if not self.frame_queue.full():
                    frame_copia = frame.copy()
                    self.frame_queue.put(frame_copia)
                else:
                    raise OverflowError("Aviso: Fila de inferência cheia. Pulando amostra.")

                proxima_amostra_time = agora + self.periodo_amostra

            with self.status_lock:
                status_para_exibir = self.status_atual
            
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
            
            cv2.imshow(NOME_JANELA_CAMERA, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27: # Tecla ESC
                break
            if cv2.getWindowProperty(NOME_JANELA_CAMERA, cv2.WND_PROP_VISIBLE) < 1:
                break
        
        self.stop()
        cap.release()

def main():
    if not solicitar_permissao_camera_gui():
        return

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
