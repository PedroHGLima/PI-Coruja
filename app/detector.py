"""
Módulo de detecção - Carregamento do modelo e inferência
"""
import cv2
import numpy as np
import os
import requests
from tqdm import tqdm

DOWNLOAD_URL = "https://drive.google.com/uc?export=download&id=1f6rv0E3_3wSQs04ZWfWNrV1fT1iDP_Vb"

class ModeloDetector:
    """Classe responsável por carregar o modelo ONNX e fazer inferências"""
    
    def __init__(self, caminho_modelo, limiar=0.0):
        """
        Inicializa o detector com o modelo ONNX.
        Tenta baixar o modelo da 'url_modelo' se não for encontrado em 'caminho_modelo'.
        
        Args:
            caminho_modelo: Caminho local onde o modelo deve estar (ex: 'model/meu_modelo.onnx')
            url_modelo: (Opcional) URL para baixar o modelo se não for encontrado localmente.
            limiar: Limiar de decisão para classificação (default: 0.0)
        """
        self.limiar = limiar
        self.caminho_modelo = caminho_modelo
        
        # Configurações do modelo (do models.py)
        self.tamanho_input = 512
        self.scale_factor = 1.0 / 255.0
        self.norm_mean = [0.485, 0.456, 0.406]
        self.norm_std = [0.229, 0.224, 0.225]
        
        if not os.path.exists(caminho_modelo):
            print(f"Modelo não encontrado em '{caminho_modelo}'.")
            print(f"Iniciando download de '{DOWNLOAD_URL}'...")
            try:
                ModeloDetector._baixar_modelo_de_link(DOWNLOAD_URL, caminho_modelo)
            except Exception as e:
                print(e)
                return None
        
        print(f"Carregando modelo ONNX de: {caminho_modelo}")
        self.net = cv2.dnn.readNetFromONNX(caminho_modelo)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        print("Modelo carregado com sucesso!")
    
    @staticmethod
    def _baixar_modelo_de_link(url, caminho_destino):
        """
        Baixa um arquivo de uma URL direta com barra de progresso (tqdm).
        Cria diretórios intermediários se não existirem.
        
        Args:
            url: URL direta para o arquivo
            caminho_destino: Caminho completo para salvar o arquivo (incluindo nome)
        """
        diretorio_pai = os.path.dirname(caminho_destino)
        if diretorio_pai and not os.path.exists(diretorio_pai):
            os.makedirs(diretorio_pai, exist_ok=True)
            print(f"Diretório '{diretorio_pai}' criado.")

        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024 * 8  # 8KB por chunk
            
            with open(caminho_destino, 'wb') as file, \
                 tqdm(total=total_size, unit='B', unit_scale=True, unit_divisor=1024,
                      desc=f"Baixando {os.path.basename(caminho_destino)}", leave=True) as progress_bar:
                
                for chunk in response.iter_content(chunk_size=block_size):
                    file.write(chunk)
                    progress_bar.update(len(chunk))
            
            if total_size != 0 and progress_bar.n != total_size:
                raise IOError("Erro: Tamanho do arquivo baixado não corresponde ao esperado.")

        except requests.exceptions.RequestException as e:
            if os.path.exists(caminho_destino):
                os.remove(caminho_destino)
            raise RuntimeError(f"Erro de rede ao baixar {url}: {e}")
        except Exception as e:
            if os.path.exists(caminho_destino):
                os.remove(caminho_destino)
            raise

    def detectar_batch(self, frames):
        """
        Realiza inferência em um batch de frames
        
        Args:
            frames: Lista de frames (numpy arrays BGR)
            
        Returns:
            tuple: (deteccao_encontrada: bool, max_confianca: float, resultados: np.ndarray)
        """
        if len(frames) == 0:
            return False, -1.0, np.array([])
        
        # Preparar blob
        blob = cv2.dnn.blobFromImages(
            frames,
            scalefactor=self.scale_factor,
            size=(self.tamanho_input, self.tamanho_input),
            mean=(0.0, 0.0, 0.0),
            swapRB=True,
            crop=False
        )
        
        # Normalização
        mean = np.array(self.norm_mean, dtype=np.float32).reshape(1, 3, 1, 1)
        std = np.array(self.norm_std, dtype=np.float32).reshape(1, 3, 1, 1)
        blob = (blob - mean) / std
        
        # Inferência
        self.net.setInput(blob)
        resultados: np.ndarray = self.net.forward()
        
        # Verificar detecção
        deteccao_encontrada = False
        max_confianca = -1.0
        
        for r in resultados:
            valor = float(r[0]) if hasattr(r, '__getitem__') else float(r)
            if valor > self.limiar:
                deteccao_encontrada = True
                max_confianca = max(max_confianca, valor)
        
        return deteccao_encontrada, max_confianca, resultados
    
    def atualizar_limiar(self, novo_limiar):
        """Atualiza o limiar de decisão"""
        self.limiar = novo_limiar
