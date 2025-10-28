"""
Módulo de detecção - Carregamento do modelo e inferência
"""
import cv2
import numpy as np
import os


class ModeloDetector:
    """Classe responsável por carregar o modelo ONNX e fazer inferências"""
    
    def __init__(self, caminho_modelo, limiar=0.0):
        """
        Inicializa o detector com o modelo ONNX
        
        Args:
            caminho_modelo: Caminho para o arquivo .onnx
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
            raise FileNotFoundError(f"Modelo não encontrado: {caminho_modelo}")
        
        print(f"Carregando modelo ONNX de: {caminho_modelo}")
        self.net = cv2.dnn.readNetFromONNX(caminho_modelo)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        print("Modelo carregado com sucesso!")
    
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
