"""
Projeto Coruja - Sistema de Detecção e Gravação de Humanos
Arquivo principal que orquestra os módulos
"""
import tkinter as tk
from detector import ModeloDetector
from batch_manager import GerenciadorBatches
from ui import CorujaApp


# Configurações padrão
MODELO_ONNX = "../model/models/teste.onnx"
PASTA_GRAVACOES = "CorujaRecordings"

# Parâmetros de detecção
DEFAULT_DURACAO_BATCH = 10.0  # segundos
DEFAULT_FREQUENCIA = 1.0  # Hz (frequência de amostragem)
DEFAULT_LIMIAR = 0.0
FPS_GRAVACAO = 30


def main():
    """Função principal"""
    print("="*60)
    print("PROJETO CORUJA - Sistema de Detecção de Humanos")
    print("="*60)
    
    # Inicializar detector
    print(f"\n[1/3] Carregando modelo: {MODELO_ONNX}")
    detector = ModeloDetector(MODELO_ONNX, limiar=DEFAULT_LIMIAR)
    print("✓ Modelo carregado com sucesso!")
    
    # Inicializar gerenciador de batches
    print(f"\n[2/3] Configurando gerenciador de batches...")
    gerenciador = GerenciadorBatches(
        detector=detector,
        duracao_batch=DEFAULT_DURACAO_BATCH,
        frequencia_amostragem=DEFAULT_FREQUENCIA,
        pasta_gravacoes=PASTA_GRAVACOES,
        fps_gravacao=FPS_GRAVACAO
    )
    print("✓ Gerenciador configurado!")
    
    # Inicializar interface
    print(f"\n[3/3] Iniciando interface gráfica...")
    root = tk.Tk()
    app = CorujaApp(root, gerenciador, detector)
    print("✓ Interface pronta!\n")
    
    print("="*60)
    print("Aplicação iniciada! Use a interface para controlar a gravação.")
    print("="*60 + "\n")
    
    # Executar aplicação
    try:
        app.run()
    except KeyboardInterrupt:
        print("\n\nEncerrando aplicação...")
    finally:
        if gerenciador.gravando:
            gerenciador.parar()
        print("Aplicação encerrada.")


if __name__ == "__main__":
    main()
