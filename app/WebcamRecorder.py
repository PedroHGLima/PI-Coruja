import cv2
import os
import datetime
import tkinter as tk
from tkinter import messagebox

# --- Configurações ---
NOME_DIRETORIO = "CorujaRecordings"
ARQUIVO_PERMISSAO = ".permissao_camera_concedida"
NOME_JANELA_CAMERA = 'Coruja Recordings - Pressione ESC ou feche esta janela para parar'

def solicitar_permissao_camera_gui():
    """
    Exibe uma caixa de diálogo gráfica para solicitar permissão da câmera.
    A permissão é salva em um arquivo oculto para não perguntar novamente.
    """
    # Se o arquivo de permissão já existe, retorna True
    if os.path.exists(ARQUIVO_PERMISSAO):
        return True

    # Inicializa Tkinter sem exibir a janela principal
    root = tk.Tk()
    root.withdraw()  # Esconde a janela principal do Tkinter

    # Exibe a caixa de mensagem
    resposta = messagebox.askyesno(
        "Solicitação de Acesso à Câmera",
        "Este programa precisa de acesso à sua webcam para gravar vídeos. Você permite?"
    )

    if resposta: # True para 'Sim'
        # Cria um arquivo vazio para "lembrar" da permissão
        with open(ARQUIVO_PERMISSAO, 'w') as f:
            pass
        messagebox.showinfo("Permissão Concedida", "Permissão concedida. Acessando a câmera...")
        return True
    else: # False para 'Não'
        messagebox.showerror("Permissão Negada", "Permissão negada. O programa será encerrado.")
        return False

def main():
    """
    Função principal que executa a gravação.
    """
    # 1. Solicitar permissão via GUI
    if not solicitar_permissao_camera_gui():
        return # Encerra o programa se a permissão for negada

    # 2. Criar o diretório de gravações se ele não existir
    if not os.path.exists(NOME_DIRETORIO):
        print(f"Criando diretório para salvar as gravações: '{NOME_DIRETORIO}'")
        os.makedirs(NOME_DIRETORIO)

    # 3. Iniciar a captura da webcam (o '0' geralmente se refere à webcam padrão)
    cap = cv2.VideoCapture(0)

    # Verificar se a webcam foi aberta corretamente
    if not cap.isOpened():
        messagebox.showerror("Erro da Câmera", "Erro: Não foi possível abrir a webcam. Verifique se ela está conectada e não está em uso por outro programa.")
        return

    # Obter as dimensões do vídeo da webcam
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 20.0 # Frames por segundo para a gravação

    # 4. Definir o codec e criar o objeto VideoWriter para salvar o vídeo
    # Gera um nome de arquivo único com base na data e hora atuais
    agora = datetime.datetime.now()
    nome_arquivo = f"gravacao_{agora.strftime('%Y-%m-%d_%H-%M-%S')}.avi"
    caminho_completo = os.path.join(NOME_DIRETORIO, nome_arquivo)

    # O codec 'XVID' é amplamente compatível
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(caminho_completo, fourcc, fps, (frame_width, frame_height))

    print(f"\nIniciando a gravação. O vídeo será salvo em: '{caminho_completo}'")
    print(f"Pressione 'ESC' ou feche a janela '{NOME_JANELA_CAMERA}' para parar a gravação.")

    while True:
        # Captura frame a frame
        ret, frame = cap.read()

        if ret:
            # Mostra o frame em uma janela
            cv2.imshow(NOME_JANELA_CAMERA, frame)

            # Grava o frame no arquivo de vídeo
            out.write(frame)

            # Verifica se a tecla 'ESC' (código ASCII 27) foi pressionada
            # ou se a janela foi fechada pelo 'X'
            # cv2.getWindowProperty verifica se a janela existe ou foi fechada
            if cv2.waitKey(1) & 0xFF == 27 or cv2.getWindowProperty(NOME_JANELA_CAMERA, cv2.WND_PROP_VISIBLE) < 1:
                break
        else:
            print("Erro ao capturar o frame da câmera. Verifique a conexão.")
            break

    # 5. Finalizar e liberar tudo ao encerrar
    print(f"\nGravação encerrada. Vídeo salvo em: '{caminho_completo}'")
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()