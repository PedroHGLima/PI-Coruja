"""
Módulo de interface gráfica - Tkinter UI para controle da aplicação
"""
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import threading
import queue


class CorujaApp:
    """Interface gráfica para o sistema de detecção e gravação"""
    
    def __init__(self, root, gerenciador_batches, detector):
        """
        Inicializa a interface
        
        Args:
            root: Janela principal do Tkinter
            gerenciador_batches: Instância de GerenciadorBatches
            detector: Instância de ModeloDetector
        """
        self.root = root
        self.root.title("Projeto Coruja - Detecção de Humanos")
        self.gerenciador = gerenciador_batches
        self.detector = detector
        
        # Queue para frames (thread-safe)
        self.ui_frame_queue = queue.Queue(maxsize=2)
        
        # Estado
        self.gravando = False
        self.current_image_ref = None
        
        # Configurar UI
        self._criar_interface()
        
        # Configurar callback do gerenciador
        self.gerenciador.frame_callback = self._processar_frame_capturado
        
        # Iniciar processamento de frames na UI
        self._processar_frames_ui()
    
    def _criar_interface(self):
        """Cria todos os elementos da interface"""
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Canvas para vídeo
        self.canvas = tk.Canvas(main_frame, width=640, height=480, bg='black')
        self.canvas.grid(row=0, column=0, columnspan=2, pady=10)
        
        # Label de status
        self.status_label = ttk.Label(main_frame, text="Status: Aguardando...", 
                                      font=('Arial', 10))
        self.status_label.grid(row=1, column=0, columnspan=2, pady=5)
        
        # Frame de controles
        controls_frame = ttk.LabelFrame(main_frame, text="Parâmetros", padding="10")
        controls_frame.grid(row=2, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))
        
        # Duração do batch
        ttk.Label(controls_frame, text="Duração do Batch (s):").grid(row=0, column=0, sticky=tk.W)
        self.duracao_var = tk.DoubleVar(value=10.0)
        self.duracao_slider = ttk.Scale(controls_frame, from_=5.0, to=30.0, 
                                        variable=self.duracao_var, orient=tk.HORIZONTAL,
                                        command=self._atualizar_duracao)
        self.duracao_slider.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
        self.duracao_value = ttk.Label(controls_frame, text="10.0")
        self.duracao_value.grid(row=0, column=2)
        
        # Frames para análise por batch
        ttk.Label(controls_frame, text="Frames para Análise:").grid(row=1, column=0, sticky=tk.W)
        self.frames_var = tk.IntVar(value=10)
        self.frames_slider = ttk.Scale(controls_frame, from_=5, to=30, 
                                       variable=self.frames_var, orient=tk.HORIZONTAL,
                                       command=self._atualizar_frames)
        self.frames_slider.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=5)
        self.frames_value = ttk.Label(controls_frame, text="10")
        self.frames_value.grid(row=1, column=2)
        
        # Limiar de decisão
        ttk.Label(controls_frame, text="Limiar de Decisão:").grid(row=2, column=0, sticky=tk.W)
        self.limiar_var = tk.DoubleVar(value=0.0)
        self.limiar_slider = ttk.Scale(controls_frame, from_=-1.0, to=1.0, 
                                       variable=self.limiar_var, orient=tk.HORIZONTAL,
                                       command=self._atualizar_limiar)
        self.limiar_slider.grid(row=2, column=1, sticky=(tk.W, tk.E), padx=5)
        self.limiar_value = ttk.Label(controls_frame, text="0.00")
        self.limiar_value.grid(row=2, column=2)
        
        # Configurar colunas expansíveis
        controls_frame.columnconfigure(1, weight=1)
        
        # Botões de controle
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, columnspan=2, pady=10)
        
        self.btn_iniciar = ttk.Button(button_frame, text="Iniciar Gravação", 
                                       command=self._toggle_gravacao)
        self.btn_iniciar.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="Sair", command=self._sair).pack(side=tk.LEFT, padx=5)
        
        # Configurar grid
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
    
    def _processar_frame_capturado(self, frame, status):
        """Callback chamado pelo gerenciador quando há novo frame"""
        # Adicionar à queue (descartar se cheia para evitar acúmulo)
        try:
            self.ui_frame_queue.put_nowait((frame, status))
        except queue.Full:
            # Descartar frames antigos
            try:
                self.ui_frame_queue.get_nowait()
                self.ui_frame_queue.put_nowait((frame, status))
            except:
                pass
    
    def _processar_frames_ui(self):
        """Processa frames da queue e atualiza UI (roda no thread principal)"""
        try:
            while not self.ui_frame_queue.empty():
                frame, status = self.ui_frame_queue.get_nowait()
                
                # Converter BGR para RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Redimensionar se necessário
                h, w = frame_rgb.shape[:2]
                if w > 640 or h > 480:
                    scale = min(640/w, 480/h)
                    new_w, new_h = int(w * scale), int(h * scale)
                    frame_rgb = cv2.resize(frame_rgb, (new_w, new_h))
                
                # Converter para ImageTk
                img = Image.fromarray(frame_rgb)
                imgtk = ImageTk.PhotoImage(image=img)
                
                # Atualizar canvas
                self.canvas.create_image(320, 240, image=imgtk)
                self.current_image_ref = imgtk  # Manter referência
                
                # Atualizar status
                self.status_label.config(text=f"Status: {status}")
        except queue.Empty:
            pass
        except Exception as e:
            print(f"Erro ao processar frame na UI: {e}")
        
        # Reagendar
        self.root.after(33, self._processar_frames_ui)  # ~30 FPS
    
    def _atualizar_duracao(self, valor):
        """Callback para atualizar duração do batch"""
        valor_float = float(valor)
        self.duracao_value.config(text=f"{valor_float:.1f}")
        self.gerenciador.atualizar_parametros(duracao_batch=valor_float)
    
    def _atualizar_frames(self, valor):
        """Callback para atualizar número de frames"""
        valor_int = int(float(valor))
        self.frames_value.config(text=str(valor_int))
        self.gerenciador.atualizar_parametros(batch_size=valor_int)
    
    def _atualizar_limiar(self, valor):
        """Callback para atualizar limiar de decisão"""
        valor_float = float(valor)
        self.limiar_value.config(text=f"{valor_float:.2f}")
        self.detector.atualizar_limiar(valor_float)
    
    def _toggle_gravacao(self):
        """Inicia ou para a gravação"""
        if not self.gravando:
            # Iniciar
            self.gravando = True
            self.gerenciador.gravando = True
            self.gerenciador.iniciar()
            self.btn_iniciar.config(text="Parar Gravação")
            self.status_label.config(text="Status: Gravação iniciada...")
        else:
            # Parar
            self.gravando = False
            self.gerenciador.gravando = False
            self.gerenciador.parar()
            self.btn_iniciar.config(text="Iniciar Gravação")
            
            # Mostrar estatísticas
            stats = self.gerenciador.obter_estatisticas()
            mensagem = (f"Gravação encerrada!\n"
                       f"Batches processados: {stats['total_batches']}\n"
                       f"Batches com detecção: {stats['batches_salvos']}\n"
                       f"Vídeos salvos: {stats['videos_salvos']}\n"
                       f"Taxa de detecção: {stats['taxa_deteccao']:.1f}%")
            print("\n" + "="*50)
            print(mensagem)
            print("="*50 + "\n")
            
            self.status_label.config(text=f"Status: {mensagem.replace(chr(10), ' | ')}")
    
    def _sair(self):
        """Encerra a aplicação"""
        if self.gravando:
            self.gerenciador.gravando = False
            self.gerenciador.parar()
        self.root.quit()
    
    def run(self):
        """Inicia o loop principal da UI"""
        self.root.mainloop()
