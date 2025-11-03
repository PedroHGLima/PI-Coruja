"""
Módulo de interface gráfica - Tkinter UI para controle da aplicação
"""
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
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
        self.ultima_deteccao = "Aguardando..."
        self.ultima_confianca = 0.0
        self.progresso_batch = 0.0
        
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
        main_frame.grid(row=0, column=0, sticky=tk.W + tk.E + tk.N + tk.S)
        
        # Frame esquerdo - Vídeo
        video_frame = ttk.Frame(main_frame)
        video_frame.grid(row=0, column=0, sticky=tk.N + tk.S + tk.W, padx=(0, 10))
        
        # Canvas para vídeo
        self.canvas = tk.Canvas(video_frame, width=640, height=480, bg='black')
        self.canvas.pack(pady=5)
        
        # Label de status
        self.status_label = ttk.Label(video_frame, text="Status: Aguardando...", 
                                      font=('Arial', 10))
        self.status_label.pack(pady=5)
        
        # Frame direito - Controles e informações
        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=0, column=1, sticky=tk.N + tk.S + tk.E + tk.W)
        
        # Frame de última detecção
        detection_frame = ttk.LabelFrame(right_frame, text="Última Detecção", padding="10")
        detection_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.detection_label = ttk.Label(detection_frame, text="Aguardando...", 
                                        font=('Arial', 12, 'bold'),
                                        foreground='gray')
        self.detection_label.pack(pady=5)
        
        self.confidence_label = ttk.Label(detection_frame, text="Confiança: 0.0%", 
                                         font=('Arial', 10))
        self.confidence_label.pack(pady=5)
        
        # Frame de progresso do batch
        progress_frame = ttk.LabelFrame(right_frame, text="Progresso do Batch", padding="10")
        progress_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.progress_bar = ttk.Progressbar(progress_frame, mode='determinate', 
                                           length=300, maximum=100)
        self.progress_bar.pack(pady=5)
        
        self.progress_label = ttk.Label(progress_frame, text="0.0s / 10.0s (0%)", 
                                       font=('Arial', 9))
        self.progress_label.pack(pady=2)
        
        # Frame de controles
        controls_frame = ttk.LabelFrame(right_frame, text="Parâmetros", padding="10")
        controls_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Duração do batch
        ttk.Label(controls_frame, text="Duração do Batch (s):").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.duracao_var = tk.DoubleVar(value=10.0)
        self.duracao_slider = ttk.Scale(controls_frame, from_=1.0, to=30.0, 
                                        variable=self.duracao_var, orient=tk.HORIZONTAL,
                                        command=self._atualizar_duracao)
        self.duracao_slider.grid(row=0, column=1, sticky=tk.W + tk.E, padx=5)
        self.duracao_entry = ttk.Entry(controls_frame, width=8)
        self.duracao_entry.insert(0, "10.0")
        self.duracao_entry.grid(row=0, column=2, padx=5)
        self.duracao_entry.bind('<Return>', self._on_duracao_entry)
        self.duracao_entry.bind('<FocusOut>', self._on_duracao_entry)
        
        # Frequência de amostragem (escala logarítmica, limitada a 2 Hz)
        ttk.Label(controls_frame, text="Frequência (Hz):").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.frequencia_var = tk.DoubleVar(value=0.0)  # 0 -> 10^0 = 1 Hz padrão
        # Escala logarítmica: slider vai de -1 (0.1 Hz) a 0.301 (2 Hz)
        # log10(2) ≈ 0.301
        self.freq_slider = ttk.Scale(controls_frame, from_=-1.0, to=0.301, 
                                       variable=self.frequencia_var, orient=tk.HORIZONTAL,
                                       command=self._atualizar_frequencia)
        self.freq_slider.grid(row=1, column=1, sticky=tk.W + tk.E, padx=5)
        self.freq_entry = ttk.Entry(controls_frame, width=8)
        self.freq_entry.insert(0, "1.0")
        self.freq_entry.grid(row=1, column=2, padx=5)
        self.freq_entry.bind('<Return>', self._on_freq_entry)
        self.freq_entry.bind('<FocusOut>', self._on_freq_entry)
        
        # Limiar de decisão
        ttk.Label(controls_frame, text="Limiar de Decisão:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.limiar_var = tk.DoubleVar(value=0.0)
        self.limiar_slider = ttk.Scale(controls_frame, from_=-1.0, to=1.0, 
                                       variable=self.limiar_var, orient=tk.HORIZONTAL,
                                       command=self._atualizar_limiar)
        self.limiar_slider.grid(row=2, column=1, sticky=tk.W + tk.E, padx=5)
        self.limiar_entry = ttk.Entry(controls_frame, width=8)
        self.limiar_entry.insert(0, "0.00")
        self.limiar_entry.grid(row=2, column=2, padx=5)
        self.limiar_entry.bind('<Return>', self._on_limiar_entry)
        self.limiar_entry.bind('<FocusOut>', self._on_limiar_entry)
        
        # Configurar colunas expansíveis
        controls_frame.columnconfigure(1, weight=1)
        
        # Botões de controle
        button_frame = ttk.Frame(right_frame)
        button_frame.pack(pady=10)
        
        self.btn_iniciar = ttk.Button(button_frame, text="Iniciar Gravação", 
                                       command=self._toggle_gravacao)
        self.btn_iniciar.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="Sair", command=self._sair).pack(side=tk.LEFT, padx=5)
        
        # Configurar grid
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
    
    def _processar_frame_capturado(self, frame, status, resultado_deteccao=None, progresso_batch=None):
        """Callback chamado pelo gerenciador quando há novo frame"""
        # Verificar se a UI ainda está ativa
        if not self.root or not self.root.winfo_exists():
            return
            
        # Adicionar à queue (descartar se cheia para evitar acúmulo)
        try:
            self.ui_frame_queue.put_nowait((frame, status, resultado_deteccao, progresso_batch))
        except queue.Full:
            # Descartar frames antigos
            try:
                self.ui_frame_queue.get_nowait()
                self.ui_frame_queue.put_nowait((frame, status, resultado_deteccao, progresso_batch))
            except:
                pass
    
    def _processar_frames_ui(self):
        """Processa frames da queue e atualiza UI (roda no thread principal)"""
        # Verificar se a janela ainda existe
        if not self.root.winfo_exists():
            return
            
        try:
            while not self.ui_frame_queue.empty():
                frame, status, resultado_deteccao, progresso_batch = self.ui_frame_queue.get_nowait()
                
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
                
                # Atualizar informações de detecção se disponíveis
                if resultado_deteccao is not None:
                    deteccao, confianca = resultado_deteccao
                    if deteccao:
                        self.detection_label.config(text="HUMANO DETECTADO", 
                                                   foreground='green')
                        # Converter confiança para porcentagem (valores entre -1 e 1, normalizar para 0-100%)
                        conf_percentual = ((confianca + 1) / 2) * 100
                        self.confidence_label.config(text=f"Confiança: {conf_percentual:.1f}%")
                    else:
                        self.detection_label.config(text="Nenhum humano", 
                                                   foreground='orange')
                        # Para não detecção, mostrar 0% ou valor baixo
                        if confianca > 0:
                            conf_percentual = ((confianca + 1) / 2) * 100
                            self.confidence_label.config(text=f"Confiança: {conf_percentual:.1f}%")
                        else:
                            self.confidence_label.config(text=f"Confiança: 0.0%")
                
                # Atualizar barra de progresso se disponível
                if progresso_batch is not None:
                    tempo_decorrido, duracao_total = progresso_batch
                    percentual = (tempo_decorrido / duracao_total * 100) if duracao_total > 0 else 0
                    percentual = min(100, max(0, percentual))  # Limitar entre 0 e 100
                    
                    self.progress_bar['value'] = percentual
                    self.progress_label.config(
                        text=f"{tempo_decorrido:.1f}s / {duracao_total:.1f}s ({percentual:.0f}%)"
                    )
        except queue.Empty:
            pass
        except Exception as e:
            print(f"Erro ao processar frame na UI: {e}")
        
        # Reagendar apenas se a janela ainda existe
        try:
            if self.root and self.root.winfo_exists():
                self.root.after(33, self._processar_frames_ui)  # ~30 FPS
        except:
            pass
    
    def _atualizar_duracao(self, valor):
        """Callback para atualizar duração do batch (via slider)"""
        valor_float = float(valor)
        self.duracao_entry.delete(0, tk.END)
        self.duracao_entry.insert(0, f"{valor_float:.1f}")
        self.gerenciador.atualizar_parametros(duracao_batch=valor_float)
    
    def _on_duracao_entry(self, event=None):
        """Callback quando usuário digita valor de duração"""
        try:
            valor = float(self.duracao_entry.get())
            # Validar limites
            if valor < 1.0:
                valor = 1.0
            elif valor > 30.0:
                valor = 30.0
            
            self.duracao_var.set(valor)
            self.duracao_entry.delete(0, tk.END)
            self.duracao_entry.insert(0, f"{valor:.1f}")
            self.gerenciador.atualizar_parametros(duracao_batch=valor)
        except ValueError:
            # Restaurar valor anterior
            self.duracao_entry.delete(0, tk.END)
            self.duracao_entry.insert(0, f"{self.duracao_var.get():.1f}")
    
    def _atualizar_frequencia(self, valor):
        """Callback para atualizar frequência de amostragem (escala logarítmica via slider)"""
        # Converter escala logarítmica: 10^valor
        # -1 -> 10^(-1) = 0.1 Hz
        #  0 -> 10^0 = 1.0 Hz
        #  0.301 -> 10^0.301 ≈ 2.0 Hz
        valor_log = float(valor)
        frequencia_hz = 10 ** valor_log
        
        self.freq_entry.delete(0, tk.END)
        self.freq_entry.insert(0, f"{frequencia_hz:.3f}")
        self.gerenciador.atualizar_parametros(frequencia_amostragem=frequencia_hz)
    
    def _on_freq_entry(self, event=None):
        """Callback quando usuário digita valor de frequência"""
        try:
            valor = float(self.freq_entry.get())
            # Validar limites (0.1 Hz a 2.0 Hz)
            if valor < 0.1:
                valor = 0.1
            elif valor > 2.0:
                valor = 2.0
            
            # Converter para escala logarítmica para o slider
            import math
            valor_log = math.log10(valor)
            self.frequencia_var.set(valor_log)
            
            self.freq_entry.delete(0, tk.END)
            self.freq_entry.insert(0, f"{valor:.3f}")
            self.gerenciador.atualizar_parametros(frequencia_amostragem=valor)
        except ValueError:
            # Restaurar valor anterior
            freq_atual = 10 ** self.frequencia_var.get()
            self.freq_entry.delete(0, tk.END)
            self.freq_entry.insert(0, f"{freq_atual:.3f}")
    
    def _atualizar_limiar(self, valor):
        """Callback para atualizar limiar de decisão (via slider)"""
        valor_float = float(valor)
        self.limiar_entry.delete(0, tk.END)
        self.limiar_entry.insert(0, f"{valor_float:.2f}")
        self.detector.atualizar_limiar(valor_float)
    
    def _on_limiar_entry(self, event=None):
        """Callback quando usuário digita valor de limiar"""
        try:
            valor = float(self.limiar_entry.get())
            # Validar limites
            if valor < -1.0:
                valor = -1.0
            elif valor > 1.0:
                valor = 1.0
            
            self.limiar_var.set(valor)
            self.limiar_entry.delete(0, tk.END)
            self.limiar_entry.insert(0, f"{valor:.2f}")
            self.detector.atualizar_limiar(valor)
        except ValueError:
            # Restaurar valor anterior
            self.limiar_entry.delete(0, tk.END)
            self.limiar_entry.insert(0, f"{self.limiar_var.get():.2f}")
    
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
