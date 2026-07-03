import sys
from pathlib import Path

src_dir = Path(__file__).parent.parent.resolve()
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QStatusBar,
    QMessageBox,
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QKeySequence, QShortcut

from config.settings import ConfigManager
from core.audio_pipeline import AudioPipeline
from core.rvc_engine import RvcEngine
from core.model_manager import ModelManager
from gui.control_panel import ControlPanel
from gui.device_selector import DeviceSelector
from gui.model_browser import ModelBrowser
from gui.audio_visualizer import AudioVisualizer


class MainWindow(QMainWindow):
    def __init__(self, config: ConfigManager):
        super().__init__()
        self.config = config
        self.setWindowTitle("Voice Changer")
        self.resize(config.settings.window_width, config.settings.window_height)

        self.rvc_engine = RvcEngine(
            sample_rate=config.settings.audio.sample_rate,
            device="cuda" if self._has_cuda() else "cpu",
        )
        self.rvc_engine.set_params(
            pitch_shift=config.settings.rvc.pitch_shift,
            index_rate=config.settings.rvc.index_rate,
            protect=config.settings.rvc.protect,
            filter_radius=config.settings.rvc.filter_radius,
            f0_method=config.settings.rvc.f0_method,
        )

        self.model_manager = ModelManager(config.settings.rvc.model_dir)

        self.audio_pipeline = AudioPipeline(
            process_callback=self._process_audio,
            sample_rate=config.settings.audio.sample_rate,
            block_size=config.settings.audio.block_size,
            input_device=config.settings.audio.input_device,
            output_device=config.settings.audio.output_device,
        )

        self._build_ui()
        self._connect_signals()
        self._setup_shortcuts()

        self._update_timer = QTimer(self)
        self._update_timer.timeout.connect(self._on_update_tick)
        self._update_timer.start(100)

    @staticmethod
    def _has_cuda() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # Left: Control Panel
        self.control_panel = ControlPanel(
            pitch=self.config.settings.rvc.pitch_shift,
            index_rate=self.config.settings.rvc.index_rate,
            protect=self.config.settings.rvc.protect,
        )
        main_layout.addWidget(self.control_panel, stretch=1)

        # Right: Device, Model, Visualizer
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        self.device_selector = DeviceSelector()
        if self.config.settings.audio.input_device is not None:
            self.device_selector.set_input_device(self.config.settings.audio.input_device)
        if self.config.settings.audio.output_device is not None:
            self.device_selector.set_output_device(self.config.settings.audio.output_device)
        right_layout.addWidget(self.device_selector)

        self.model_browser = ModelBrowser(self.model_manager)
        right_layout.addWidget(self.model_browser)

        self.visualizer = AudioVisualizer()
        right_layout.addWidget(self.visualizer, stretch=1)

        self.sr_label = QLabel(f"采样率: {self.config.settings.audio.sample_rate} Hz")
        right_layout.addWidget(self.sr_label)

        main_layout.addWidget(right_widget, stretch=2)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("状态: 就绪")
        self._gpu_label = QLabel("")
        self.status_bar.addPermanentWidget(self._gpu_label)
        self._update_gpu_info()

    def _update_gpu_info(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            name = pynvml.nvmlDeviceGetName(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            used_mb = mem.used // 1024 // 1024
            total_mb = mem.total // 1024 // 1024
            self._gpu_label.setText(f"GPU: {name} | 显存: {used_mb}/{total_mb} MB")
        except Exception:
            self._gpu_label.setText("GPU: 未检测到")

    def _connect_signals(self):
        self.control_panel.toggle_changed.connect(self._on_toggle)
        self.control_panel.pitch_changed.connect(self._on_pitch_changed)
        self.control_panel.index_changed.connect(self._on_index_changed)
        self.control_panel.protect_changed.connect(self._on_protect_changed)

        self.device_selector.input_device_changed.connect(self._on_input_device_changed)
        self.device_selector.output_device_changed.connect(self._on_output_device_changed)

        self.model_browser.model_selected.connect(self._on_model_selected)

    def _setup_shortcuts(self):
        shortcut = QShortcut(QKeySequence("Space"), self)
        shortcut.activated.connect(lambda: self.control_panel.toggle_btn.toggle())

    def _process_audio(self, audio: np.ndarray) -> np.ndarray:
        # Push to visualizer (input side)
        self.visualizer.push_audio(audio)
        # Run through RVC / fallback
        return self.rvc_engine.infer(audio)

    def _on_toggle(self, checked: bool):
        if checked:
            try:
                self.audio_pipeline.start()
                self.control_panel.set_running(True)
                self.status_bar.showMessage("状态: 运行中")
            except Exception as e:
                self.control_panel.set_running(False)
                QMessageBox.critical(self, "错误", f"无法启动音频流:\n{e}")
        else:
            self.audio_pipeline.stop()
            self.control_panel.set_running(False)
            self.status_bar.showMessage("状态: 就绪")

    def _on_pitch_changed(self, value: int):
        self.control_panel.update_pitch_label(value)
        self.config.settings.rvc.pitch_shift = value
        self.rvc_engine.set_params(pitch_shift=value)

    def _on_index_changed(self, value: int):
        self.control_panel.update_index_label(value)
        self.config.settings.rvc.index_rate = value / 100.0
        self.rvc_engine.set_params(index_rate=value / 100.0)

    def _on_protect_changed(self, value: int):
        self.control_panel.update_protect_label(value)
        self.config.settings.rvc.protect = value / 100.0
        self.rvc_engine.set_params(protect=value / 100.0)

    def _on_input_device_changed(self, device_id: int):
        self.config.settings.audio.input_device = device_id
        self.audio_pipeline.input_device = device_id

    def _on_output_device_changed(self, device_id: int):
        self.config.settings.audio.output_device = device_id
        self.audio_pipeline.output_device = device_id

    def _on_model_selected(self, name: str):
        model = self.model_manager.set_current_model(name)
        if model:
            self.sr_label.setText(f"采样率: {model.sample_rate} Hz")
            # Load model into engine (non-blocking would be better, but sync for now)
            try:
                self.rvc_engine.load_model(
                    pth_path=model.pth_path,
                    index_path=model.index_path,
                    hubert_path=self.config.settings.rvc.hubert_path,
                )
                self.status_bar.showMessage(f"已加载模型: {model.name}")
            except Exception as e:
                QMessageBox.warning(self, "模型加载失败", str(e))

    def _on_update_tick(self):
        if self.audio_pipeline.is_running():
            latency = self.rvc_engine.get_latency_ms()
            self.control_panel.update_latency(f"当前延迟: {latency:.1f} ms")
        else:
            self.control_panel.update_latency("当前延迟: -- ms")
        self._update_gpu_info()

    def closeEvent(self, event):
        self.audio_pipeline.stop()
        # Persist window size
        self.config.settings.window_width = self.width()
        self.config.settings.window_height = self.height()
        event.accept()
