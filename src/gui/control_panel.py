from typing import Optional

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QGroupBox,
)
from PyQt6.QtCore import Qt, pyqtSignal


class ControlPanel(QGroupBox):
    toggle_changed = pyqtSignal(bool)
    pitch_changed = pyqtSignal(int)
    index_changed = pyqtSignal(int)
    protect_changed = pyqtSignal(int)

    def __init__(
        self,
        pitch: int = 0,
        index_rate: float = 0.75,
        protect: float = 0.33,
        parent: Optional[QWidget] = None,
    ):
        super().__init__("控制面板", parent)
        self._build_ui(pitch, index_rate, protect)

    def _build_ui(self, pitch: int, index_rate: float, protect: float):
        layout = QVBoxLayout(self)

        self.toggle_btn = QPushButton("启动变声")
        self.toggle_btn.setCheckable(True)
        self.toggle_btn.setMinimumHeight(48)
        self.toggle_btn.toggled.connect(self._on_toggle)
        layout.addWidget(self.toggle_btn)

        layout.addWidget(QLabel("音调"))
        self.pitch_slider = QSlider(Qt.Orientation.Horizontal)
        self.pitch_slider.setRange(-24, 24)
        self.pitch_slider.setValue(pitch)
        self.pitch_slider.valueChanged.connect(self._on_pitch_moved)
        self.pitch_label = QLabel(str(pitch))
        layout.addWidget(self.pitch_slider)
        layout.addWidget(self.pitch_label)

        layout.addWidget(QLabel("索引强度"))
        self.index_slider = QSlider(Qt.Orientation.Horizontal)
        self.index_slider.setRange(0, 100)
        self.index_slider.setValue(int(index_rate * 100))
        self.index_slider.valueChanged.connect(self._on_index_moved)
        self.index_label = QLabel(f"{int(index_rate * 100)}%")
        layout.addWidget(self.index_slider)
        layout.addWidget(self.index_label)

        layout.addWidget(QLabel("清音保护"))
        self.protect_slider = QSlider(Qt.Orientation.Horizontal)
        self.protect_slider.setRange(0, 50)
        self.protect_slider.setValue(int(protect * 100))
        self.protect_slider.valueChanged.connect(self._on_protect_moved)
        self.protect_label = QLabel(f"{protect:.2f}")
        layout.addWidget(self.protect_slider)
        layout.addWidget(self.protect_label)

        self.latency_label = QLabel("当前延迟: -- ms")
        layout.addWidget(self.latency_label)

        layout.addStretch()

    def _on_toggle(self, checked: bool):
        self.toggle_btn.setText("停止变声" if checked else "启动变声")
        self.toggle_changed.emit(checked)

    def _on_pitch_moved(self, value: int):
        self.pitch_label.setText(str(value))
        self.pitch_changed.emit(value)

    def _on_index_moved(self, value: int):
        self.index_label.setText(f"{value}%")
        self.index_changed.emit(value)

    def _on_protect_moved(self, value: int):
        self.protect_label.setText(f"{value / 100.0:.2f}")
        self.protect_changed.emit(value)

    def set_running(self, running: bool):
        self.toggle_btn.blockSignals(True)
        self.toggle_btn.setChecked(running)
        self.toggle_btn.setText("停止变声" if running else "启动变声")
        self.toggle_btn.blockSignals(False)

    def update_pitch_label(self, value: int):
        self.pitch_label.setText(str(value))

    def update_index_label(self, value: int):
        self.index_label.setText(f"{value}%")

    def update_protect_label(self, value: int):
        self.protect_label.setText(f"{value / 100.0:.2f}")

    def update_latency(self, text: str):
        self.latency_label.setText(text)
