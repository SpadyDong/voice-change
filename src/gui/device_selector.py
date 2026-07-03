from typing import Callable, Optional

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QComboBox,
    QPushButton,
)
from PyQt6.QtCore import pyqtSignal

import sounddevice as sd


class DeviceSelector(QWidget):
    devices_changed = pyqtSignal()
    input_device_changed = pyqtSignal(int)
    output_device_changed = pyqtSignal(int)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._build_ui()
        self.refresh_devices()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("输入设备"))
        self.input_combo = QComboBox()
        self.input_combo.currentIndexChanged.connect(self._on_input_changed)
        layout.addWidget(self.input_combo)

        layout.addWidget(QLabel("输出设备"))
        self.output_combo = QComboBox()
        self.output_combo.currentIndexChanged.connect(self._on_output_changed)
        layout.addWidget(self.output_combo)

        self.refresh_btn = QPushButton("刷新设备列表")
        self.refresh_btn.clicked.connect(self.refresh_devices)
        layout.addWidget(self.refresh_btn)

        layout.addStretch()

    def refresh_devices(self):
        self.input_combo.clear()
        self.output_combo.clear()

        try:
            devices = sd.query_devices()
        except Exception:
            devices = []

        default_input = sd.default.device[0] if sd.default.device else None
        default_output = sd.default.device[1] if sd.default.device else None

        input_idx = 0
        output_idx = 0

        for i, dev in enumerate(devices):
            name = dev.get("name", "Unknown")
            max_in = dev.get("max_input_channels", 0)
            max_out = dev.get("max_output_channels", 0)

            if max_in > 0:
                self.input_combo.addItem(f"{name}", i)
                if i == default_input:
                    input_idx = self.input_combo.count() - 1

            if max_out > 0:
                self.output_combo.addItem(f"{name}", i)
                if i == default_output:
                    output_idx = self.output_combo.count() - 1

        self.input_combo.setCurrentIndex(input_idx)
        self.output_combo.setCurrentIndex(output_idx)
        self.devices_changed.emit()

    def _on_input_changed(self, index: int):
        if index >= 0:
            device_id = self.input_combo.itemData(index)
            if device_id is not None:
                self.input_device_changed.emit(device_id)

    def _on_output_changed(self, index: int):
        if index >= 0:
            device_id = self.output_combo.itemData(index)
            if device_id is not None:
                self.output_device_changed.emit(device_id)

    def get_input_device(self) -> Optional[int]:
        idx = self.input_combo.currentIndex()
        return self.input_combo.itemData(idx) if idx >= 0 else None

    def get_output_device(self) -> Optional[int]:
        idx = self.output_combo.currentIndex()
        return self.output_combo.itemData(idx) if idx >= 0 else None

    def set_input_device(self, device_id: int):
        for i in range(self.input_combo.count()):
            if self.input_combo.itemData(i) == device_id:
                self.input_combo.setCurrentIndex(i)
                break

    def set_output_device(self, device_id: int):
        for i in range(self.output_combo.count()):
            if self.output_combo.itemData(i) == device_id:
                self.output_combo.setCurrentIndex(i)
                break
