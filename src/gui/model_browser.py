from typing import Optional, Callable

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QComboBox,
    QPushButton,
    QHBoxLayout,
)
from PyQt6.QtCore import pyqtSignal

from core.model_manager import ModelManager, ModelInfo


class ModelBrowser(QWidget):
    model_selected = pyqtSignal(str)
    models_refreshed = pyqtSignal()

    def __init__(self, model_manager: ModelManager, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.model_manager = model_manager
        self._build_ui()
        self.refresh_models()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        top = QHBoxLayout()
        top.addWidget(QLabel("模型"))
        self.refresh_btn = QPushButton("刷新")
        self.refresh_btn.setToolTip("重新扫描模型目录")
        self.refresh_btn.clicked.connect(self.refresh_models)
        top.addWidget(self.refresh_btn)
        layout.addLayout(top)

        self.model_combo = QComboBox()
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        layout.addWidget(self.model_combo)

        self.info_label = QLabel("版本: -- 采样率: --")
        layout.addWidget(self.info_label)

        layout.addStretch()

    def refresh_models(self):
        self.model_combo.clear()
        models = self.model_manager.scan_models()
        for m in models:
            self.model_combo.addItem(m.name)
        if models:
            self.model_combo.setCurrentIndex(0)
        self.models_refreshed.emit()

    def _on_model_changed(self, name: str):
        if not name:
            self.info_label.setText("版本: -- 采样率: --")
            return
        info = self.model_manager.get_model_by_name(name)
        if info:
            self.info_label.setText(f"版本: {info.version} 采样率: {info.sample_rate} Hz")
            self.model_selected.emit(name)
        else:
            self.info_label.setText("版本: -- 采样率: --")

    def get_current_model(self) -> Optional[ModelInfo]:
        name = self.model_combo.currentText()
        return self.model_manager.get_model_by_name(name) if name else None

    def select_model(self, name: str):
        idx = self.model_combo.findText(name)
        if idx >= 0:
            self.model_combo.setCurrentIndex(idx)
