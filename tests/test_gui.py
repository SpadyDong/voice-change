import sys
from pathlib import Path

src_dir = Path(__file__).parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import pytest
from PyQt6.QtCore import Qt

from config.settings import ConfigManager
from gui.control_panel import ControlPanel
from gui.model_browser import ModelBrowser
from gui.device_selector import DeviceSelector
from core.model_manager import ModelManager


@pytest.fixture
def qapp():
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


def test_control_panel_signals(qtbot, qapp):
    panel = ControlPanel()
    qtbot.addWidget(panel)

    with qtbot.waitSignal(panel.pitch_changed, timeout=500):
        panel.pitch_slider.setValue(5)

    assert panel.pitch_label.text() == "5"


def test_control_panel_toggle(qtbot, qapp):
    panel = ControlPanel()
    qtbot.addWidget(panel)

    with qtbot.waitSignal(panel.toggle_changed, timeout=500):
        panel.toggle_btn.setChecked(True)

    assert panel.toggle_btn.text() == "停止变声"


def test_device_selector_refresh(qtbot, qapp):
    selector = DeviceSelector()
    qtbot.addWidget(selector)
    # Should populate without crash even without hardware
    assert selector.input_combo.count() >= 0
    assert selector.output_combo.count() >= 0


def test_model_browser_empty(qtbot, qapp):
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = ModelManager(tmpdir)
        browser = ModelBrowser(mgr)
        qtbot.addWidget(browser)
        assert browser.model_combo.count() == 0
