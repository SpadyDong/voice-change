import sys
from pathlib import Path

# Ensure src directory is in path for imports
src_dir = Path(__file__).parent.resolve()
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from PyQt6.QtWidgets import QApplication

from gui.main_window import MainWindow
from config.settings import ConfigManager


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Voice Changer")
    app.setApplicationVersion("0.1.0")

    # Load stylesheet if present
    qss_path = Path(__file__).parent.parent / "assets" / "style.qss"
    if qss_path.exists():
        with open(qss_path, "r", encoding="utf-8") as f:
            app.setStyleSheet(f.read())

    config = ConfigManager()
    config.load()

    window = MainWindow(config)
    window.show()

    exit_code = app.exec()
    config.save()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
