import numpy as np
from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPainter, QColor, QPen


class AudioVisualizer(QWidget):
    def __init__(self, parent=None, history_size: int = 2048):
        super().__init__(parent)
        self.history_size = history_size
        self._buffer = np.zeros(history_size, dtype=np.float32)
        self._write_pos = 0
        self.setMinimumHeight(120)
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(self.backgroundRole(), QColor(30, 30, 30))
        self.setPalette(palette)

        # Update at 30 FPS
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.update)
        self._timer.start(33)

    def push_audio(self, audio: np.ndarray):
        audio = np.asarray(audio, dtype=np.float32)
        n = len(audio)
        if n >= self.history_size:
            self._buffer[:] = audio[-self.history_size:]
            self._write_pos = 0
        else:
            end = self._write_pos + n
            if end <= self.history_size:
                self._buffer[self._write_pos:end] = audio
            else:
                self._buffer[self._write_pos:] = audio[: self.history_size - self._write_pos]
                self._buffer[: end - self.history_size] = audio[self.history_size - self._write_pos :]
            self._write_pos = end % self.history_size

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = self.rect()
        width = rect.width()
        height = rect.height()
        mid_y = height // 2

        # Background
        painter.fillRect(rect, QColor(30, 30, 30))

        # Grid line
        painter.setPen(QPen(QColor(60, 60, 60), 1, Qt.PenStyle.SolidLine))
        painter.drawLine(0, mid_y, width, mid_y)

        # Waveform
        painter.setPen(QPen(QColor(0, 200, 100), 2, Qt.PenStyle.SolidLine))

        if width <= 0:
            painter.end()
            return

        # Reorder buffer so oldest sample is at index 0
        ordered = np.concatenate((self._buffer[self._write_pos:], self._buffer[: self._write_pos]))

        step = max(1, len(ordered) // width)
        points = []
        for x in range(width):
            idx_start = x * step
            idx_end = min(idx_start + step, len(ordered))
            if idx_start < len(ordered):
                val = np.max(np.abs(ordered[idx_start:idx_end])) if idx_end > idx_start else 0.0
            else:
                val = 0.0
            y = mid_y - int(val * (height // 2 - 4))
            points.append((x, y))

        if len(points) > 1:
            for i in range(len(points) - 1):
                painter.drawLine(points[i][0], points[i][1], points[i + 1][0], points[i + 1][1])

        painter.end()
