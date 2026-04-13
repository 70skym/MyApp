
"""
Real-time voice changer GUI (PyQt + sounddevice + pyqtgraph)

Updated version
---------------
- Pitch Shift mode:
    ratio ~= 1.0 => exact bypass (output == input)
- Harmonizer mode:
    adds one pitch-shifted harmony voice to the dry input
    ratio ~= 1.0 => exact bypass
- Four live plots:
    input waveform / input spectrum / output waveform / output spectrum

Install
-------
pip install numpy scipy sounddevice pyqtgraph PyQt5
"""

from __future__ import annotations

import sys
import threading
from dataclasses import dataclass

import numpy as np
import sounddevice as sd

try:
    from PyQt5.QtCore import QTimer, Qt
    from PyQt5.QtWidgets import (
        QApplication,
        QButtonGroup,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QSlider,
        QVBoxLayout,
        QWidget,
    )
except ImportError:
    from PyQt6.QtCore import QTimer, Qt
    from PyQt6.QtWidgets import (
        QApplication,
        QButtonGroup,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QSlider,
        QVBoxLayout,
        QWidget,
    )

import pyqtgraph as pg

FS = 48000
BLOCKSIZE = 1024
CHANNELS = 1
WAVE_HISTORY_SEC = 0.12
PLOT_UPDATE_MS = 33
SPECTRUM_MAX_HZ = 20000
EPS = 1e-9


@dataclass
class SharedState:
    input_wave: np.ndarray
    output_wave: np.ndarray
    input_spec: np.ndarray
    output_spec: np.ndarray
    freqs: np.ndarray


class VoiceProcessor:
    """
    Slider mapping:
        0      -> ratio 0.5
        10000  -> ratio 1.0
        20000  -> ratio 2.0

    ratio == 1.0 is handled as exact bypass for both modes.
    """

    def __init__(self, fs: int, blocksize: int):
        self.fs = fs
        self.blocksize = blocksize
        self.window = np.hanning(blocksize).astype(np.float32)
        self.freqs = np.fft.rfftfreq(blocksize, d=1.0 / fs)

    @staticmethod
    def slider_to_ratio(slider_value: int) -> float:
        slider_value = int(np.clip(slider_value, 0, 20000))
        return 2.0 ** ((slider_value - 10000) / 10000.0)

    @staticmethod
    def is_bypass_ratio(ratio: float) -> bool:
        return abs(ratio - 1.0) < 1e-6

    def process(self, x: np.ndarray, mode: str, slider_value: int) -> np.ndarray:
        ratio = self.slider_to_ratio(slider_value)

        # Exact bypass at ratio = 1
        if self.is_bypass_ratio(ratio):
            return x.astype(np.float32, copy=True)

        if mode == "pitch":
            y = self.pitch_shift_fft(x, ratio)
        elif mode == "harmonizer":
            y = self.harmonizer(x, ratio)
        else:
            y = x.copy()

        # Only protect against clipping when needed
        peak = np.max(np.abs(y)) + EPS
        if peak > 0.98:
            y = 0.98 * y / peak

        return y.astype(np.float32)

    def pitch_shift_fft(self, x: np.ndarray, ratio: float) -> np.ndarray:
        """
        Educational FFT-domain pitch shift approximation.
        Moves visible peaks in the spectrum, which is easy to understand visually.
        """
        win_x = x * self.window
        spec = np.fft.rfft(win_x)
        out_spec = np.zeros_like(spec)

        # Preserve DC explicitly
        out_spec[0] = spec[0]

        for k in range(1, len(spec)):
            new_k = int(round(k * ratio))
            if 0 < new_k < len(out_spec):
                out_spec[new_k] += spec[k]

        y = np.fft.irfft(out_spec, n=len(x))

        # Match rough block energy to input without forcing identical gain
        in_rms = np.sqrt(np.mean(x * x) + EPS)
        out_rms = np.sqrt(np.mean(y * y) + EPS)
        y *= in_rms / out_rms

        return y

    def harmonizer(self, x: np.ndarray, ratio: float) -> np.ndarray:
        """
        Add one harmony voice:
        output = dry + pitch-shifted copy

        ratio > 1.0 => upper harmony
        ratio < 1.0 => lower harmony
        """
        harmony = self.pitch_shift_fft(x, ratio)

        # Dry/wet mix chosen for audible harmony while keeping speech intelligible
        y = 0.72 * x + 0.55 * harmony
        return y

    def compute_spectrum_db(self, x: np.ndarray) -> np.ndarray:
        mag = np.abs(np.fft.rfft(x * self.window))
        db = 20 * np.log10(np.maximum(mag, 1e-8))
        return db.astype(np.float32)


class AudioEngine:
    def __init__(self, processor: VoiceProcessor, shared: SharedState):
        self.processor = processor
        self.shared = shared
        self.mode = "pitch"
        self.slider_value = 10000
        self.stream = None
        self.lock = threading.Lock()

    def set_mode(self, mode: str) -> None:
        with self.lock:
            self.mode = mode

    def set_slider_value(self, value: int) -> None:
        with self.lock:
            self.slider_value = int(value)

    def start(self) -> None:
        self.stream = sd.Stream(
            samplerate=FS,
            blocksize=BLOCKSIZE,
            channels=CHANNELS,
            dtype="float32",
            callback=self.audio_callback,
            latency="low",
        )
        self.stream.start()

    def stop(self) -> None:
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    @staticmethod
    def _push_history(hist: np.ndarray, new_block: np.ndarray) -> None:
        n = len(new_block)
        hist[:-n] = hist[n:]
        hist[-n:] = new_block

    def audio_callback(self, indata, outdata, frames, time_info, status):
        if status:
            pass

        x = indata[:, 0].copy()

        with self.lock:
            mode = self.mode
            slider_value = self.slider_value

        y = self.processor.process(x, mode, slider_value)
        outdata[:, 0] = y

        in_spec = self.processor.compute_spectrum_db(x)
        out_spec = self.processor.compute_spectrum_db(y)

        with self.lock:
            self._push_history(self.shared.input_wave, x)
            self._push_history(self.shared.output_wave, y)
            self.shared.input_spec[:] = in_spec
            self.shared.output_spec[:] = out_spec


class PlotWidgetGroup(QWidget):
    def __init__(self, shared: SharedState):
        super().__init__()
        self.shared = shared

        layout = QGridLayout(self)

        self.input_wave_plot = pg.PlotWidget(title="Microphone Input Waveform")
        self.input_spec_plot = pg.PlotWidget(title="Microphone Input Spectrum")
        self.output_wave_plot = pg.PlotWidget(title="Speaker Output Waveform")
        self.output_spec_plot = pg.PlotWidget(title="Speaker Output Spectrum")

        layout.addWidget(self.input_wave_plot, 0, 0)
        layout.addWidget(self.input_spec_plot, 0, 1)
        layout.addWidget(self.output_wave_plot, 1, 0)
        layout.addWidget(self.output_spec_plot, 1, 1)

        self._setup_plots()

    def _setup_plots(self) -> None:
        pg.setConfigOptions(antialias=True)

        self.input_wave_curve = self.input_wave_plot.plot(pen=pg.mkPen(width=2))
        self.output_wave_curve = self.output_wave_plot.plot(pen=pg.mkPen(width=2))
        self.input_spec_curve = self.input_spec_plot.plot(pen=pg.mkPen(width=2))
        self.output_spec_curve = self.output_spec_plot.plot(pen=pg.mkPen(width=2))

        for p in [self.input_wave_plot, self.output_wave_plot]:
            p.setLabel("bottom", "Time", units="s")
            p.setLabel("left", "Amplitude")
            p.setYRange(-1.05, 1.05)
            p.showGrid(x=True, y=True, alpha=0.25)

        for p in [self.input_spec_plot, self.output_spec_plot]:
            p.setLabel("bottom", "Frequency", units="Hz")
            p.setLabel("left", "Magnitude", units="dB")
            p.setXRange(0, SPECTRUM_MAX_HZ)
            p.setYRange(-100, 50)
            p.showGrid(x=True, y=True, alpha=0.25)

        n = len(self.shared.input_wave)
        self.wave_t = np.arange(n) / FS - (n / FS)

    def refresh(self) -> None:
        self.input_wave_curve.setData(self.wave_t, self.shared.input_wave)
        self.output_wave_curve.setData(self.wave_t, self.shared.output_wave)
        self.input_spec_curve.setData(self.shared.freqs, self.shared.input_spec)
        self.output_spec_curve.setData(self.shared.freqs, self.shared.output_spec)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-Time Voice Changer")
        self.resize(1400, 900)

        history_len = int(FS * WAVE_HISTORY_SEC)
        dummy = np.zeros(BLOCKSIZE // 2 + 1, dtype=np.float32)

        self.shared = SharedState(
            input_wave=np.zeros(history_len, dtype=np.float32),
            output_wave=np.zeros(history_len, dtype=np.float32),
            input_spec=dummy.copy(),
            output_spec=dummy.copy(),
            freqs=np.fft.rfftfreq(BLOCKSIZE, d=1.0 / FS).astype(np.float32),
        )

        self.processor = VoiceProcessor(FS, BLOCKSIZE)
        self.engine = AudioEngine(self.processor, self.shared)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        root.addWidget(self._build_controls())
        self.plots = PlotWidgetGroup(self.shared)
        root.addWidget(self.plots, stretch=1)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.plots.refresh)
        self.timer.start(PLOT_UPDATE_MS)

        self._update_slider_label(10000)

        try:
            self.engine.start()
        except Exception as e:
            QMessageBox.critical(
                self,
                "Audio Error",
                "Audio stream could not be started.\n\n"
                f"{type(e).__name__}: {e}\n\n"
                "Check microphone/speaker permissions and installed audio devices."
            )

    def _build_controls(self) -> QWidget:
        box = QGroupBox("Controls")
        layout = QHBoxLayout(box)

        mode_layout = QVBoxLayout()
        mode_label = QLabel("Mode")
        self.pitch_btn = QPushButton("Pitch Shift")
        self.harmony_btn = QPushButton("Harmonizer")
        self.pitch_btn.setCheckable(True)
        self.harmony_btn.setCheckable(True)
        self.pitch_btn.setChecked(True)

        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)
        self.mode_group.addButton(self.pitch_btn)
        self.mode_group.addButton(self.harmony_btn)

        self.pitch_btn.clicked.connect(lambda: self.engine.set_mode("pitch"))
        self.harmony_btn.clicked.connect(lambda: self.engine.set_mode("harmonizer"))

        mode_layout.addWidget(mode_label)
        mode_layout.addWidget(self.pitch_btn)
        mode_layout.addWidget(self.harmony_btn)

        slider_layout = QVBoxLayout()
        self.slider_title = QLabel("Shift Amount")
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 20000)
        self.slider.setValue(10000)
        self.slider.setTickInterval(1000)
        self.slider.setTickPosition(QSlider.TickPosition.TicksBelow)

        self.slider_value_label = QLabel()
        self.slider.valueChanged.connect(self._on_slider_changed)

        slider_layout.addWidget(self.slider_title)
        slider_layout.addWidget(self.slider)
        slider_layout.addWidget(self.slider_value_label)

        help_layout = QVBoxLayout()
        help_title = QLabel("Slider Meaning")
        help_text = QLabel(
            "0 = lower / darker\n"
            "10000 = bypass\n"
            "20000 = higher / brighter\n\n"
            "Internal ratio mapping:\n"
            "0.5x ... 1.0x ... 2.0x\n\n"
            "Pitch Shift: shifted voice only\n"
            "Harmonizer: dry voice + shifted voice"
        )
        help_layout.addWidget(help_title)
        help_layout.addWidget(help_text)

        layout.addLayout(mode_layout, 0)
        layout.addSpacing(30)
        layout.addLayout(slider_layout, 1)
        layout.addSpacing(30)
        layout.addLayout(help_layout, 0)

        return box

    def _on_slider_changed(self, value: int) -> None:
        self.engine.set_slider_value(value)
        self._update_slider_label(value)

    def _update_slider_label(self, value: int) -> None:
        ratio = self.processor.slider_to_ratio(value)
        text = (
            f"Slider: {value} / 20000    "
            f"Ratio: {ratio:.3f}x"
        )
        if abs(ratio - 1.0) < 1e-6:
            text += "    (bypass)"
        self.slider_value_label.setText(text)

    def closeEvent(self, event) -> None:
        self.engine.stop()
        event.accept()

def main():
    app = QApplication(sys.argv)
    print("QApplication created:", app)

    w = MainWindow()
    print("MainWindow created")

    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
