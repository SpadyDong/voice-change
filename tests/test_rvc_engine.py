import sys
from pathlib import Path

src_dir = Path(__file__).parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import numpy as np
import pytest

from core.rvc_engine import RvcEngine


def test_rvc_engine_fallback():
    engine = RvcEngine(sample_rate=48000, device="cpu")
    assert not engine.is_loaded()

    audio = np.random.randn(1024).astype(np.float32)
    out = engine.infer(audio)
    assert out.dtype == np.float32
    assert len(out) == len(audio)
    assert engine.get_latency_ms() >= 0.0


def test_rvc_engine_params():
    engine = RvcEngine()
    engine.set_params(pitch_shift=6, index_rate=0.5, protect=0.2, filter_radius=5, f0_method="harvest")
    assert engine.pitch_shift == 6
    assert engine.index_rate == 0.5
    assert engine.protect == 0.2
    assert engine.filter_radius == 5
    assert engine.f0_method == "harvest"


def test_rvc_engine_pitch_shift():
    engine = RvcEngine(sample_rate=22050, device="cpu")
    engine.set_params(pitch_shift=3)
    audio = np.sin(2 * np.pi * 440 * np.arange(2048) / 22050).astype(np.float32)
    out = engine.infer(audio)
    assert out.dtype == np.float32
    assert len(out) > 0
