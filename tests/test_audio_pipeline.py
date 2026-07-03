import sys
from pathlib import Path

src_dir = Path(__file__).parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import numpy as np
import pytest

from core.audio_pipeline import AudioPipeline


def test_audio_pipeline_lifecycle():
    pipeline = AudioPipeline(sample_rate=48000, block_size=512)
    assert not pipeline.is_running()
    # Start may fail without real devices; ensure graceful handling
    try:
        pipeline.start()
        assert pipeline.is_running()
        pipeline.stop()
        assert not pipeline.is_running()
    except RuntimeError:
        # Expected in CI without audio hardware
        pass


def test_process_callback():
    def dummy_process(audio):
        return audio * 2.0

    pipeline = AudioPipeline(process_callback=dummy_process)
    assert pipeline.process_callback is dummy_process

    pipeline.set_process_callback(None)
    assert pipeline.process_callback is None


def test_query_devices():
    devices = AudioPipeline.query_devices()
    assert isinstance(devices, list)
