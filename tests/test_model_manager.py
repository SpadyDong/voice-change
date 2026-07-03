import sys
from pathlib import Path
import tempfile

src_dir = Path(__file__).parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import pytest

from core.model_manager import ModelManager


def test_scan_empty_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = ModelManager(tmpdir)
        models = mgr.scan_models()
        assert models == []
        assert mgr.get_model_names() == []


def test_scan_with_models():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dummy model files
        pth = Path(tmpdir) / "test_model_v2_48k.pth"
        pth.write_text("dummy")
        idx = Path(tmpdir) / "test_model_v2_48k.index"
        idx.write_text("dummy")

        mgr = ModelManager(tmpdir)
        models = mgr.scan_models()
        assert len(models) == 1
        assert models[0].name == "test_model_v2_48k"
        assert models[0].version == "v2"
        assert models[0].sample_rate == 48000
        assert models[0].index_path == str(idx)


def test_select_model():
    with tempfile.TemporaryDirectory() as tmpdir:
        pth = Path(tmpdir) / "model_a.pth"
        pth.write_text("dummy")
        mgr = ModelManager(tmpdir)
        mgr.scan_models()

        selected = mgr.set_current_model("model_a")
        assert selected is not None
        assert selected.name == "model_a"

        missing = mgr.set_current_model("nonexistent")
        assert missing is None
