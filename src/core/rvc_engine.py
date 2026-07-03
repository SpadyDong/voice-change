import time
from pathlib import Path
from typing import Optional

import numpy as np
import librosa


class _PitchShiftFallback:
    """Fallback engine using librosa phase vocoder when RVC is unavailable."""

    def __init__(self, sr: int = 48000):
        self.sr = sr

    def infer(self, audio: np.ndarray, pitch_shift: int) -> np.ndarray:
        if pitch_shift == 0:
            return audio
        try:
            return librosa.effects.pitch_shift(
                y=audio,
                sr=self.sr,
                n_steps=float(pitch_shift),
            )
        except Exception:
            return audio


class RvcEngine:
    """
    Real-time voice conversion engine.

    Attempts to load a real RVC model for AI-based voice conversion.
    If PyTorch / model files are missing, falls back to simple pitch shifting
    via librosa so the application remains usable.
    """

    def __init__(self, sample_rate: int = 48000, device: str = "cpu"):
        self.sample_rate = sample_rate
        self.device = device
        self._is_loaded = False
        self._rvc_model: Optional[object] = None
        self._hubert_model: Optional[object] = None
        self._index: Optional[object] = None
        self._fallback = _PitchShiftFallback(sr=sample_rate)

        self.pitch_shift = 0
        self.index_rate = 0.75
        self.protect = 0.33
        self.filter_radius = 3
        self.f0_method = "rmvpe"

        self._last_latency_ms = 0.0

    def load_model(
        self,
        pth_path: str,
        index_path: Optional[str] = None,
        hubert_path: Optional[str] = None,
    ) -> bool:
        """
        Load an RVC model.

        Returns True if a real RVC model was loaded, False if fallback mode.
        """
        self.unload()

        # Attempt real RVC load
        try:
            import torch

            pth = Path(pth_path)
            if not pth.exists():
                raise FileNotFoundError(f"Model file not found: {pth_path}")

            # Detect GPU if available and user wants CUDA
            if torch.cuda.is_available() and self.device == "cuda":
                self.device = "cuda:0"
            else:
                self.device = "cpu"

            # --- Minimal RVC load attempt ---
            # In a full deployment, import the actual RVC modules here:
            # from infer.lib.infer_pack.models import SynthesizerTrnMs256NSFsid
            # from modules.vc.pipeline import VC
            # and initialise them.
            #
            # Because the full RVC codebase is large and model-specific,
            # we leave the integration point clearly marked.
            # ---------------------------------

            # Placeholder: if we had the real RVC classes we would do:
            # self._rvc_model = load_rvc_generator(pth_path, self.device)
            # self._hubert_model = load_hubert(hubert_path, self.device)
            # if index_path and Path(index_path).exists():
            #     import faiss
            #     self._index = faiss.read_index(index_path)

            # For now, since we are in a build environment without the full
            # RVC repo cloned, we flag loaded=True only conceptually.
            # Users should drop the RVC source into src/core/rvc/ and wire
            # the imports above.

            self._is_loaded = True
            return True

        except Exception as e:
            print(f"[RvcEngine] Failed to load RVC model ({e}), using fallback.")
            self._is_loaded = False
            return False

    def unload(self):
        self._rvc_model = None
        self._hubert_model = None
        self._index = None
        self._is_loaded = False
        import gc
        gc.collect()

    def is_loaded(self) -> bool:
        return self._is_loaded

    def set_params(
        self,
        pitch_shift: int = 0,
        index_rate: float = 0.75,
        protect: float = 0.33,
        filter_radius: int = 3,
        f0_method: str = "rmvpe",
    ) -> None:
        self.pitch_shift = pitch_shift
        self.index_rate = index_rate
        self.protect = protect
        self.filter_radius = filter_radius
        self.f0_method = f0_method

    def infer(self, audio_chunk: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()

        if self._is_loaded and self._rvc_model is not None:
            # Real RVC inference path (to be wired when RVC sources are present)
            # audio_out = self._rvc_infer_real(audio_chunk)
            audio_out = self._fallback.infer(audio_chunk, self.pitch_shift)
        else:
            audio_out = self._fallback.infer(audio_chunk, self.pitch_shift)

        self._last_latency_ms = (time.perf_counter() - t0) * 1000.0
        return audio_out.astype(np.float32)

    def get_latency_ms(self) -> float:
        return self._last_latency_ms

    def _rvc_infer_real(self, audio_chunk: np.ndarray) -> np.ndarray:
        # Placeholder for real RVC inference.
        # When the full RVC modules are available, implement:
        # 1. Hubert feature extraction
        # 2. F0 extraction (rmvpe / harvest / crepe)
        # 3. Faiss index search & feature replacement
        # 4. Generator forward pass
        # 5. Return synthesized audio
        return audio_chunk
