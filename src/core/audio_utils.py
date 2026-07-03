import numpy as np
import librosa


def resample_audio(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int,
) -> np.ndarray:
    if orig_sr == target_sr:
        return audio
    return librosa.resample(
        y=audio,
        orig_sr=orig_sr,
        target_sr=target_sr,
    )


def ensure_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim > 1:
        return np.mean(audio, axis=1)
    return audio


def ensure_float32(audio: np.ndarray) -> np.ndarray:
    if audio.dtype != np.float32:
        if np.issubdtype(audio.dtype, np.integer):
            max_val = np.iinfo(audio.dtype).max
            audio = audio.astype(np.float32) / max_val
        else:
            audio = audio.astype(np.float32)
    return audio


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    max_val = np.max(np.abs(audio))
    if max_val > 1.0:
        audio = audio / max_val
    return audio


def preprocess_audio(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int,
) -> np.ndarray:
    audio = ensure_float32(audio)
    audio = ensure_mono(audio)
    audio = resample_audio(audio, orig_sr, target_sr)
    audio = normalize_audio(audio)
    return audio
