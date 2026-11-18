"""Small audio helpers (WAV in/out, resampling, loudness)."""
import io
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ASR_SR = 16000


def read_wav_bytes(data: bytes):
    x, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    return x.mean(axis=1), int(sr)


def read_wav(path):
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return x.mean(axis=1), int(sr)


def write_wav(path, x, sr):
    sf.write(str(path), np.asarray(x, dtype=np.float32), int(sr), subtype="PCM_16")


def write_wav_bytes(x, sr) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.clip(np.asarray(x, dtype=np.float32), -1, 1), int(sr), format="WAV", subtype="PCM_16")
    return buf.getvalue()


def resample(x, sr, target):
    if sr == target or len(x) == 0:
        return np.asarray(x, dtype=np.float32)
    g = gcd(int(sr), int(target))
    return resample_poly(x, target // g, sr // g).astype(np.float32)


def rms(x):
    return float(np.sqrt(np.mean(np.square(x)))) if len(x) else 0.0


def normalize_rms(x, target=0.05):
    r = rms(x)
    if r < 1e-8:
        return x
    y = x * (target / r)
    peak = np.abs(y).max()
    return (y / peak * 0.99 if peak > 0.99 else y).astype(np.float32)


def peak_normalize(x, peak=0.9):
    m = np.abs(x).max() if len(x) else 0
    return (x / m * peak).astype(np.float32) if m > 1e-8 else np.asarray(x, dtype=np.float32)


def trim_silence(x, sr, top_db=35.0):
    """Remove leading/trailing silence (frame energy more than top_db below the loudest frame)."""
    if len(x) == 0:
        return x
    hop = int(0.01 * sr)
    frames = max(1, len(x) // hop)
    e = np.array([np.sqrt(np.mean(x[i * hop:(i + 1) * hop] ** 2) + 1e-12) for i in range(frames)])
    db = 20 * np.log10(e / e.max())
    idx = np.where(db > -top_db)[0]
    if len(idx) == 0:
        return x
    a, b = max(0, (idx[0] - 5) * hop), min(len(x), (idx[-1] + 6) * hop)
    return x[a:b]
