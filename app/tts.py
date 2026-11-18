"""Text -> clear speech.
  * speaker's own voice : XTTS-v2 voice cloning from a reference recording
  * standard clear voice: XTTS-v2 built-in speaker  ->  MMS-TTS  ->  operating-system voice (pyttsx3)
XTTS-v2 is released under the Coqui Public Model License (non-commercial). It only loads when the
environment variable COQUI_TOS_AGREED=1 is set (start.bat sets it; see README)."""
import importlib.util
import os
import tempfile
import threading

import numpy as np

from . import config
from .audio_io import read_wav


def trim_output(x, sr, text):
    """Remove silence at both ends; for a single word, cut any stray sound after it (notebook 05 v2)."""
    from .audio_io import trim_silence
    x = trim_silence(x, sr, top_db=35.0)
    if len(text.split()) == 1:
        cap = int((0.6 + 0.09 * len(text)) * sr)
        if len(x) > cap:
            x = x[:cap].copy()
            f = int(0.03 * sr)
            x[-f:] *= np.linspace(1, 0, f)
    return x.astype(np.float32)


def clone_installed() -> bool:
    return importlib.util.find_spec("TTS") is not None


class Synthesizer:
    def __init__(self):
        self._xtts = None
        self._xtts_error = None
        self._mms = None
        self._lock = threading.Lock()
        self.last_clone_error = None

    # ---------- XTTS-v2 ----------
    def _load_xtts(self):
        if self._xtts is not None:
            return self._xtts
        if self._xtts_error:
            raise RuntimeError(self._xtts_error)
        try:
            if os.environ.get("COQUI_TOS_AGREED") != "1":
                raise RuntimeError("XTTS licence not accepted (set COQUI_TOS_AGREED=1 - see README).")
            import torch
            from TTS.api import TTS
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            self._xtts = TTS(config.XTTS_ID).to(dev)
            return self._xtts
        except Exception as e:
            self._xtts_error = f"{type(e).__name__}: {str(e)[:200]}"
            raise

    def warm_up(self):
        """Load XTTS in the background so the first conversion is not slow."""
        try:
            with self._lock:
                self._load_xtts()
        except Exception:
            pass

    @staticmethod
    def _prep(text):
        t = text.strip()
        t = t[:1].upper() + t[1:]
        return t + ("" if t[-1:] in ".!?" else ".")      # a closing full stop stabilises single words

    def _xtts_say(self, text, reference=None):
        with self._lock:
            m = self._load_xtts()
            kw = dict(speaker_wav=reference) if reference else dict(speaker=config.PRESET_SPEAKER)
            try:
                wav = m.tts(text=self._prep(text), language="en", split_sentences=False,
                            temperature=config.XTTS_TEMPERATURE, top_p=0.85, **kw)
            except TypeError:
                wav = m.tts(text=self._prep(text), language="en", **kw)
        sr = int(getattr(getattr(m, "synthesizer", None), "output_sample_rate", 24000) or 24000)
        x = trim_output(np.asarray(wav, dtype=np.float32).reshape(-1), sr, text)
        if len(x) < 1600:
            raise RuntimeError("XTTS produced no audio")
        return x, sr

    # ---------- fallbacks ----------
    def _mms_say(self, text):
        import torch
        from transformers import AutoTokenizer, VitsModel
        if self._mms is None:
            self._mms = (AutoTokenizer.from_pretrained(config.MMS_ID), VitsModel.from_pretrained(config.MMS_ID).eval())
        tok, model = self._mms
        torch.manual_seed(0)
        with torch.no_grad():
            wav = model(**tok(text, return_tensors="pt")).waveform[0].cpu().numpy()
        return wav.astype(np.float32), int(model.config.sampling_rate)

    @staticmethod
    def _os_say(text):
        import pyttsx3
        path = os.path.join(tempfile.gettempdir(), "whisper_dysarthric_tts.wav")
        eng = pyttsx3.init(); eng.setProperty("rate", 150); eng.save_to_file(text, path); eng.runAndWait()
        return read_wav(path)

    # ---------- public ----------
    def _best_of(self, make, text, scorer):
        """Single words: several takes, keep the one the scorer likes best (notebook 05 v2)."""
        n = config.TAKES_SINGLE_WORD if (scorer and len(text.split()) == 1) else 1
        best, best_s = None, -1.0
        for _ in range(max(1, n)):
            x, sr = make()
            s = scorer(x, sr) if n > 1 else 0.0
            if best is None or s > best_s:
                best, best_s = (x, sr), s
            if s >= 0.999:
                break
        return best

    def synthesize(self, text, reference=None, scorer=None):
        """Returns (audio float32, sample_rate, engine). engine == 'xtts-clone' when the speaker's voice was used.
        scorer(audio, sr) -> 0..1 (how well the recogniser understands the take), used to pick the best take."""
        self.last_clone_error = None
        if reference:
            try:
                x, sr = self._best_of(lambda: self._xtts_say(text, reference), text, scorer)
                return x, sr, "xtts-clone"
            except Exception as e:
                self.last_clone_error = f"{type(e).__name__}: {str(e)[:200]}"
        errors = []
        for name, fn in (("xtts", lambda: self._best_of(lambda: self._xtts_say(text), text, scorer)),
                         ("mms", lambda: self._mms_say(text)),
                         ("system", lambda: self._os_say(text))):
            try:
                x, sr = fn()
                return x, sr, name
            except Exception as e:
                errors.append(f"{name}: {type(e).__name__}: {str(e)[:120]}")
        raise RuntimeError("No text-to-speech engine worked. " + " | ".join(errors))
