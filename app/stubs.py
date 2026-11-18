"""Developer test mode only (CLARITY_STUB_MODELS=1): fake models so the web page can be tried without
downloading anything. Never used in normal running."""
import numpy as np


class StubRecognizer:
    device, finetuned, model_path = "cpu", True, "stub"
    mode = "STUB (developer test mode)"

    def transcribe(self, audio16k, n_best=5):
        return ["the quick brown fox", "the quick brown socks", "a quick brown fox"][:n_best]


class StubSynthesizer:
    last_clone_error = None

    def warm_up(self):
        pass

    def synthesize(self, text, reference=None, scorer=None):
        if scorer: scorer(np.zeros(2400, np.float32), 24000)
        sr = 24000
        t = np.arange(int(0.12 * sr * max(4, len(text)))) / sr
        env = np.abs(np.sin(2 * np.pi * 3 * t))
        x = (0.3 * env * np.sin(2 * np.pi * (180 + 40 * np.sin(2 * np.pi * 0.7 * t)) * t)).astype(np.float32)
        return x, sr, ("xtts-clone" if reference else "xtts")
