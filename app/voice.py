"""Saved voice sample used for cloning ("Speak in my voice")."""
import numpy as np

from . import config
from .audio_io import ASR_SR, read_wav, resample, trim_silence, write_wav


class VoiceProfile:
    path = config.VOICE_PROFILE

    def exists(self):
        return self.path.exists() and self.seconds() >= 1.0

    def seconds(self):
        if not self.path.exists():
            return 0.0
        x, sr = read_wav(self.path)
        return round(len(x) / sr, 1)

    def add(self, x16, replace=False):
        x = trim_silence(x16, ASR_SR)
        if len(x) < 0.5 * ASR_SR:
            raise ValueError("That recording has almost no speech in it.")
        if self.path.exists() and not replace:
            old, sr = read_wav(self.path)
            old = resample(old, sr, ASR_SR)
            x = np.concatenate([old, np.zeros(int(0.3 * ASR_SR), np.float32), x])
        x = x[-int(config.VOICE_PROFILE_MAX_S * ASR_SR):]           # keep the most recent 60 s
        try:                                                        # cleaner sample -> clearer cloned voice (notebook 05 v2)
            import noisereduce as nr
            x = nr.reduce_noise(y=x, sr=ASR_SR, stationary=True, prop_decrease=0.8).astype(np.float32)
        except Exception:
            pass
        write_wav(self.path, x, ASR_SR)
        return self.seconds()

    def clear(self):
        if self.path.exists():
            self.path.unlink()

    def temp_reference(self, x16):
        """Use the current recording as the reference if it is long enough."""
        x = trim_silence(x16, ASR_SR)
        if len(x) < config.MIN_REFERENCE_S * ASR_SR:
            return None
        p = config.DATA_DIR / "current_reference.wav"
        write_wav(p, x, ASR_SR)
        return str(p)


PROFILE = VoiceProfile()
