"""dysarthric audio -> fine-tuned Whisper -> (prompt matching) -> clear voice."""
import base64
import difflib
import time

import numpy as np

from . import config
from .audio_io import ASR_SR, normalize_rms, peak_normalize, read_wav_bytes, resample, rms, write_wav_bytes
from .text_utils import normalize_text
from .voice import PROFILE


class Pipeline:
    def __init__(self, recognizer, synthesizer, lexicon):
        self.rec, self.tts, self.lex = recognizer, synthesizer, lexicon

    def run(self, wav_bytes, snap=True, voice=False):
        try:
            x, sr = read_wav_bytes(wav_bytes)
        except Exception:
            return dict(ok=False, error="Could not read that audio. Please record with the Record button or upload a WAV/MP3 file.")
        x = resample(x, sr, ASR_SR)
        truncated = len(x) > config.MAX_SECONDS * ASR_SR
        x = x[: int(config.MAX_SECONDS * ASR_SR)]
        out = dict(input_seconds=round(len(x) / ASR_SR, 2), truncated=bool(truncated))
        if len(x) < 0.15 * ASR_SR or rms(x) < 1e-4:
            return dict(out, ok=False, error="No speech detected (the recording is silent or too short).")
        x = normalize_rms(x)

        t0 = time.time()
        cands = self.rec.transcribe(x)
        out["asr_seconds"] = round(time.time() - t0, 2)
        raw = normalize_text(cands[0]) if cands else ""
        text, how = raw, "off"
        if snap and len(self.lex):
            text, how = self.lex.snap(cands)
        out.update(raw_text=raw, text=text, snap=how,
                   candidates=[c for c in dict.fromkeys(normalize_text(c) for c in cands) if c])
        if not text:
            return dict(out, ok=False, error="Could not recognise any words. Try speaking a little closer to the microphone.")

        reference, source = None, None
        if voice:
            if PROFILE.exists():
                reference, source = str(PROFILE.path), "saved voice sample"
            else:
                reference = PROFILE.temp_reference(x)
                source = "this recording" if reference else None
        return self._speak(text, out, reference, source, voice)

    def speak(self, text, voice=False):
        """Text -> voice only (used when the user corrects the words or picks another guess)."""
        text = normalize_text(text)
        if not text:
            return dict(ok=False, error="Nothing to say.")
        reference = str(PROFILE.path) if (voice and PROFILE.exists()) else None
        return self._speak(text, dict(text=text, snap="off", candidates=[]), reference,
                           "saved voice sample" if reference else None, voice)

    def _scorer(self, text):
        def score(y, sr):
            try:
                hyp = self.rec.transcribe(resample(y, sr, ASR_SR), n_best=1)
                return difflib.SequenceMatcher(None, text, normalize_text(hyp[0]) if hyp else "").ratio()
            except Exception:
                return 0.0
        return score

    def _speak(self, text, out, reference, source, voice_requested):
        t0 = time.time()
        y, ysr, engine = self.tts.synthesize(text, reference=reference, scorer=self._scorer(text))
        y = peak_normalize(y, 0.9)
        pad = np.zeros(int(0.15 * ysr), np.float32)
        y = np.concatenate([pad, y, pad])
        cloned = engine == "xtts-clone"
        note = None
        if voice_requested and not cloned:
            note = ("No voice sample yet: save at least a few seconds with 'Save this audio as my voice sample', "
                    "or record 3 s or more. The standard voice was used." if reference is None else
                    "Your-voice mode failed, so the standard voice was used: " + str(self.tts.last_clone_error))
        out.update(ok=True, tts_seconds=round(time.time() - t0, 2), engine=engine,
                   voice="cloned" if cloned else "standard", voice_source=source if cloned else None,
                   voice_note=note, output_seconds=round(len(y) / ysr, 2), output_sr=int(ysr),
                   audio_b64=base64.b64encode(write_wav_bytes(y, ysr)).decode("ascii"))
        out.setdefault("asr_seconds", 0)
        return out
