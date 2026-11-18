"""All settings in one place. Most can be overridden with environment variables (see README)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT / "web"
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

HOST = os.environ.get("CLARITY_HOST", "127.0.0.1")
PORT = int(os.environ.get("CLARITY_PORT", "8000"))

# ---- speech recognition model (fine-tuned Whisper from notebook 04) ----
FALLBACK_ASR = "openai/whisper-small"          # generic Whisper, used only if the trained model is missing
BASE_PROCESSOR = "openai/whisper-small"        # feature extractor + tokenizer (unchanged by fine-tuning)


def find_asr_model() -> str:
    env = os.environ.get("CLARITY_ASR_MODEL")
    if env:
        return env
    for name in ("whisper-dysarthric-torgoUASpeech", "whisper-dysarthric"):
        p = ROOT / "models" / name
        if (p / "config.json").exists():
            return str(p)
    return FALLBACK_ASR


N_BEST = 5                  # recognition alternatives shown as "other guesses"
MAX_SECONDS = 30.0          # Whisper works on <= 30 s windows

# ---- text-to-speech ----
XTTS_ID = "tts_models/multilingual/multi-dataset/xtts_v2"
PRESET_SPEAKER = os.environ.get("CLARITY_PRESET_SPEAKER", "Ana Florence")   # XTTS built-in "standard" voice
MMS_ID = "facebook/mms-tts-eng"                                              # light fallback voice
XTTS_TEMPERATURE = 0.65          # calmer than the default 0.75 (notebook 05 v2)
TAKES_SINGLE_WORD = 2            # single words: make 2 takes, keep the one the recogniser understands best
VOICE_PROFILE = DATA_DIR / "voice_profile.wav"
VOICE_PROFILE_MAX_S = 60.0
MIN_REFERENCE_S = 3.0
