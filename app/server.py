"""FastAPI server: serves the web page and the /api endpoints."""
import json
import os
import threading
import traceback

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse

from . import config
from .audio_io import ASR_SR, read_wav_bytes, resample
from .lexicon import Lexicon
from .voice import PROFILE

app = FastAPI(title="Whisper-Dysarthric")
STATE = dict(state="loading", message="Starting…", device="cpu", asr_mode="", finetuned=False,
             lexicon_size=0, model_path="")
PIPE = {}
_lock = threading.Lock()          # one conversion at a time (models are not thread-safe)


def _load_models():
    try:
        if os.environ.get("CLARITY_STUB_MODELS") == "1":          # developer test mode, no downloads
            from .stubs import StubRecognizer as Recognizer, StubSynthesizer as Synthesizer
        else:
            from .asr import Recognizer
            from .tts import Synthesizer
        from .pipeline import Pipeline
        STATE.update(message="Loading the speech recogniser…")
        rec = Recognizer()
        lex = Lexicon.from_model_dir(rec.model_path) if rec.finetuned else Lexicon([])
        tts = Synthesizer()
        PIPE["p"] = Pipeline(rec, tts, lex)
        STATE.update(state="ready", message="ready", device=rec.device, asr_mode=rec.mode,
                     finetuned=rec.finetuned, lexicon_size=len(lex), model_path=rec.model_path)
        print(f"[whisper-dysarthric] ready on {rec.device} | {rec.mode} | {len(lex)} prompts | {rec.model_path}")
        threading.Thread(target=tts.warm_up, daemon=True).start()   # load the voice model in the background
    except Exception as e:
        traceback.print_exc()
        STATE.update(state="error", message=f"{type(e).__name__}: {e}")


@app.on_event("startup")
def _startup():
    threading.Thread(target=_load_models, daemon=True).start()


@app.get("/")
def index():
    return FileResponse(config.WEB_DIR / "index.html")


@app.get("/api/health")
def health():
    from .tts import clone_installed
    return dict(STATE, clone_installed=clone_installed() or os.environ.get("CLARITY_STUB_MODELS") == "1")


def _not_ready():
    return JSONResponse(dict(ok=False, error="The models are still loading — please wait a moment."), status_code=503)


@app.post("/api/convert")
async def convert(request: Request, snap: int = 1, voice: int = 0):
    if STATE["state"] != "ready":
        return _not_ready()
    body = await request.body()

    def work():
        with _lock:
            return PIPE["p"].run(body, snap=bool(snap), voice=bool(voice))
    try:
        return await run_in_threadpool(work)
    except Exception as e:
        traceback.print_exc()
        return dict(ok=False, error=f"Conversion failed: {type(e).__name__}: {str(e)[:200]}")


@app.post("/api/speak")
async def speak(request: Request, voice: int = 0):
    if STATE["state"] != "ready":
        return _not_ready()
    try:
        text = json.loads((await request.body()) or b"{}").get("text", "")
    except Exception:
        return dict(ok=False, error="Bad request.")

    def work():
        with _lock:
            return PIPE["p"].speak(text, voice=bool(voice))
    try:
        return await run_in_threadpool(work)
    except Exception as e:
        traceback.print_exc()
        return dict(ok=False, error=f"Speaking failed: {type(e).__name__}: {str(e)[:200]}")


@app.get("/api/voice")
def voice_info():
    return dict(has_profile=PROFILE.exists(), seconds=PROFILE.seconds())


@app.post("/api/voice/profile")
async def voice_add(request: Request, mode: str = "add"):
    try:
        x, sr = read_wav_bytes(await request.body())
        secs = PROFILE.add(resample(x, sr, ASR_SR), replace=(mode == "replace"))
        return dict(ok=True, seconds=secs)
    except Exception as e:
        return dict(ok=False, error=str(e))


@app.post("/api/voice/clear")
def voice_clear():
    PROFILE.clear()
    return dict(ok=True)
