# Whisper-Dysarthric — dysarthric speech → text → clear speech

Record or upload dysarthric speech. A **Whisper-small model fine-tuned on TORGO + UA-Speech** (notebook 04) recognises the words; the app shows the text and speaks it back in a clear voice — a standard voice, or **the speaker's own voice** (XTTS-v2 voice cloning).

```
microphone / file ──► fine-tuned Whisper ──► prompt matching (optional) ──► XTTS-v2 ──► clear speech
                         (5 best guesses)      (1,241 known prompts)       (standard / cloned voice)
```

---

## 1. One-time setup (Windows)

1. **Python 3.12** must be installed (python.org, tick *"Add python.exe to PATH"*). You already have it if your old `venv312` worked.
2. **Copy your trained model** into this app:
   `whisper-dysarthric\models\whisper-dysarthric-torgoUASpeech\`
   The folder must directly contain `config.json`, `model.safetensors`, `lexicon.txt`, … (exactly the folder you unzipped from `whisper-dysarthric.zip`).
3. **Double-click `setup.bat`.** It creates a private Python environment (`.venv`), installs PyTorch (GPU version if you have an NVIDIA card, otherwise CPU) and everything else. Takes 5–15 minutes; needs internet.

## 2. Start the app (every time)

**Double-click `start.bat`.** The browser opens **http://localhost:8000** automatically (if not, open it yourself).

- The status pill (top right) shows *Loading models…*, then **Model ready · CUDA/CPU · 1241 prompts**.
- **First start only:** it downloads the voice model XTTS-v2 (~1.9 GB) and a few small Whisper files. Later starts are offline and much faster.
- To stop: close the black terminal window (or press Ctrl+C in it).

## 3. Using it

1. **Record** (press again to stop) or **Choose file** / drag & drop a recording.
2. Options:
   - **Match known prompts** – snaps the result to the closest phrase from the TORGO/UA-Speech prompt list (best for the corpus words; turn off for free speech).
   - **Speak in my voice** (off by default) – clones the speaker's voice. In our evaluation the standard voice was clearly more intelligible (57% vs 31% word accuracy), while the cloned voice keeps ~89% of the speaker similarity, so use it when sounding like the speaker matters more than clarity. Under *Manage sample* save 20–30 s of the speaker's clearest speech; without a saved sample, the current recording is used if it is ≥ 3 s.
3. **Convert** → recognised text in large letters, **Play clear voice / Play original / Play both**, other guesses, timings, and waveform + spectrogram of input vs output.
4. **Wrong word?** Click one of the *other guesses*, or type the correct text in the box and press **Speak this text**.

**Voice-output improvements (from notebook 05 v2):** single words are generated twice and the take the recogniser understands best is kept; stray sound after short words is trimmed; XTTS runs with calmer sampling; saved voice samples are denoised.

## 4. Speed

| Hardware | Recognition | Voice (per short sentence) |
|---|---|---|
| NVIDIA GPU | ~0.3–1 s | ~1–3 s |
| CPU only | ~1–4 s | ~5–30 s (single words are faster) |

## 5. Troubleshooting

| Problem | Fix |
|---|---|
| Yellow banner *"trained model was not found"* | The model folder is not at `models\whisper-dysarthric-torgoUASpeech\config.json`. Move it there, or edit the `CLARITY_ASR_MODEL` line in `start.bat` to point to where it is. |
| *"Voice cloning is not installed"* / *standard voice was used* | Check the terminal for the error. Make sure `start.bat` still contains `set COQUI_TOS_AGREED=1`. |
| Port 8000 busy | Add `set CLARITY_PORT=8010` in `start.bat` before `python run.py`, then open http://localhost:8010 |
| Microphone not working | Allow microphone access in the browser (lock icon in the address bar). It only works on `localhost`, not on a network IP. |
| `setup.bat` failed | Send the red error lines from the window to Claude. |

## 6. Folder layout

```
whisper-dysarthric/
├── setup.bat / start.bat     one-time install / start the app
├── run.py                    starts the server and opens the browser
├── requirements.txt
├── app/
│   ├── server.py             FastAPI: /, /api/health, /api/convert, /api/speak, /api/voice…
│   ├── pipeline.py           audio → ASR → prompt matching → TTS
│   ├── asr.py                fine-tuned Whisper (5-best beam search)
│   ├── lexicon.py            prompt matching (closest known phrase)
│   ├── tts.py                XTTS-v2 cloned / standard voice, MMS-TTS + system-voice fallbacks
│   ├── voice.py              saved voice sample for cloning
│   ├── audio_io.py, text_utils.py, config.py
│   └── stubs.py              developer test mode only
├── web/index.html            the whole user interface
├── models/                   ← put whisper-dysarthric-torgoUASpeech here
└── data/                     saved voice sample (created automatically)
```

## 7. Licences and ethics

- **XTTS-v2** (voice cloning): Coqui Public Model License — **non-commercial use only**. `start.bat` accepts it via `COQUI_TOS_AGREED=1`; delete that line to use only the fallback voices.
- **TORGO** and **UA-Speech** are for academic, non-commercial research; cite them in any publication.
- Only clone a person's voice with their consent.
