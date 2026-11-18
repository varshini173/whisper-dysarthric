"""Start the app:  python run.py   (then it opens http://localhost:8000 in your browser)."""
import os
import sys
import threading
import time
import webbrowser

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import config  # noqa: E402


def _open_browser():
    time.sleep(2.0)
    try:
        webbrowser.open(f"http://localhost:{config.PORT}")
    except Exception:
        pass


if __name__ == "__main__":
    import uvicorn
    print(f"\n  Whisper-Dysarthric  ->  http://localhost:{config.PORT}\n"
          f"  Speech model: {config.find_asr_model()}\n  (press Ctrl+C here to stop)\n")
    if "--no-browser" not in sys.argv:
        threading.Thread(target=_open_browser, daemon=True).start()
    uvicorn.run("app.server:app", host=config.HOST, port=config.PORT, log_level="warning")
