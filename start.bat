@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
    echo Please run setup.bat first.
    pause
    exit /b 1
)
call .venv\Scripts\activate.bat

REM ---- Voice cloning model (XTTS-v2) licence ----------------------------------------------
REM XTTS-v2 is released under the Coqui Public Model License: NON-COMMERCIAL use only.
REM Keeping the next line means you accept that licence. Delete it to use only the fallback voices.
set COQUI_TOS_AGREED=1

REM ---- Optional: use a model folder somewhere else (remove REM and edit the path) ----------
REM set CLARITY_ASR_MODEL=C:\Users\varsh\Desktop\whisper-dysarthric-torgoUASpeech

python run.py
pause
