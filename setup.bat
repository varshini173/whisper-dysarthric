@echo off
setlocal
cd /d "%~dp0"
echo.
echo ===== Whisper-Dysarthric : one-time setup =====
echo.

if exist .venv\Scripts\python.exe goto have_venv
echo Creating a private Python environment in .venv ...
where py >nul 2>nul
if %errorlevel%==0 (
    py -3.12 -m venv .venv 2>nul
    if not exist .venv\Scripts\python.exe py -3 -m venv .venv
) else (
    python -m venv .venv
)
if not exist .venv\Scripts\python.exe (
    echo.
    echo Could not create the Python environment.
    echo Install Python 3.12 from https://www.python.org/downloads/ ^(tick "Add python.exe to PATH"^) and run setup.bat again.
    pause
    exit /b 1
)

:have_venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip

where nvidia-smi >nul 2>nul
if %errorlevel%==0 (
    echo.
    echo NVIDIA GPU found - installing PyTorch with CUDA support ...
    pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126
) else (
    echo.
    echo No NVIDIA GPU found - installing the CPU version of PyTorch ^(works, just slower^) ...
    pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cpu
)
if errorlevel 1 goto failed

echo.
echo Installing the remaining packages ...
pip install -r requirements.txt
if errorlevel 1 goto failed

echo.
if exist models\whisper-dysarthric-torgoUASpeech\config.json (
    echo Trained model found in models\whisper-dysarthric-torgoUASpeech
) else (
    echo NOTE: copy your trained model folder "whisper-dysarthric-torgoUASpeech" into the "models" folder
    echo       ^(so that models\whisper-dysarthric-torgoUASpeech\config.json exists^) before starting.
)
echo.
echo ===== Setup finished. Now double-click start.bat =====
pause
exit /b 0

:failed
echo.
echo Setup failed - scroll up to see the error, then send it to Claude.
pause
exit /b 1
