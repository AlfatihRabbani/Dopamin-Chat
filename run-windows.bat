@echo off
REM Dopamine Chat - web UI launcher (Windows)
REM Self-installing: creates .venv (main) and .venv-applio (voice only) on first run.
setlocal enabledelayedexpansion
cd /d "%~dp0"

REM ---------- main python (chat/web) ----------
set "PY_CMD="
where py >nul 2>&1
if not errorlevel 1 (
    set "PY_CMD=py -3"
) else (
    where python >nul 2>&1
    if not errorlevel 1 set "PY_CMD=python"
)

REM ---------- voice python: pin to 3.11 (RVC/fairseq compat) ----------
set "PY311="
where py >nul 2>&1
if not errorlevel 1 (
    py -3.10 -V >nul 2>&1
    if not errorlevel 1 (
        set "PY311=py -3.10"
    ) else (
        py -3.11 -V >nul 2>&1
        if not errorlevel 1 set "PY311=py -3.11"
    )
)

REM ---------- main venv ----------
call :need_main
if errorlevel 1 (
    call :install_main
    if errorlevel 1 (
        echo [run-windows] Main install failed.
        pause
        exit /b 1
    )
)
call ".venv\Scripts\activate.bat"

python -c "import rich, torch, transformers, flask, diffusers" 2>nul
if errorlevel 1 (
    echo [run-windows] Core deps missing - re-running main install
    call :install_main
)

REM ---------- voice venv (best effort, never blocks chat) ----------
call :need_voice
if errorlevel 1 (
    if defined PY311 (
        call :install_voice
    ) else (
        echo [run-windows] Python 3.11 not found - skipping voice venv.
        echo   Install with: winget install -e --id Python.Python.3.11
    )
)

python web.py %*
endlocal
exit /b 0

REM ==========================================================================
:need_main
if not exist ".venv\Scripts\activate.bat" exit /b 1
exit /b 0

REM ==========================================================================
:need_voice
if not exist ".venv-applio\Scripts\activate.bat" exit /b 1
if not exist "Applio_src\core.py" exit /b 1
".venv-applio\Scripts\python.exe" -c "import sys, os; sys.path.insert(0, os.path.abspath('Applio_src')); os.chdir('Applio_src'); from core import run_infer_script; import piper" >nul 2>&1
if errorlevel 1 exit /b 1
exit /b 0

REM ==========================================================================
:install_main
if not defined PY_CMD (
    echo [run-windows] ERROR: python not found.
    echo   Install Python 3.10+ from https://www.python.org/downloads/
    echo   IMPORTANT: tick "Add Python to PATH" during install.
    exit /b 1
)

if not exist ".venv\Scripts\activate.bat" (
    echo [run-windows] Creating main virtualenv at .venv
    %PY_CMD% -m venv .venv
    if errorlevel 1 (
        echo [run-windows] ERROR: venv creation failed
        exit /b 1
    )
)
call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip wheel setuptools >nul

set "CUDA_TAG="
set "LLAMA_CUDA_INDEX="
set "GPU_KIND=cpu"
where nvidia-smi >nul 2>&1
if not errorlevel 1 (
    echo [run-windows] NVIDIA GPU detected
    set "GPU_KIND=nvidia"
    for /f "tokens=*" %%v in ('nvidia-smi 2^>nul ^| findstr /C:"CUDA Version"') do set "NVSMI_LINE=%%v"
    set "CUDA_VER="
    for /f "tokens=9" %%a in ("!NVSMI_LINE!") do set "CUDA_VER=%%a"
    if not defined CUDA_VER set "CUDA_VER=12.4"
    for /f "tokens=1,2 delims=." %%a in ("!CUDA_VER!") do (
        set "CUDA_MAJ=%%a"
        set "CUDA_MIN=%%b"
    )
    if !CUDA_MAJ! GEQ 13 (
        set "CUDA_TAG=cu128"
    ) else if !CUDA_MAJ! EQU 12 (
        if !CUDA_MIN! GEQ 8 ( set "CUDA_TAG=cu128"
        ) else if !CUDA_MIN! GEQ 4 ( set "CUDA_TAG=cu124"
        ) else ( set "CUDA_TAG=cu121" )
    ) else (
        set "CUDA_TAG=cu118"
    )
    echo [run-windows] CUDA !CUDA_VER! -^> tag !CUDA_TAG!
    set "LLAMA_CUDA_INDEX=https://abetlen.github.io/llama-cpp-python/whl/!CUDA_TAG!"
) else (
    echo [run-windows] No NVIDIA GPU. AMD ROCm on native Windows is unsupported.
    echo   AMD users: use WSL2 + Ubuntu and run run-linux.sh inside WSL.
    echo   Continuing with CPU torch ^(very slow^)...
)

echo [run-windows] Installing torch ...
set "TORCH_OK=0"
if defined CUDA_TAG (
    pip install torch --index-url https://download.pytorch.org/whl/!CUDA_TAG!
    if not errorlevel 1 set "TORCH_OK=1"
    if "!TORCH_OK!"=="0" if not "!CUDA_TAG!"=="cu124" (
        echo [run-windows]   !CUDA_TAG! failed, trying cu124 ...
        pip install torch --index-url https://download.pytorch.org/whl/cu124
        if not errorlevel 1 (
            set "TORCH_OK=1"
            set "CUDA_TAG=cu124"
            set "LLAMA_CUDA_INDEX=https://abetlen.github.io/llama-cpp-python/whl/cu124"
        )
    )
    if "!TORCH_OK!"=="0" (
        echo [run-windows]   CUDA wheels failed, falling back to CPU torch ...
        pip install torch
        if not errorlevel 1 (
            set "TORCH_OK=1"
            set "LLAMA_CUDA_INDEX="
        )
    )
) else (
    pip install torch
    if not errorlevel 1 set "TORCH_OK=1"
)
if "!TORCH_OK!"=="0" (
    echo [run-windows] ERROR: torch install failed
    exit /b 1
)

echo [run-windows] Installing core stack ...
findstr /V /B /C:"llama-cpp-python" requirements.txt > "%TEMP%\dopamine_req.txt"
pip install -r "%TEMP%\dopamine_req.txt"
if errorlevel 1 (
    del "%TEMP%\dopamine_req.txt"
    echo [run-windows] ERROR: core stack install failed
    exit /b 1
)
del "%TEMP%\dopamine_req.txt"

echo [run-windows] Installing llama-cpp-python (GGUF backend, prebuilt only)
set "LLAMA_OK=0"
if defined LLAMA_CUDA_INDEX (
    echo [run-windows]   trying CUDA wheel ...
    pip install --only-binary=llama-cpp-python llama-cpp-python --extra-index-url %LLAMA_CUDA_INDEX%
    if not errorlevel 1 set "LLAMA_OK=1"
)
if "!LLAMA_OK!"=="0" (
    echo [run-windows]   trying CPU wheel ...
    pip install --only-binary=llama-cpp-python llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
    if not errorlevel 1 set "LLAMA_OK=1"
)
if "!LLAMA_OK!"=="0" (
    pip install --only-binary=llama-cpp-python llama-cpp-python
    if not errorlevel 1 set "LLAMA_OK=1"
)
if "!LLAMA_OK!"=="0" (
    echo [run-windows] WARNING: llama-cpp-python not installed from any prebuilt wheel.
    echo   GGUF model support disabled. HF safetensors models still work.
)

echo [run-windows] Main install complete. GPU: %GPU_KIND%  llama-cpp-python: !LLAMA_OK!
exit /b 0

REM ==========================================================================
:install_voice
echo [run-windows] === Installing voice stack into .venv-applio (Python 3.11 + Applio) ===
if not exist "Applio_src\core.py" (
    echo [run-windows] [voice] Cloning Applio repo ...
    where git >nul 2>&1
    if errorlevel 1 (
        echo [run-windows] [voice] ERROR: git not found. Install Git for Windows.
        exit /b 1
    )
    git clone --depth 1 --branch 3.6.2 https://github.com/IAHispano/Applio.git Applio_src
    if errorlevel 1 (
        echo [run-windows] [voice] ERROR: git clone failed
        exit /b 1
    )
)

if not exist ".venv-applio\Scripts\activate.bat" (
    echo [run-windows] [voice] Creating voice virtualenv at .venv-applio
    %PY311% -m venv .venv-applio
    if errorlevel 1 (
        echo [run-windows] [voice] ERROR: .venv-applio creation failed
        exit /b 1
    )
)
".venv-applio\Scripts\python.exe" -m pip install --upgrade pip wheel >nul

echo [run-windows] [voice] Installing Applio requirements (~3GB, CUDA torch + deps) ...
".venv-applio\Scripts\python.exe" -m pip install -r Applio_src\requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 (
    echo [run-windows] [voice] ERROR: Applio requirements install failed
    exit /b 1
)

echo [run-windows] [voice] Installing piper-tts ...
".venv-applio\Scripts\python.exe" -m pip install piper-tts

echo [run-windows] [voice] Downloading Applio inference prerequisites (rmvpe + embedder) ...
pushd Applio_src
"..\.venv-applio\Scripts\python.exe" core.py prerequisites --models True --pretraineds_hifigan False
if errorlevel 1 echo [run-windows] [voice] WARNING: prerequisites download failed (RVC may fail on first run)
popd

echo [run-windows] [voice] Voice install complete.
exit /b 0
