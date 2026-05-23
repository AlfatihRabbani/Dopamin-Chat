#!/usr/bin/env bash
# Dopamine Chat — web UI launcher (Linux/WSL)
# Self-installing: creates .venv (main) and .venv-applio (voice only) on first run.
set -u
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

PY="${PYTHON:-python3}"
PY311="$(command -v python3.10 || command -v python3.11 || true)"

install_main() {
    if ! command -v "$PY" >/dev/null 2>&1; then
        echo "[run-linux] ERROR: python3 not found. Install Python 3.10+:"
        echo "  Arch:   sudo pacman -S python python-pip"
        echo "  Ubuntu: sudo apt install python3 python3-pip python3-venv"
        echo "  Fedora: sudo dnf install python3 python3-pip"
        exit 1
    fi

    if [ ! -d ".venv" ]; then
        echo "[run-linux] Creating main virtualenv at .venv"
        "$PY" -m venv .venv || { echo "[run-linux] venv creation failed"; exit 1; }
    fi
    # shellcheck disable=SC1091
    source .venv/bin/activate
    python -m pip install --upgrade pip wheel setuptools >/dev/null

    local TORCH_INDEX="" LLAMA_CUDA_INDEX="" GPU_KIND="cpu" CUDA_TAG=""
    if command -v nvidia-smi >/dev/null 2>&1; then
        echo "[run-linux] NVIDIA GPU detected"
        GPU_KIND="nvidia"
        local CUDA_VER MAJ MIN
        CUDA_VER="$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | head -1)"
        CUDA_VER="${CUDA_VER:-12.4}"
        MAJ="${CUDA_VER%.*}"; MIN="${CUDA_VER#*.}"
        if   [ "$MAJ" -ge 13 ];                              then CUDA_TAG="cu128"
        elif [ "$MAJ" -eq 12 ] && [ "$MIN" -ge 8 ];          then CUDA_TAG="cu128"
        elif [ "$MAJ" -eq 12 ] && [ "$MIN" -ge 4 ];          then CUDA_TAG="cu124"
        elif [ "$MAJ" -eq 12 ];                              then CUDA_TAG="cu121"
        else                                                      CUDA_TAG="cu118"
        fi
        TORCH_INDEX="https://download.pytorch.org/whl/${CUDA_TAG}"
        LLAMA_CUDA_INDEX="https://abetlen.github.io/llama-cpp-python/whl/${CUDA_TAG}"
    elif command -v rocm-smi >/dev/null 2>&1 || [ -d "/opt/rocm" ]; then
        echo "[run-linux] AMD ROCm detected"
        GPU_KIND="amd"
        TORCH_INDEX="https://download.pytorch.org/whl/rocm6.0"
    else
        echo "[run-linux] No GPU detected — CPU only (slow)"
    fi

    echo "[run-linux] Installing torch ..."
    if [ -n "$TORCH_INDEX" ]; then
        pip install torch --index-url "$TORCH_INDEX" || {
            echo "[run-linux] ERROR: torch install failed"; exit 1; }
    else
        pip install torch || { echo "[run-linux] ERROR: torch install failed"; exit 1; }
    fi

    echo "[run-linux] Installing core stack (transformers, diffusers, flask, ...)"
    grep -v '^llama-cpp-python' requirements.txt > "/tmp/dopamine_req_$$.txt"
    pip install -r "/tmp/dopamine_req_$$.txt" || {
        echo "[run-linux] ERROR: core stack install failed"
        rm -f "/tmp/dopamine_req_$$.txt"
        exit 1
    }
    rm -f "/tmp/dopamine_req_$$.txt"

    echo "[run-linux] Installing llama-cpp-python (GGUF backend, prebuilt only)"
    local LLAMA_OK=0
    if [ -n "$LLAMA_CUDA_INDEX" ]; then
        echo "[run-linux]   trying CUDA wheel (${CUDA_TAG}) ..."
        if pip install --only-binary=llama-cpp-python llama-cpp-python \
                --extra-index-url "$LLAMA_CUDA_INDEX"; then LLAMA_OK=1; fi
    fi
    if [ "$LLAMA_OK" = "0" ]; then
        echo "[run-linux]   trying CPU wheel ..."
        if pip install --only-binary=llama-cpp-python llama-cpp-python \
                --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu; then LLAMA_OK=1; fi
    fi
    if [ "$LLAMA_OK" = "0" ]; then
        if pip install --only-binary=llama-cpp-python llama-cpp-python; then LLAMA_OK=1; fi
    fi
    if [ "$LLAMA_OK" = "0" ]; then
        echo "[run-linux] WARNING: llama-cpp-python could not be installed."
        echo "  GGUF model support disabled. HF safetensors still work."
    fi

    echo "[run-linux] Main install complete. GPU: $GPU_KIND  llama-cpp-python: $([ "$LLAMA_OK" = "1" ] && echo "yes" || echo "no")"
}

install_voice() {
    if [ -z "$PY311" ]; then
        echo "[run-linux] python3.11 not found - skipping voice venv."
        echo "  Install:"
        echo "    Ubuntu: sudo apt install python3.11 python3.11-venv"
        echo "    Arch:   yay -S python311  (AUR)"
        echo "    Fedora: sudo dnf install python3.11"
        return 0
    fi
    echo "[run-linux] === Installing voice stack into .venv-applio (Python 3.11 + Applio) ==="

    if [ ! -f "Applio_src/core.py" ]; then
        echo "[run-linux] [voice] Cloning Applio repo ..."
        if ! command -v git >/dev/null 2>&1; then
            echo "[run-linux] [voice] ERROR: git not found"
            return 1
        fi
        git clone --depth 1 --branch 3.6.2 https://github.com/IAHispano/Applio.git Applio_src || {
            echo "[run-linux] [voice] git clone failed"; return 1; }
    fi

    if [ ! -d ".venv-applio" ]; then
        echo "[run-linux] [voice] Creating voice virtualenv at .venv-applio"
        "$PY311" -m venv .venv-applio || { echo "[run-linux] .venv-applio creation failed"; return 1; }
    fi
    .venv-applio/bin/python -m pip install --upgrade pip wheel >/dev/null

    echo "[run-linux] [voice] Installing Applio requirements (~3GB) ..."
    .venv-applio/bin/python -m pip install -r Applio_src/requirements.txt \
        --extra-index-url https://download.pytorch.org/whl/cu128 || {
        echo "[run-linux] [voice] ERROR: Applio requirements install failed"; return 1; }

    echo "[run-linux] [voice] Installing piper-tts ..."
    .venv-applio/bin/python -m pip install piper-tts || true

    echo "[run-linux] [voice] Downloading Applio inference prerequisites ..."
    (cd Applio_src && ../.venv-applio/bin/python core.py prerequisites --models True --pretraineds_hifigan False) || \
        echo "[run-linux] [voice] WARNING: prerequisites download failed (RVC may fail on first run)"

    echo "[run-linux] [voice] Voice install complete."
}

# Main venv
if [ ! -d ".venv" ]; then
    install_main
fi
# shellcheck disable=SC1091
source .venv/bin/activate

if ! python -c "import rich, torch, transformers, flask, diffusers" 2>/dev/null; then
    echo "[run-linux] Core deps missing or broken — re-running main install"
    install_main
fi

# Voice venv (best effort, never blocks chat)
voice_ok=0
if [ -x ".venv-applio/bin/python" ]; then
    if [ -f "Applio_src/core.py" ] && .venv-applio/bin/python -c "import sys, os; sys.path.insert(0, os.path.abspath('Applio_src')); os.chdir('Applio_src'); from core import run_infer_script; import piper" 2>/dev/null; then
        voice_ok=1
    fi
fi
if [ "$voice_ok" = "0" ]; then
    install_voice || true
fi

exec python web.py "$@"
