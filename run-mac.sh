#!/usr/bin/env bash
# Dopamine Chat — web UI launcher (macOS)
# Self-installing. Apple Silicon uses Metal (MPS); Intel falls back to CPU.
set -u
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

PY="${PYTHON:-python3}"

install() {
    if ! command -v "$PY" >/dev/null 2>&1; then
        echo "[run-mac] python3 not found. Install via:  brew install python@3.11"
        exit 1
    fi
    if [ ! -d ".venv" ]; then
        echo "[run-mac] Creating virtualenv at .venv"
        "$PY" -m venv .venv || { echo "[run-mac] venv creation failed"; exit 1; }
    fi
    # shellcheck disable=SC1091
    source .venv/bin/activate
    python -m pip install --upgrade pip wheel setuptools >/dev/null

    echo "[run-mac] Installing torch (Metal/CPU)..."
    pip install torch || { echo "[run-mac] torch install failed"; exit 1; }

    echo "[run-mac] Installing core stack ..."
    grep -v '^llama-cpp-python' requirements.txt > "/tmp/dopamine_req_$$.txt"
    pip install -r "/tmp/dopamine_req_$$.txt" || {
        echo "[run-mac] core stack install failed"
        rm -f "/tmp/dopamine_req_$$.txt"
        exit 1
    }
    rm -f "/tmp/dopamine_req_$$.txt"

    echo "[run-mac] Installing llama-cpp-python (Metal prebuilt) ..."
    if ! pip install --only-binary=llama-cpp-python llama-cpp-python \
            --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/metal; then
        echo "[run-mac]   metal wheel unavailable — trying CPU wheel"
        if ! pip install --only-binary=llama-cpp-python llama-cpp-python \
                --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu; then
            echo "[run-mac] WARNING: llama-cpp-python unavailable; GGUF disabled."
        fi
    fi
    echo "[run-mac] Install complete."
}

if [ ! -d ".venv" ]; then
    install
fi
# shellcheck disable=SC1091
source .venv/bin/activate

if ! python -c "import rich, torch, transformers, flask, diffusers" 2>/dev/null; then
    echo "[run-mac] Core deps missing or broken — re-running install"
    install
fi

exec python web.py "$@"
