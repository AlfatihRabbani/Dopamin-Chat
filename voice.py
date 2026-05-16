"""
TTS (piper) + Voice Conversion (RVC) pipeline.

Voice deps live in a separate Python 3.11 venv (.venv-voice) because the RVC
ecosystem (fairseq, numpy<=1.25) does not build on Python 3.13. This module
proxies all TTS/RVC calls into that venv via voice_worker.py as a subprocess.

Layout:
    voices/
        piper/<voice>.onnx
        piper/<voice>.onnx.json
        rvc/<character>/<name>.pth
        rvc/<character>/<name>.index
        rvc/rmvpe.pt
    .venv-voice/   (separate Py 3.11 venv with piper-tts + rvc-python)
"""
import os
import sys
import time
import subprocess
import tempfile
from pathlib import Path


def _log(msg: str) -> None:
    print(f"[voice] {msg}", flush=True)

ROOT = Path(__file__).resolve().parent
VOICES_DIR = ROOT / "voices"
PIPER_DIR = VOICES_DIR / "piper"
RVC_DIR = VOICES_DIR / "rvc"
VOICES_DIR.mkdir(exist_ok=True)
PIPER_DIR.mkdir(exist_ok=True)
RVC_DIR.mkdir(exist_ok=True)

# Walk both Applio variants — local clone (Applio_src/logs) and the user's
# model folder (Applio/<name>/*.pth). On Windows these dir-name lookups are
# case-insensitive; we list both spellings for portability.
APPLIO_MODEL_DIRS = [
    ROOT / "Applio",
    ROOT / "applio",
    ROOT / "Applio_src" / "logs",
]

WORKER = ROOT / "applio_worker.py"
VOICE_VENV = ".venv-applio"


def _voice_python() -> Path | None:
    """Return path to voice venv python, or None if missing."""
    if os.name == "nt":
        p = ROOT / VOICE_VENV / "Scripts" / "python.exe"
    else:
        p = ROOT / VOICE_VENV / "bin" / "python"
    return p if p.exists() else None


def _run_worker(args: list[str]) -> None:
    """Invoke voice_worker.py in .venv-voice with the given args. Raises on failure."""
    py = _voice_python()
    if py is None:
        raise RuntimeError(
            "Voice venv (.venv-voice) not found. Run run-windows.bat (or run-linux.sh) "
            "to create it. Needs Python 3.11 installed system-wide first."
        )
    if not WORKER.exists():
        raise RuntimeError(f"voice_worker.py missing: {WORKER}")
    t0 = time.monotonic()
    proc = subprocess.run(
        [str(py), str(WORKER), *args],
        capture_output=True,
        text=True,
    )
    dt = time.monotonic() - t0
    op = args[0] if args else "?"
    if proc.returncode != 0:
        msg = (proc.stderr or proc.stdout or "").strip() or f"exit {proc.returncode}"
        _log(f"{op} FAILED in {dt:.1f}s: {msg}")
        raise RuntimeError(f"voice worker failed: {msg}")
    # Echo worker's stderr (its log channel) so the launcher console sees it.
    if proc.stderr:
        for line in proc.stderr.rstrip().splitlines():
            _log(f"{op} | {line}")
    _log(f"{op} done in {dt:.1f}s")


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def list_piper_voices() -> list[dict]:
    out = []
    if not PIPER_DIR.exists():
        return out
    for f in PIPER_DIR.glob("*.onnx"):
        out.append({"name": f.stem, "path": str(f)})
    return sorted(out, key=lambda x: x["name"])


def list_rvc_models() -> list[dict]:
    """Walk voices/rvc/ + Applio model folders for .pth files; pair with same-stem .index."""
    out = []
    seen = set()
    seen_paths = set()  # de-dupe Windows case-insensitive collisions
    for root in [RVC_DIR, *APPLIO_MODEL_DIRS]:
        if not root.exists():
            continue
        resolved = str(root.resolve()).lower()
        if resolved in seen_paths:
            continue
        seen_paths.add(resolved)
        for pth in root.rglob("*.pth"):
            key = pth.name
            if key in seen:
                continue
            seen.add(key)
            stem = pth.stem
            idx = None
            for cand in pth.parent.glob(f"{stem}*.index"):
                idx = cand
                break
            if idx is None:
                for cand in pth.parent.glob("*.index"):
                    idx = cand
                    break
            out.append({
                "name": stem,
                "pth": str(pth),
                "index": str(idx) if idx else "",
                "folder": str(pth.parent),
            })
    return sorted(out, key=lambda x: x["name"])


# ---------------------------------------------------------------------------
# Availability checks
# ---------------------------------------------------------------------------

def piper_available() -> tuple[bool, str]:
    py = _voice_python()
    if py is None:
        return False, ("Voice venv (.venv-applio) not found. Run run-windows.bat / "
                       "run-linux.sh to create it.")
    r = subprocess.run([str(py), "-c", "import piper"], capture_output=True)
    if r.returncode != 0:
        return False, "piper-tts not installed in .venv-voice. Re-run the launcher."
    return True, ""


def rvc_available() -> tuple[bool, str]:
    py = _voice_python()
    if py is None:
        return False, ("Voice venv (.venv-applio) not found. Run run-windows.bat / "
                       "run-linux.sh to create it.")
    applio_src = ROOT / "Applio_src"
    if not (applio_src / "core.py").exists():
        return False, "Applio_src/ missing. Re-run the launcher to clone it."
    r = subprocess.run(
        [str(py), "-c",
         f"import sys, os; sys.path.insert(0, r'{applio_src}'); os.chdir(r'{applio_src}'); from core import run_infer_script"],
        capture_output=True,
    )
    if r.returncode != 0:
        return False, "Applio not installed in .venv-applio. Re-run the launcher."
    return True, ""


def rvc_device_info() -> dict:
    """Query .venv-applio's torch + Applio Config to report inference device.

    Returns {device, gpu_name, vram_gb, cuda_available} or {error}.
    """
    py = _voice_python()
    applio_src = ROOT / "Applio_src"
    if py is None or not (applio_src / "core.py").exists():
        return {"device": "?", "error": "voice venv missing"}
    probe = (
        "import sys, os, json; "
        f"sys.path.insert(0, r'{applio_src}'); "
        f"os.chdir(r'{applio_src}'); "
        "import torch; "
        "from rvc.configs.config import Config; "
        "c = Config(); "
        "print(json.dumps({"
        "'cuda_available': torch.cuda.is_available(), "
        "'device': c.device, "
        "'gpu_name': c.gpu_name, "
        "'vram_gb': c.gpu_mem, "
        "'torch_version': torch.__version__, "
        "'cuda_runtime': torch.version.cuda, "
        "}))"
    )
    r = subprocess.run([str(py), "-c", probe], capture_output=True, text=True)
    if r.returncode != 0:
        return {"device": "?", "error": (r.stderr or "probe failed").strip()[:200]}
    import json as _json
    try:
        return _json.loads(r.stdout.strip().splitlines()[-1])
    except Exception as e:
        return {"device": "?", "error": f"parse: {e}"}


# ---------------------------------------------------------------------------
# TTS — piper
# ---------------------------------------------------------------------------

def piper_synthesize(text: str, voice: str) -> bytes:
    """Generate WAV bytes from text using the named piper voice (in .venv-voice)."""
    if not text.strip():
        raise RuntimeError("empty text")
    model_path = PIPER_DIR / f"{voice}.onnx"
    if not model_path.exists():
        raise RuntimeError(f"piper voice not found: {model_path}")
    _log(f"piper synth: voice={voice} chars={len(text)}")
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w", encoding="utf-8") as tf:
        tf.write(text)
        text_path = tf.name
    out_path = text_path + ".wav"
    try:
        _run_worker(["piper", voice, text_path, out_path])
        with open(out_path, "rb") as f:
            return f.read()
    finally:
        for p in (text_path, out_path):
            try: os.remove(p)
            except OSError: pass


# ---------------------------------------------------------------------------
# RVC voice conversion (Applio-compatible models)
# ---------------------------------------------------------------------------

def rvc_convert(wav_bytes: bytes,
                pth_path: str,
                index_path: str = "",
                pitch: int = 0,
                index_rate: float = 0.75,
                extractor: str = "rmvpe") -> bytes:
    """Run RVC inference over input WAV (via .venv-voice subprocess)."""
    if not Path(pth_path).exists():
        raise RuntimeError(f"RVC .pth not found: {pth_path}")
    _log(f"rvc convert: pth={Path(pth_path).name} wav_bytes={len(wav_bytes)} "
         f"pitch={pitch} extractor={extractor}")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as in_f:
        in_f.write(wav_bytes)
        in_path = in_f.name
    out_path = in_path + ".out.wav"
    try:
        _run_worker([
            "rvc", in_path, pth_path, index_path or "",
            str(pitch), str(index_rate), extractor, out_path,
        ])
        with open(out_path, "rb") as f:
            return f.read()
    finally:
        for p in (in_path, out_path):
            try: os.remove(p)
            except OSError: pass


# ---------------------------------------------------------------------------
# Convenience: text → piper → RVC
# ---------------------------------------------------------------------------

def speak_as_character(text: str,
                       piper_voice: str,
                       rvc_pth: str = "",
                       rvc_index: str = "",
                       pitch: int = 0,
                       index_rate: float = 0.75,
                       extractor: str = "rmvpe") -> bytes:
    """End-to-end: text → piper WAV → optional RVC → final WAV."""
    wav = piper_synthesize(text, piper_voice)
    if rvc_pth:
        wav = rvc_convert(wav, rvc_pth, rvc_index, pitch, index_rate, extractor)
    return wav
