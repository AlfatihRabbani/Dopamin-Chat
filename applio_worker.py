"""
Voice worker — runs inside .venv-applio (Python 3.11 + Applio stack).

Invoked as subprocess by voice.py in main .venv. Communicates via argv +
files on disk. stdout is reserved for the output WAV path on success; all
log/info goes to stderr. Exit 0 = success, non-zero = failure.

Usage:
    python applio_worker.py piper <voice_name> <text_file> <out_wav>
    python applio_worker.py rvc <in_wav> <pth> <index_or_empty> <pitch> <index_rate> <extractor> <out_wav>

For rvc, this calls Applio's core.run_infer_script which uses the
rvc.infer.infer.VoiceConverter pipeline (more stable than rvc-python).
"""
import os
import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent
APPLIO_SRC = ROOT / "Applio_src"
PIPER_DIR = ROOT / "voices" / "piper"

# Applio modules expect to be imported with Applio_src as cwd / on sys.path.
sys.path.insert(0, str(APPLIO_SRC))


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def cmd_piper(voice: str, text_file: str, out_wav: str) -> int:
    from piper import PiperVoice
    model = PIPER_DIR / f"{voice}.onnx"
    if not model.exists():
        log(f"piper voice not found: {model}")
        return 2
    text = Path(text_file).read_text(encoding="utf-8")
    if not text.strip():
        log("empty text")
        return 3
    pv = PiperVoice.load(str(model))
    with wave.open(out_wav, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(pv.config.sample_rate)
        pv.synthesize(text, wf)
    return 0


def cmd_rvc(in_wav: str, pth: str, index: str, pitch: str,
            index_rate: str, extractor: str, out_wav: str) -> int:
    pth_path = Path(pth)
    if not pth_path.exists():
        log(f"RVC .pth not found: {pth}")
        return 2

    # Applio's run_infer_script expects to be run with Applio_src as cwd
    # (it resolves rmvpe/contentvec paths relative to cwd).
    os.chdir(str(APPLIO_SRC))

    # Device probe — surfaces whether RVC will run on GPU or CPU.
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            log(f"DEVICE: cuda:0 ({name}, {mem:.1f} GB)")
        else:
            log("DEVICE: cpu  (torch.cuda.is_available() == False)")
    except Exception as e:
        log(f"DEVICE: probe failed: {e}")

    # Probe Applio's own Config — what device the inference pipeline will use.
    try:
        from rvc.configs.config import Config
        cfg = Config()
        log(f"APPLIO_DEVICE: {cfg.device}  gpu_name={cfg.gpu_name}  vram={cfg.gpu_mem}GB")
    except Exception as e:
        log(f"APPLIO_DEVICE: probe failed: {e}")

    from core import run_infer_script
    run_infer_script(
        pitch=int(pitch),
        index_rate=float(index_rate),
        volume_envelope=1.0,
        protect=0.5,
        f0_method=extractor,
        input_path=str(Path(in_wav).resolve()),
        output_path=str(Path(out_wav).resolve()),
        pth_path=str(pth_path.resolve()),
        index_path=str(Path(index).resolve()) if index else "",
        split_audio=False,
        f0_autotune=False,
        f0_autotune_strength=1.0,
        proposed_pitch=False,
        proposed_pitch_threshold=155.0,
        clean_audio=False,
        clean_strength=0.7,
        export_format="WAV",
        embedder_model="contentvec",
    )
    return 0


def main(argv):
    if len(argv) < 2:
        log("usage: applio_worker.py {piper|rvc} ...")
        return 1
    op = argv[1]
    try:
        if op == "piper":
            _, _, voice, text_file, out_wav = argv
            return cmd_piper(voice, text_file, out_wav)
        if op == "rvc":
            _, _, in_wav, pth, index, pitch, index_rate, extractor, out_wav = argv
            return cmd_rvc(in_wav, pth, index, pitch, index_rate, extractor, out_wav)
        log(f"unknown op: {op}")
        return 1
    except Exception as e:
        import traceback
        log(f"{type(e).__name__}: {e}")
        log(traceback.format_exc())
        return 10


if __name__ == "__main__":
    sys.exit(main(sys.argv))
