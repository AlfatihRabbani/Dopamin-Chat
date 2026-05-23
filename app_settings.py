"""
Persistent global settings for the Dopamine Chat web UI.

Stored at:  dopamine_chat/settings.json
Loaded once per process; mutated via /api/settings endpoint.
"""
import json
from pathlib import Path

SETTINGS_PATH = Path(__file__).resolve().parent / "settings.json"
USER_PFP_PATH = Path(__file__).resolve().parent / "user_pfp"
USER_PFP_PATH.mkdir(exist_ok=True)

DEFAULTS = {
    "theme": "dark",             # dark | light | ash | onyx
    "user_name": "",
    "user_description": "",
    "user_pfp": "",              # filename inside user_pfp/
    "tts_engine": "piper",       # piper | edge | none
    "tts_enabled": False,
    "tts_voice": "",             # piper voice .onnx basename
    "rvc_enabled": False,
    "rvc_pth": "",
    "rvc_index": "",
    "rvc_pitch": 0,              # semitones
    "rvc_index_rate": 0.75,
    "rvc_pitch_extractor": "rmvpe",  # rmvpe | crepe | pm
    "auto_rename_chats": True,
    "load_full_model": True,     # informational; backend still tier-falls-back
    "gpu_backends": ["cuda", "cpu"],          # legacy, kept for back-compat
    "gpu_device_index": 0,                    # legacy
    "gpu_compute_backend": "none",            # none|cuda|optix|rocm|intel|metal|vulkan
    "gpu_devices_enabled": [],                # e.g. ["cuda:0", "cpu"]
    # Image generation
    "imggen_mode": "local",                   # local | server
    "imggen_kind": "diffusers",               # diffusers|safetensors|transformers|gguf|lora (UI sub-tab)
    "imggen_backend": "none",                 # local: 'local'  |  server: 'comfyui' | 'sd_cpp'
    "imggen_model_path": "",
    "imggen_negative": "",
    "imggen_width": 512,
    "imggen_height": 512,
    "imggen_steps": 20,
    "imggen_strength": 0.75,                  # img2img only
    "imggen_offload_mode": "auto",            # auto|off|model|sequential
    "imggen_loras": [],                       # [{path, scale}, ...]
    "imggen_comfy_url": "http://127.0.0.1:8188",
    "imggen_comfy_model": "",
}


def load() -> dict:
    try:
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}
    merged = {**DEFAULTS, **data}
    return merged


def save(data: dict) -> dict:
    cur = load()
    cur.update({k: v for k, v in data.items() if k in DEFAULTS})
    with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(cur, f, indent=2, ensure_ascii=False)
    return cur
