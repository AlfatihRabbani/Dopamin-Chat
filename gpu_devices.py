"""
Cross-vendor compute device detection (Blender-style backend list).

Each backend reports: name, kind, available, devices, notes.
Used by web UI Settings → GPU tab.
"""
import os
import shutil
import subprocess
from pathlib import Path


def _safe(cmd: list[str], timeout: int = 3) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout).stdout
    except Exception:
        return ""


def _has(bin_name: str) -> bool:
    return shutil.which(bin_name) is not None


# ---------------------------------------------------------------------------
# Backend detectors
# ---------------------------------------------------------------------------

def detect_cuda() -> dict:
    available = False
    devices = []
    note = ""
    try:
        import torch  # noqa
        torch_ver = getattr(torch, "__version__", "?")
        torch_cuda_build = getattr(torch.version, "cuda", None)
        is_hip = bool(getattr(torch.version, "hip", None))
        if torch.cuda.is_available() and not is_hip:
            available = True
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                devices.append({
                    "index": i, "name": p.name,
                    "memory_gib": round(p.total_memory / (1024 ** 3), 1),
                })
        elif _has("nvidia-smi") and not torch_cuda_build:
            note = (f"nvidia-smi present but torch {torch_ver} is the CPU build "
                    f"(torch.version.cuda=None). Reinstall:  "
                    "pip install --force-reinstall torch --index-url "
                    "https://download.pytorch.org/whl/cu124")
        elif _has("nvidia-smi") and torch_cuda_build:
            note = (f"nvidia-smi present, torch built for CUDA {torch_cuda_build}, "
                    "but torch.cuda.is_available()==False. Driver/runtime "
                    "mismatch — check  nvidia-smi  shows CUDA Version: ≥"
                    f"{torch_cuda_build}. Reboot if drivers were just installed.")
        elif not _has("nvidia-smi"):
            note = ("no NVIDIA driver detected. Install with: "
                    "sudo bash /home/wolgm/aitest/install-nvidia.sh  then reboot.")
        else:
            note = "no NVIDIA driver detected"
    except ImportError:
        note = "torch not installed. Run dopamine_chat/install.sh"
    return {"name": "CUDA", "kind": "cuda", "available": available,
            "devices": devices, "note": note}


def detect_optix() -> dict:
    """OptiX is NVIDIA raytracing (Blender). Not used for LLM inference. We
    include it for parity but mark inference_supported=False."""
    cu = detect_cuda()
    return {
        "name": "OptiX", "kind": "optix",
        "available": cu["available"],
        "devices": cu["devices"],
        "note": "OptiX is for raytracing only — not used by LLM inference. "
                "Selecting it falls back to CUDA.",
        "inference_supported": False,
    }


def detect_rocm() -> dict:
    available = False
    devices = []
    note = ""
    try:
        import torch  # noqa
        is_hip = bool(getattr(torch.version, "hip", None))
        if is_hip and torch.cuda.is_available():
            available = True
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                devices.append({
                    "index": i, "name": p.name,
                    "memory_gib": round(p.total_memory / (1024 ** 3), 1),
                })
        elif _has("rocm-smi") or Path("/opt/rocm").exists():
            note = "ROCm present but torch was not built with HIP. Reinstall torch with rocm wheel."
        else:
            note = "no AMD/ROCm runtime detected"
    except ImportError:
        note = "torch not installed"
    return {"name": "HIP (ROCm)", "kind": "rocm", "available": available,
            "devices": devices, "note": note}


def detect_intel() -> dict:
    available = False
    devices = []
    note = ""
    # Intel Extension for PyTorch (IPEX) provides XPU
    try:
        import intel_extension_for_pytorch as ipex  # noqa
        import torch  # noqa
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            available = True
            for i in range(torch.xpu.device_count()):
                devices.append({
                    "index": i,
                    "name": getattr(torch.xpu, "get_device_name", lambda x: f"xpu:{x}")(i),
                })
    except ImportError:
        if _has("sycl-ls") or Path("/opt/intel/oneapi").exists():
            note = "Intel oneAPI present — install intel-extension-for-pytorch for inference"
        else:
            note = "no Intel oneAPI / Arc GPU detected"
    return {"name": "Intel oneAPI", "kind": "intel", "available": available,
            "devices": devices, "note": note}


def detect_metal() -> dict:
    available = False
    note = ""
    try:
        import torch  # noqa
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            available = True
        else:
            note = "no Apple Metal device"
    except ImportError:
        note = "torch not installed"
    return {"name": "Metal", "kind": "metal", "available": available,
            "devices": [{"index": 0, "name": "Apple Metal"}] if available else [],
            "note": note}


def detect_vulkan() -> dict:
    """llama-cpp-python can be built with Vulkan; not directly via torch."""
    available = _has("vulkaninfo")
    return {"name": "Vulkan", "kind": "vulkan", "available": available,
            "devices": [], "note": "GGUF only; requires Vulkan-built llama-cpp-python"}


def detect_cpu() -> dict:
    cpu_name = ""
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu_name = line.split(":", 1)[1].strip()
                    break
    except Exception:
        pass
    if not cpu_name:
        cpu_name = "CPU"
    try:
        import psutil
        cores = psutil.cpu_count(logical=False) or psutil.cpu_count()
    except Exception:
        cores = os.cpu_count() or 0
    return {"name": "CPU", "kind": "cpu", "available": True,
            "devices": [{"index": 0, "name": cpu_name, "cores": cores}],
            "note": "always available; slowest path"}


# ---------------------------------------------------------------------------
# Public
# ---------------------------------------------------------------------------

def list_backends() -> list[dict]:
    return [
        detect_cuda(),
        detect_optix(),
        detect_rocm(),
        detect_intel(),
        detect_metal(),
        detect_vulkan(),
        detect_cpu(),
    ]


def apply_selection(selected_kinds: list[str], device_index: int = 0) -> dict:
    """Legacy multi-kind apply. Kept for back-compat."""
    eff = {"selected": list(selected_kinds), "device_index": device_index}
    if "cuda" in selected_kinds or "optix" in selected_kinds:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_index)
        eff["CUDA_VISIBLE_DEVICES"] = str(device_index)
    if "rocm" in selected_kinds:
        os.environ["HIP_VISIBLE_DEVICES"] = str(device_index)
        eff["HIP_VISIBLE_DEVICES"] = str(device_index)
    if selected_kinds == ["cpu"]:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        eff["CUDA_VISIBLE_DEVICES"] = ""
    return eff


def apply_blender_style(backend: str, enabled_devices: list[str]) -> dict:
    """Apply user's Blender-style selection.

    IMPORTANT: only sets CUDA/HIP_VISIBLE_DEVICES when at least one GPU index
    is explicitly selected. Never blanks the env (blanking after torch is
    imported has no effect on the running process, and blanking before would
    hide the card from later restarts of subprocesses).
    """
    eff = {"backend": backend, "enabled_devices": list(enabled_devices)}
    # Only collect indices whose kind matches the selected backend; dedupe.
    # cuda and optix alias the same physical NVIDIA card — never emit both.
    allowed = {
        "cuda":  ("cuda", "optix"),
        "optix": ("cuda", "optix"),
        "rocm":  ("rocm",),
        "intel": ("intel",),
        "metal": ("metal",),
    }.get(backend, ())
    seen = set()
    gpu_indices = []
    for d in enabled_devices:
        if ":" not in d:
            continue
        kind, idx = d.split(":", 1)
        if kind in allowed and idx.isdigit() and idx not in seen:
            seen.add(idx)
            gpu_indices.append(int(idx))

    if backend in ("cuda", "optix") and gpu_indices:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in gpu_indices)
        eff["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]
    elif backend == "rocm" and gpu_indices:
        os.environ["HIP_VISIBLE_DEVICES"] = ",".join(str(i) for i in gpu_indices)
        eff["HIP_VISIBLE_DEVICES"] = os.environ["HIP_VISIBLE_DEVICES"]

    if backend == "optix":
        eff["note"] = "OptiX is raytracing-only — runs as CUDA for LLM inference."
    if backend == "none":
        eff["note"] = "No GPU explicitly selected. Detection runs with default env."
    if (backend in ("cuda", "optix", "rocm")) and not gpu_indices:
        eff["warning"] = ("backend chosen but no device ticked. Tick a GPU in "
                          "the Settings → GPU device list and Apply.")
    return eff
