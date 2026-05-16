"""
Image generation backends.

Lazy-imported. Picks one of:
  - diffusers   : local Stable Diffusion / SDXL / Flux via HF diffusers library
  - comfyui     : forward prompt to a running ComfyUI HTTP API (default localhost:8188)
  - sd_cpp_cli  : shell out to `sd` binary from leejet/stable-diffusion.cpp
  - none        : error
"""
import os
import io
import json
import time
import uuid
import base64
import urllib.request
import urllib.parse
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parent
IMG_OUT = ROOT / "generated_images"
IMG_OUT.mkdir(exist_ok=True)
SD_MODELS = ROOT / "ImageGen_Models"
SD_MODELS.mkdir(exist_ok=True)
LEGACY_SD_MODELS = ROOT / "sd_models"
LORAS_DIR = ROOT / "ImageGen_LoRAs"
LORAS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Public availability checks
# ---------------------------------------------------------------------------

def diffusers_available() -> tuple[bool, str]:
    try:
        import diffusers  # noqa: F401
        import torch     # noqa: F401
        return True, ""
    except ImportError:
        return False, "diffusers/torch not installed. pip install diffusers transformers accelerate"


def comfyui_available(url: str = "http://127.0.0.1:8188") -> tuple[bool, str]:
    try:
        with urllib.request.urlopen(f"{url}/system_stats", timeout=1.5) as r:
            if r.status == 200:
                return True, ""
    except Exception as e:
        return False, f"ComfyUI not reachable at {url}: {type(e).__name__}"
    return False, "ComfyUI server not responding"


def sd_cpp_cli_available() -> tuple[bool, str]:
    import shutil
    p = shutil.which("sd") or shutil.which("stable-diffusion")
    if p:
        return True, p
    return False, "stable-diffusion.cpp CLI not on PATH"


def list_diffusers_models() -> list[dict]:
    """Walk ImageGen_Models/ (and legacy sd_models/) for HF dirs, .safetensors, .gguf.

    Format mapping:
      HF dir (model_index.json present)  → diffusers AutoPipeline
      .safetensors                       → diffusers single-file loader
      .gguf                              → routed to stable-diffusion.cpp CLI
    """
    out = []
    seen = set()
    for root in (SD_MODELS, LEGACY_SD_MODELS):
        if not root.exists():
            continue
        for entry in sorted(root.iterdir()):
            key = entry.name
            if key in seen:
                continue
            seen.add(key)
            if entry.is_dir():
                if (entry / "model_index.json").exists():
                    out.append({"name": entry.name, "kind": "hf-diffusers",
                                "path": str(entry)})
                elif (entry / "config.json").exists():
                    # transformers-style (e.g. HiDream-O1, Bagel, Janus). Not
                    # loadable via diffusers AutoPipeline.
                    out.append({
                        "name": entry.name,
                        "kind": "hf-transformers-NOT-DIFFUSERS",
                        "path": str(entry),
                        "warning": ("This is a transformers multimodal LLM, "
                                    "not a diffusion pipeline. Diffusers cannot "
                                    "load it. Use ComfyUI with HiDream custom "
                                    "nodes, OR fetch a real diffusers release "
                                    "(e.g. HiDream-ai/HiDream-I1-Full)."),
                    })
            elif entry.is_file() and entry.suffix == ".safetensors":
                out.append({"name": entry.stem, "kind": "safetensors", "path": str(entry)})
            elif entry.is_file() and entry.suffix == ".gguf":
                is_text = _looks_like_text_llm(str(entry))
                out.append({
                    "name": entry.stem,
                    "kind": "gguf-text-llm-WRONG-FOLDER" if is_text else "gguf",
                    "path": str(entry),
                    "warning": "Text LLM in image-gen folder — move to dopamine_chat/models/" if is_text else "",
                })
    return out


def _detect_kind(model_path: str) -> str:
    p = Path(model_path)
    if p.is_dir():
        return "hf-dir"
    if p.suffix == ".safetensors":
        return "safetensors"
    if p.suffix == ".gguf":
        return "gguf"
    return "unknown"


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

def _slugify_prompt(prompt: str, max_len: int = 40) -> str:
    """Build a filesystem-safe slug from a generation prompt.
    'A simple tree at sunset' → 'a_simple_tree_at_sunset'."""
    if not prompt:
        return "img"
    s = prompt.strip().lower()
    # Drop the bracketed [Scene context …] suffix we add for RP awareness
    cut = s.find("\n[scene context")
    if cut > 0:
        s = s[:cut]
    out = "".join(c if (c.isalnum() or c in " _-") else " " for c in s)
    out = "_".join(out.split())[:max_len].strip("_")
    return out or "img"


def _save_png(img_bytes_or_pil, prefix: str = "img",
              prompt: str | None = None, out_dir=None) -> str:
    """Save PNG and return its path. When `prompt` is given the filename
    is derived from a slug of the prompt instead of a random hash, so
    images land as `a_simple_tree.png`, `the_daisy_flowers.png`, etc.
    When `out_dir` is given, the image is saved there (used to land the
    image inside the active chat folder)."""
    base_dir = Path(out_dir) if out_dir else IMG_OUT
    base_dir.mkdir(parents=True, exist_ok=True)
    if prompt:
        slug = _slugify_prompt(prompt)
        # Disambiguate collisions with a short suffix.
        candidate = base_dir / f"{slug}.png"
        i = 2
        while candidate.exists():
            candidate = base_dir / f"{slug}_{i}.png"
            i += 1
        p = candidate
    else:
        ts = time.strftime("%Y%m%d-%H%M%S")
        p = base_dir / f"{prefix}_{ts}_{uuid.uuid4().hex[:6]}.png"
    if isinstance(img_bytes_or_pil, (bytes, bytearray)):
        p.write_bytes(img_bytes_or_pil)
    else:
        img_bytes_or_pil.save(str(p), "PNG")
    return str(p)


# In-process pipeline cache so we don't reload on every call.
_PIPE_CACHE: dict[tuple, object] = {}


def _free_vram_gib() -> float:
    try:
        import torch
        if not torch.cuda.is_available():
            return 0.0
        free_b, _ = torch.cuda.mem_get_info()
        return free_b / (1024 ** 3)
    except Exception:
        return 0.0


# Approximate VRAM needs at fp16 for full GPU residency. Below this, we
# auto-fall to CPU offload. Above the heavy threshold, we force sequential
# offload to survive 8GB cards.
_VRAM_HEAVY_GIB = 12.0     # below this → enable_model_cpu_offload
_VRAM_TINY_GIB  = 4.0      # below this → enable_sequential_cpu_offload


def _apply_offload(pipe, offload_mode: str, dtype, device: str):
    """offload_mode ∈ {'auto','off','model','sequential'}."""
    import torch
    if device != "cuda":
        # CPU path: nothing to offload to.
        return pipe.to("cpu")

    mode = (offload_mode or "auto").lower()
    if mode == "auto":
        free = _free_vram_gib()
        if free < _VRAM_TINY_GIB:
            mode = "sequential"
        elif free < _VRAM_HEAVY_GIB:
            mode = "model"
        else:
            mode = "off"

    if mode == "off":
        return pipe.to(device)
    if mode == "model" and hasattr(pipe, "enable_model_cpu_offload"):
        pipe.enable_model_cpu_offload()
        return pipe
    if mode == "sequential" and hasattr(pipe, "enable_sequential_cpu_offload"):
        pipe.enable_sequential_cpu_offload()
        return pipe
    # Fallback if pipeline doesn't support requested mode.
    return pipe.to(device)


def _load_pipeline(model_path: str, img2img: bool, dtype, device,
                   offload_mode: str = "auto"):
    """Load (or fetch cached) a diffusers pipeline. Picks Image2Image when
    img2img=True. Applies CPU offload per `offload_mode`."""
    kind = _detect_kind(model_path)
    key = (model_path, "img2img" if img2img else "text2img", offload_mode)
    if key in _PIPE_CACHE:
        print(f"[imggen] cached pipeline reused: {Path(model_path).name} ({kind}, "
              f"{'img2img' if img2img else 'text2img'})", flush=True)
        return _PIPE_CACHE[key]

    print(f"[imggen] loading pipeline: {model_path}", flush=True)
    print(f"[imggen]   kind={kind} mode={'img2img' if img2img else 'text2img'} "
          f"dtype={dtype} device={device} offload={offload_mode}", flush=True)
    t0 = time.time()

    from diffusers import (
        AutoPipelineForText2Image, AutoPipelineForImage2Image,
        StableDiffusionPipeline, StableDiffusionImg2ImgPipeline,
    )

    AutoCls = AutoPipelineForImage2Image if img2img else AutoPipelineForText2Image
    SDCls   = StableDiffusionImg2ImgPipeline if img2img else StableDiffusionPipeline

    try:
        if kind == "hf-dir":
            pipe = AutoCls.from_pretrained(model_path, torch_dtype=dtype)
        elif kind == "safetensors":
            pipe = SDCls.from_single_file(model_path, torch_dtype=dtype)
        else:
            raise RuntimeError(f"diffusers cannot load this format: {kind} "
                               "(use ComfyUI or sd.cpp for GGUF)")
    except Exception as e:
        print(f"[imggen] LOAD FAILED: {type(e).__name__}: {e}", flush=True)
        raise RuntimeError(f"pipeline load failed: {type(e).__name__}: {e}")

    pipe = _apply_offload(pipe, offload_mode, dtype, device)
    print(f"[imggen] pipeline ready in {time.time()-t0:.1f}s", flush=True)

    # Memory-saving extras when offloading
    if offload_mode in ("auto", "model", "sequential"):
        for fn_name in ("enable_attention_slicing", "enable_vae_slicing",
                        "enable_vae_tiling"):
            fn = getattr(pipe, fn_name, None)
            if callable(fn):
                try: fn()
                except Exception: pass

    _PIPE_CACHE[key] = pipe
    return pipe


def _load_init_image(src):
    """Accept a filesystem path or http(s) URL; return a PIL Image."""
    from PIL import Image
    if not src:
        return None
    s = str(src)
    if s.startswith(("http://", "https://")):
        import urllib.request as ur
        with ur.urlopen(s, timeout=15) as r:
            return Image.open(io.BytesIO(r.read())).convert("RGB")
    return Image.open(s).convert("RGB")


def gen_local(prompt: str,
              model_path: str,
              negative: str = "",
              width: int = 512, height: int = 512,
              steps: int = 20, seed: int | None = None,
              guidance: float = 7.5,
              init_image: str | None = None,
              strength: float = 0.75,
              offload_mode: str = "auto",
              loras: list[dict] | None = None,
              out_dir: str | None = None) -> str:
    """Local diffusers generation.

    text-to-image       : init_image=None
    image-text-to-image : init_image is a file path or URL (strength 0..1)
    """
    kind = _detect_kind(model_path)
    if kind == "gguf":
        if _looks_like_text_llm(model_path):
            raise RuntimeError(
                f"'{Path(model_path).name}' looks like a TEXT LLM, not an image "
                "model. Move it to dopamine_chat/models/ and pick it under "
                "Settings → Models (chat). ImageGen_Models/ is for diffusion "
                "weights only (SD/SDXL/Flux/HiDream)."
            )
        # Image GGUFs go through sd.cpp.
        return gen_sd_cpp_cli(prompt, model_path, negative, width, height, steps, seed)

    # transformers multimodal LLM (HiDream-O1, Bagel, Janus): try the
    # transformers AutoModel + trust_remote_code path.
    p = Path(model_path)
    if p.is_dir() and not (p / "model_index.json").exists() and (p / "config.json").exists():
        return gen_transformers_mllm(
            prompt, str(p),
            width=width, height=height, steps=steps,
            seed=seed, negative=negative, guidance=guidance,
            init_image=init_image, offload_mode=offload_mode,
        )

    ok, msg = diffusers_available()
    if not ok:
        raise RuntimeError(msg)
    if not Path(model_path).exists():
        raise RuntimeError(f"model not found: {model_path}")

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = _load_pipeline(model_path, img2img=bool(init_image), dtype=dtype,
                          device=device, offload_mode=offload_mode)

    # LoRA application — load + activate; unload after generation to keep
    # pipeline cache clean for next call.
    _unload_loras(pipe)
    applied_loras = _apply_loras(pipe, loras or [])

    gen = None
    if seed is not None:
        gen = torch.Generator(device=device).manual_seed(int(seed))

    kwargs = dict(
        prompt=prompt, negative_prompt=negative or None,
        num_inference_steps=steps, guidance_scale=guidance,
        generator=gen,
    )
    if init_image:
        img = _load_init_image(init_image)
        if img is None:
            raise RuntimeError(f"init_image not loadable: {init_image}")
        # Most img2img pipes take `image` + `strength`, no explicit width/height.
        kwargs["image"] = img.resize((width, height))
        kwargs["strength"] = float(strength)
    else:
        kwargs["width"] = width
        kwargs["height"] = height

    try:
        result = pipe(**kwargs)
    finally:
        _unload_loras(pipe)
    return _save_png(result.images[0], prompt=prompt, out_dir=out_dir)


def gen_transformers_mllm(prompt: str, model_dir: str,
                          width: int = 1024, height: int = 1024,
                          steps: int = 28, seed: int | None = None,
                          negative: str = "", guidance: float = 5.0,
                          init_image: str | None = None,
                          offload_mode: str = "auto",
                          out_dir: str | None = None) -> str:
    """Loader for transformers-style multimodal image LLMs (HiDream-O1 etc).

    Requires the repo's *.py files (trust_remote_code=True). Calls one of the
    plausible image-generation entry points; raises a clear error if none of
    them exist on the loaded model.
    """
    try:
        import torch
        from transformers import AutoModel, AutoProcessor, AutoTokenizer
    except ImportError:
        raise RuntimeError("transformers not installed. pip install transformers")

    # Need at least one *.py file in the model dir for trust_remote_code.
    if not any(Path(model_dir).glob("*.py")):
        raise RuntimeError(
            f"{Path(model_dir).name} has no *.py custom code. Re-download with "
            "the HF download button (now includes *.py), or pull via:\n"
            "  huggingface-cli download <repo_id> --local-dir " + model_dir
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    try:
        model = AutoModel.from_pretrained(
            model_dir, trust_remote_code=True, torch_dtype=dtype,
        )
    except Exception as e:
        raise RuntimeError(f"transformers AutoModel load failed: {e}")

    try:
        processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    except Exception:
        processor = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    except Exception:
        tokenizer = None

    # VRAM offload — transformers has its own
    if device == "cuda":
        if offload_mode in ("auto", "model", "sequential"):
            try:
                if hasattr(model, "enable_model_cpu_offload"):
                    model.enable_model_cpu_offload()
                else:
                    model = model.to(device)
            except Exception:
                model = model.to(device)
        else:
            model = model.to(device)
    else:
        model = model.to("cpu")

    if seed is not None:
        torch.manual_seed(int(seed))

    # Try the common entry-point names in order. Custom MLLM repos expose
    # something different per project, so we attempt several.
    img = None
    last_err = None
    candidates = ("generate_image", "image_generate", "txt2img",
                  "generate_images", "synthesize", "sample")
    for fn_name in candidates:
        fn = getattr(model, fn_name, None)
        if not callable(fn):
            continue
        try:
            # Most accept (prompt, height, width, num_inference_steps, guidance_scale)
            kw = dict(height=height, width=width,
                      num_inference_steps=steps, guidance_scale=guidance)
            if negative:
                kw["negative_prompt"] = negative
            out = fn(prompt, **kw)
            # Normalize to PIL
            from PIL import Image as PILImage
            if isinstance(out, list):
                out = out[0]
            if hasattr(out, "save"):
                img = out
            elif isinstance(out, dict) and out.get("images"):
                img = out["images"][0]
            else:
                # Tensor → PIL
                import numpy as np
                arr = out
                if hasattr(arr, "cpu"): arr = arr.cpu().numpy()
                if arr.ndim == 4: arr = arr[0]
                if arr.dtype != "uint8":
                    arr = (arr.clip(0, 1) * 255).astype("uint8") if arr.max() <= 1.0 else arr.astype("uint8")
                if arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
                    arr = arr.transpose(1, 2, 0)
                img = PILImage.fromarray(arr)
            break
        except Exception as e:
            last_err = f"{fn_name}: {type(e).__name__}: {e}"
            continue

    if img is None:
        # Last resort: try transformers.pipeline("image-generation")
        try:
            from transformers import pipeline
            pipe = pipeline("image-generation", model=model_dir,
                            trust_remote_code=True, torch_dtype=dtype)
            out = pipe(prompt, height=height, width=width,
                       num_inference_steps=steps, guidance_scale=guidance)
            if isinstance(out, list): out = out[0]
            img = out.get("image") if isinstance(out, dict) else out
        except Exception as e:
            raise RuntimeError(
                f"could not generate via transformers MLLM. Last error: {last_err or e}. "
                "This model likely needs its own inference script — clone the repo and "
                "follow its README, or use ComfyUI with the matching custom nodes."
            )
    return _save_png(img, prompt=prompt, out_dir=out_dir)


def gen_comfyui(prompt: str,
                negative: str = "",
                width: int = 512, height: int = 512,
                steps: int = 20, seed: int | None = None,
                model: str = "",
                url: str = "http://127.0.0.1:8188") -> str:
    """Submit a minimal text2image workflow to ComfyUI and poll until done."""
    ok, msg = comfyui_available(url)
    if not ok:
        raise RuntimeError(msg)
    cid = uuid.uuid4().hex
    if seed is None:
        seed = int(time.time())

    # Minimal SDXL-or-SD1.5 workflow (CheckpointLoaderSimple → KSampler → VAEDecode → SaveImage)
    wf = {
        "3": {"class_type": "KSampler",
              "inputs": {"seed": int(seed), "steps": int(steps),
                         "cfg": 7.0, "sampler_name": "euler",
                         "scheduler": "normal", "denoise": 1.0,
                         "model": ["4", 0], "positive": ["6", 0],
                         "negative": ["7", 0], "latent_image": ["5", 0]}},
        "4": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": model or "v1-5-pruned-emaonly.safetensors"}},
        "5": {"class_type": "EmptyLatentImage",
              "inputs": {"width": int(width), "height": int(height), "batch_size": 1}},
        "6": {"class_type": "CLIPTextEncode",
              "inputs": {"text": prompt, "clip": ["4", 1]}},
        "7": {"class_type": "CLIPTextEncode",
              "inputs": {"text": negative, "clip": ["4", 1]}},
        "8": {"class_type": "VAEDecode",
              "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
        "9": {"class_type": "SaveImage",
              "inputs": {"filename_prefix": "dopamine", "images": ["8", 0]}},
    }
    body = json.dumps({"prompt": wf, "client_id": cid}).encode("utf-8")
    req = urllib.request.Request(f"{url}/prompt", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            resp = json.loads(r.read().decode())
    except Exception as e:
        raise RuntimeError(f"comfyui submit failed: {e}")
    pid = resp.get("prompt_id")
    if not pid:
        raise RuntimeError(f"comfyui returned no prompt_id: {resp}")

    # Poll history
    deadline = time.time() + 180  # 3 min
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/history/{pid}", timeout=5) as r:
                hist = json.loads(r.read().decode())
        except Exception:
            time.sleep(1); continue
        if pid in hist and "outputs" in hist[pid]:
            outputs = hist[pid]["outputs"]
            for node_out in outputs.values():
                for img in node_out.get("images", []):
                    qp = urllib.parse.urlencode({
                        "filename": img["filename"],
                        "subfolder": img.get("subfolder", ""),
                        "type": img.get("type", "output"),
                    })
                    with urllib.request.urlopen(f"{url}/view?{qp}", timeout=15) as r:
                        return _save_png(r.read(), prefix="comfy")
            raise RuntimeError("comfyui finished but no images returned")
        time.sleep(1)
    raise RuntimeError("comfyui timeout (>3 min)")


def list_loras() -> list[dict]:
    """Walk ImageGen_LoRAs/ for .safetensors LoRA files."""
    out = []
    if not LORAS_DIR.exists():
        return out
    for entry in sorted(LORAS_DIR.rglob("*.safetensors")):
        out.append({"name": entry.stem, "path": str(entry)})
    return out


def _apply_loras(pipe, loras: list[dict]) -> list[dict]:
    """`loras` = [{path,scale}, ...]. Returns the applied list (with name).
    Diffusers needs unique adapter names; we strip + slugify the filename."""
    applied = []
    if not loras:
        return applied
    if not hasattr(pipe, "load_lora_weights"):
        raise RuntimeError("this pipeline does not support LoRAs")
    names = []
    scales = []
    for i, item in enumerate(loras):
        path = item.get("path") or ""
        scale = float(item.get("scale", 1.0))
        if not path:
            continue
        p = Path(path)
        if not p.exists():
            raise RuntimeError(f"lora not found: {path}")
        slug = "".join(c if c.isalnum() else "_" for c in p.stem)[:40] or f"lora{i}"
        print(f"[imggen] applying LoRA: {p.name} @ scale={scale}", flush=True)
        try:
            pipe.load_lora_weights(str(p.parent), weight_name=p.name, adapter_name=slug)
        except TypeError:
            # older diffusers: no adapter_name kwarg
            pipe.load_lora_weights(str(p))
        names.append(slug)
        scales.append(scale)
        applied.append({"name": slug, "path": path, "scale": scale})
    if names and hasattr(pipe, "set_adapters"):
        try:
            pipe.set_adapters(names, adapter_weights=scales)
        except Exception as e:
            print(f"[imggen]   set_adapters failed: {e}", flush=True)
    return applied


def preload(settings: dict) -> dict:
    """Eagerly load the configured image-gen pipeline + LoRAs into memory
    and report status. Called by /api/imggen/load (the "Load" button)."""
    mode = (settings.get("imggen_mode") or "local").lower()
    backend = (settings.get("imggen_backend") or "none").lower()
    model_path = settings.get("imggen_model_path") or ""
    loras = settings.get("imggen_loras") or []
    offload = settings.get("imggen_offload_mode", "auto")

    info = {"mode": mode, "backend": backend, "model_path": model_path,
            "loras": loras, "loaded": False}

    print(f"[imggen] === LOAD === mode={mode} backend={backend}", flush=True)
    print(f"[imggen]   model: {model_path or '(none)'}", flush=True)
    if loras:
        for l in loras:
            print(f"[imggen]   lora : {l.get('path')} @ {l.get('scale')}", flush=True)

    if mode != "local":
        info["note"] = "server mode — nothing to preload locally."
        print("[imggen] server mode — skip local preload.", flush=True)
        return info
    if not model_path:
        info["error"] = ("no base model selected. Pick one in Diffusers/SafeTensors/"
                         "Transformers/GGUF sub-tab first.")
        print(f"[imggen] LOAD ABORT: {info['error']}", flush=True)
        return info

    kind = _detect_kind(model_path)
    p = Path(model_path)

    # transformers MLLM (HiDream-O1 etc.) — load via AutoModel
    if p.is_dir() and not (p / "model_index.json").exists() and (p / "config.json").exists():
        try:
            import torch
            from transformers import AutoModel
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.bfloat16 if device == "cuda" else torch.float32
            print(f"[imggen]   loading transformers AutoModel ({device}, {dtype})…", flush=True)
            t0 = time.time()
            _PIPE_CACHE[(model_path, "mllm", offload)] = AutoModel.from_pretrained(
                model_path, trust_remote_code=True, torch_dtype=dtype)
            print(f"[imggen]   ready in {time.time()-t0:.1f}s", flush=True)
            info["loaded"] = True
            info["device"] = device
        except Exception as e:
            info["error"] = f"transformers load failed: {type(e).__name__}: {e}"
            print(f"[imggen] LOAD FAIL: {info['error']}", flush=True)
        return info

    if kind == "gguf":
        info["note"] = "GGUF runs via stable-diffusion.cpp CLI — no preload."
        print("[imggen] GGUF backend — no preload needed.", flush=True)
        return info

    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        pipe = _load_pipeline(model_path, img2img=False, dtype=dtype,
                              device=device, offload_mode=offload)
        if loras:
            _unload_loras(pipe)
            applied = _apply_loras(pipe, loras)
            info["loras_applied"] = applied
        info["loaded"] = True
        info["device"] = device
    except Exception as e:
        info["error"] = f"{type(e).__name__}: {e}"
        print(f"[imggen] LOAD FAIL: {info['error']}", flush=True)
    return info


def _unload_loras(pipe):
    for fn in ("unload_lora_weights", "disable_lora"):
        f = getattr(pipe, fn, None)
        if callable(f):
            try: f()
            except Exception: pass


_TEXT_LLM_HINTS = ("qwen", "llama", "gemma", "mistral", "phi", "claude",
                   "reasoning", "distill", "instruct", "chat", "opus", "sonnet",
                   "haiku", "deepseek", "yi-", "command-r", "starcoder")


def _looks_like_text_llm(model_path: str) -> bool:
    """Cheap filename heuristic — text reasoning LLMs vs diffusion GGUFs."""
    name = Path(model_path).stem.lower()
    return any(h in name for h in _TEXT_LLM_HINTS)


def gen_sd_cpp_cli(prompt: str,
                   model_path: str,
                   negative: str = "",
                   width: int = 512, height: int = 512,
                   steps: int = 20, seed: int | None = None) -> str:
    if _looks_like_text_llm(model_path):
        raise RuntimeError(
            f"'{Path(model_path).name}' looks like a TEXT LLM (qwen/llama/etc), "
            "not an image diffusion model. Text LLMs cannot generate images. "
            "Move it to dopamine_chat/models/ and pick it as your chat model. "
            "For image GGUFs, use community quants of SD/SDXL/Flux/HiDream "
            "and place them in ImageGen_Models/."
        )
    import subprocess, shutil
    sd = shutil.which("sd") or shutil.which("stable-diffusion")
    if not sd:
        raise RuntimeError(
            "stable-diffusion.cpp CLI not on PATH. To use .gguf image models, "
            "build leejet/stable-diffusion.cpp:\n"
            "  git clone --recursive https://github.com/leejet/stable-diffusion.cpp\n"
            "  cd stable-diffusion.cpp && cmake -B build -DSD_CUDA=ON && cmake --build build -j\n"
            "  sudo install build/bin/sd /usr/local/bin/\n"
            "Or use the Local-model path with a .safetensors checkpoint (diffusers handles those)."
        )
    out_path = IMG_OUT / f"sdcpp_{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:6]}.png"
    args = [sd, "-m", model_path, "-p", prompt,
            "-W", str(width), "-H", str(height),
            "--steps", str(steps), "-o", str(out_path)]
    if negative:
        args += ["-n", negative]
    if seed is not None:
        args += ["-s", str(seed)]
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=600)
    except Exception as e:
        raise RuntimeError(f"sd.cpp run failed: {e}")
    if r.returncode != 0:
        raise RuntimeError(f"sd.cpp exit {r.returncode}: {r.stderr[:500]}")
    if not out_path.exists():
        raise RuntimeError("sd.cpp finished but no output file")
    return str(out_path)


# ---------------------------------------------------------------------------
# Dispatch by settings
# ---------------------------------------------------------------------------

def generate(prompt: str, settings: dict, out_dir: str | None = None, **overrides) -> dict:
    """Returns {'path': str, 'backend': str, 'prompt': str, ...}.

    Mode is read from settings.imggen_mode:
      'local'  → run model in-process (safetensors / HF dir / gguf-via-sd.cpp)
      'server' → external (ComfyUI or sd.cpp CLI subprocess)
    settings.imggen_backend picks the concrete backend within each mode.
    """
    mode    = (overrides.get("mode") or settings.get("imggen_mode") or "local").lower()
    backend = (overrides.get("backend") or settings.get("imggen_backend") or "none").lower()
    model_path = overrides.get("model") or settings.get("imggen_model_path") or ""
    width = int(overrides.get("width", settings.get("imggen_width", 512)))
    height = int(overrides.get("height", settings.get("imggen_height", 512)))
    steps = int(overrides.get("steps", settings.get("imggen_steps", 20)))
    seed = overrides.get("seed")
    negative = overrides.get("negative") or settings.get("imggen_negative", "")
    init_image = overrides.get("init_image") or ""
    strength = float(overrides.get("strength", settings.get("imggen_strength", 0.75)))
    offload = overrides.get("offload_mode") or settings.get("imggen_offload_mode", "auto")
    loras = overrides.get("loras")
    if loras is None:
        loras = settings.get("imggen_loras") or []
    comfy_url = settings.get("imggen_comfy_url") or "http://127.0.0.1:8188"

    if backend == "none":
        raise RuntimeError("image generation not configured. Settings → Image gen.")

    if mode == "local":
        if not model_path:
            raise RuntimeError("no local base model selected. Pick one in Settings → Image gen → Diffusers/SafeTensors/Transformers/GGUF sub-tab and press Load. LoRAs require a base model.")
        path = gen_local(prompt, model_path, negative, width, height,
                         steps, seed, init_image=init_image, strength=strength,
                         offload_mode=offload, loras=loras, out_dir=out_dir)
    elif mode == "server":
        if backend == "comfyui":
            path = gen_comfyui(prompt, negative, width, height, steps, seed,
                               model=settings.get("imggen_comfy_model", ""),
                               url=comfy_url)
        elif backend == "sd_cpp":
            if not model_path:
                raise RuntimeError("no model selected for sd.cpp")
            path = gen_sd_cpp_cli(prompt, model_path, negative, width, height, steps, seed)
        else:
            raise RuntimeError(f"unknown server backend: {backend}")
    else:
        raise RuntimeError(f"unknown imggen mode: {mode}")

    return {"path": path, "mode": mode, "backend": backend, "prompt": prompt,
            "width": width, "height": height, "steps": steps, "seed": seed,
            "init_image": init_image, "strength": strength if init_image else None,
            "offload_mode": offload if mode == "local" else None,
            "loras": loras if mode == "local" else None}
