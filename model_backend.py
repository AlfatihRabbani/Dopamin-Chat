"""
Model backends for Dopamine Chat.

Two backends share a common stream() interface:

  HFBackend    — transformers + bitsandbytes 4-bit. Supports VLM image input
                 when an AutoProcessor + processor_config.json is present.
                 3-tier VRAM degrade: gpu → cpu-offload → cpu-fp32.

  GGUFBackend  — llama-cpp-python. Text only. Uses the chat template baked
                 into the .gguf file. Single-file model.
"""

from __future__ import annotations

import os
import sys
import gc
import json
from pathlib import Path
from threading import Thread
from typing import Iterator, Optional

import torch
from rich.console import Console

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODELS_DIR = SCRIPT_DIR / "models"
OFFLOAD_DIR = SCRIPT_DIR / "offload"


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def models_root() -> Path:
    env = os.environ.get("DOPAMINE_MODELS_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return DEFAULT_MODELS_DIR


def discover_models() -> list[dict]:
    """Return list of {name, path, format} for every model under models_root().

    A directory containing config.json is "hf". A *.gguf file is "gguf".
    Legacy single-model env var DOPAMINE_MODEL_DIR is also honored.
    """
    found: list[dict] = []

    single = os.environ.get("DOPAMINE_MODEL_DIR")
    if single:
        p = Path(single).expanduser().resolve()
        if p.is_file() and p.suffix.lower() == ".gguf":
            found.append({"name": p.stem, "path": p, "format": "gguf"})
        elif p.is_dir() and (p / "config.json").exists():
            found.append({"name": p.name, "path": p, "format": "hf"})
        return found

    root = models_root()
    if not root.exists():
        return found
    # Case A: HF weights laid directly inside models_root() (config.json
    # alongside tokenizer + safetensors / index). Treat the whole folder
    # as one HF model.
    if (root / "config.json").exists():
        found.append({"name": root.name + " (hf)", "path": root, "format": "hf"})
    for entry in sorted(root.iterdir()):
        if entry.name.startswith("."):
            continue
        if entry.is_file() and entry.suffix.lower() == ".gguf":
            found.append({"name": entry.name, "path": entry, "format": "gguf"})
        elif entry.is_dir() and (entry / "config.json").exists():
            found.append({"name": entry.name, "path": entry, "format": "hf"})
    return found


# ---------------------------------------------------------------------------
# Memory probes
# ---------------------------------------------------------------------------

def free_vram_gib() -> float:
    if not torch.cuda.is_available():
        return 0.0
    free, _ = torch.cuda.mem_get_info()
    return free / (1024 ** 3)


def host_ram_gib() -> float:
    try:
        import psutil
        return psutil.virtual_memory().available / (1024 ** 3)
    except Exception:
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) / (1024 ** 2)
        except Exception:
            pass
        return 16.0


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

class Backend:
    """Common interface."""
    label: str = ""
    has_vision: bool = False
    tokenizer = None    # provided for tool-call parsing helpers (may be None for GGUF)
    processor = None
    n_ctx: int = 4096   # backend-reported context window in tokens

    def stream(self, messages: list[dict], temp: float, max_new: int,
               image_pil=None, seed: int | None = None) -> Iterator[str]:
        raise NotImplementedError

    def count_context_tokens(self, messages: list[dict]) -> int:
        """Approximate context length in tokens. Override per backend."""
        try:
            txt = ""
            for m in messages:
                c = m.get("content", "")
                if isinstance(c, list):
                    c = " ".join(p.get("text", "") for p in c
                                 if isinstance(p, dict) and p.get("type") == "text")
                txt += str(c) + "\n"
            return max(1, len(txt) // 4)
        except Exception:
            return 0


class HFBackend(Backend):
    """transformers + bitsandbytes 4-bit."""

    def __init__(self, model_dir: Path, console: Console):
        from transformers import (
            AutoModelForCausalLM, AutoTokenizer, AutoProcessor, BitsAndBytesConfig,
        )
        try:
            from transformers import AutoModelForImageTextToText
        except ImportError:
            AutoModelForImageTextToText = None

        self.model_dir = Path(model_dir)
        console.print(f"[cyan]Loading HF model: {self.model_dir}[/]")
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        try:
            mlen = int(getattr(self.tokenizer, "model_max_length", 4096))
            if mlen > 200000:  # tokenizers often report 1e30; sanity clamp
                mlen = 16384
            self.n_ctx = mlen
        except Exception:
            self.n_ctx = 4096
        try:
            self.processor = AutoProcessor.from_pretrained(str(self.model_dir))
        except Exception:
            self.processor = None
            console.print("[dim]No AutoProcessor — text only.[/]")

        # Independent vision capability detection. Set early so even if the
        # ImageTextToText loader path falls back to CausalLM, has_vision
        # still reflects the real model shape.
        self._vision_capable = False
        try:
            import json as _json
            cfg_path = self.model_dir / "config.json"
            if cfg_path.exists():
                cfg = _json.loads(cfg_path.read_text(encoding="utf-8"))
                arch = " ".join(cfg.get("architectures") or [])
                mtype = (cfg.get("model_type") or "").lower()
                vision_hints = ("VL", "Vision", "ImageText", "Llava", "Qwen2VL",
                                "Qwen25VL", "Idefics", "PaliGemma", "Gemma3",
                                "InternVL", "MiniCPMV", "Bunny", "Phi3V",
                                "CogVLM", "Janus", "Florence")
                if any(h.lower() in arch.lower() for h in vision_hints):
                    self._vision_capable = True
                if any(h in mtype for h in (
                    "vl", "vision", "image_text", "llava", "idefics",
                    "paligemma", "gemma3", "gemma4", "internvl", "minicpmv",
                    "phi3_v", "florence", "qwen2_vl", "qwen2_5_vl",
                    "assistant")):
                    self._vision_capable = True
                # Definitive marker: any image_token_id in config means the
                # model was trained with image tokens. Trumps architecture
                # name lookups (catches new repos like Gemma4Assistant).
                if cfg.get("image_token_id") is not None:
                    self._vision_capable = True
        except Exception:
            pass
        if self.processor is not None and (
                hasattr(self.processor, "image_processor") or
                hasattr(self.processor, "feature_extractor")):
            self._vision_capable = True
        if self._vision_capable:
            console.print("[cyan]Vision capable: processor has image input.[/]")

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            console.print(f"[dim]VRAM: {free_vram_gib():.1f}/{total:.1f} GiB free  |  "
                          f"Host RAM: {host_ram_gib():.1f} GiB free[/]")
        else:
            console.print("[red]No CUDA device — CPU only.[/]")

        candidates = []
        is_vlm = []
        if AutoModelForImageTextToText is not None:
            candidates.append(("ImageTextToText", AutoModelForImageTextToText))
            is_vlm.append(True)
        candidates.append(("CausalLM", AutoModelForCausalLM))
        is_vlm.append(False)

        common = dict(
            quantization_config=bnb,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )

        # ----- Tier 1: pure GPU -----
        last_err: Optional[Exception] = None
        if torch.cuda.is_available():
            for (tag, cls), vlm in zip(candidates, is_vlm):
                try:
                    console.print(f"[cyan]Tier 1: {tag} pure-GPU 4-bit…[/]")
                    self.model = cls.from_pretrained(str(self.model_dir), device_map="auto", **common)
                    self.label = f"hf-gpu-{tag.lower()}"
                    self.has_vision = bool(self._vision_capable or (vlm and self.processor))
                    self.model.eval()
                    return
                except (torch.cuda.OutOfMemoryError, RuntimeError, ValueError) as e:
                    last_err = e
                    msg = str(e).lower()
                    if "out of memory" in msg or "cuda" in msg:
                        console.print(f"[yellow]Tier 1 OOM with {tag} → next tier[/]")
                        gc.collect()
                        torch.cuda.empty_cache()
                        break
                    console.print(f"[yellow]{tag} unavailable: {type(e).__name__}[/]")

        # ----- Tier 2: CPU offload -----
        if torch.cuda.is_available():
            OFFLOAD_DIR.mkdir(exist_ok=True)
            vram = max(1, int(free_vram_gib() - 1))
            ram = max(8, int(host_ram_gib() - 2))
            max_memory = {0: f"{vram}GiB", "cpu": f"{ram}GiB"}
            console.print(f"[cyan]Tier 2: CPU offload, max_memory={max_memory}[/]")
            for (tag, cls), vlm in zip(candidates, is_vlm):
                try:
                    self.model = cls.from_pretrained(
                        str(self.model_dir),
                        device_map="auto",
                        max_memory=max_memory,
                        offload_folder=str(OFFLOAD_DIR),
                        offload_state_dict=True,
                        **common,
                    )
                    self.label = f"hf-offload-{tag.lower()}"
                    self.has_vision = bool(self._vision_capable or (vlm and self.processor))
                    self.model.eval()
                    return
                except Exception as e:
                    last_err = e
                    console.print(f"[yellow]Offload {tag} failed: {type(e).__name__}[/]")
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        # ----- Tier 3: CPU only -----
        console.print("[red]Tier 3: CPU-only fp32 (very slow).[/]")
        for (tag, cls), vlm in zip(candidates, is_vlm):
            try:
                self.model = cls.from_pretrained(
                    str(self.model_dir),
                    torch_dtype=torch.float32,
                    device_map={"": "cpu"},
                    low_cpu_mem_usage=True,
                )
                self.label = f"hf-cpu-{tag.lower()}"
                self.has_vision = bool(vlm and self.processor)
                self.model.eval()
                return
            except Exception as e:
                last_err = e

        console.print(f"[red]All HF load tiers failed. Last error: {last_err}[/]")
        sys.exit(1)

    def _encode(self, messages: list[dict], has_image: bool):
        if has_image and self.processor is not None:
            try:
                return self.processor.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=True,
                    return_dict=True, return_tensors="pt",
                )
            except Exception:
                pass
        # Text-only fallback
        flat = []
        for m in messages:
            c = m["content"]
            if isinstance(c, list):
                c = " ".join(p.get("text", "") for p in c
                             if isinstance(p, dict) and p.get("type") == "text")
            flat.append({"role": m["role"], "content": c})
        text = self.tokenizer.apply_chat_template(flat, tokenize=False,
                                                  add_generation_prompt=True)
        return self.tokenizer(text, return_tensors="pt")

    def stream(self, messages, temp, max_new, image_pil=None, seed=None):
        from transformers import TextIteratorStreamer
        if seed is not None:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))

        # Inject the image (if any) into the last user turn
        if image_pil is not None:
            messages = list(messages)
            last = messages[-1]
            if isinstance(last.get("content"), str):
                messages[-1] = {"role": last["role"], "content": [
                    {"type": "image", "image": image_pil},
                    {"type": "text",  "text":  last["content"]},
                ]}

        inputs = self._encode(messages, has_image=(image_pil is not None))
        device = next(self.model.parameters()).device
        inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True,
                                        skip_special_tokens=False)
        kwargs = dict(
            **inputs, streamer=streamer, do_sample=True,
            temperature=temp, top_p=0.9, max_new_tokens=max_new,
            repetition_penalty=1.05,
        )
        thread = Thread(target=self.model.generate, kwargs=kwargs)
        thread.start()
        try:
            for tok in streamer:
                yield tok
        finally:
            thread.join()

    def count_context_tokens(self, messages):
        try:
            flat = []
            for m in messages:
                c = m["content"]
                if isinstance(c, list):
                    c = " ".join(p.get("text", "") for p in c
                                 if isinstance(p, dict) and p.get("type") == "text")
                flat.append({"role": m["role"], "content": c})
            text = self.tokenizer.apply_chat_template(flat, tokenize=False,
                                                      add_generation_prompt=True)
            return int(len(self.tokenizer(text)["input_ids"]))
        except Exception:
            return super().count_context_tokens(messages)


class GGUFBackend(Backend):
    """llama-cpp-python. Text only. Uses GGUF's embedded chat template."""

    def __init__(self, gguf_path: Path, console: Console):
        try:
            from llama_cpp import Llama
        except ImportError:
            console.print("[red]llama-cpp-python is not installed.[/]")
            console.print("Install (CPU):   pip install llama-cpp-python")
            console.print("Install (CUDA):  pip install llama-cpp-python "
                          "--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121")
            sys.exit(1)

        self.gguf_path = Path(gguf_path)
        console.print(f"[cyan]Loading GGUF: {self.gguf_path}[/]")
        if torch.cuda.is_available():
            free = free_vram_gib()
            console.print(f"[dim]VRAM free: {free:.1f} GiB → will try full GPU offload[/]")
            n_gpu = -1
        else:
            console.print("[dim]No CUDA — CPU only[/]")
            n_gpu = 0

        # Auto-detect an mmproj-*.gguf next to the model — wire it as a
        # llava-style chat handler so image_pil works through llama.cpp.
        mmproj_files = sorted(self.gguf_path.parent.glob("mmproj*.gguf"))
        chat_handler = None
        if mmproj_files:
            mmproj = mmproj_files[0]
            try:
                from llama_cpp.llama_chat_format import Llava15ChatHandler
                chat_handler = Llava15ChatHandler(clip_model_path=str(mmproj),
                                                  verbose=False)
                console.print(f"[cyan]Vision handler wired via {mmproj.name}[/]")
            except Exception as e:
                console.print(f"[yellow]mmproj load failed ({type(e).__name__}: {e}); "
                              f"falling back to text-only[/]")
                chat_handler = None

        # flash_attn=True silences the "V embeddings have different sizes
        # across layers and FA is not enabled - padding V cache to 2048"
        # warning and lets per-layer V-cache run at native size on GPU.
        common_kwargs = dict(
            n_gpu_layers=n_gpu,
            n_ctx=int(os.environ.get("DOPAMINE_N_CTX", "16384")),
            flash_attn=True,
            verbose=False,
        )
        if chat_handler is not None:
            common_kwargs["chat_handler"] = chat_handler
            common_kwargs["logits_all"] = True

        try:
            self.llm = Llama(model_path=str(self.gguf_path), **common_kwargs)
            self.label = f"gguf-{'gpu' if n_gpu == -1 else 'cpu'}"
        except Exception as e:
            console.print(f"[yellow]GPU offload failed ({e}) → falling back to CPU[/]")
            try:
                cpu_kwargs = dict(common_kwargs)
                cpu_kwargs["n_gpu_layers"] = 0
                self.llm = Llama(model_path=str(self.gguf_path), **cpu_kwargs)
                self.label = "gguf-cpu"
            except Exception as e2:
                # Last resort: drop flash_attn (older llama-cpp wheels may
                # not have the kwarg) and try again.
                console.print(f"[yellow]Retry without flash_attn ({e2})[/]")
                cpu_kwargs.pop("flash_attn", None)
                try:
                    self.llm = Llama(model_path=str(self.gguf_path), **cpu_kwargs)
                    self.label = "gguf-cpu"
                except Exception as e3:
                    console.print(f"[red]GGUF load failed: {e3}[/]")
                    sys.exit(1)

        # Provide a stub tokenizer for tool-call regex (just splits on whitespace).
        self.tokenizer = None
        self.processor = None
        self._chat_handler = chat_handler
        self.has_vision = chat_handler is not None
        if not self.has_vision:
            # Probe config.json sibling for image_token_id — warn if user
            # loaded a vision-trained model without a projector.
            cfg_path = self.gguf_path.parent / "config.json"
            if cfg_path.exists():
                import json as _json
                try:
                    cfg = _json.loads(cfg_path.read_text(encoding="utf-8"))
                    if cfg.get("image_token_id") is not None:
                        console.print("[yellow]Model config marks this as a vision model "
                                      "(image_token_id present) but no mmproj-*.gguf was "
                                      "loaded. Drop the matching mmproj file in this folder "
                                      "or use the HF safetensors release.[/]")
                except Exception:
                    pass
        try:
            self.n_ctx = int(self.llm.n_ctx())
        except Exception:
            self.n_ctx = int(os.environ.get("DOPAMINE_N_CTX", "16384"))
        console.print(f"[dim]GGUF n_ctx = {self.n_ctx}[/]")

    def stream(self, messages, temp, max_new, image_pil=None, seed=None):
        # Vision path: if a chat_handler is wired and we got a PIL image,
        # attach it to the latest user turn as a data-URL image_url part.
        # llama-cpp's Llava15ChatHandler reads OpenAI-style multipart content.
        image_data_url = None
        if image_pil is not None and self.has_vision and self._chat_handler is not None:
            try:
                import io as _io, base64 as _b64
                buf = _io.BytesIO()
                image_pil.convert("RGB").save(buf, format="PNG")
                image_data_url = ("data:image/png;base64,"
                                  + _b64.b64encode(buf.getvalue()).decode("ascii"))
            except Exception:
                image_data_url = None

        # Flatten any list content (vision attachments) to plain text — UNLESS
        # this turn includes a real image we'll re-attach as multipart below.
        flat = []
        for m in messages:
            c = m["content"]
            if isinstance(c, list):
                c = " ".join(p.get("text", "") for p in c
                             if isinstance(p, dict) and p.get("type") == "text")
            flat.append({"role": m["role"], "content": c})
        if image_data_url:
            # Rewrite the last user message into OpenAI multipart format.
            for i in range(len(flat) - 1, -1, -1):
                if flat[i]["role"] == "user":
                    flat[i] = {"role": "user", "content": [
                        {"type": "text", "text": flat[i]["content"]},
                        {"type": "image_url", "image_url": {"url": image_data_url}},
                    ]}
                    break

        kwargs = dict(
            messages=flat,
            temperature=temp,
            max_tokens=max_new,
            top_p=0.9,
            repeat_penalty=1.05,
            stream=True,
        )
        if seed is not None:
            kwargs["seed"] = int(seed)
        try:
            stream = self.llm.create_chat_completion(**kwargs)
            for chunk in stream:
                delta = chunk["choices"][0].get("delta", {})
                tok = delta.get("content")
                if tok:
                    yield tok
        except TypeError:
            # Some llama-cpp versions reject seed kw — retry without.
            kwargs.pop("seed", None)
            try:
                stream = self.llm.create_chat_completion(**kwargs)
                for chunk in stream:
                    delta = chunk["choices"][0].get("delta", {})
                    tok = delta.get("content")
                    if tok:
                        yield tok
            except Exception as e:
                yield f"\n[GGUF generation error: {e}]"
        except Exception as e:
            yield f"\n[GGUF generation error: {e}]"

    def count_context_tokens(self, messages):
        try:
            flat_text = ""
            for m in messages:
                c = m.get("content", "")
                if isinstance(c, list):
                    c = " ".join(p.get("text", "") for p in c
                                 if isinstance(p, dict) and p.get("type") == "text")
                flat_text += f"{m.get('role','')}: {c}\n"
            toks = self.llm.tokenize(flat_text.encode("utf-8", "replace"))
            return len(toks)
        except Exception:
            return super().count_context_tokens(messages)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def load_backend(entry: dict, console: Console) -> Backend:
    fmt = entry["format"]
    if fmt == "hf":
        return HFBackend(entry["path"], console)
    if fmt == "gguf":
        return GGUFBackend(entry["path"], console)
    raise ValueError(f"unknown format: {fmt}")
