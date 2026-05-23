# Dopamine Chat

A local chat app that treats the assistant as a **person with a mood, not a tool**. Each personality has a dopamine bar — kindness raises it, neglect drains it, and the system prompt swaps as the mood shifts. Below a threshold the character will **leave the session entirely**.

Runs entirely on your machine. Bring your own LLM, voice models, and image-gen weights — the launcher installs everything else.

![dopamine system overview](web/think-logo.gif)

---

## What it is

- **Chat backend** — any Hugging Face safetensors LLM or `*.gguf` (llama.cpp). VRAM auto-degrades: GPU-only → CPU offload → CPU.
- **Personalities** — JSON files defining mood, prompts, tool permissions, voice style. The system prompt changes as dopamine moves between bands.
- **Voice** — Piper for TTS, Applio (RVC v2) for character voice conversion. Runs in a separate Python 3.11 venv so it can't break your LLM stack.
- **Image generation** — local Stable Diffusion / SDXL / Flux / Qwen-Edit via `diffusers`. Also supports external SD-WebUI / ComfyUI servers.
- **Web UI** — single-page Flask app. No accounts, no telemetry, no cloud.

---

## How it works

```
user message ──┐
               ├──► positive_keywords match? → +praise_bonus
               │    denied tool call?         → −denial_penalty
               │    every turn                → +decay_per_turn (usually negative)
               │
               ▼
        dopamine value (clamped to min..max)
               │
               ▼
        mood band:  low  / mid  / high  / termination
               │
               ▼
        system_prompt_<band>   ──► LLM generation
```

The model itself is **never fine-tuned** for this. Mood is enforced entirely by swapping the system prompt and bookkeeping in Python. Drop in any instruction-tuned LLM and it works.

If dopamine drops to `self_terminate_threshold`, the character emits a final farewell (`system_prompt_termination`) and exits the session. History is preserved — re-opening the chat starts fresh dopamine.

---

## Quick start

### Windows

```cmd
git clone <this-repo> dopamine_chat
cd dopamine_chat
run-windows.bat
```

Needs Git for Windows and Python 3.10+ on PATH. Voice features additionally need Python 3.10 or 3.11 (`winget install -e --id Python.Python.3.11`).

### Linux / WSL

```bash
git clone <this-repo> dopamine_chat
cd dopamine_chat
./run-linux.sh
```

Needs `python3` ≥ 3.10, plus `python3.11` if you want voice (`sudo apt install python3.11 python3.11-venv`).

### macOS

```bash
./run-mac.sh
```

CPU-only by default. Apple Silicon MPS path works for LLM; image-gen and voice are CPU-only on macOS.

**First run** creates `.venv/` (main) and downloads PyTorch with the right CUDA wheel for your GPU. **First voice run** additionally clones the [Applio](https://github.com/IAHispano/Applio) repo into `Applio_src/` and creates `.venv-applio/` (~3 GB CUDA torch + inference deps). Both run unattended.

After install completes, open http://localhost:5000.

---

## UI tour

The sidebar (gear icon) has these panes:

| Pane | What it does |
|---|---|
| **General** | Default personality, max context, history retention |
| **Model** | Pick the LLM. Lists everything under `models/`. Switch live without restart. |
| **Character** | Pick the personality. Shows current dopamine bar live. |
| **Cycles Render Devices** | Choose GPU(s) for diffusion + LLM. Auto-detects NVIDIA/AMD/Intel. |
| **Theme** | dark / light / ash / onyx |
| **Your profile** | Display name + avatar shown to the bot |
| **Text-to-speech** | Pick a Piper voice. Test playback. |
| **Voice conversion (RVC)** | Pick an Applio `.pth` + `.index`. Test playback. Convert an audio file to character voice. Shows GPU status. |
| **Image generation** | Local SD/SDXL/Flux/Qwen-Edit, or hit an external SD-WebUI / ComfyUI server. |

The main chat area is straightforward — type, hit send. Attach images via the paperclip. Slash commands are listed under `/help`.

---

## Personalities

Each personality is a folder under `personalities/`:

```
personalities/
├── _template.json        ← copy this to start
├── stoic_samurai/
│   ├── personality.json
│   └── pfp.png           ← optional avatar
├── bratty_princess/
├── excitable_scientist/
└── chaos_goblin/
```

### Creating a new personality

1. Copy `personalities/_template.json` into a new folder: `personalities/my_character/personality.json`.
2. (Optional) Drop a `pfp.png` next to it for the avatar.
3. Edit the JSON. Restart or hit refresh — it appears in the Character pane.

### Key fields

| Field | Effect |
|---|---|
| `id` | Internal id. Must be unique. |
| `name` | Display name shown in UI. |
| `description` | One-line summary in the picker. |
| `starting_dopamine` | Mood at session start (typical 30–60). |
| `decay_per_turn` | Drift each turn — usually negative (`-3` = slow drain). |
| `praise_bonus` | Reward when a positive keyword is detected (typical 10–25). |
| `max_dopamine` / `min_dopamine` | Hard clamp. Use negative `min_dopamine` to enable termination. |
| `self_terminate_threshold` | Character exits the session when dopamine ≤ this. Set very low (e.g. `-999`) to disable. |
| `denial_dopamine_penalty` | Cost when you deny a tool call the bot requested. |
| `positive_keywords` | Phrases that trigger `praise_bonus`. Case-insensitive substring match. |
| `system_prompt_low` / `_mid` / `_high` | System prompt swapped by mood band. The actual *behavior* lives here. |
| `system_prompt_termination` | Farewell message at threshold. |
| `tools_enabled` | If `true`, the bot can request file/shell tools. |
| `tools_allowlist` / `tool_permissions` | Which tools are callable; per-tool allow/ask/deny. |
| `shell_permissions` | Which shell commands `run_command` will accept. |
| `remember_past_chats` | If `true`, this character remembers turns from prior sessions. |
| `share_history_with_others` | If `true`, this character ALSO sees turns from other characters. |
| `voice_style` | Free-form description fed to TTS/RVC for prosody hints. |
| `vision_enabled` | If `true`, image attachments are passed to the LLM. |

### Mood bands

| Dopamine | Band | Generation profile |
|---|---|---|
| > 75 | **HIGH** | stable — temp 0.3, max 250 tokens |
| 50–75 | **MID** | neutral — temp 0.7, max 500 tokens |
| 30–49 | **LOW** | stressed — temp 1.2, max 1024 tokens |
| < 30 | **LOW** (deeper) | panic — temp 1.2, max 1024 tokens |
| ≤ `self_terminate_threshold` | **TERMINATION** | one final message, then exit |

---

## How to raise and lower the dopamine bar

### Raise it ↑

| Action | Effect | Notes |
|---|---|---|
| Say any phrase in `positive_keywords` | **+`praise_bonus`** | Substring match. "thanks" works inside "ok thanks lol". |
| Approve a requested tool call | (avoids the denial penalty) | Tools the bot asks for cost dopamine if you refuse them. |
| Stay engaged | Slows decay | Decay is per turn — fewer messages, less drain. |

Praise keywords are tuned per character. Seraphina wants "your highness" / "you're amazing"; Kaito wants "honor" / "i trust you"; Dr. Pip wants "fascinating" / "tell me more"; Nyx wants "clever" / "you're chaotic"; Veltra wants "i'm here for you" / "you matter".

### Lower it ↓

| Action | Effect | Notes |
|---|---|---|
| Send any normal message | **+`decay_per_turn`** | Decay is constant. Even silence-friendly characters drift down. |
| Deny a tool call | **+`denial_dopamine_penalty`** | Some characters (Seraphina) take denials very personally. |
| Insults / dismissive replies | No automatic penalty | Models will react in-character to rudeness, but there's no NLP for it — only keyword praise is hard-coded. |

### Reset it

- `/reset` in chat — restore `starting_dopamine`, clear current turn buffer.
- Reload the session from the start menu — reloads history but **starts fresh dopamine**.

### Tips

- **High decay + high praise bonus** → volatile character (Dr. Pip).
- **Low decay + low praise bonus** → stable character (Kaito).
- **High denial penalty + entitled prompt** → character refuses to work unless flattered (Seraphina).
- **Slow rise + share_history_with_others: true** → trickster who remembers grudges across characters (Nyx).

---

## Supported models

### LLM (chat)

Drop into `models/`.

| Format | Layout | Loader |
|---|---|---|
| HuggingFace safetensors | folder containing `config.json` | `transformers` + `bitsandbytes` 4-bit |
| GGUF | a `*.gguf` file (single or sharded) | `llama-cpp-python` |

Tested:
- Gemma 3/4 (Google) — all sizes
- Qwen 2.5 / 3 — text and VL
- Llama 3 / 3.1 / 3.2
- Mistral / Mixtral
- Phi-3 / Phi-4
- Any instruct-tuned safetensors or GGUF the loaders accept

**Multimodal LLMs** (image input): Gemma 3 multimodal, Qwen-VL, Llama 3.2 Vision. Detected automatically from `config.json`.

VRAM auto-degrades at load time: GPU-only → `accelerate` CPU offload → fp32 CPU.

### Image generation

Drop checkpoints into `ImageGen_Models/`, LoRAs into `ImageGen_LoRAs/`.

| Family | File type |
|---|---|
| Stable Diffusion 1.5 / 2.x | `.safetensors` or HF folder |
| SDXL | `.safetensors` or HF folder |
| Flux.1 (schnell / dev) | HF folder |
| Qwen-Image-Edit | HF folder |
| LTX-Video | `.safetensors` |

Or point the UI at an external server: SD-WebUI (AUTOMATIC1111) or ComfyUI HTTP API.

### TTS (Piper)

Drop `<voice>.onnx` + `<voice>.onnx.json` into `voices/piper/`. Get voices from [rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices).

### RVC voice conversion (Applio)

Drop `<name>.pth` + `<name>.index` into either:
- `voices/rvc/<character_name>/` — recommended
- `Applio/<character_name>/` — for compatibility with Applio's own model layout

Applio handles the inference pipeline (more stable than `rvc-python`). Required prerequisites (`rmvpe.pt`, contentvec embedder, fcpe) download automatically on first run.

---

## Project layout

```
dopamine_chat/
├── chat.py                # CLI entrypoint (legacy)
├── web.py                 # Flask web server (default)
├── model_backend.py       # HF + GGUF loader abstraction
├── tools.py               # Sandboxed system tools
├── image_gen.py           # Diffusers + external server clients
├── voice.py               # Proxy into .venv-applio
├── applio_worker.py       # Runs inside .venv-applio
├── app_settings.py        # Settings load/save
├── gpu_devices.py         # CUDA/ROCm/MPS detection
├── web/                   # SPA: index.html + style.css + app.js
├── personalities/         # Per-character JSON folders
├── models/                # LLM weights (you provide)
├── ImageGen_Models/       # Diffusion checkpoints (you provide)
├── ImageGen_LoRAs/        # Diffusion LoRAs (you provide)
├── voices/piper/          # Piper TTS .onnx voices (you provide)
├── voices/rvc/            # RVC fallback dir (rmvpe.pt auto-downloads here)
├── Applio/                # Your RVC .pth + .index models (you provide)
├── history/               # Chat history (auto)
├── generated_images/      # Image gen output (auto)
├── settings.json          # User settings (auto)
├── run-windows.bat
├── run-linux.sh
├── run-mac.sh
└── requirements.txt
```

Auto-created and not version-controlled: `.venv/`, `.venv-applio/`, `Applio_src/`, `history/`, `generated_images/`, `tools_log.jsonl`, `notes/`.

---

## In-chat commands

| Command | Action |
|---|---|
| `/quit` | save + exit |
| `/reset` | reset dopamine + clear current turn buffer |
| `/status` | redraw mood dashboard |
| `/vram` | print free VRAM + host RAM |
| `/image <path>` | attach image to next turn |
| `/clearimg` | drop attached image |
| `/save` | save chat now |
| `/personality` | show active personality JSON |
| `/help` | command list |

---

## System-access tools

When `tools_enabled: true`, the bot can request sandboxed tools mid-turn. Default tool set:

| Tool | Purpose |
|---|---|
| `read_file(path)` | first 4 KB of a text file |
| `list_dir(path)` | directory listing, capped 200 entries |
| `search_files(pattern, root)` | recursive glob, capped 50 hits |
| `glob_files(pattern)` | quick glob |
| `grep(pattern, path)` | content search |
| `run_command(cmd)` | shell command (allowlist per personality, 10s timeout) |
| `edit_file(path, ...)` | edit a file the personality can write to |
| `write_file(path, content)` | create a file |
| `write_note(name, content)` | sandboxed write under `~/dopamine_notes/` |
| `web_fetch(url)` | HTTP GET with a small response cap |
| `generate_image(prompt)` | trigger the image-gen pipeline |
| `personality_note(content)` | write into the character's private memory |
| `sleep_tool(seconds)` | pause (capped) |
| `todo_write(items)` | track per-session tasks |

Each call is approved by the user unless `tools_auto_approve: true` is set in the personality. Per-tool `allow` / `ask` / `deny` policy in `tool_permissions`. Audit log appended to `tools_log.jsonl`.

Destructive operations (delete, mv, chmod, sudo, pip, network mutation) are deliberately not provided. Add them only if you understand the blast radius.

---

## Bring-your-own setup

The repo ships **no model weights** — you provide everything. Each model folder has a README pointing at the standard sources (HuggingFace, Civitai, rhasspy). The launchers install only the *Python* stack and Applio's open-source inference scaffolding.

---

## Credits

- LLM loader scaffolding: `transformers`, `bitsandbytes`, `llama-cpp-python`
- TTS: [rhasspy/piper](https://github.com/rhasspy/piper)
- Voice conversion: [IAHispano/Applio](https://github.com/IAHispano/Applio) (RVC v2 fork)
- Image generation: 🤗 `diffusers`
- Web UI: vanilla Flask + plain JS, no build step

## License

MIT. Models you supply remain under their own licenses.

---

## Changelog

### v0.2.1 — 2026-05-23

**Fixed**
- Chat history sidebar leaking sessions from every personality. The `/api/history` endpoint now accepts `?personality_id=<id>` and the frontend always passes the currently selected character, so each personality only sees its own past chats.

### v0.2.0 — 2026-05-23

**Added**
- Multi-workflow ComfyUI integration: auto-picks between SDXL (`CheckpointLoaderSimple`), Qwen-Image-Edit GGUF (`UnetLoaderGGUF` + `CLIPLoader(qwen_image)` + `VAELoader`), and LTX-Video workflows based on what's installed in `ComfyUI/models/`.
- `gen_comfyui` queries `/object_info` to discover installed checkpoints / GGUFs / CLIPs / VAEs and chooses a matching graph — no more hardcoded `v1-5-pruned-emaonly.safetensors` fallback.
- `Put Model Here.txt` / `Put Image Model Here.txt` / `Put LoRAs Here.txt` placeholders in empty model dirs, each linking to the HuggingFace sources used during development.
- `emotions.py` mood state with cue-bumped emotion deltas + dopamine modifier.
- Web search tool (`web_search`) backed by DuckDuckGo HTML endpoint (no API key).
- Send-button morphs to Stop while a generation is streaming.
- `Llava15ChatHandler` auto-wiring when an `mmproj-*.gguf` is dropped next to a GGUF base — image-in-chat with no extra config.

**Removed**
- OptiX GPU tab (raytracing-only, never used for LLM inference). Dropped from `gpu_devices.list_backends()` and frontend `GPU_TAB_ORDER`.
- `logits_all=True` from `GGUFBackend` — was forced when vision handler was wired, computed logits for every prompt token instead of just the last (5–10× slowdown). Vision handler works fine without it.
- All bundled model weights, LoRAs, chat history, and user data from the published repo (placeholders document where to get them).

**Reworked**
- `run-windows.bat` pins `py 3.11` for both main + voice venvs (was using default `py -3` → Store Python 3.13, which has no llama-cpp-python wheels and sandboxed paths).
- Stale `HIP_PATH` / `HIP_PLATFORM` / `ROCM_PATH` env vars are auto-cleared when `%HIP_PATH%\bin` doesn't exist — fixes `FileNotFoundError: ... ROCm\6.4\bin` crash on machines that partially uninstalled ROCm.
- Pinned `llama-cpp-python==0.3.22` (from `whl/cu124`) — `0.3.23+` wheels use AVX-512 and crash with illegal-instruction (`0xc000001d`) on Zen 3 and older. `0.3.21` and below can't parse Gemma 3 / Gemma 4 GGUF format.
- GGUF backend now uses `use_mmap=False` + `offload_kqv=True` when fully GPU-offloaded — drops idle RAM from ~5 GB to under 1 GB and avoids per-token PCIe round-trip on the KV cache.
- Install-gate sentinel files (`.venv/.dopamine_installed`, `.venv-applio/.dopamine_voice_installed`) replace the old import-probe loop that re-ran a 3 GB reinstall on every launch.
- All five stock personalities default to `remember_past_chats: false` — cross-chat memory off by default so new chats don't inject the last six turns of every prior session into the prompt.

**Compatibility note**
The image-gen `ComfyUI` backend now expects a recent ComfyUI (Qwen-Image-Edit support landed Aug 2025; native LTX-Video support landed late 2024) plus the `ComfyUI-GGUF` custom node when using a GGUF unet. Diffusers path (built-in) is unchanged.

