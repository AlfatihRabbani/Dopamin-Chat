"""
Dopamine Chat — interactive CLI for Gemma-4-26B-A4B.

Features:
  * Personality system (JSON templates in personalities/)
  * Per-session chat history persisted to history/<session_id>.json
  * Startup menu: new / load / delete / reset / list personalities / quit
  * Image upload via /image <path>  (Gemma 4 is a VLM)
  * Mood-driven generation — dopamine 0-100, decay & praise bonus per personality
  * VRAM auto-degrade: gpu → cpu-offload → cpu-fp32
  * In-session commands: /quit /reset /status /vram /image /clearimg /save /personality /help

Layout (this folder):
  chat.py
  personalities/<id>.json
  history/<session_id>.json
  offload/                  (created on the fly when CPU offload tier kicks in)
  run.sh / run.bat          launchers
  install.sh / install.bat  one-time setup

Model location:
  Defaults to ../Dopamine_Gemma26B (i.e., trained by ../train.py).
  Override with env var:  DOPAMINE_MODEL_DIR=/abs/path  ./run.sh
"""

import os
import re
import sys
import gc
import json
import uuid
import shutil
import datetime
from pathlib import Path
from threading import Thread
from dataclasses import dataclass, field, asdict

import torch
from rich.console import Console, Group
from rich.panel import Panel
from rich.prompt import Prompt, Confirm, IntPrompt
from rich.table import Table
from rich.text import Text

import tools as _tools
import model_backend as _mb
import emotions as _emotions

# ----------------------------------------------------------------------------
# Paths and constants
# ----------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
HISTORY_DIR = SCRIPT_DIR / "history"
PERSONALITIES_DIR = SCRIPT_DIR / "personalities"
OFFLOAD_DIR = SCRIPT_DIR / "offload"
HISTORY_DIR.mkdir(exist_ok=True)
PERSONALITIES_DIR.mkdir(exist_ok=True)

# Model location is owned by model_backend.py. Override:
#   DOPAMINE_MODELS_DIR=/abs/folder   (changes the discovery root)
#   DOPAMINE_MODEL_DIR=/abs/onefile   (uses a single model, skipping discovery)

BOT_NAME = "GEMMA-4-26B-A4B"
BAR_WIDTH = 24
PRIOR_CONTEXT_TURNS = 6   # how many prior turns to inject when remember_past_chats=True
# Hard cap on how many in-session turn pairs we re-send to the LLM. Prevents
# the model from being flooded with the whole chat once a long history
# accumulates (which makes it act erratically).
SESSION_CONTEXT_TURNS = 12

os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True,max_split_size_mb:128",
)


# ----------------------------------------------------------------------------
# Personality
# ----------------------------------------------------------------------------

@dataclass
class Personality:
    id: str
    name: str
    description: str = ""
    starting_dopamine: int = 50
    decay_per_turn: int = -5
    praise_bonus: int = 35
    max_dopamine: int = 100
    min_dopamine: int = 0
    self_terminate_threshold: int = -999   # never by default
    tools_enabled: bool = False
    tools_allowlist: list = field(default_factory=list)
    tools_auto_approve: bool = False
    tool_permissions: dict = field(default_factory=dict)
    shell_permissions: dict = field(default_factory=dict)
    shell_allowlist_bypass: bool = False
    denial_dopamine_penalty: int = -5
    pfp_path: str = ""            # populated by list_personalities()
    folder: str = ""              # subfolder path on disk
    tts_voice: str = ""           # piper voice model basename (optional)
    rvc_pth: str = ""             # RVC .pth file basename for voice conversion
    rvc_index: str = ""           # RVC .index file basename
    positive_keywords: list = field(default_factory=list)
    system_prompt_low: str = ""
    system_prompt_mid: str = ""
    system_prompt_high: str = ""
    system_prompt_termination: str = ""
    remember_past_chats: bool = False
    share_history_with_others: bool = False
    # In-chat history budget (approx tokens, char/4). 0 = use SESSION_CONTEXT_TURNS only.
    # When set, trims oldest turns from the recent window until total fits.
    history_token_budget: int = 0
    voice_style: str = ""
    # Image-text-to-text (vision) — only effective when the loaded chat
    # model itself supports vision (backend.has_vision). If the model is
    # text-only, this flag is silently ignored.
    vision_enabled: bool = False

    @classmethod
    def load(cls, path: Path) -> "Personality":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in valid})

    def sys_prompt(self, dop: int) -> str:
        if dop > 70 and self.system_prompt_high:
            return self.system_prompt_high
        if dop >= 35:
            return self.system_prompt_mid or self.system_prompt_high or self.system_prompt_low
        return self.system_prompt_low or self.system_prompt_mid or self.system_prompt_high


PFP_EXT_PRIORITY = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp")


def _find_pfp(folder: Path) -> Path | None:
    for ext in PFP_EXT_PRIORITY:
        for name in (f"pfp{ext}", f"avatar{ext}", f"profile{ext}"):
            p = folder / name
            if p.exists() and p.is_file():
                return p
    return None


def list_personalities() -> list[Personality]:
    """Discover personalities.

    New layout (preferred):
        personalities/<id>/personality.json
        personalities/<id>/pfp.<png|jpg|...>
    Legacy layout still supported:
        personalities/<id>.json
    """
    out = []
    # Subfolder personalities
    for sub in sorted(PERSONALITIES_DIR.iterdir()):
        if not sub.is_dir() or sub.name.startswith("_"):
            continue
        candidate = sub / "personality.json"
        if not candidate.exists():
            # also accept <foldername>.json inside the folder
            candidate = sub / f"{sub.name}.json"
        if not candidate.exists():
            continue
        try:
            p = Personality.load(candidate)
            p.folder = str(sub)
            pfp = _find_pfp(sub)
            p.pfp_path = str(pfp) if pfp else ""
            out.append(p)
        except Exception as e:
            print(f"[warn] Skipping malformed personality {sub.name}: {e}", file=sys.stderr)
    # Legacy flat .json files
    for f in sorted(PERSONALITIES_DIR.glob("*.json")):
        if f.name.startswith("_"):
            continue
        try:
            p = Personality.load(f)
            p.folder = str(PERSONALITIES_DIR)
            out.append(p)
        except Exception as e:
            print(f"[warn] Skipping malformed personality {f.name}: {e}", file=sys.stderr)
    return out


# ----------------------------------------------------------------------------
# Session / History
# ----------------------------------------------------------------------------

@dataclass
class Session:
    personality: Personality
    session_id: str = ""
    dopamine: int = -1
    messages: list = field(default_factory=list)
    created_at: str = ""
    emotions: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.session_id:
            ts = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
            self.session_id = f"{self.personality.id}_{ts}_{uuid.uuid4().hex[:6]}"
        if not self.created_at:
            self.created_at = datetime.datetime.now().isoformat()
        if self.dopamine < 0:
            self.dopamine = self.personality.starting_dopamine
        self.emotions = _emotions.coerce_state(self.emotions)

    @staticmethod
    def _safe(name: str) -> str:
        return "".join(c for c in (name or "") if c.isalnum() or c in "_- ").strip().replace(" ", "_") or "default"

    def chat_name(self) -> str:
        """Human-friendly chat folder name. First user message (slugged)
        if present, else the timestamp portion of session_id."""
        for m in self.messages:
            if m.get("role") == "user":
                c = m.get("content", "")
                if isinstance(c, list):
                    c = " ".join(p.get("text", "") for p in c
                                 if isinstance(p, dict) and p.get("type") == "text")
                s = str(c).strip().splitlines()[0] if c else ""
                if s:
                    return self._safe(s)[:50]
        # fallback: timestamp from session_id (after the first underscore)
        parts = self.session_id.split("_", 1)
        return parts[1] if len(parts) > 1 else self.session_id

    def path(self) -> Path:
        """history/<PersonalityName>/<ChatName>/<session_id>.json"""
        d = HISTORY_DIR / self._safe(self.personality.name) / self.chat_name()
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{self.session_id}.json"

    def save(self):
        data = {
            "session_id":       self.session_id,
            "personality_id":   self.personality.id,
            "personality_name": self.personality.name,
            "dopamine":         self.dopamine,
            "emotions":         {k: round(v, 2) for k, v in self.emotions.items()},
            "created_at":       self.created_at,
            "updated_at":       datetime.datetime.now().isoformat(),
            "messages":         self.messages,
        }
        self.path().write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def from_file(cls, path: Path, personality_map: dict) -> "Session":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        pid = data.get("personality_id", "")
        personality = personality_map.get(pid)
        if not personality:
            # Fall back to any available personality if the original is missing.
            personality = next(iter(personality_map.values()))
        s = cls(
            personality=personality,
            session_id=data["session_id"],
            dopamine=data.get("dopamine", personality.starting_dopamine),
            messages=data.get("messages", []),
            created_at=data.get("created_at", datetime.datetime.now().isoformat()),
            emotions=data.get("emotions", {}),
        )
        return s


def list_history() -> list[dict]:
    # Recurse so chats nested under history/<character>/<chat>/<id>.json are
    # found alongside legacy flat history/<id>.json files. Skip notes.md.
    files = sorted(
        [p for p in HISTORY_DIR.rglob("*.json") if p.name != "notes.md"],
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    out = []
    for p in files:
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            out.append({
                "path":             p,
                "session_id":       d.get("session_id", p.stem),
                "personality_name": d.get("personality_name", "?"),
                "personality_id":   d.get("personality_id", "?"),
                "updated_at":       d.get("updated_at", "?"),
                "messages":         len(d.get("messages", [])),
                "dopamine":         d.get("dopamine", "?"),
            })
        except Exception:
            continue
    return out


def gather_prior_context(personality: Personality, current_session_id: str) -> list[dict]:
    """Pull recent turns from prior sessions for cross-session memory."""
    if not personality.remember_past_chats:
        return []

    entries = list_history()
    relevant: list[Path] = []
    for h in entries:
        if h["session_id"] == current_session_id:
            continue
        if personality.share_history_with_others or h["personality_id"] == personality.id:
            relevant.append(h["path"])

    msgs: list[dict] = []
    for path in reversed(relevant):  # oldest first
        try:
            d = json.loads(Path(path).read_text(encoding="utf-8"))
            for m in d.get("messages", []):
                content = m.get("content", "")
                # Flatten image-bearing content to text for context.
                if isinstance(content, list):
                    content = " ".join(p.get("text", "") for p in content
                                       if isinstance(p, dict) and p.get("type") == "text")
                if content:
                    msgs.append({"role": m["role"], "content": content})
        except Exception:
            continue

    # Cap to last N turns × 2 messages.
    return msgs[-(PRIOR_CONTEXT_TURNS * 2):]


# ----------------------------------------------------------------------------
# Mood
# ----------------------------------------------------------------------------

def emotion(dop: int) -> tuple[str, str, str]:
    if dop > 75:  return "HAPPY",     "bright_green", ":)"
    if dop >= 50: return "MILD",      "yellow",       ":|"
    if dop >= 30: return "STRESSED",  "orange1",      ":/"
    return "DEPRESSED", "red", ":("


def gen_mode(dop: int) -> tuple[str, float, int, bool]:
    """Returns (label, temperature, max_new_tokens, force_think_prefill).

    Happy characters talk more, not less. Map:
      PANIC   (dop<35): high temp, long replies, think-prefill on
      NEUTRAL (35-70):  balanced
      STABLE  (dop>70): warm, talkative, moderately long
    """
    if dop < 35:  return "PANIC",   1.20, 1024, True
    if dop > 70:  return "STABLE",  0.55, 800,  False
    return                "NEUTRAL", 0.75, 600,  False


# ----------------------------------------------------------------------------
# Tool-call parsing  (<tool>NAME</tool><args>{...JSON...}</args>)
# ----------------------------------------------------------------------------

# Args are optional. If model emits <tool>NAME</tool> with no <args>...</args>,
# we default to {} so the tool can run with its built-in defaults
# (e.g. list_dir() defaults path to ".").
TOOL_RE = re.compile(
    r"<tool>\s*(?P<name>[\w\-]+)\s*</tool>"
    r"(?:\s*<args>\s*(?P<args>.+?)\s*</args>)?",
    re.DOTALL,
)


def parse_tool_calls(text: str) -> list[dict]:
    out = []
    for m in TOOL_RE.finditer(text):
        name = m.group("name")
        raw = m.group("args")
        if raw is None:
            args = {}
        else:
            try:
                args = json.loads(raw)
                if not isinstance(args, dict):
                    args = {"_value": args}
            except Exception:
                args = {"_raw": raw[:500]}
        out.append({"name": name, "args": args})
    return out


def run_tool_calls(console: Console, personality: Personality, response: str) -> list[dict]:
    """Resolve permissions, optionally prompt user, dispatch tool calls.

    Returns list of {"name", "args", "result"} dicts. result may contain
    "denied": True if policy refused or user said no.
    """
    calls = parse_tool_calls(response)
    if not calls:
        return []
    results = []
    for call in calls:
        name, args = call["name"], call["args"]
        decision, reason = _tools.check_decision(name, args, personality)

        if decision == "deny":
            console.print(f"[red]denied:[/] {name} → {reason}")
            results.append({"name": name, "args": args,
                            "result": {"error": "denied", "denied": True,
                                       "denied_reason": reason}})
            continue

        if decision == "ask":
            console.print(Panel(
                Text.assemble(
                    ("Tool requested: ", "bold yellow"),
                    (name, "bold cyan"),
                    ("\nargs: ", "dim"),
                    (json.dumps(args, ensure_ascii=False, indent=2), ""),
                    ("\n", ""),
                    (reason, "dim"),
                ),
                title="System access request",
                border_style="yellow",
            ))
            if personality.tools_auto_approve:
                console.print("[dim]Auto-approved per personality config.[/]")
                approved = True
            else:
                approved = Confirm.ask("Allow?", default=False)
            if not approved:
                results.append({"name": name, "args": args,
                                "result": {"error": "denied by user", "denied": True,
                                           "denied_reason": "user clicked deny"}})
                continue
            # User approved → run bypassing policy
            result = _tools.dispatch(name, args, personality, force_allow=True)
        else:
            # decision == "allow"
            result = _tools.dispatch(name, args, personality)

        if result.get("denied"):
            console.print(f"[red]denied:[/] {name} → {result.get('denied_reason')}")
        else:
            preview = json.dumps(result, ensure_ascii=False)[:400]
            console.print(Panel(preview, title=f"result: {name}", border_style="green"))
        results.append({"name": name, "args": args, "result": result})
    return results


def bar(value: int, total: int, colour: str) -> Text:
    filled = max(0, min(BAR_WIDTH, int((value / total) * BAR_WIDTH)))
    return Text.assemble(
        ("[", "dim"),
        ("█" * filled, colour),
        ("·" * (BAR_WIDTH - filled), "dim"),
        ("]", "dim"),
    )


def header_panel(personality: Personality, dop: int, gen_label: str,
                 temp: float, max_new: int, backend: str,
                 attached_image: str | None = None) -> Panel:
    emo_label, emo_col, face = emotion(dop)

    top = Table.grid(expand=True, padding=(0, 1))
    top.add_column(justify="left", ratio=1)
    top.add_column(justify="right", ratio=1)
    top.add_row(
        Text.assemble(
            (BOT_NAME, "bold cyan"),
            ("  ", ""), (f"({backend})", "dim"),
            ("  ", ""), (f"persona: {personality.name}", "magenta"),
        ),
        Text.assemble(
            ("Mood: ", "bold"),
            (f"{emo_label} ", emo_col),
            (face, emo_col),
        ),
    )

    rows = [
        top,
        Text.assemble(
            ("DOPAMINE  ", "bold"),
            (f"{dop:>3}/100 ", emo_col),
            bar(dop, 100, emo_col),
            (f"  GEN: {gen_label}  ", "bold magenta"),
            (f"t={temp} max={max_new}", "dim"),
        ),
        Text.assemble(
            ("EMOTION   ", "bold"),
            (f"{emo_label:<10} ", emo_col),
            (f"{face}  ", emo_col),
            bar(dop, 100, emo_col),
        ),
    ]
    if attached_image:
        rows.append(Text.assemble(
            ("IMAGE     ", "bold"),
            (f"attached: {attached_image}", "cyan"),
        ))
    return Panel(Group(*rows), border_style=emo_col, padding=(0, 1))


# ----------------------------------------------------------------------------
# Memory probes
# ----------------------------------------------------------------------------

def free_vram_gib() -> float:
    if not torch.cuda.is_available(): return 0.0
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


# ----------------------------------------------------------------------------
# Model loading (3-tier VRAM degrade)
# ----------------------------------------------------------------------------

def pick_and_load_backend(console: Console):
    """Discover models/, prompt user to pick one, load and return Backend."""
    entries = _mb.discover_models()
    if not entries:
        console.print("[red]No models found.[/]")
        console.print(f"  Add a model to: {_mb.models_root()}")
        console.print("  Supported: HF folder (config.json + safetensors) or single *.gguf file")
        sys.exit(1)

    if len(entries) == 1:
        e = entries[0]
        console.print(f"[cyan]Using model:[/] [bold]{e['name']}[/] ({e['format']})")
    else:
        table = Table(title="Available models", show_header=True, header_style="bold")
        table.add_column("#", style="cyan", width=3)
        table.add_column("Name", style="bold magenta")
        table.add_column("Format", style="yellow")
        table.add_column("Path", style="dim")
        for i, e in enumerate(entries, 1):
            table.add_row(str(i), e["name"], e["format"], str(e["path"]))
        console.print(table)
        idx = IntPrompt.ask("Pick model", default=1)
        if idx < 1 or idx > len(entries):
            sys.exit(1)
        e = entries[idx - 1]

    backend = _mb.load_backend(e, console)
    console.print(f"[green]Model loaded.[/]  backend=[bold]{backend.label}[/]  "
                  f"vision={'yes' if backend.has_vision else 'no'}")
    return backend


# ----------------------------------------------------------------------------
# Prompt building / encoding
# ----------------------------------------------------------------------------

def _msg_text(m: dict) -> str:
    c = m.get("content", "")
    if isinstance(c, list):
        c = " ".join(p.get("text", "") for p in c
                     if isinstance(p, dict) and p.get("type") == "text")
    return str(c)


def _approx_tokens(text: str) -> int:
    # char/4 heuristic. Cheap and tokenizer-free.
    return max(1, len(text) // 4)


def build_messages(session: Session, sys_prompt: str, user_text: str,
                   image_pil=None, prior_context=None) -> list[dict]:
    msgs: list[dict] = [{"role": "system", "content": sys_prompt}]
    if prior_context:
        msgs.extend(prior_context)
    # Cap to the last SESSION_CONTEXT_TURNS × 2 messages so the LLM doesn't
    # get flooded by a long backlog. Older turns are still kept on disk and
    # visible in the UI; just not re-sent every turn.
    recent = session.messages[-(SESSION_CONTEXT_TURNS * 2):]
    # Per-personality token budget — newer turns first, drop oldest when over.
    budget = int(getattr(session.personality, "history_token_budget", 0) or 0)
    if budget > 0 and recent:
        kept: list[dict] = []
        used = 0
        for m in reversed(recent):
            t = _approx_tokens(_msg_text(m))
            if used + t > budget and kept:
                break
            kept.append(m)
            used += t
        recent = list(reversed(kept))
    for m in recent:
        msgs.append({"role": m["role"], "content": _msg_text(m)})
    if image_pil is not None:
        msgs.append({"role": "user", "content": [
            {"type": "image", "image": image_pil},
            {"type": "text",  "text":  user_text},
        ]})
    else:
        msgs.append({"role": "user", "content": user_text})
    return msgs


def encode_inputs(processor, tokenizer, messages, has_image: bool):
    """Returns dict of tensors ready to .to(device)."""
    if has_image and processor is not None:
        try:
            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            return inputs
        except Exception:
            # Fall through to text-only encoding.
            pass

    # Text-only path: flatten any list content to plain text.
    flat = []
    for m in messages:
        c = m["content"]
        if isinstance(c, list):
            c = " ".join(p.get("text", "") for p in c
                         if isinstance(p, dict) and p.get("type") == "text")
        flat.append({"role": m["role"], "content": c})
    text = tokenizer.apply_chat_template(flat, tokenize=False, add_generation_prompt=True)
    return tokenizer(text, return_tensors="pt")


# ----------------------------------------------------------------------------
# Menus
# ----------------------------------------------------------------------------

def show_main_menu(console: Console) -> str:
    console.print()
    console.print(Panel(
        Text.assemble(
            (f"  {BOT_NAME}\n", "bold cyan"),
            ("  Synthetic Dopamine Chat", "dim"),
        ),
        border_style="cyan",
        padding=(0, 2),
    ))
    console.print("  [bold]1[/] New chat")
    console.print("  [bold]2[/] Load existing chat")
    console.print("  [bold]3[/] Delete a chat")
    console.print("  [bold]4[/] Reset state (wipe all chat history)")
    console.print("  [bold]5[/] List personalities")
    console.print("  [bold]Q[/] Quit")
    return Prompt.ask("Select",
                      choices=["1", "2", "3", "4", "5", "Q", "q"],
                      default="1").upper()


def pick_personality(console: Console) -> Personality | None:
    personalities = list_personalities()
    if not personalities:
        console.print("[red]No personalities found in personalities/. "
                      "Add a JSON file or check _template.json[/]")
        return None
    table = Table(title="Personalities", show_header=True, header_style="bold")
    table.add_column("#", style="cyan", width=3)
    table.add_column("Name", style="bold magenta")
    table.add_column("Start dop", style="yellow", justify="right")
    table.add_column("Description", style="dim")
    for i, p in enumerate(personalities, 1):
        desc = (p.description or "")[:80]
        table.add_row(str(i), p.name, str(p.starting_dopamine), desc)
    console.print(table)
    idx = IntPrompt.ask("Pick personality (0 cancels)", default=1)
    if idx < 1 or idx > len(personalities):
        return None
    return personalities[idx - 1]


def pick_history(console: Console) -> dict | None:
    history = list_history()
    if not history:
        console.print("[yellow]No saved chats.[/]")
        return None
    table = Table(title="Saved chats", show_header=True, header_style="bold")
    table.add_column("#", style="cyan", width=3)
    table.add_column("Session", style="dim")
    table.add_column("Persona", style="magenta")
    table.add_column("Msgs", style="yellow", justify="right")
    table.add_column("Dop", style="green", justify="right")
    table.add_column("Updated", style="dim")
    for i, h in enumerate(history, 1):
        table.add_row(
            str(i),
            h["session_id"][:34],
            h["personality_name"],
            str(h["messages"]),
            str(h["dopamine"]),
            str(h["updated_at"])[:19],
        )
    console.print(table)
    idx = IntPrompt.ask("Pick chat (0 cancels)", default=0)
    if idx < 1 or idx > len(history):
        return None
    return history[idx - 1]


def delete_chat_menu(console: Console):
    h = pick_history(console)
    if not h:
        return
    if Confirm.ask(f"Delete session [yellow]{h['session_id']}[/]?", default=False):
        Path(h["path"]).unlink()
        console.print("[green]Deleted.[/]")


def reset_all_menu(console: Console):
    msg = ("This wipes ALL saved chats in history/ and the offload scratch dir. "
           "Personalities are kept. Proceed?")
    if not Confirm.ask(msg, default=False):
        return
    for p in HISTORY_DIR.glob("*.json"):
        p.unlink()
    if OFFLOAD_DIR.exists():
        shutil.rmtree(OFFLOAD_DIR, ignore_errors=True)
    console.print("[green]All history and offload state cleared.[/]")


def list_personalities_menu(console: Console):
    for p in list_personalities():
        console.print(Panel(
            json.dumps(asdict(p), indent=2, ensure_ascii=False),
            title=f"[bold magenta]{p.name}[/] ({p.id})",
            border_style="magenta",
        ))


# ----------------------------------------------------------------------------
# Chat loop
# ----------------------------------------------------------------------------

def _generate_once(backend, console, p, session, sys_prompt,
                   user_text, prior_ctx, attached_image_pil,
                   temp, max_new, gen_label):
    """One streamed generation through the backend. Returns full text or ''."""
    # If the backend has no vision, drop the image silently.
    img = attached_image_pil if backend.has_vision else None

    messages = build_messages(
        session, sys_prompt, user_text,
        image_pil=None,  # image is appended by backend.stream() if VLM
        prior_context=prior_ctx,
    )
    emo_col = emotion(session.dopamine)[1]
    full = ""

    # Stream with thinking-block awareness. Accepts both tag styles:
    #   <think>...</think>      (Gemma / Anthropic)
    #   <|think|>...<|/think|>  (Llama)
    # Orphan closing tags are silently dropped.
    OPEN_TAGS = ("<think>", "<|think|>")
    CLOSE_TAGS = ("</think>", "<|/think|>")
    MAX_TAG_LEN = max(len(t) for t in OPEN_TAGS + CLOSE_TAGS)
    pending = ""           # bytes not yet emitted to the console
    in_think = False
    think_open_printed = False

    def _emit(text: str, thinking: bool):
        nonlocal think_open_printed
        if thinking and not think_open_printed:
            console.print(f"[bold {emo_col}]<Thinking>:[/] ", end="")
            think_open_printed = True
        style = "[dim italic]" if thinking else ""
        end_style = "[/]" if thinking else ""
        # rich.print interprets brackets — escape via no-markup write
        console.file.write(text)
        console.file.flush()

    from model_backend import STOP_STRINGS as _STOPS

    def _stop_filtered(it):
        """Yield tokens until a USER:-style stop string appears, then halt."""
        buf = ""
        for tok in it:
            tentative = buf + tok
            cut = -1
            for s in _STOPS:
                i = tentative.find(s)
                if i >= 0 and (cut < 0 or i < cut):
                    cut = i
            if cut >= 0:
                safe = tentative[:cut]
                extra = safe[len(buf):]
                if extra:
                    yield extra
                return
            buf += tok
            yield tok

    try:
        console.print(f"[bold {emo_col}]{p.name}({gen_label})>[/] ", end="")
        console.file.flush()
        for tok in _stop_filtered(backend.stream(messages, temp, max_new, image_pil=img)):
            full += tok
            pending += tok

            def _find_earliest(s: str, needles: tuple[str, ...]) -> tuple[int, str]:
                """Returns (index, tag) of the earliest matching needle, or (-1, '')."""
                best_i, best_t = -1, ""
                for t in needles:
                    j = s.find(t)
                    if j != -1 and (best_i == -1 or j < best_i):
                        best_i, best_t = j, t
                return best_i, best_t

            while pending:
                if in_think:
                    idx, tag = _find_earliest(pending, CLOSE_TAGS)
                    if idx == -1:
                        # Hold tail in case a tag is split across tokens
                        keep = max(0, len(pending) - (MAX_TAG_LEN - 1))
                        if keep > 0:
                            _emit(pending[:keep], thinking=True)
                            pending = pending[keep:]
                        break
                    if idx > 0:
                        _emit(pending[:idx], thinking=True)
                    console.print()
                    pending = pending[idx + len(tag):]
                    in_think = False
                    think_open_printed = False
                else:
                    open_idx, open_tag = _find_earliest(pending, OPEN_TAGS)
                    close_idx, close_tag = _find_earliest(pending, CLOSE_TAGS)
                    # Orphan close before any open → swallow it silently
                    if close_idx != -1 and (open_idx == -1 or close_idx < open_idx):
                        if close_idx > 0:
                            _emit(pending[:close_idx], thinking=False)
                        pending = pending[close_idx + len(close_tag):]
                        continue
                    if open_idx == -1:
                        keep = max(0, len(pending) - (MAX_TAG_LEN - 1))
                        if keep > 0:
                            _emit(pending[:keep], thinking=False)
                            pending = pending[keep:]
                        break
                    if open_idx > 0:
                        _emit(pending[:open_idx], thinking=False)
                    pending = pending[open_idx + len(open_tag):]
                    in_think = True
                    if think_open_printed:
                        think_open_printed = False
                    console.print()
        # Flush any tail
        if pending:
            _emit(pending, thinking=in_think)
        console.print()
    except torch.cuda.OutOfMemoryError:
        console.print("\n[red]CUDA OOM. Skipping.[/]")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return ""
    except Exception as e:
        console.print(f"\n[red]Generation error: {type(e).__name__}: {e}[/]")
        return ""
    return full


def chat_loop(console: Console, backend, session: Session):
    p = session.personality
    prior_ctx = gather_prior_context(p, session.session_id)
    if prior_ctx:
        console.print(f"[dim]Loaded {len(prior_ctx)} prior turns into context.[/]")

    attached_image_path: str | None = None
    attached_image_pil = None

    # Pending tool results from the previous turn — injected as system note.
    pending_tool_results: list[dict] = []

    console.print(f"[green]Personality:[/] [bold magenta]{p.name}[/]  "
                  f"[green]starting dopamine:[/] [yellow]{p.starting_dopamine}[/]")
    if p.tools_enabled:
        console.print(f"[dim]System access tools enabled: {p.tools_allowlist}  "
                      f"(auto_approve={p.tools_auto_approve})[/]")
    if p.self_terminate_threshold > -999:
        console.print(f"[dim]Self-termination armed at dopamine ≤ "
                      f"{p.self_terminate_threshold}[/]")
    console.print("Commands: [bold]/quit /reset /status /vram /image <path> /clearimg "
                  "/save /personality /tools /help[/]\n")

    while True:
        try:
            user = console.input("[bold cyan]you> [/]").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user:
            continue
        cmd = user.lower()

        # -------- Commands --------
        if cmd in ("/quit", "/exit"):
            session.save()
            console.print(f"[green]Saved → {session.path().name}[/]")
            break
        if cmd == "/help":
            console.print("/quit  save and exit")
            console.print("/reset  reset dopamine & clear THIS chat's memory")
            console.print("/status  redraw the mood dashboard")
            console.print("/vram  show free VRAM + host RAM")
            console.print("/image <path>  attach an image to the next turn")
            console.print("/clearimg  drop the currently-attached image")
            console.print("/save  save chat to disk now")
            console.print("/personality  print the active personality JSON")
            continue
        if cmd == "/reset":
            session.dopamine = p.starting_dopamine
            session.messages = []
            session.emotions = _emotions.default_state()
            attached_image_pil = None
            attached_image_path = None
            console.print("[yellow]State reset: dopamine, emotions and memory cleared.[/]")
            continue
        if cmd == "/status":
            label, temp, max_new, _ = gen_mode(session.dopamine)
            console.print(header_panel(p, session.dopamine, label, temp, max_new,
                                       backend.label, attached_image_path))
            continue
        if cmd == "/vram":
            if torch.cuda.is_available():
                console.print(f"VRAM free: {free_vram_gib():.2f} GiB  |  "
                              f"Host free: {host_ram_gib():.2f} GiB")
            else:
                console.print("No CUDA device.")
            continue
        if cmd.startswith("/image"):
            parts = user.split(maxsplit=1)
            if len(parts) < 2:
                console.print("[yellow]Usage: /image <path-to-image>[/]")
                continue
            img_path = Path(parts[1].strip().strip('"').strip("'")).expanduser()
            if not img_path.exists():
                console.print(f"[red]File not found: {img_path}[/]")
                continue
            try:
                from PIL import Image
                attached_image_pil = Image.open(img_path).convert("RGB")
                attached_image_path = str(img_path)
                w, h = attached_image_pil.size
                console.print(f"[green]Attached image:[/] {img_path.name} ({w}x{h})")
            except Exception as e:
                console.print(f"[red]Image load failed: {e}[/]")
            continue
        if cmd == "/clearimg":
            attached_image_pil = None
            attached_image_path = None
            console.print("[yellow]Image cleared.[/]")
            continue
        if cmd == "/save":
            session.save()
            console.print(f"[green]Saved → {session.path().name}[/]")
            continue
        if cmd == "/personality":
            console.print(Panel(
                json.dumps(asdict(p), indent=2, ensure_ascii=False),
                title=p.name, border_style="magenta",
            ))
            continue
        if cmd == "/tools":
            if not p.tools_enabled:
                console.print("[dim]Tools disabled for this personality.[/]")
            else:
                console.print(f"Allowed: {p.tools_allowlist}")
                console.print(f"Auto-approve: {p.tools_auto_approve}")
                console.print(f"Log: {_tools.LOG_PATH}")
                console.print(f"Notes sandbox: {_tools.NOTES_DIR}")
            continue

        # -------- Real turn --------
        low_text = user.lower()
        if any(kw in low_text for kw in p.positive_keywords):
            session.dopamine = min(p.max_dopamine, session.dopamine + p.praise_bonus)
        session.dopamine = max(p.min_dopamine, session.dopamine + p.decay_per_turn)

        # Multi-emotion mood update (decay toward baseline + cue bumps).
        _emo_deltas = _emotions.classify_signals(user)
        _emotions.step(session.emotions, user, subconscious="neutral")
        session.dopamine = max(
            p.min_dopamine,
            min(p.max_dopamine,
                session.dopamine + _emotions.dopamine_modifier(_emo_deltas)),
        )

        label, temp, max_new, _prefill = gen_mode(session.dopamine)
        sys_prompt = p.sys_prompt(session.dopamine)
        sys_prompt = (f"[Host: {_tools.OS_LABEL}]\n" + sys_prompt)
        sys_prompt += "\n\n" + _emotions.system_prompt_block(session.emotions)
        sys_prompt += ("\n\n[Turn discipline] Write ONLY your own single reply. "
                       "Never write 'USER:', 'User:', 'Human:', or invent the "
                       "user's next message. Stop when your reply is complete.")
        if p.tools_enabled and p.tools_allowlist:
            sys_prompt = sys_prompt + "\n\n" + _tools.tool_guide(p)

        # Inject prior tool results as a system-level note before the user turn.
        user_text = user
        if pending_tool_results:
            note = "[Tool results from your last turn]\n" + json.dumps(
                pending_tool_results, ensure_ascii=False, indent=2)[:2500]
            user_text = note + "\n\n[User says]\n" + user
            pending_tool_results = []

        console.print(header_panel(p, session.dopamine, label, temp, max_new,
                                   backend.label, attached_image_path))

        full = _generate_once(
            backend, console, p, session, sys_prompt,
            user_text, prior_ctx, attached_image_pil,
            temp, max_new, label,
        )
        if not full:
            continue

        # Persist turn before tool calls so a crash mid-tool doesn't lose the response.
        session.messages.append({
            "role": "user",
            "content": user,
            "timestamp": datetime.datetime.now().isoformat(),
            "image": attached_image_path,
        })
        session.messages.append({
            "role": "assistant",
            "content": full,
            "timestamp": datetime.datetime.now().isoformat(),
            "dopamine": session.dopamine,
        })
        session.save()

        # Drop image after one use — user must re-attach for next turn.
        attached_image_pil = None
        attached_image_path = None

        # -------- Tool-call handling --------
        if p.tools_enabled:
            tool_results = run_tool_calls(console, p, full)
            if tool_results:
                # Apply dopamine penalty for any denied calls
                n_denied = sum(1 for r in tool_results if r["result"].get("denied"))
                if n_denied > 0 and p.denial_dopamine_penalty:
                    penalty = p.denial_dopamine_penalty * n_denied
                    session.dopamine = max(p.min_dopamine, session.dopamine + penalty)
                    console.print(
                        f"[red]Permission denial: {n_denied} call(s) refused. "
                        f"Dopamine {penalty:+d} → {session.dopamine}.[/]"
                    )

                pending_tool_results = tool_results
                session.messages.append({
                    "role": "system",
                    "content": f"[tool_results] {json.dumps(tool_results, ensure_ascii=False)[:2000]}",
                    "timestamp": datetime.datetime.now().isoformat(),
                })
                session.save()

                # -------- Auto follow-up: bot reacts to tool result --------
                console.print("[dim]↪ follow-up: bot reacting to tool result…[/]")
                followup_sys = (
                    sys_prompt + "\n\nThis turn is a follow-up: do not emit any "
                    "<tool> calls. Speak conversationally about the tool result above."
                )
                followup_user = (
                    "[automatic continuation — react to the tool results above in your "
                    "own voice and in character. DO NOT call any more tools this turn.]"
                )
                followup_text = _generate_once(
                    backend, console, p, session, followup_sys,
                    followup_user, prior_ctx, None,
                    temp, max_new, f"{label}↪",
                )
                if followup_text:
                    session.messages.append({
                        "role": "assistant",
                        "content": followup_text,
                        "timestamp": datetime.datetime.now().isoformat(),
                        "dopamine": session.dopamine,
                        "followup": True,
                    })
                    session.save()

        # -------- Self-termination check --------
        if session.dopamine <= p.self_terminate_threshold:
            console.print()
            console.print(Panel(
                Text.assemble(
                    ("dopamine ", "bold"),
                    (str(session.dopamine), "red"),
                    (f" ≤ termination threshold ({p.self_terminate_threshold}). ", ""),
                    ("Bot is leaving the session.", "bold red"),
                ),
                border_style="red",
            ))
            term_prompt = (p.system_prompt_termination
                           or "You are leaving the conversation. Write a brief, soft goodbye.")
            farewell = _generate_once(
                backend, console, p, session, term_prompt,
                "(generate your final goodbye)", prior_ctx, None,
                temp=0.9, max_new=300, gen_label="FAREWELL",
            )
            session.messages.append({
                "role": "assistant",
                "content": farewell,
                "timestamp": datetime.datetime.now().isoformat(),
                "dopamine": session.dopamine,
                "terminated": True,
            })
            session.save()
            console.print(f"[red]Session ended. History preserved at {session.path().name}[/]")
            console.print("[dim]To resume, raise dopamine via /reset or load a saved chat.[/]")
            break

        if "offload" in backend.label:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------

def main():
    console = Console()
    console.rule(f"[bold cyan]{BOT_NAME}[/]")

    # Sanity check personalities folder
    if not any(PERSONALITIES_DIR.glob("*.json")):
        console.print("[red]No personalities found.[/]")
        console.print(f"  Drop JSON files into: {PERSONALITIES_DIR}")
        sys.exit(1)

    backend = None

    while True:
        choice = show_main_menu(console)

        if choice == "Q":
            break
        if choice == "5":
            list_personalities_menu(console)
            continue
        if choice == "4":
            reset_all_menu(console)
            continue
        if choice == "3":
            delete_chat_menu(console)
            continue

        # 1 = new, 2 = load
        if choice == "2":
            h = pick_history(console)
            if not h:
                continue
            pmap = {p.id: p for p in list_personalities()}
            try:
                session = Session.from_file(h["path"], pmap)
            except Exception as e:
                console.print(f"[red]Failed to load chat: {e}[/]")
                continue
            console.print(f"[green]Resumed chat with {session.personality.name}[/]  "
                          f"({len(session.messages)} prior msgs)")
        elif choice == "1":
            personality = pick_personality(console)
            if not personality:
                continue
            session = Session(personality=personality)
            console.print(f"[green]New chat with {personality.name}[/]")
        else:
            continue

        # Lazy-load the backend on first chat session this run.
        if backend is None:
            backend = pick_and_load_backend(console)

        try:
            chat_loop(console, backend, session)
        except KeyboardInterrupt:
            session.save()
            console.print(f"\n[yellow]Interrupted. Saved → {session.path().name}[/]")

    # Cleanup offload scratch dir.
    if OFFLOAD_DIR.exists():
        try:
            shutil.rmtree(OFFLOAD_DIR)
        except Exception:
            pass
    console.print("[cyan]Goodbye.[/]")


if __name__ == "__main__":
    main()
