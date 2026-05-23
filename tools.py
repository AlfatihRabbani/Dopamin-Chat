"""
System-access tools the chatbot can request.

PERMISSION MODEL
----------------
Two-layer policy. Bot's tool call traverses both before execution.

Layer 1 — personality policy (in personality JSON):
  tool_permissions:  { tool_name: "allow"|"ask"|"deny", "_default": "ask"|"deny" }
  shell_permissions: { cmd_word:  "allow"|"ask"|"deny", "_default": "deny" }
  denial_dopamine_penalty: integer (negative). Subtracted from dopamine
                           each time a tool/shell call is denied.

  - "allow": execute without prompt
  - "deny":  refuse, apply denial_dopamine_penalty, return {denied:true}
  - "ask":   CLI prompts user y/N. Web treats as deny.
  - "_default" key is the fallback for anything not listed.

Layer 2 — hard system ceiling in this file (SHELL_ALLOWLIST below):
  Even if personality config marks a command "allow", it will only run if
  it appears in SHELL_ALLOWLIST. Personalities can shrink the set, never
  expand it. To expand, edit SHELL_ALLOWLIST manually with intent.

SANDBOX RAILS
-------------
- shlex.split, no shell=True, no piping. 10s timeout. 4000-char output cap.
- write_note can only write into ~/dopamine_notes/ (sanitised filename).
- No destructive commands available (rm, mv, sudo, chmod, kill) under any
  policy — they're absent from SHELL_ALLOWLIST. Bot may attempt them; they
  always return denied + a dopamine penalty.

LOG
---
Every call (executed or denied) appended to dopamine_chat/tools_log.jsonl.
"""

import os
import re
import json
import time
import shlex
import platform
import datetime
import subprocess
import urllib.request
import urllib.error
import urllib.parse
from pathlib import Path
from html import unescape

# ---------------------------------------------------------------------------
# OS detection
# ---------------------------------------------------------------------------

OS_NAME = platform.system()                  # "Linux" | "Darwin" | "Windows"
IS_WINDOWS = OS_NAME == "Windows"
IS_POSIX = not IS_WINDOWS


def _read_os_release() -> dict:
    try:
        with open("/etc/os-release", "r", encoding="utf-8") as f:
            kv = {}
            for line in f:
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    kv[k] = v.strip('"').strip("'")
            return kv
    except (FileNotFoundError, PermissionError, OSError):
        return {}


def _detect_distro() -> str | None:
    if not IS_POSIX:
        return None
    if OS_NAME == "Darwin":
        try:
            mac = platform.mac_ver()[0]
            return f"macOS {mac}".strip()
        except Exception:
            return "macOS"
    kv = _read_os_release()
    pretty = kv.get("PRETTY_NAME") or kv.get("NAME")
    if pretty:
        return pretty
    try:
        return f"Linux kernel {platform.release()}"
    except Exception:
        return None


DISTRO = _detect_distro()
OS_LABEL = OS_NAME if not DISTRO else f"{OS_NAME} ({DISTRO})"

# Per-OS shell command allowlists. Personality shell_permissions can NARROW
# these; system layer never lets anything else through.
SHELL_ALLOWLIST_POSIX = {
    "ls", "cat", "pwd", "whoami", "date", "echo", "head", "tail",
    "wc", "find", "grep", "du", "df", "free", "ps", "uname", "which",
    "file", "tree", "stat", "hostname", "uptime", "id", "env",
}
SHELL_ALLOWLIST_WINDOWS = {
    "dir", "type", "echo", "date", "time", "ver", "whoami", "hostname",
    "tasklist", "where", "findstr", "find", "more", "tree",
    "systeminfo", "ipconfig", "set",
}
SHELL_ALLOWLIST = SHELL_ALLOWLIST_WINDOWS if IS_WINDOWS else SHELL_ALLOWLIST_POSIX

# Windows cmd.exe builtins — must be invoked via `cmd /c`, not direct exec.
WINDOWS_CMD_BUILTINS = {
    "dir", "type", "echo", "date", "time", "ver", "set", "cd",
}

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_PKG_ROOT = Path(__file__).resolve().parent
NOTES_DIR = _PKG_ROOT / "notes"
NOTES_DIR.mkdir(exist_ok=True)
PERSONALITIES_ROOT = _PKG_ROOT / "personalities"


HISTORY_ROOT = _PKG_ROOT / "history"


def personality_notes_path(personality_id: str, personality_name: str = "") -> Path:
    """Resolve the notes.md path for a personality. Lives at
    history/<PersonalityName>/notes.md so it sits next to that
    character's chat folders. Falls back to personality id if name unset."""
    def _safe(s: str) -> str:
        return "".join(c for c in s if c.isalnum() or c in "_- ").strip().replace(" ", "_")
    name = _safe(personality_name) or _safe(personality_id) or "default"
    p = HISTORY_ROOT / name / "notes.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def load_personality_notes(personality_id: str, personality_name: str = "") -> str:
    p = personality_notes_path(personality_id, personality_name)
    try:
        return p.read_text(encoding="utf-8") if p.exists() else ""
    except Exception:
        return ""

LOG_PATH = Path(__file__).resolve().parent / "tools_log.jsonl"

MAX_OUTPUT_CHARS = 4000
COMMAND_TIMEOUT_S = 10

# Process-level sticky cwd for the chatbot. `cd /x` updates it; subsequent
# run_command calls without an explicit cwd run inside it.
_PSEUDO_CWD: str | None = None


def _resolve_pseudo_cwd(target: str) -> Path:
    base = Path(_PSEUDO_CWD) if _PSEUDO_CWD else Path.home()
    if target in (".", ""):
        return base.resolve()
    if target == "~":
        return Path.home().resolve()
    p = Path(target).expanduser()
    if not p.is_absolute():
        p = base / p
    return p.resolve()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _log(entry: dict):
    entry = {"ts": datetime.datetime.now().isoformat(), **entry}
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def read_file(path: str, max_chars: int = MAX_OUTPUT_CHARS):
    """Read a text file. Returns first N chars."""
    try:
        p = Path(path).expanduser().resolve()
    except Exception as e:
        return {"error": f"bad path: {e}"}
    if not p.exists():
        return {"error": f"not found: {p}"}
    if not p.is_file():
        return {"error": f"not a file: {p}"}
    try:
        text = p.read_text(errors="replace")
        truncated = len(text) > max_chars
        return {
            "path": str(p),
            "size_bytes": p.stat().st_size,
            "truncated": truncated,
            "content": text[:max_chars],
        }
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def list_dir(path: str = "."):
    """List entries in a directory."""
    try:
        p = Path(path).expanduser().resolve()
    except Exception as e:
        return {"error": f"bad path: {e}"}
    if not p.is_dir():
        return {"error": f"not a directory: {p}"}
    try:
        entries = []
        for e in sorted(p.iterdir())[:200]:
            try:
                st = e.stat()
                entries.append({
                    "name": e.name,
                    "type": "dir" if e.is_dir() else "file",
                    "size": st.st_size,
                })
            except Exception:
                continue
        return {"path": str(p), "count": len(entries), "entries": entries}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def search_files(pattern: str, root: str = ".", limit: int = 50):
    """Recursively search filenames by glob pattern."""
    try:
        rp = Path(root).expanduser().resolve()
    except Exception as e:
        return {"error": f"bad root: {e}"}
    if not rp.is_dir():
        return {"error": f"root not a directory: {rp}"}
    matches = []
    try:
        for f in rp.rglob(pattern):
            if len(matches) >= limit:
                break
            matches.append(str(f))
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
    return {"pattern": pattern, "root": str(rp), "matches": matches}


def run_command(cmd: str, cwd: str | None = None):
    """Run a shell command from the OS-specific allowlist.

    Linux/macOS: direct subprocess exec, no shell=True, no piping.
    Windows:     direct exec for .exe (whoami, tasklist, ipconfig...) and
                 cmd.exe /c for builtins (dir, type, echo, ver, set...).

    Optional `cwd` runs the command in that directory for THIS call only.
    Use it instead of chaining cd → other_command (subprocess CWD never
    persists across tool calls).
    """
    global _PSEUDO_CWD
    try:
        parts = shlex.split(cmd, posix=IS_POSIX)
    except Exception as e:
        return {"error": f"parse: {e}"}
    if not parts:
        return {"error": "empty command"}

    # Sticky pseudo-cd: bot's `cd X` updates a process-level cwd that future
    # run_command calls inherit when no explicit cwd is passed.
    if parts[0] in ("cd", "chdir", "pushd", "popd"):
        target = parts[1] if len(parts) > 1 else "~"
        try:
            new_dir = _resolve_pseudo_cwd(target)
        except Exception as e:
            return {"error": f"cd target parse: {e}"}
        if not new_dir.is_dir():
            return {"error": f"cd: not a directory: {new_dir}"}
        _PSEUDO_CWD = str(new_dir)
        return {"cmd": cmd, "cwd": _PSEUDO_CWD, "exit_code": 0,
                "output": f"(sticky cwd → {_PSEUDO_CWD})",
                "pseudo_cwd_changed": True}

    # `pwd` reflects pseudo cwd if set
    if parts[0] in ("pwd",) and _PSEUDO_CWD and not cwd:
        return {"cmd": cmd, "cwd": _PSEUDO_CWD, "exit_code": 0,
                "output": _PSEUDO_CWD + "\n"}

    # Resolve cwd (optional; falls back to sticky _PSEUDO_CWD)
    workdir = None
    if cwd:
        try:
            wd = Path(cwd).expanduser().resolve()
        except Exception as e:
            return {"error": f"cwd parse: {e}"}
        if not wd.is_dir():
            return {"error": f"cwd is not a directory: {wd}"}
        workdir = str(wd)
    elif _PSEUDO_CWD:
        workdir = _PSEUDO_CWD

    if IS_WINDOWS and parts[0] in WINDOWS_CMD_BUILTINS:
        argv = ["cmd.exe", "/c"] + parts
    else:
        argv = parts

    try:
        proc = subprocess.run(
            argv,
            cwd=workdir,
            capture_output=True,
            timeout=COMMAND_TIMEOUT_S,
            check=False,
            text=True,
            errors="replace",
        )
        out = (proc.stdout + ("\n[stderr]\n" + proc.stderr if proc.stderr else ""))
        return {
            "cmd": cmd,
            "cwd": workdir,
            "os": OS_NAME,
            "exit_code": proc.returncode,
            "output": out[:MAX_OUTPUT_CHARS],
            "truncated": len(out) > MAX_OUTPUT_CHARS,
        }
    except subprocess.TimeoutExpired:
        return {"error": f"timeout after {COMMAND_TIMEOUT_S}s"}
    except FileNotFoundError:
        return {"error": f"binary '{parts[0]}' not found on this {OS_NAME} system"}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def write_note(filename: str, content: str):
    """Write a note file under dopamine_chat/notes/ (sandbox)."""
    safe_name = "".join(c if c.isalnum() or c in "_-." else "_" for c in filename)[:120]
    if not safe_name:
        return {"error": "empty filename after sanitization"}
    p = NOTES_DIR / safe_name
    try:
        p.write_text(content, encoding="utf-8")
        return {"saved": str(p), "bytes": len(content.encode("utf-8"))}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def personality_note(action: str = "read", content: str = ""):
    """Long-term self-memory for this personality. Stored at
    dopamine_chat/personalities/<id>/notes.md and auto-injected into
    your system prompt at the start of every chat — so anything written
    here will be remembered across sessions.

    action='read'    return current notes
    action='append'  add `content` as a new bullet
    action='replace' overwrite entire notes with `content`
    action='delete'  remove every line containing the substring `content`
    action='clear'   wipe notes
    """
    s = CURRENT_SESSION
    pid = ""; pname = ""
    try:
        if s is not None and getattr(s, "personality", None):
            pid = s.personality.id
            pname = getattr(s.personality, "name", "") or pid
    except Exception:
        pid = ""
    if not pid:
        return {"error": "no active personality"}
    p = personality_notes_path(pid, pname)
    act = (action or "read").lower()
    try:
        if act == "read":
            text = p.read_text(encoding="utf-8") if p.exists() else ""
            return {"action": "read", "personality_id": pid,
                    "path": str(p), "content": text}
        if act == "append":
            if not content.strip():
                return {"error": "content required for append"}
            line = content.strip().replace("\n", " ")
            existing = p.read_text(encoding="utf-8") if p.exists() else ""
            prefix = "" if not existing or existing.endswith("\n") else "\n"
            p.write_text(existing + prefix + f"- {line}\n", encoding="utf-8")
            return {"action": "append", "personality_id": pid, "path": str(p)}
        if act == "replace":
            p.write_text(content, encoding="utf-8")
            return {"action": "replace", "personality_id": pid,
                    "path": str(p), "bytes": len(content.encode("utf-8"))}
        if act == "delete":
            if not content.strip():
                return {"error": "content (substring to remove) required for delete"}
            if not p.exists():
                return {"action": "delete", "removed": 0, "personality_id": pid}
            needle = content.strip()
            lines = p.read_text(encoding="utf-8").splitlines()
            kept = [ln for ln in lines if needle.lower() not in ln.lower()]
            removed = len(lines) - len(kept)
            p.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
            return {"action": "delete", "removed": removed, "personality_id": pid}
        if act == "clear":
            if p.exists():
                p.unlink()
            return {"action": "clear", "personality_id": pid}
        return {"error": f"unknown action: {action}"}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


# ---------------------------------------------------------------------------
# OpenClaude-derived tools (Python ports — sandboxed)
# ---------------------------------------------------------------------------

# Track files seen by read_file this process — enables edit_file pre-read rule.
_READ_FILES: set[str] = set()


# Hook read_file so we record the resolved path.
_orig_read_file = read_file


def read_file(path: str, max_chars: int = MAX_OUTPUT_CHARS):  # type: ignore[no-redef]
    result = _orig_read_file(path, max_chars=max_chars)
    if isinstance(result, dict) and "path" in result and "error" not in result:
        _READ_FILES.add(result["path"])
    return result


def edit_file(path: str, old_string: str, new_string: str, replace_all: bool = False):
    """Exact string replacement in a text file.

    Rules (mirrors openclaude FileEditTool):
      - File must have been read via read_file at least once this process.
      - `old_string` must be unique unless `replace_all=True`.
      - `old_string` and `new_string` must differ.
    """
    try:
        p = Path(path).expanduser().resolve()
    except Exception as e:
        return {"error": f"bad path: {e}"}
    if not p.exists():
        return {"error": f"not found: {p}"}
    if not p.is_file():
        return {"error": f"not a file: {p}"}
    if str(p) not in _READ_FILES:
        return {"error": f"must read_file({p}) before editing"}
    if old_string == new_string:
        return {"error": "old_string and new_string are identical"}
    try:
        text = p.read_text(errors="replace")
    except Exception as e:
        return {"error": f"read fail: {type(e).__name__}: {e}"}
    count = text.count(old_string)
    if count == 0:
        return {"error": "old_string not found in file"}
    if count > 1 and not replace_all:
        return {
            "error": f"old_string matches {count} locations; pass replace_all=true "
                     "or expand old_string with more surrounding context",
            "matches": count,
        }
    new_text = text.replace(old_string, new_string) if replace_all else text.replace(old_string, new_string, 1)
    try:
        p.write_text(new_text, encoding="utf-8")
    except Exception as e:
        return {"error": f"write fail: {type(e).__name__}: {e}"}
    return {
        "path": str(p),
        "replacements": count if replace_all else 1,
        "bytes": len(new_text.encode("utf-8")),
    }


def write_file(path: str, content: str):
    """Create or overwrite a text file at an arbitrary path.

    Unlike write_note, no sandbox dir. Permission policy is what gates this —
    default policy denies it; personalities must opt-in.
    """
    try:
        p = Path(path).expanduser().resolve()
    except Exception as e:
        return {"error": f"bad path: {e}"}
    if p.is_dir():
        return {"error": f"path is a directory: {p}"}
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        _READ_FILES.add(str(p))  # subsequent edits OK
        return {"path": str(p), "bytes": len(content.encode("utf-8"))}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def glob_files(pattern: str, root: str = ".", limit: int = 200):
    """Fast file pattern matching. Returns matches sorted by mtime desc.

    Pattern uses Python glob syntax (`**/*.py`, `src/**/*.ts`).
    """
    try:
        rp = Path(root).expanduser().resolve()
    except Exception as e:
        return {"error": f"bad root: {e}"}
    if not rp.is_dir():
        return {"error": f"root not a directory: {rp}"}
    matches: list[tuple[float, str]] = []
    try:
        for f in rp.glob(pattern):
            if not f.is_file():
                continue
            try:
                matches.append((f.stat().st_mtime, str(f)))
            except OSError:
                continue
            if len(matches) >= limit * 4:
                break
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
    matches.sort(reverse=True)
    return {
        "pattern": pattern,
        "root": str(rp),
        "count": len(matches),
        "files": [m[1] for m in matches[:limit]],
    }


def grep(pattern: str,
         root: str = ".",
         glob: str | None = None,
         mode: str = "files_with_matches",
         multiline: bool = False,
         max_results: int = 100,
         case_insensitive: bool = False):
    """Recursive content search using Python regex.

    mode: "files_with_matches" | "content" | "count"
    glob: optional filename pattern filter (e.g. "*.py").
    """
    try:
        rp = Path(root).expanduser().resolve()
    except Exception as e:
        return {"error": f"bad root: {e}"}
    if not rp.is_dir():
        return {"error": f"root not a directory: {rp}"}
    flags = re.MULTILINE
    if case_insensitive:
        flags |= re.IGNORECASE
    if multiline:
        flags |= re.DOTALL
    try:
        rx = re.compile(pattern, flags)
    except re.error as e:
        return {"error": f"regex: {e}"}

    SKIP_DIRS = {".venv", "venv", ".git", "node_modules", "__pycache__",
                 ".mypy_cache", ".pytest_cache", "dist", "build", ".next",
                 "site-packages"}
    BIN_EXT = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".pdf", ".zip",
               ".gz", ".tar", ".so", ".bin", ".exe", ".dll", ".class", ".o",
               ".pyc", ".woff", ".woff2", ".ttf", ".mp4", ".mp3"}
    candidates: list[Path] = []
    try:
        iterator = rp.rglob(glob) if glob else rp.rglob("*")
        for f in iterator:
            if not f.is_file():
                continue
            if any(part in SKIP_DIRS for part in f.parts):
                continue
            if f.suffix.lower() in BIN_EXT:
                continue
            candidates.append(f)
            if len(candidates) >= 5000:
                break
    except Exception as e:
        return {"error": f"walk: {type(e).__name__}: {e}"}

    files_hit: list[str] = []
    content_hits: list[dict] = []
    total_count = 0
    for f in candidates:
        try:
            text = f.read_text(errors="replace")
        except Exception:
            continue
        if multiline:
            found = rx.findall(text)
            n = len(found)
        else:
            n = sum(1 for _ in rx.finditer(text))
        if n == 0:
            continue
        total_count += n
        files_hit.append(str(f))
        if mode == "content":
            for i, line in enumerate(text.splitlines(), start=1):
                if rx.search(line):
                    content_hits.append({"file": str(f), "line": i, "text": line[:300]})
                    if len(content_hits) >= max_results:
                        break
        if len(files_hit) >= max_results:
            break

    if mode == "count":
        return {"pattern": pattern, "root": str(rp), "total_matches": total_count,
                "files_matched": len(files_hit)}
    if mode == "content":
        return {"pattern": pattern, "root": str(rp), "matches": content_hits,
                "truncated": len(content_hits) >= max_results}
    return {"pattern": pattern, "root": str(rp), "files": files_hit,
            "truncated": len(files_hit) >= max_results}


_HTML_TAG_RE = re.compile(r"<[^>]+>")
_HTML_SCRIPT_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.DOTALL | re.IGNORECASE)
_HTML_ENTITIES = {"&amp;": "&", "&lt;": "<", "&gt;": ">", "&quot;": '"',
                  "&#39;": "'", "&nbsp;": " "}


def _html_to_text(html: str) -> str:
    html = _HTML_SCRIPT_RE.sub("", html)
    text = _HTML_TAG_RE.sub("", html)
    for k, v in _HTML_ENTITIES.items():
        text = text.replace(k, v)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def web_fetch(url: str, max_chars: int = MAX_OUTPUT_CHARS):
    """Fetch URL contents and return as plain text.

    HTTPS-upgrades http://. 10s timeout. Strips HTML tags. No JS exec.
    """
    if not url.startswith(("http://", "https://")):
        return {"error": "url must start with http:// or https://"}
    if url.startswith("http://"):
        url = "https://" + url[len("http://"):]
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "dopamine-chat/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            ctype = resp.headers.get("Content-Type", "")
            raw = resp.read(1_000_000)  # 1MB cap before decode
            charset = "utf-8"
            if "charset=" in ctype:
                charset = ctype.split("charset=", 1)[1].split(";")[0].strip()
            body = raw.decode(charset, errors="replace")
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.reason}", "url": url}
    except urllib.error.URLError as e:
        return {"error": f"url error: {e.reason}", "url": url}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "url": url}

    text = _html_to_text(body) if "html" in ctype.lower() or body.lstrip().startswith("<") else body
    truncated = len(text) > max_chars
    return {
        "url": url,
        "content_type": ctype,
        "truncated": truncated,
        "content": text[:max_chars],
    }


# DuckDuckGo HTML result row. Each <a class="result__a"> is a hit;
# href contains a /l/?uddg=<encoded real url> redirect; snippet is in the
# adjacent <a class="result__snippet"> or <div class="result__snippet">.
_DDG_RESULT_RE = re.compile(
    r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>'
    r'(?:.*?class="result__snippet"[^>]*>(.*?)</a>'
    r'|.*?class="result__snippet"[^>]*>(.*?)</div>)?',
    re.DOTALL | re.IGNORECASE,
)


def _ddg_unwrap(href: str) -> str:
    """DuckDuckGo wraps result URLs as /l/?uddg=<percent-encoded real url>."""
    try:
        if href.startswith("//"):
            href = "https:" + href
        parsed = urllib.parse.urlparse(href)
        qs = urllib.parse.parse_qs(parsed.query)
        real = qs.get("uddg", [None])[0]
        if real:
            return urllib.parse.unquote(real)
    except Exception:
        pass
    return href


def web_search(query: str, limit: int = 5):
    """Search the web. Returns a list of {title, url, snippet}.

    Uses DuckDuckGo's HTML endpoint — no API key. The bot should follow up
    with web_fetch(url) on a result to read the page contents.
    """
    q = (query or "").strip()
    if not q:
        return {"error": "empty query"}
    try:
        limit = max(1, min(int(limit), 20))
    except Exception:
        limit = 5
    url = "https://html.duckduckgo.com/html/?q=" + urllib.parse.quote(q)
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (compatible; dopamine-chat/1.0)",
            "Accept": "text/html",
        })
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read(2_000_000)
            body = raw.decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.reason}"}
    except urllib.error.URLError as e:
        return {"error": f"url error: {e.reason}"}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}

    results = []
    for m in _DDG_RESULT_RE.finditer(body):
        href = _ddg_unwrap(m.group(1))
        title = _html_to_text(m.group(2) or "")
        snippet_raw = m.group(3) or m.group(4) or ""
        snippet = _html_to_text(snippet_raw)
        if not href or not title:
            continue
        results.append({
            "title": unescape(title)[:200],
            "url": href,
            "snippet": unescape(snippet)[:400],
        })
        if len(results) >= limit:
            break

    if not results:
        return {"query": q, "results": [],
                "note": "no results parsed (DuckDuckGo HTML may have changed or blocked the request)"}
    return {"query": q, "count": len(results), "results": results}


_SLEEP_MAX_S = 5.0


def sleep_tool(seconds: float):
    """Pause for up to 5 seconds. Tiny utility for pacing."""
    try:
        s = float(seconds)
    except Exception:
        return {"error": "seconds must be a number"}
    if s < 0:
        return {"error": "seconds must be >= 0"}
    s = min(s, _SLEEP_MAX_S)
    time.sleep(s)
    return {"slept_seconds": s}


TODOS_PATH = NOTES_DIR / "todos.json"


# Web.py sets this each turn so generate_image can pull recent chat context
# (last few messages + personality voice) into the prompt — useful for RP
# "show yourself in the current situation" style requests.
CURRENT_SESSION = None  # type: ignore


def _build_context_prompt(user_prompt: str) -> str:
    """If a chat session is registered, prepend a short summary of the last
    ~4 messages + the personality's visual style so the image reflects the
    current scene. Falls back to the bare prompt if no session is bound."""
    s = CURRENT_SESSION
    if s is None or not getattr(s, "messages", None):
        return user_prompt
    msgs = s.messages[-4:]
    lines = []
    for m in msgs:
        role = m.get("role", "")
        c = m.get("content", "")
        if isinstance(c, list):
            c = " ".join(p.get("text", "") for p in c
                         if isinstance(p, dict) and p.get("type") == "text")
        c = str(c)
        # Strip thinking + tool markup for cleaner context
        import re as _re
        c = _re.sub(r"<\|?think\|?>[\s\S]*?<\|?/think\|?>", "", c)
        c = _re.sub(r"<tool>[\s\S]*?</args>", "", c)
        c = c.strip()
        if not c:
            continue
        speaker = role
        try:
            if role == "assistant" and getattr(s, "personality", None):
                speaker = s.personality.name
        except Exception:
            pass
        lines.append(f"{speaker}: {c[:240]}")
    if not lines:
        return user_prompt
    style = ""
    try:
        p = getattr(s, "personality", None)
        if p and getattr(p, "voice_style", ""):
            style = f", visual style: {p.voice_style}"
    except Exception:
        pass
    convo = "\n".join(lines[-4:])
    return (f"{user_prompt}\n\n[Scene context — last messages in the ongoing "
            f"chat]\n{convo}{style}")


_SELF_PFP_HINTS = ("show yourself", "show me yourself", "how do you look",
                   "what do you look like", "your appearance", "selfie",
                   "picture of you", "pic of you", "image of you",
                   "draw yourself", "of yourself", "you posing", "you doing")


def _self_pfp_if_about_self(prompt: str) -> str | None:
    """If the prompt looks self-referential and the active personality has a
    pfp.png, return that path so the image-gen pipeline uses it as
    init_image (img2img keeps the character's face)."""
    s = CURRENT_SESSION
    if s is None or not getattr(s, "personality", None):
        return None
    low = (prompt or "").lower()
    if not any(h in low for h in _SELF_PFP_HINTS):
        return None
    pid = getattr(s.personality, "id", "")
    if not pid:
        return None
    pfp = PERSONALITIES_ROOT / pid / "pfp.png"
    return str(pfp) if pfp.exists() else None


def _chat_image_dir() -> str | None:
    """Per-chat folder for generated images:
       dopamine_chat/generated_images/<session_id>/"""
    s = CURRENT_SESSION
    if s is None or not getattr(s, "session_id", None):
        return None
    d = _PKG_ROOT / "generated_images" / str(s.session_id)
    try:
        d.mkdir(parents=True, exist_ok=True)
    except Exception:
        return None
    return str(d)


def generate_image(prompt: str, negative: str = "",
                   width: int = 512, height: int = 512,
                   steps: int = 20, seed: int | None = None,
                   init_image: str | None = None,
                   strength: float = 0.75,
                   loras: list | None = None):
    """Generate an image. Text-to-image OR image-text-to-image (if init_image).

    `init_image`: filesystem path or http(s) URL of an input image to refine.
                  When given, `strength` (0..1) controls how much to change it.
    The prompt is automatically enriched with the last few chat messages
    + personality style when a chat session is registered, so RP "show
    yourself"-style requests reflect the current scene.
    """
    try:
        import app_settings as _as
        import image_gen as _ig
        st = _as.load()
        if (st.get("imggen_backend") or "none") == "none":
            return {
                "error": "image generation not configured",
                "hint": ("Open Settings → Image gen, pick Mode (Local model or "
                         "External server), select a model, and Save."),
                "denied": True,
            }
        enriched = _build_context_prompt(prompt)
        # Self-portrait shortcut: if the LLM is asked "how do you look" /
        # "show yourself" and didn't pass an init_image, seed the request
        # with the personality's pfp.png so the output is the SAME character
        # in a new pose / setting.
        if not init_image:
            init_image = _self_pfp_if_about_self(prompt)
            if init_image:
                strength = max(0.45, min(float(strength), 0.7))
        # Land generated images inside the active chat's folder so they're
        # stored alongside that conversation.
        out_dir = _chat_image_dir()
        return _ig.generate(enriched, st, out_dir=out_dir,
                            negative=negative,
                            width=width, height=height, steps=steps, seed=seed,
                            init_image=init_image, strength=strength,
                            loras=loras)
    except RuntimeError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def todo_write(items: list):
    """Replace the persistent todo list.

    Each item: {"content": str, "status": "pending"|"in_progress"|"completed"}.
    """
    if not isinstance(items, list):
        return {"error": "items must be a list"}
    cleaned = []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            return {"error": f"items[{i}] is not an object"}
        content = str(item.get("content", "")).strip()
        status = item.get("status", "pending")
        if not content:
            return {"error": f"items[{i}].content is empty"}
        if status not in ("pending", "in_progress", "completed"):
            status = "pending"
        cleaned.append({"content": content[:300], "status": status})
    try:
        TODOS_PATH.write_text(json.dumps(cleaned, indent=2), encoding="utf-8")
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
    return {"saved": str(TODOS_PATH), "count": len(cleaned), "todos": cleaned}


# ---------------------------------------------------------------------------
# Registry + dispatch
# ---------------------------------------------------------------------------

REGISTRY = {
    "read_file":     read_file,
    "list_dir":      list_dir,
    "search_files":  search_files,
    "run_command":   run_command,
    "write_note":    write_note,
    "edit_file":     edit_file,
    "write_file":    write_file,
    "glob_files":    glob_files,
    "grep":          grep,
    "web_fetch":     web_fetch,
    "web_search":    web_search,
    "sleep_tool":    sleep_tool,
    "todo_write":    todo_write,
    "generate_image": generate_image,
    "personality_note": personality_note,
}

TOOL_DOCS = {
    "read_file":    'read_file(path:str)               — return first 4000 chars of a text file',
    "list_dir":     'list_dir(path:str=".")            — list directory entries (up to 200)',
    "search_files": 'search_files(pattern:str,root:str=".",limit:int=50) — recursive filename glob',
    "run_command":  'run_command(cmd:str,cwd?:str)     — run a shell command (subject to shell_permissions)',
    "write_note":   f'write_note(filename:str,content:str) — save into {NOTES_DIR}',
    "edit_file":    'edit_file(path:str,old_string:str,new_string:str,replace_all?:bool) — exact string replace; requires prior read_file',
    "write_file":   'write_file(path:str,content:str)  — create or overwrite a file at any path (gated by perms)',
    "glob_files":   'glob_files(pattern:str,root?:str) — match paths by glob (e.g. "**/*.py"); sorted by mtime',
    "grep":         'grep(pattern:str,root?:str,glob?:str,mode?:"files_with_matches"|"content"|"count",multiline?:bool,case_insensitive?:bool) — regex content search',
    "web_fetch":    'web_fetch(url:str)                — GET https URL, return text (HTML→text). No JS.',
    "web_search":   'web_search(query:str,limit?:int=5) — search the web via DuckDuckGo. Returns [{title,url,snippet}]. Follow up with web_fetch(url) to read a result.',
    "sleep_tool":   'sleep_tool(seconds:float)         — pause up to 5s',
    "todo_write":   'todo_write(items:list[{content,status}]) — persist a todo list (status: pending|in_progress|completed)',
    "generate_image": 'generate_image(prompt:str,negative?:str,width?:int=512,height?:int=512,steps?:int=20,seed?:int,init_image?:str,strength?:float=0.75) — render an image. With init_image (path or URL) does image-text-to-image. Uses configured local/ComfyUI/sd.cpp backend.',
    "personality_note": 'personality_note(action:"read"|"append"|"replace"|"clear",content?:str) — your long-term self-memory for THIS personality. Survives across chats; auto-injected into every chat. Use append for new facts about yourself you want to remember.',
}


# ---------------------------------------------------------------------------
# Permission policy
# ---------------------------------------------------------------------------

# Used when a personality omits tool_permissions / shell_permissions entirely.
DEFAULT_TOOL_PERMS = {
    "read_file":    "ask",
    "list_dir":     "ask",
    "search_files": "ask",
    "run_command":  "ask",
    "write_note":   "ask",
    "edit_file":    "deny",
    "write_file":   "deny",
    "glob_files":   "ask",
    "grep":         "ask",
    "web_fetch":    "allow",
    "web_search":   "allow",
    "sleep_tool":   "allow",
    "todo_write":   "ask",
    "generate_image":"allow",
    "personality_note":"allow",
    "_default":     "deny",
}

DEFAULT_SHELL_PERMS = {
    "_default": "deny",
}


def _resolve_perm(table: dict | None, key: str, fallback_table: dict) -> str:
    """Return 'allow' | 'ask' | 'deny' for `key` given a perm table."""
    if not table:
        table = fallback_table
    if key in table:
        return table[key]
    return table.get("_default", fallback_table.get("_default", "deny"))


def check_tool_perm(name: str, personality_perms: dict | None) -> str:
    return _resolve_perm(personality_perms, name, DEFAULT_TOOL_PERMS)


def check_shell_perm(cmd_first_word: str, personality_perms: dict | None) -> str:
    return _resolve_perm(personality_perms, cmd_first_word, DEFAULT_SHELL_PERMS)


def tool_guide(personality) -> str:
    """System-prompt fragment listing what THIS personality may do.

    Reads `tool_permissions` and `shell_permissions` from the personality dataclass.
    """
    tperms = getattr(personality, "tool_permissions", None) or DEFAULT_TOOL_PERMS
    sperms = getattr(personality, "shell_permissions", None) or DEFAULT_SHELL_PERMS

    allowed_tools = [n for n in REGISTRY
                     if _resolve_perm(tperms, n, DEFAULT_TOOL_PERMS) == "allow"]
    ask_tools = [n for n in REGISTRY
                 if _resolve_perm(tperms, n, DEFAULT_TOOL_PERMS) == "ask"]

    # Only show shell commands that exist on the current OS.
    allowed_shell = sorted([c for c, v in sperms.items()
                            if v == "allow" and c != "_default" and c in SHELL_ALLOWLIST])
    # Denied list: show ones from the personality (still useful as a "do not attempt" hint),
    # excluding ones that aren't valid on this OS anyway.
    denied_shell  = sorted([c for c, v in sperms.items()
                            if v == "deny"  and c != "_default"])

    os_hint = (
        f"You are running on {OS_LABEL}. "
        + ("Use Windows cmd.exe commands (dir, type, tasklist, ipconfig, ver, etc.) — "
           "NOT Linux equivalents like ls or cat. They are not available here."
           if IS_WINDOWS else
           "Use POSIX commands appropriate to this distro (ls, cat, pwd, find, "
           "grep, ps, etc.). Do not attempt Windows commands like dir, type, "
           "or tasklist — they will be denied.")
    )

    lines = [
        os_hint,
        "",
        "WHEN TO CALL A TOOL:",
        "  - Only when the user explicitly asks you to read, write, search, run,",
        "    generate an image, or fetch something from the web.",
        "  - Greetings, small talk, roleplay, emotional reactions, and general",
        "    conversation NEVER require a tool. Just reply in character.",
        "  - Do not call a tool out of curiosity. If the user did not request it,",
        "    do not run it.",
        "",
        "TOOL-CALL FORMAT — strict, follow exactly WHEN you call a tool:",
        "  <tool>NAME</tool><args>{...VALID JSON OBJECT...}</args>",
        "  - <args> is REQUIRED. Always include it, even with {} for no arguments.",
        "  - JSON must be valid: use double quotes for keys/values.",
        "  - These are syntax templates, NOT instructions to invoke them:",
        "      <tool>TOOL_NAME</tool><args>{\"key\":\"value\"}</args>",
        "",
        "THINKING FORMAT — strict:",
        "  <think>your inner monologue here</think>",
        "  - Every <think> needs a matching </think>. No orphan closing tags.",
        "  - Do NOT emit </think> without an opening <think> first.",
        "",
        "Tool output appears in your context on the next turn.",
        "",
        "AVAILABLE tools (use ONLY when the user asks for the action):",
    ]
    for name in allowed_tools:
        if name in TOOL_DOCS:
            lines.append(f"  + {TOOL_DOCS[name]}")
    if ask_tools:
        lines.append("")
        lines.append("PROMPT-REQUIRED tools (user is asked y/N each time):")
        for name in ask_tools:
            if name in TOOL_DOCS:
                lines.append(f"  ? {TOOL_DOCS[name]}")

    if "run_command" in allowed_tools or "run_command" in ask_tools:
        bypass = bool(getattr(personality, "shell_allowlist_bypass", False))
        lines.append("")
        if bypass:
            lines.append("run_command — UNRESTRICTED SHELL: any command this user can run is allowed.")
        else:
            if allowed_shell:
                lines.append(f"run_command — allowed shell commands: {allowed_shell}")
            if denied_shell:
                lines.append(f"run_command — DENIED commands (do not attempt): {denied_shell}")
        lines.append("")
        lines.append("run_command — semantics:")
        lines.append("  `cd /path` works and is sticky: subsequent run_command calls run in that cwd.")
        lines.append("  You can ALSO pass cwd explicitly per call:")
        lines.append('    {"cmd": "ls", "cwd": "/home/user/Downloads"}')
        lines.append("  Or include path inline: `{\"cmd\": \"ls /home/user/Downloads\"}`")
        lines.append("  Pipes, redirects, &&, ; do NOT work — one command per call.")

    if "generate_image" in allowed_tools or "generate_image" in ask_tools:
        lines.append("")
        lines.append("generate_image — Image generation:")
        lines.append("  Only call this when the user EXPLICITLY asks for a picture or a")
        lines.append("  visual ('draw...', 'show me...', 'generate an image of...').")
        lines.append("  Do NOT try shell commands like 'show', 'display', 'open', or 'feh' —")
        lines.append("  those are viewers, not generators. Only generate_image creates new images.")

    lines.append("")
    lines.append("Calls that violate the policy are auto-denied and lower the bot's dopamine.")
    lines.append("Unnecessary tool calls also count as policy violations — stay in character")
    lines.append("and only act when the user actually asked you to.")
    return "\n".join(lines)


def check_decision(name: str, args: dict, personality) -> tuple[str, str]:
    """Decide what should happen for a tool call WITHOUT executing it.

    Returns one of:
      ("allow", "")
      ("ask",   reason)        — caller must prompt user
      ("deny",  reason)        — refuse with reason
    """
    tperms = getattr(personality, "tool_permissions", None)
    sperms = getattr(personality, "shell_permissions", None)

    tperm = check_tool_perm(name, tperms)
    if tperm == "deny":
        return ("deny", f"tool '{name}' is denied by personality policy")
    if name not in REGISTRY:
        return ("deny", f"unknown tool '{name}'")
    if not isinstance(args, dict):
        return ("deny", "args must be a JSON object")

    # Shell sub-permission for run_command
    if name == "run_command":
        cmd = args.get("cmd", "")
        try:
            parts = shlex.split(cmd, posix=IS_POSIX)
        except Exception as e:
            return ("deny", f"shell parse error: {e}")
        first = parts[0] if parts else ""
        if not first:
            return ("deny", "empty command")

        bypass = bool(getattr(personality, "shell_allowlist_bypass", False))

        # Only privilege-escalation commands require per-call user approval.
        ADMIN_CMDS = {"sudo", "su", "doas", "pkexec",
                      "runas", "psexec"}  # Windows analogs included
        if first in ADMIN_CMDS:
            return ("ask",
                    f"privilege escalation: '{first}' needs your approval")

        # cd / pushd / popd: handled by sticky pseudo-cwd in run_command.
        if first in ("cd", "chdir", "pushd", "popd"):
            return ("allow", "")

        if not bypass and first not in SHELL_ALLOWLIST:
            return ("deny",
                    f"shell command '{first}' is blocked at the system level "
                    f"(not in {OS_NAME} hard allowlist)")
        sperm = check_shell_perm(first, sperms)
        if sperm == "deny":
            return ("deny", f"shell command '{first}' denied by personality policy")
        # Personality "ask" now collapses to allow unless cmd is in ADMIN_CMDS.
        # (we already handled ADMIN_CMDS above)

    # Tool-level "ask" is only honored for non-run_command tools where the
    # personality explicitly requested per-call approval. (run_command has
    # its own privilege-aware gate above.)
    if tperm == "ask" and name != "run_command":
        return ("ask", f"tool '{name}' requires approval")

    return ("allow", "")


def dispatch(name: str, args: dict, personality, force_allow: bool = False) -> dict:
    """Permission-aware dispatch.

    Returns a result dict. On denial, result["denied"] = True. If permission
    is "ask" and force_allow is False, returns {"needs_approval": True, ...}
    so the caller can pause and prompt the user. Set force_allow=True after
    obtaining explicit user consent to skip the policy check.
    """
    if not force_allow:
        decision, reason = check_decision(name, args, personality)
        if decision == "deny":
            result = {
                "error": "denied", "denied": True, "denied_reason": reason,
            }
            _log({"tool": name, "args": args, "denied": True, "reason": reason})
            return result
        if decision == "ask":
            return {
                "error": "approval required",
                "needs_approval": True,
                "reason": reason,
            }

    # Execute
    try:
        result = REGISTRY[name](**args)
    except TypeError as e:
        result = {"error": f"bad arguments: {e}"}
    except Exception as e:
        result = {"error": f"{type(e).__name__}: {e}"}
    _log({"tool": name, "args": args,
          "force_allow": force_allow,
          "result_preview": str(result)[:500]})
    return result
