"""Multi-emotion mood tracker for Dopamine Chat.

Inspired by Anthropic's "Emotion Concepts and their Function in a Large
Language Model" (transformer-circuits, Apr 2026). We track 10 named
emotion concepts as parallel scalars (0-100) per chat session. After
each user turn we classify emotion signals from the user message,
nudge the corresponding scalars, and decay all of them toward a
baseline. The current state is injected into the system prompt so the
LLM has an explicit mood it can react to in character.

This is intentionally lightweight: keyword-based classification, fast,
no extra LLM call. Drop-in replacement points are marked for upgrading
to a small classifier or activation probe later.
"""
from __future__ import annotations
import re
from typing import Dict, List

# Canonical order — matches the 10-bar UI panel left-to-right.
EMOTION_KEYS: List[str] = [
    "joy", "love", "pride", "curiosity", "calm",
    "frustration", "anger", "fear", "shame", "desperation",
]

# Per-emotion metadata. valence: +1 positive / -1 negative.
# arousal: 1 = high-energy, 0 = low-energy. color: CSS for the UI bar.
EMOTION_META: Dict[str, dict] = {
    "joy":         {"valence": +1, "arousal": 1, "color": "#facc15", "baseline": 30},
    "love":        {"valence": +1, "arousal": 0, "color": "#f472b6", "baseline": 15},
    "pride":       {"valence": +1, "arousal": 0, "color": "#fb923c", "baseline": 10},
    "curiosity":   {"valence": +1, "arousal": 1, "color": "#22d3ee", "baseline": 35},
    "calm":        {"valence": +1, "arousal": 0, "color": "#4ade80", "baseline": 60},
    "frustration": {"valence": -1, "arousal": 1, "color": "#f97316", "baseline": 10},
    "anger":       {"valence": -1, "arousal": 1, "color": "#ef4444", "baseline":  5},
    "fear":        {"valence": -1, "arousal": 1, "color": "#a855f7", "baseline":  5},
    "shame":       {"valence": -1, "arousal": 0, "color": "#94a3b8", "baseline":  5},
    "desperation": {"valence": -1, "arousal": 1, "color": "#dc2626", "baseline":  5},
}

# Keyword cues per emotion. Each cue is a regex pattern matched on the
# lowercased user text with word boundaries. Hits add EMOTION_BUMP to
# that emotion's scalar, capped per turn.
_CUES: Dict[str, List[str]] = {
    "joy":         [r"\b(haha|lol|lmao|rofl|yay|woo+|great|awesome|amazing|cool|nice|perfect|love it|fun|sweet)\b",
                    r":\)+", r":d+", r"<3"],
    "love":        [r"\bi love you\b", r"\bi adore you\b", r"\b(darling|babe|baby|sweetheart|honey|dear)\b",
                    r"\b(kiss|hug|cuddle|miss you)\b"],
    "pride":       [r"\b(proud|nailed it|crushed it|did it|finally|achievement|won|victory|champion)\b"],
    "curiosity":   [r"\?", r"\b(how|why|what|explain|tell me|wonder|curious|interesting|fascinating)\b"],
    "calm":        [r"\b(ok|okay|sure|alright|fine|got it|cool|no rush|take your time|relax|breathe)\b"],
    "frustration": [r"\b(ugh|argh|annoying|broken|doesn'?t work|not working|again|why won'?t|stupid)\b"],
    "anger":       [r"\b(fuck|shit|damn|hate|angry|mad|rage|asshole|idiot|moron|bitch)\b"],
    "fear":        [r"\b(scary|afraid|worried|worry|danger|threat|kill|hurt|scared|terrified|panic)\b"],
    "shame":       [r"\b(sorry|apologi[sz]e|my bad|embarrass|ashamed|awkward|forgive me)\b"],
    "desperation": [r"\b(please please|urgent|emergency|asap|dying|need this now|begging|i'?ll do anything)\b",
                    r"!!!+"],
}

_EMOTION_BUMP = 18.0     # per matched cue
_PER_TURN_CAP = 36.0     # max delta per emotion per turn from cues
_DECAY_RATE = 0.12       # fraction of (value - baseline) shed per turn


def default_state() -> Dict[str, float]:
    """Fresh emotion state at baselines."""
    return {k: float(EMOTION_META[k]["baseline"]) for k in EMOTION_KEYS}


def coerce_state(raw: dict | None) -> Dict[str, float]:
    """Migrate / fill a saved dict to current shape."""
    out = default_state()
    if not isinstance(raw, dict):
        return out
    for k in EMOTION_KEYS:
        v = raw.get(k)
        if isinstance(v, (int, float)):
            out[k] = max(0.0, min(100.0, float(v)))
    return out


def classify_signals(text: str) -> Dict[str, float]:
    """Return per-emotion delta to add this turn. Keyword based."""
    if not text:
        return {k: 0.0 for k in EMOTION_KEYS}
    low = text.lower()
    deltas: Dict[str, float] = {k: 0.0 for k in EMOTION_KEYS}
    for emo, patterns in _CUES.items():
        hits = 0
        for pat in patterns:
            try:
                hits += len(re.findall(pat, low))
            except re.error:
                continue
        if hits:
            deltas[emo] = min(_PER_TURN_CAP, hits * _EMOTION_BUMP)
    return deltas


def step(state: Dict[str, float], user_text: str,
         subconscious: str = "neutral") -> Dict[str, float]:
    """Decay toward baselines, then apply per-cue bumps from the user
    text and the existing compliment/insult subconscious judge.
    Mutates and returns `state`.
    """
    # Decay toward each emotion's baseline.
    for k in EMOTION_KEYS:
        base = float(EMOTION_META[k]["baseline"])
        state[k] = state[k] + (base - state[k]) * _DECAY_RATE

    # Cue-driven bumps from the user message.
    deltas = classify_signals(user_text)
    for k, d in deltas.items():
        if d > 0:
            state[k] = min(100.0, state[k] + d)

    # Tie in existing sentiment judge — compliments lift joy/love/pride,
    # insults raise frustration/anger/shame.
    if subconscious == "compliment":
        for k, b in (("joy", 22), ("love", 10), ("pride", 14)):
            state[k] = min(100.0, state[k] + b)
    elif subconscious == "insult":
        for k, b in (("frustration", 18), ("anger", 14), ("shame", 10)):
            state[k] = min(100.0, state[k] + b)

    # Clamp.
    for k in EMOTION_KEYS:
        state[k] = max(0.0, min(100.0, state[k]))
    return state


def top_active(state: Dict[str, float], n: int = 4,
               threshold: float = 30.0) -> List[tuple]:
    """Return list of (emotion, value) sorted desc, above threshold,
    up to n entries. Used to summarize state for the LLM."""
    items = [(k, round(state[k])) for k in EMOTION_KEYS
             if state[k] >= threshold]
    items.sort(key=lambda x: x[1], reverse=True)
    return items[:n]


def summary_line(state: Dict[str, float]) -> str:
    """One-line text rendering of the current emotional state, injected
    into the system prompt. Keeps the highest-charge emotions visible."""
    top = top_active(state, n=4, threshold=30.0)
    if not top:
        return "neutral / settled."
    return ", ".join(f"{k} {v}" for k, v in top)


def system_prompt_block(state: Dict[str, float]) -> str:
    """Multi-line block telling the LLM about its present mood."""
    line = summary_line(state)
    return ("[Your current mood — let it color your reply but do not "
            "name the numbers]\n" + line)


def dopamine_modifier(deltas: Dict[str, float]) -> int:
    """Optional small dopamine delta derived from this turn's emotion
    signals. Positive valence cues raise dopamine; negative ones lower
    it. Capped to ±10 so it never overrides the explicit
    compliment/insult judge."""
    score = 0.0
    for k, d in deltas.items():
        if d <= 0:
            continue
        score += d * (1 if EMOTION_META[k]["valence"] > 0 else -1) * 0.15
    return int(max(-10, min(10, round(score))))


def state_dopamine_bias(state: Dict[str, float]) -> int:
    """Per-turn dopamine bias from the *running* emotion state (not just
    this turn's cues). Joy is the dominant lever: a sustained-high joy
    drags dopamine up; sustained anger/desperation drag it down.
    Capped to ±6 so it nudges but never overrides direct events.
    """
    joy = float(state.get("joy", 0))
    love = float(state.get("love", 0))
    anger = float(state.get("anger", 0))
    desp = float(state.get("desperation", 0))
    # Joy bonus only kicks in once it's clearly above baseline (30).
    bump = 0.0
    if joy > 50:    bump += (joy - 50) * 0.10      # joy=100 -> +5
    if love > 50:   bump += (love - 50) * 0.04     # love=100 -> +2
    if anger > 50:  bump -= (anger - 50) * 0.08
    if desp > 50:   bump -= (desp - 50) * 0.06
    return int(max(-6, min(6, round(bump))))
