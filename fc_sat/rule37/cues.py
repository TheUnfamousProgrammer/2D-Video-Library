"""Every visual and audio cue, keyed to the narration's word times."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

FPS = 60
TAIL = 1.1  # seconds after the last word, so the loop pose can settle
N = 100
R = 37


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9-]", "", text.lower())


def success_curve(n: int = N) -> np.ndarray:
    """P(pick the best) when the first r of n are rejected, r = 0..n-1 (exact)."""
    out = np.zeros(n)
    out[0] = 1.0 / n
    for r in range(1, n):
        out[r] = (r / n) * sum(1.0 / i for i in range(r, n))
    return out


def _perm_success(rng: np.random.Generator) -> tuple[list[int], int]:
    """Scores where the research max is beaten first by the overall best, landing mid-pack."""
    while True:
        s = rng.permutation(N) + 1
        bar = s[:R].max()
        later = [j for j in range(R, N) if s[j] > bar]
        if later and s[later[0]] == N and 52 <= later[0] <= 64 and bar >= 80:
            return s.tolist(), later[0]


def _perm_catch(rng: np.random.Generator) -> tuple[list[int], int]:
    while True:
        s = rng.permutation(N) + 1
        top = int(np.argmax(s))
        if 18 <= top <= 24:
            return s.tolist(), top


@dataclass
class Cues:
    t: dict[str, float]
    end: float
    n_frames: int
    scores_a: list[int]
    pick_a: int
    scores_b: list[int]
    best_b: int
    swipes: list[float]
    sfx: list[tuple[float, str, float]] = field(default_factory=list)


def load(words_path: Path) -> Cues:
    words = json.loads(words_path.read_text())
    seen: dict[str, int] = {}
    t: dict[str, float] = {}
    t1: dict[str, float] = {}
    for w in words:
        k = _norm(w["text"])
        seen[k] = seen.get(k, 0) + 1
        t[f"{k}{seen[k]}"] = w["t0"]
        t1[f"{k}{seen[k]}"] = w["t1"]

    def g(word: str, nth: int = 1) -> float:
        return t[f"{word}{nth}"]

    c = {
        "math": g("math"), "dump": g("dump"), "date": g("date"), "on": g("on"), "purpose": g("purpose"),
        "even": g("even"), "perfect": g("perfect"), "its": g("its"), "secretary": g("secretary"),
        "because": g("because"), "terrible": g("terrible"), "things": g("things"), "meet": g("meet"),
        "fumble": g("fumble"), "theres": g("theres"), "back": g("back"), "so": g("so"),
        "hundred": g("hundred"), "people2": g("people", 3), "respectfully": g("respectfully"),
        "touch": g("touch"), "but1": g("but"), "do": g("do"), "first2": g("first", 2),
        "t37b": g("thirty-seven", 2), "pure": g("pure"), "zero": g("zero"), "vibes": g("vibes"),
        "then": g("then"), "lock": g("lock"), "all": g("all"), "them": g("them"), "that": g("that"),
        "t37c": g("thirty-seven", 3), "shot": g("shot"), "best": g("best"), "no2": g("no", 2),
        "better": g("better"), "dating": g("dating"), "eighteen": g("eighteen"), "forty": g("forty"),
        "your": g("your"), "twentysix": g("twenty-six"), "but2": g("but", 2), "catch": g("catch"),
        "also": g("also"), "t37d": g("thirty-seven", 4), "one3": g("one", 3), "phase2": g("phase", 2),
        "and3": g("and", 2), "let": g("let"), "go": g("go"), "so2": g("so", 2), "how": g("how"),
        "rejecting": g("rejecting"),
    }
    end = t1[f"rejecting1"] + TAIL
    n_frames = int(round(end * FPS))
    rng = np.random.default_rng(37)
    scores_a, pick_a = _perm_success(rng)
    scores_b, best_b = _perm_catch(rng)

    # Hook swipes accelerate from "dump" to "date": 14 cards, geometric spacing.
    k = 14
    span = c["date"] - c["dump"]
    ratios = 0.82 ** np.arange(k - 1)
    steps = ratios / ratios.sum() * span
    swipes = [c["dump"]] + list(c["dump"] + np.cumsum(steps))
    cues = Cues(c, end, n_frames, scores_a, pick_a, scores_b, best_b, swipes)
    cues.sfx = _sfx(cues)
    return cues


# --- derived schedules shared by picture and sound -------------------------------------------

def reveal_a(cues: Cues, i: int) -> float:
    """When bar i of the first chart starts to grow."""
    c = cues.t
    if i < R:
        return c["pure"] + i * 0.04
    k = cues.pick_a
    if i <= k:
        step = (c["all"] - c["then"]) / (k - R + 1)
        return c["then"] + (i - R + 1) * step
    return c["that"] + (i - k - 1) * (0.75 / (N - k - 1))


def reveal_b(cues: Cues, i: int) -> float:
    c = cues.t
    if i < R:
        # The crown bar lands on "one".
        start = c["one3"] - 0.08 - cues.best_b * 0.03
        return start + i * 0.03
    return c["and3"] + (i - R) * (0.72 / (N - R))


def _sfx(cues: Cues) -> list[tuple[float, str, float]]:
    c = cues.t
    ev: list[tuple[float, str, float]] = [(0.0, "whoosh_soft", 0.0)]
    for i, s in enumerate(cues.swipes):
        ev.append((s, "swipe", i / len(cues.swipes)))
    ev.append((c["on"] - 0.02, "slam", 0.0))
    ev.append((c["even"], "sparkle", 0.0))
    ev.append((c["perfect"] + 0.35, "swipe", 0.0))
    ev.append((c["perfect"] + 0.38, "bonk", 0.0))
    ev.append((c["its"], "whoosh", 0.0))
    ev.append((c["secretary"], "pop", 0.6))
    ev.append((c["terrible"], "scribble", 0.0))
    ev.append((c["things"] + 0.15, "pop", 0.9))
    q0 = c["meet"]
    for j in range(6):
        ev.append((q0 + j * 0.62, "step", j))
    ev.append((c["theres"], "gate", 0.0))
    ev.append((c["back"], "lock", 0.0))
    for d in range(19):
        ev.append((grid_pop(cues, d), "pop", d / 18))
    ev.append((c["touch"] - 0.12, "grass", 0.0))
    ev.append((c["but1"], "grass_back", 0.0))
    ev.append((c["first2"] - 0.25, "whoosh", 0.0))
    run = 0
    for i in range(N):
        t = reveal_a(cues, i)
        h = cues.scores_a[i]
        if i < R:
            rec = h > run
            run = max(run, h)
            ev.append((t, "record" if rec else "tick", h / N))
        elif i < cues.pick_a:
            ev.append((t, "tick_low", h / N))
        elif i == cues.pick_a:
            ev.append((t, "ding", 0.0))
        else:
            if i % 3 == 0:
                ev.append((t, "tick_soft", h / N))
    ev.append((c["zero"], "fade_down", 0.0))
    ev.append((c["vibes"], "wobble", 0.0))
    ev.append((c["best"], "crown", 0.0))
    ev.append((c["no2"], "riser", 0.0))
    ev.append((curve_peak_time(cues), "pop", 1.0))
    for j in range(12):
        ev.append((label_time(cues, j), "tick", 0.3 + j / 24))
    ev.append((c["your"], "slide", 0.0))
    ev.append((c["twentysix"], "confetti", 0.0))
    ev.append((c["but2"], "drop", 0.0))
    ev.append((c["also"], "pop", 0.5))
    for i in range(N):
        t = reveal_b(cues, i)
        if i == cues.best_b:
            ev.append((t, "crown", 0.0))
        elif i < R:
            ev.append((t, "tick", cues.scores_b[i] / N))
        elif i % 2 == 0:
            ev.append((t, "tick_low", cues.scores_b[i] / N))
    ev.append((c["go"] + 0.05, "sad", 0.0))
    ev.append((c["go"] + 0.45, "crack", 0.0))
    ev.append((c["so2"], "whoosh", 0.0))
    for j in range(spin_count(cues)):
        ev.append((spin_time(cues, j), "tick", 0.5))
    ev.append((cues.end - 0.55, "settle", 0.0))
    return sorted(ev)


def grid_pop(cues: Cues, diag: int) -> float:
    c = cues.t
    return c["so"] + 0.05 + diag * (c["people2"] + 0.2 - c["so"]) / 19


def curve_peak_time(cues: Cues) -> float:
    return cues.t["no2"] + 0.85


def label_time(cues: Cues, j: int) -> float:
    c = cues.t
    return c["eighteen"] + j * (c["forty"] - c["eighteen"]) / 11


def spin_count(cues: Cues) -> int:
    return 14


def spin_time(cues: Cues, j: int) -> float:
    c = cues.t
    a, b = c["how"], cues.end - 0.6
    x = j / (spin_count(cues) - 1)
    return a + (b - a) * (x ** 1.6)
