"""Cue times for 'Why the other line is always faster', read from the narration's word times."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

FPS = 60
TAIL = 1.0

# Queueing model shown in the fix scene: 3 registers, 80 % busy, 1 min mean service.
SERVERS = 3
RHO = 0.8


def mm1_wait(rho: float) -> float:
    """Mean time in queue for one M/M/1 lane (service time 1)."""
    return rho / (1 - rho)


def mmc_wait(c: int, rho: float) -> float:
    """Mean time in queue for one shared line feeding c registers (Erlang C, service time 1)."""
    a = c * rho
    s = sum(a ** k / math.factorial(k) for k in range(c))
    top = a ** c / math.factorial(c) / (1 - rho)
    erlang_c = top / (s + top)
    return erlang_c / (c - a)


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9-]", "", text.lower())


@dataclass
class Cues:
    t: dict[str, float]
    end: float
    n_frames: int
    trials: list[int]  # winning lane (0..2) of each 3-lane race; lane 1 is yours
    sfx: list[tuple[float, str, float]] = field(default_factory=list)


def _trials(seed: int = 3) -> list[int]:
    rng = np.random.default_rng(seed)
    while True:
        w = rng.integers(0, 3, size=30).tolist()
        if sum(1 for x in w if x == 1) == 10 and w[0] != 1:
            return w


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
        "always": g("always"), "slowest": g("slowest"), "universe": g("universe"), "targeting": g("targeting"),
        "its": g("its"), "math": g("math"),
        "three1": g("three"), "odds": g("odds"), "fastest": g("fastest"), "one1": g("one"), "three2": g("three", 2),
        "so1": g("so"), "two": g("two"), "some": g("some"), "wins": g("wins"),
        "ten": g("ten"), "store": g("store"), "lose": g("lose"), "ninety": g("ninety"),
        "main": g("main"), "energy": g("energy"), "npc": g("npc"), "luck": g("luck"),
        "traffic": g("traffic"), "worse": g("worse"), "researchers": g("researchers"), "video": g("video"),
        "seventy": g("seventy"), "said": g("said"), "faster": g("faster"), "when1": g("when"),
        "actually": g("actually"), "slower": g("slower"),
        "heres": g("heres"), "slow": g("slow"), "cars": g("cars"), "ages": g("ages"), "when3": g("when", 3),
        "blow": g("blow"), "seconds": g("seconds"), "so2": g("so", 2), "receipts": g("receipts"), "ls": g("ls"),
        "fix": g("fix"), "single": g("single"), "feeding": g("feeding"), "register": g("register"),
        "busy": g("busy"), "cut": g("cut"), "half": g("half"), "banks": g("banks"), "airports": g("airports"),
        "snake": g("snake"), "so3": g("so", 3), "honest": g("honest"), "lane2": g("lane", 2),
        "switcher": g("switcher"),
    }
    end = t1["switcher1"] + TAIL
    cues = Cues(c, end, int(round(end * FPS)), _trials())
    cues.sfx = _sfx(cues)
    return cues


# --- schedules shared by picture and sound ----------------------------------------------------

def side_serves(cues: Cues) -> list[tuple[float, int]]:
    """(time, lane) when a side-lane customer finishes during the hook. Lane 1 (yours) never moves."""
    out = []
    end = cues.t["three1"]
    for lane, (first, gap) in ((0, (0.55, 1.05)), (2, (0.95, 1.2))):
        s = first
        while s < end - 0.3:
            out.append((s, lane))
            s += gap
    return sorted(out)


def trial_time(cues: Cues, k: int) -> float:
    c = cues.t
    a, b = c["two"] - 0.1, c["wins"] + 0.2
    return a + (b - a) * k / 30


def ten_finish(cues: Cues, lane: int) -> float:
    """When each of the ten lanes lights up as finished; yours (lane 3) is last."""
    c = cues.t
    order = [7, 1, 9, 5, 0, 8, 2, 6, 4]
    if lane == 3:
        return c["lose"] + 1.3
    a, b = c["lose"], c["lose"] + 1.15
    return a + (b - a) * order.index(lane) / 8


def hops(cues: Cues) -> list[tuple[float, int]]:
    """Outro lane-switching hops: (time, lane)."""
    c = cues.t
    a = c["lane2"]
    return [(a - 0.05, 0), (a + 0.3, 2), (a + 0.62, 1)]


def _sfx(cues: Cues) -> list[tuple[float, str, float]]:
    c = cues.t
    ev: list[tuple[float, str, float]] = [(0.0, "bwomp", 0.0)]
    for s, lane in side_serves(cues):
        ev.append((s, "beep", lane / 2))
    ev.append((c["universe"], "thunder", 0.0))
    ev.append((c["targeting"], "zap", 0.0))
    ev.append((c["math"], "pop", 0.9))
    ev.append((c["three1"], "whoosh", 0.0))
    ev.append((c["odds"] + 0.1, "riser", 0.0))
    ev.append((c["one1"], "slam", 0.0))
    for k, w in enumerate(cues.trials):
        ev.append((trial_time(cues, k), "tick" if w == 1 else "tick_low", 0.4 + 0.02 * k))
    ev.append((c["ten"] - 0.15, "whoosh", 0.0))
    for lane in range(10):
        if lane != 3:
            ev.append((ten_finish(cues, lane), "beep", lane / 9))
    ev.append((ten_finish(cues, 3), "bonk", 0.0))
    ev.append((c["main"], "sparkle", 0.0))
    ev.append((c["npc"], "fade_down", 0.0))
    ev.append((c["traffic"] - 0.1, "drop", 0.0))
    ev.append((c["video"], "pop", 0.5))
    for k in range(7):
        ev.append((c["seventy"] + 0.08 * k, "pop", 0.3 + 0.08 * k))
    ev.append((c["slower"], "slam", 0.0))
    for k in range(9):
        ev.append((c["cars"] + 0.17 * k, "car", 0.0))
    for k in range(4):
        ev.append((c["blow"] + 0.15 * k, "car_fast", 0.0))
    ev.append((c["receipts"] - 0.2, "printer", 0.0))
    ev.append((c["fix"], "ding", 0.0))
    ev.append((c["single"], "whoosh", 0.0))
    ev.append((c["half"], "slam", 0.0))
    ev.append((c["banks"], "pop", 0.6))
    ev.append((c["airports"], "pop", 0.8))
    ev.append((c["snake"], "whoosh", 0.0))
    for tt, lane in hops(cues):
        ev.append((tt, "swipe", 0.3))
    ev.append((cues.end - 0.5, "settle", 0.0))
    return sorted(ev)
