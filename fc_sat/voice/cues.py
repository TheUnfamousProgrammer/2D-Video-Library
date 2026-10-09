"""Cue times for 'Why does your voice sound so weird on recordings?'."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

FPS = 60
TAIL = 1.0


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9-]", "", text.lower())


@dataclass
class Cues:
    t: dict[str, float]
    end: float
    n_frames: int
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
        "head1": g("head"), "nobody": g("nobody"), "heard": g("heard"), "once": g("once"),
        "mum": g("mum"), "friends": g("friends"), "ex": g("ex"), "voice2": g("voice", 2), "know": g("know"),
        "notes1": g("notes"), "yeah": g("yeah"), "that": g("that"), "when": g("when"), "ways": g("ways"),
        "through1": g("through"), "everyone": g("everyone"), "through2": g("through", 2), "skull": g("skull"),
        "bones": g("bones"), "deep": g("deep"), "frequencies": g("frequencies"), "inner": g("inner"),
        "head2": g("head", 2), "bass1": g("bass"), "boost1": g("boost"), "recording": g("recording"),
        "air2": g("air", 2), "no": g("no"), "bass2": g("bass", 2), "thinner": g("thinner"), "higher": g("higher"),
        "nasal": g("nasal"), "filter": g("filter"), "life": g("life"), "plot": g("plot"), "study": g("study"),
        "researchers": g("researchers"), "secretly": g("secretly"), "lineup": g("lineup"), "most": g("most"),
        "themselves": g("themselves"), "rated": g("rated"), "more": g("more"), "listeners": g("listeners"),
        "so3": g("so", 3), "hate1": g("hate"), "you_hate": g("hate", 2), "unfiltered": g("unfiltered"),
        "so4": g("so", 4), "how": g("how"), "rerecorded": g("re-recorded"),
    }
    end = t1["re-recorded1"] + TAIL
    cues = Cues(c, end, int(round(end * FPS)))
    cues.sfx = _sfx(cues)
    return cues


def rerecord_times(cues: Cues, section: str) -> list[float]:
    c = cues.t
    if section == "hook":
        a, b, n = c["yeah"], c["when"] - 0.15, 6
    else:
        a, b, n = c["how"], cues.end - 0.75, 9
    return [a + (b - a) * (k / (n - 1)) ** 1.4 for k in range(n)]


def _sfx(cues: Cues) -> list[tuple[float, str, float]]:
    c = cues.t
    ev: list[tuple[float, str, float]] = [(0.0, "hum", 0.0)]
    ev.append((c["nobody"], "pop", 0.3))
    ev.append((c["once"], "lock", 0.0))
    for k in ("mum", "friends", "ex"):
        ev.append((c[k] + 0.05, "stamp", 1.0 if k == "ex" else 0.6))
    ev.append((c["voice2"], "whoosh", 0.0))
    ev.append((c["notes1"], "tap", 0.0))
    ev.append((c["yeah"] - 0.05, "scratch", 0.0))
    for tt in rerecord_times(cues, "hook"):
        ev.append((tt, "tick", 0.6))
    ev.append((c["when"] - 0.1, "whoosh", 0.0))
    ev.append((c["through1"], "air", 0.0))
    ev.append((c["skull"], "thump", 0.0))
    ev.append((c["deep"], "sub", 0.0))
    ev.append((c["bass1"], "sub", 0.0))
    ev.append((c["no"], "cut", 0.0))
    ev.append((c["nasal"], "boing", 0.0))
    ev.append((c["filter"], "toggle", 0.0))
    ev.append((c["plot"] - 0.05, "drop", 0.0))
    for k in range(10):
        ev.append((c["secretly"] + 0.1 * k, "pop", 0.3 + 0.05 * k))
    ev.append((c["more"], "ding", 0.0))
    ev.append((c["unfiltered"], "toggle", 0.0))
    for tt in rerecord_times(cues, "outro"):
        ev.append((tt, "tick", 0.7))
    ev.append((cues.end - 0.5, "settle", 0.0))
    return sorted(ev)
