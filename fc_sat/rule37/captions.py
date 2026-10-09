"""Burned-in word-by-word captions: short uppercase chunks, the spoken word lit up."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import skia

ROOT = Path(__file__).resolve().parents[2]
WHITE = (0xFF, 0xFF, 0xFF)
PINK = (0xFF, 0x4F, 0x8B)
GOLD = (0xFF, 0xC2, 0x4B)
CENTER_Y = 1470.0
MAX_W = 800.0
SIZE = 84.0
MAX_WORDS = 3

NUMBERS = {"thirty-seven": "37", "eighteen": "18", "forty": "40", "twenty-six": "26", "hundred": "100"}
# Words that get the gold accent while spoken, beyond the numbers.
ACCENT = {"DUMP", "PURPOSE", "PERFECT", "TERRIBLE", "FUMBLE", "GRASS", "VIBES", "ALL", "BEST", "CATCH", "ALSO", "GO",
          "REJECTING", "LOCK", "SECRETARY"}


@dataclass
class Token:
    text: str
    t0: float
    t1: float
    brk: bool  # chunk ends after this token


def tokens(words: list[dict]) -> list[Token]:
    out: list[Token] = []
    i = 0
    while i < len(words):
        w = words[i]
        raw = w["text"]
        core = re.sub(r"[^A-Za-z0-9'-]", "", raw).lower()
        tail = re.sub(r"^[A-Za-z0-9'-]+", "", raw)
        brk = bool(re.search(r"[.,?!]|\.\.\.", tail))
        text = NUMBERS.get(core, core).upper()
        t0, t1 = w["t0"], w["t1"]
        nxt = words[i + 1]["text"].lower().strip(".,?") if i + 1 < len(words) else ""
        if core == "thirty-seven" and nxt == "percent":
            text = "37%"
            t1 = words[i + 1]["t1"]
            brk = bool(re.search(r"[.,?!]", words[i + 1]["text"]))
            i += 1
        if core == "hundred" and out and out[-1].text == "A":
            out.pop()
        if "?" in tail:
            text += "?"
        out.append(Token(text, t0, t1, brk))
        i += 1
    return out


def chunks(toks: list[Token]) -> list[list[Token]]:
    """Split at punctuation and long pauses, then balance each phrase so no word is left alone."""
    phrases: list[list[Token]] = []
    cur: list[Token] = []
    for k, tok in enumerate(toks):
        cur.append(tok)
        gap = toks[k + 1].t0 - tok.t1 if k + 1 < len(toks) else 9
        if tok.brk or gap > 0.35:
            phrases.append(cur)
            cur = []
    if cur:
        phrases.append(cur)
    groups: list[list[Token]] = []
    for ph in phrases:
        n = len(ph)
        parts = -(-n // MAX_WORDS)
        base, extra = divmod(n, parts)
        i = 0
        for p in range(parts):
            size = base + (1 if p < extra else 0)
            groups.append(ph[i:i + size])
            i += size
    return groups


class Captions:
    def __init__(self, words_path: Path, end: float) -> None:
        self.groups = chunks(tokens(json.loads(words_path.read_text())))
        self.end = end
        self.face = skia.Typeface.MakeFromFile(str(ROOT / "assets" / "fonts" / "Montserrat-ExtraBold.ttf"))

    def _span(self, gi: int) -> tuple[float, float]:
        g = self.groups[gi]
        start = g[0].t0 - 0.04
        stop = self.groups[gi + 1][0].t0 - 0.04 if gi + 1 < len(self.groups) else self.end - 0.6
        return start, min(stop, g[-1].t1 + 0.6)

    def draw(self, c: skia.Canvas, t: float) -> None:
        for gi, g in enumerate(self.groups):
            a, b = self._span(gi)
            if a <= t < b:
                self._draw_group(c, g, t, t - a)
                return

    def _draw_group(self, c, g: list[Token], t: float, age: float) -> None:
        size = SIZE
        font = skia.Font(self.face, size)
        space = font.measureText(" ")
        widths = [font.measureText(tok.text) for tok in g]
        total = sum(widths) + space * (len(g) - 1)
        if total > MAX_W:
            size *= MAX_W / total
            font = skia.Font(self.face, size)
            space = font.measureText(" ")
            widths = [font.measureText(tok.text) for tok in g]
            total = sum(widths) + space * (len(g) - 1)
        k = min(1.0, age / 0.12)
        pop = 1 + 0.12 * math.sin(math.pi * k) if k < 1 else 1.0
        c.save()
        c.translate(540, CENTER_Y)
        c.scale(pop, pop)
        x = -total / 2
        y = size * 0.36
        for tok, w in zip(g, widths):
            live = tok.t0 - 0.03 <= t
            now = tok.t0 - 0.03 <= t < tok.t1 + 0.05
            accent = any(ch.isdigit() for ch in tok.text) or tok.text.strip("?") in ACCENT
            color = WHITE
            if now:
                color = GOLD if accent else PINK
            elif live and accent:
                color = GOLD
            alpha = 255 if live else 110
            blob = skia.TextBlob.MakeFromString(tok.text, font)
            shadow = skia.Paint(AntiAlias=True, Color=skia.Color(0, 0, 0, int(alpha * 0.55)))
            shadow.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, 10))
            c.drawTextBlob(blob, x, y + 6, shadow)
            stroke = skia.Paint(AntiAlias=True, Color=skia.Color(10, 6, 20, alpha), Style=skia.Paint.kStroke_Style,
                                StrokeWidth=size * 0.16, StrokeJoin=skia.Paint.kRound_Join)
            c.drawTextBlob(blob, x, y, stroke)
            c.drawTextBlob(blob, x, y, skia.Paint(AntiAlias=True, Color=skia.Color(*color, alpha)))
            x += w + space
        c.restore()
