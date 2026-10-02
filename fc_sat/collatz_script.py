"""Script load, caption chunking, and the wording checks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from fc_sat.collatz_claims import ClaimBook, lint_text, number_tokens

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "configs" / "collatz_script.yaml"
BANNED = ("impossible", "nobody has tried", "<break", "—", "(")


@dataclass
class Line:
    id: str
    scene: str
    offset: float
    slot_max_s: float
    tts_text: str
    caption_text: str
    claims: list[str]
    alt_text: str
    tag: str
    tag_candidates: list[str]
    stability: float | None = None

    def spoken(self) -> str:
        tag = self.tag.strip()
        if tag:
            return f"{tag} {self.tts_text}"
        return self.tts_text

    def word_count(self) -> int:
        return len(self.tts_text.replace("...", " ").split())


def load_script(path: Path | None = None) -> list[Line]:
    raw = yaml.safe_load((path or SCRIPT_PATH).read_text())
    lines = []
    for body in raw["lines"]:
        lines.append(
            Line(
                id=body["id"],
                scene=body["scene"],
                offset=float(body["offset"]),
                slot_max_s=float(body["slot_max_s"]),
                tts_text=body["tts_text"],
                caption_text=body["caption_text"],
                claims=list(body.get("claims") or []),
                alt_text=body.get("alt_text", body["tts_text"]),
                tag=body.get("tag", ""),
                tag_candidates=list(body.get("tag_candidates") or [""]),
                stability=body.get("stability"),
            )
        )
    return lines


def chunk_caption(text: str, limit: int = 28) -> list[str]:
    clean = " ".join(text.split())
    if len(clean) <= limit:
        return [clean]
    words = clean.split()
    best: tuple[str, str] | None = None
    best_gap = 10**9
    for cut in range(1, len(words)):
        left = " ".join(words[:cut])
        right = " ".join(words[cut:])
        if len(left) <= limit and len(right) <= limit:
            gap = abs(len(left) - len(right))
            if gap < best_gap:
                best = (left, right)
                best_gap = gap
    if best is None:
        raise ValueError(f"caption does not fit in two lines of {limit}: {clean!r}")
    return [best[0], best[1]]


def _caps_words(text: str) -> list[str]:
    return [word for word in re.findall(r"[A-Za-z']+", text) if word.isupper() and len(word) > 1]


def lint_script(lines: list[Line], book: ClaimBook) -> list[str]:
    errors: list[str] = []
    for line in lines:
        if line.word_count() > 12:
            errors.append(f"{line.id}: {line.word_count()} words")
        if len(line.tag_candidates) < 1:
            errors.append(f"{line.id}: missing tag candidates")
        if line.tag and line.tag not in line.tag_candidates:
            errors.append(f"{line.id}: tag {line.tag} is not a candidate")
        if line.tag and not (line.tag.startswith("[") and line.tag.endswith("]")):
            errors.append(f"{line.id}: tag must be in square brackets")
        spoken = line.tts_text
        if spoken.count("[") > 0:
            errors.append(f"{line.id}: tag belongs in the tag field, not tts_text")
        for banned in BANNED:
            if banned in spoken or banned in line.caption_text:
                errors.append(f"{line.id}: banned {banned!r}")
        caps = _caps_words(spoken)
        if len(caps) > 1:
            errors.append(f"{line.id}: more than one emphasized word {caps}")
        errors.extend(lint_text(spoken, line.claims, book, f"{line.id} tts"))
        errors.extend(lint_text(line.caption_text, line.claims, book, f"{line.id} caption"))
        errors.extend(lint_text(line.alt_text, line.claims, book, f"{line.id} alt"))
        try:
            rows = chunk_caption(line.caption_text)
        except ValueError as exc:
            errors.append(str(exc))
            rows = []
        if len(rows) > 2:
            errors.append(f"{line.id}: more than 2 caption lines")
    joined = " ".join(line.tts_text for line in lines)
    if "still can't prove" not in joined:
        errors.append("voiceover never says still can't prove")
    if "impossible" in joined.lower() or "nobody has tried" in joined.lower():
        errors.append("banned proof language")
    return errors


def script_markdown(lines: list[Line]) -> str:
    rows = ["# Collatz script", ""]
    for line in lines:
        rows.append(
            f"- {line.id} {line.offset:.2f}s slot {line.slot_max_s:.2f}s "
            f"[{line.scene}] {line.spoken()}"
        )
        rows.append(f"  caption: {line.caption_text}")
        if line.claims:
            rows.append(f"  claims: {', '.join(line.claims)}")
    rows.append("")
    return "\n".join(rows)


def estimate_characters(lines: list[Line], takes: int = 3) -> int:
    return sum(len(line.spoken()) for line in lines) * takes
