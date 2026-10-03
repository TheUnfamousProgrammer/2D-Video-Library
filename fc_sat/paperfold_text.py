"""On-screen sentences. Digits are filled from claims, then linted.

The top slot is at most 3 lines of 14 characters. The bottom sub-line is separate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from fc_sat.paperfold_claims import ClaimBook, lint_text, load_claims

ROOT = Path(__file__).resolve().parents[1]
TEXT_PATH = ROOT / "configs" / "paperfold_text.yaml"
_SLOT = re.compile(r"\{([A-Za-z0-9_]+)\}")
LINE_LIMIT = 14


@dataclass(frozen=True)
class Card:
    id: str
    start: int
    end: int
    lines: tuple[str, ...]
    claims: tuple[str, ...]


@dataclass(frozen=True)
class SubLine:
    id: str
    start: int
    end: int
    template: str
    text: str
    claims: tuple[str, ...]


@dataclass
class Script:
    hooks: dict[str, tuple[str, ...]]
    hook_claims: dict[str, tuple[str, ...]]
    cards: tuple[Card, ...]
    subs: tuple[SubLine, ...]


def _fill(template: str, book: ClaimBook) -> str:
    def replace(match: re.Match[str]) -> str:
        claim = book.get(match.group(1))
        value = claim.value
        if isinstance(value, bool):
            raise ValueError(f"{claim.id} is not a number")
        if isinstance(value, float):
            text = f"{value:.6g}"
            if text.endswith(".0"):
                text = text[:-2]
            return text
        return str(value)

    text = _SLOT.sub(replace, template)
    return re.sub(r"[ \t]+([.!?])", r"\1", text)


def load_script(path: Path | None = None, book: ClaimBook | None = None) -> Script:
    path = path or TEXT_PATH
    book = book or load_claims()
    raw = yaml.safe_load(path.read_text())
    hooks = {}
    hook_claims = {}
    for key, body in raw["hooks"].items():
        claims = tuple(body.get("claims") or [])
        hooks[key] = tuple(_fill(line, book) for line in body["lines"])
        hook_claims[key] = claims
    cards = []
    for body in raw["cards"]:
        claims = tuple(body.get("claims") or [])
        cards.append(
            Card(
                id=body["id"],
                start=int(body["start"]),
                end=int(body["end"]),
                lines=tuple(_fill(line, book) for line in body["lines"]),
                claims=claims,
            )
        )
    subs = []
    for body in raw.get("subs") or []:
        claims = tuple(body.get("claims") or [])
        template = str(body["template"])
        subs.append(
            SubLine(
                id=body["id"],
                start=int(body["start"]),
                end=int(body["end"]),
                template=template,
                text=_fill(template, book),
                claims=claims,
            )
        )
    return Script(hooks, hook_claims, tuple(cards), tuple(subs))


def lint_script(script: Script, book: ClaimBook) -> list[str]:
    errors: list[str] = []
    for key, lines in script.hooks.items():
        if len(lines) > 3:
            errors.append(f"hook {key} has {len(lines)} lines")
        claims = list(script.hook_claims.get(key, ()))
        for line in lines:
            if len(line) > LINE_LIMIT:
                errors.append(f"hook {key} line {line!r} is {len(line)} characters")
            errors.extend(lint_text(line, claims, book, f"hook {key}"))
    for card in script.cards:
        if len(card.lines) > 3:
            errors.append(f"{card.id} has {len(card.lines)} lines")
        for line in card.lines:
            if len(line) > LINE_LIMIT:
                errors.append(f"{card.id} line {line!r} is {len(line)} characters")
            errors.extend(lint_text(line, list(card.claims), book, card.id))
    for sub in script.subs:
        errors.extend(lint_text(sub.text, list(sub.claims), book, sub.id))
    return errors


def top_lines(script: Script, frame: int, hook: str) -> tuple[str, ...]:
    if frame <= 180 or frame >= 1800:
        return script.hooks[hook]
    for card in script.cards:
        if card.start <= frame < card.end:
            return card.lines
    return ()
