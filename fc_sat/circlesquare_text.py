"""On-screen sentences. Digits are filled from claims, then linted."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from fc_sat.circlesquare_claims import ClaimBook, digit_sequences, lint_text, load_claims
from fc_sat.polycircle_format import format_count, format_gap

ROOT = Path(__file__).resolve().parents[1]
TEXT_PATH = ROOT / "configs" / "circlesquare_text.yaml"
_SLOT = re.compile(r"\{([A-Za-z0-9_]+)\}")


@dataclass(frozen=True)
class Card:
    id: str
    start: int
    end: int
    lines: tuple[str, ...]
    claims: tuple[str, ...]
    gold: tuple[str, ...] = ()


@dataclass(frozen=True)
class SubLine:
    id: str
    start: int
    end: int
    template: str
    claims: tuple[str, ...]
    dynamic: str


@dataclass
class Script:
    hooks: dict[str, tuple[str, ...]]
    cards: tuple[Card, ...]
    subs: tuple[SubLine, ...]


def _fill(template: str, book: ClaimBook) -> str:
    def replace(match: re.Match[str]) -> str:
        claim = book.get(match.group(1))
        value = claim.value
        if isinstance(value, bool):
            raise ValueError(f"{claim.id} has nothing to print")
        if isinstance(value, int):
            return str(value) if value < 10000 else format_count(value)
        if isinstance(value, float):
            return format_gap(value)
        raise ValueError(f"{claim.id} has nothing to print")

    return _SLOT.sub(replace, template)


def load_script(path: Path | None = None, book: ClaimBook | None = None) -> Script:
    path = path or TEXT_PATH
    book = book or load_claims()
    raw = yaml.safe_load(path.read_text())
    hooks = {key: tuple(lines) for key, lines in raw["hooks"].items()}
    cards = []
    for body in raw["cards"]:
        claims = tuple(body.get("claims") or [])
        lines = tuple(_fill(line, book) for line in body["lines"])
        cards.append(
            Card(
                id=body["id"],
                start=int(body["start"]),
                end=int(body["end"]),
                lines=lines,
                claims=claims,
                gold=tuple(_fill(item, book) for item in (body.get("gold") or [])),
            )
        )
    subs = []
    for body in raw.get("subs") or []:
        subs.append(
            SubLine(
                id=body["id"],
                start=int(body["start"]),
                end=int(body["end"]),
                template=str(body["template"]),
                claims=tuple(body.get("claims") or []),
                dynamic=str(body.get("dynamic") or ""),
            )
        )
    return Script(hooks=hooks, cards=tuple(cards), subs=tuple(subs))


def lint_script(script: Script, book: ClaimBook) -> list[str]:
    errors: list[str] = []
    for key, lines in script.hooks.items():
        if len(lines) > 3:
            errors.append(f"hook {key} has {len(lines)} lines")
        for line in lines:
            if len(line) > 16:
                errors.append(f"hook {key} line {line!r} is {len(line)} characters")
            errors.extend(lint_text(line, [], book, f"hook {key}"))
    for card in script.cards:
        if len(card.lines) > 3:
            errors.append(f"{card.id} has {len(card.lines)} lines")
        for line in card.lines:
            if len(line) > 16:
                errors.append(f"{card.id} line {line!r} is {len(line)} characters")
            errors.extend(lint_text(line, list(card.claims), book, card.id))
    for sub in script.subs:
        if sub.dynamic:
            continue
        errors.extend(lint_text(sub.template, list(sub.claims), book, sub.id))
    return errors


def top_lines(script: Script, frame: int, hook: str) -> tuple[str, ...] | None:
    if frame <= 191 or frame >= 1800:
        return script.hooks[hook]
    for card in script.cards:
        if card.start <= frame < card.end:
            return card.lines
    return None


def sub_template(script: Script, frame: int) -> SubLine | None:
    for sub in script.subs:
        if sub.start <= frame < sub.end:
            return sub
    return None
