"""On-screen sentences. Digits are filled from claims, then linted."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from fc_sat.coinspin_claims import ClaimBook, lint_text, load_claims
from fc_sat.polycircle_format import format_count

ROOT = Path(__file__).resolve().parents[1]
TEXT_PATH = ROOT / "configs" / "coinspin_text.yaml"
_SLOT = re.compile(r"\{([A-Za-z0-9_]+)\}")
LINE_LIMIT = 16
HOOK_END = 144
CARDS_END = 1800


@dataclass(frozen=True)
class Card:
    id: str
    start: int
    end: int
    lines: tuple[str, ...]
    claims: tuple[str, ...]
    gold: tuple[str, ...] = ()


@dataclass
class Script:
    hooks: dict[str, tuple[str, ...]]
    cards: tuple[Card, ...]


def _fill(template: str, book: ClaimBook) -> str:
    def replace(match: re.Match[str]) -> str:
        claim = book.get(match.group(1))
        value = claim.value
        if claim.type == "definitional" and claim.surfaces:
            return claim.surfaces[0]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{claim.id} has nothing to print")
        # Years and small counts print plain: 1982, not 1,982.
        return str(value) if value < 10000 else format_count(value)

    return _SLOT.sub(replace, template)


def load_script(path: Path | None = None, book: ClaimBook | None = None) -> Script:
    path = path or TEXT_PATH
    book = book or load_claims()
    raw = yaml.safe_load(path.read_text())
    hooks = {key: tuple(lines) for key, lines in raw["hooks"].items()}
    cards = []
    for body in raw["cards"]:
        cards.append(
            Card(
                id=body["id"],
                start=int(body["start"]),
                end=int(body["end"]),
                lines=tuple(_fill(line, book) for line in body["lines"]),
                claims=tuple(body.get("claims") or []),
                gold=tuple(_fill(item, book) for item in (body.get("gold") or [])),
            )
        )
    return Script(hooks=hooks, cards=tuple(cards))


def lint_script(script: Script, book: ClaimBook) -> list[str]:
    errors: list[str] = []
    for key, lines in script.hooks.items():
        if len(lines) > 3:
            errors.append(f"hook {key} has {len(lines)} lines")
        for line in lines:
            if len(line) > LINE_LIMIT:
                errors.append(f"hook {key} line {line!r} is {len(line)} characters")
            errors.extend(lint_text(line, ["unit"], book, f"hook {key}"))
    previous_end = HOOK_END
    for card in script.cards:
        if card.start != previous_end:
            errors.append(f"{card.id} starts at {card.start}, after a gap or overlap at {previous_end}")
        previous_end = card.end
        if len(card.lines) > 3:
            errors.append(f"{card.id} has {len(card.lines)} lines")
        for line in card.lines:
            if len(line) > LINE_LIMIT:
                errors.append(f"{card.id} line {line!r} is {len(line)} characters")
            errors.extend(lint_text(line, list(card.claims), book, card.id))
        for gold in card.gold:
            if not any(gold in line for line in card.lines):
                errors.append(f"{card.id} gold {gold!r} is not on a line")
    if previous_end != CARDS_END:
        errors.append(f"cards end at {previous_end}, not {CARDS_END}")
    return errors


def top_lines(script: Script, frame: int, hook: str) -> tuple[str, ...]:
    card = card_at(script, frame)
    if card is None:
        return script.hooks[hook]
    return card.lines


def card_at(script: Script, frame: int) -> Card | None:
    if frame < HOOK_END or frame >= CARDS_END:
        return None
    for card in script.cards:
        if card.start <= frame < card.end:
            return card
    return None
