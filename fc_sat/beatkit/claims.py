"""Claims book. A digit on screen, in a title, or in a description must name a claim."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import yaml

_COMMA = re.compile(r"\d{1,3}(?:,\d{3})+")
_DECIMAL = re.compile(r"\d+\.\d+")
_DIGITS = re.compile(r"\d+")


def digit_sequences(text: str) -> list[str]:
    occupied = [False] * (len(text) + 1)
    found: list[tuple[int, str]] = []

    def take(pattern: re.Pattern[str]) -> None:
        for match in pattern.finditer(text):
            if any(occupied[match.start() : match.end()]):
                continue
            found.append((match.start(), match.group()))
            for index in range(match.start(), match.end()):
                occupied[index] = True

    take(_COMMA)
    take(_DECIMAL)
    take(_DIGITS)
    found.sort()
    return [token for _, token in found]


def _default_int(value: int) -> str:
    return f"{int(value):,}"


def _default_float(value: float) -> str:
    return str(value)


@dataclass
class Claim:
    id: str
    type: str
    text: str
    value: object
    source: str
    surfaces: list[str]
    used_in: list[str]
    fn: str | None = None
    args: list = field(default_factory=list)
    format_int: Callable[[int], str] = _default_int
    format_float: Callable[[float], str] = _default_float

    def tokens(self) -> set[str]:
        allowed = {item.lower() for item in self.surfaces}
        allowed |= {item.lower() for item in digit_sequences(self.text)}
        allowed |= {item.lower() for item in digit_sequences(self.source)}
        if isinstance(self.value, bool):
            return allowed
        if isinstance(self.value, int):
            allowed.add(str(self.value))
            allowed.add(self.format_int(self.value).lower())
        elif isinstance(self.value, float):
            allowed.add(self.format_float(self.value).lower())
        elif isinstance(self.value, list):
            for item in self.value:
                if isinstance(item, int):
                    allowed.add(str(item))
                    allowed.add(self.format_int(item).lower())
                elif isinstance(item, float):
                    allowed.add(self.format_float(item).lower())
        return allowed


@dataclass
class ClaimBook:
    claims: dict[str, Claim]
    whitelist: set[str]
    path: Path
    fns: dict[str, Callable]
    format_int: Callable[[int], str] = _default_int
    format_float: Callable[[float], str] = _default_float

    def get(self, claim_id: str) -> Claim:
        if claim_id not in self.claims:
            raise KeyError(claim_id)
        return self.claims[claim_id]


def load_claims(
    path: Path,
    fns: dict[str, Callable],
    *,
    format_int: Callable[[int], str] = _default_int,
    format_float: Callable[[float], str] = _default_float,
) -> ClaimBook:
    raw = yaml.safe_load(path.read_text())
    claims: dict[str, Claim] = {}
    for claim_id, body in raw["claims"].items():
        used = body.get("used_in")
        if used is None:
            used = body.get("usage") or []
        claims[claim_id] = Claim(
            id=claim_id,
            type=body["type"],
            text=body.get("text", ""),
            value=body.get("value"),
            source=body.get("source", ""),
            surfaces=list(body.get("surfaces") or []),
            used_in=list(used),
            fn=body.get("fn"),
            args=list(body.get("args") or []),
            format_int=format_int,
            format_float=format_float,
        )
    whitelist = {str(item) for item in raw.get("whitelist", [])}
    return ClaimBook(claims, whitelist, path, fns, format_int, format_float)


def _close(expected, got) -> bool:
    if isinstance(expected, bool) or isinstance(got, bool):
        return expected == got
    if isinstance(expected, list):
        if not isinstance(got, list) or len(expected) != len(got):
            return False
        return all(_close(a, b) for a, b in zip(expected, got))
    if isinstance(expected, float) or isinstance(got, float):
        a, b = float(expected), float(got)
        scale = max(1.0, abs(a), abs(b))
        return abs(a - b) <= 1e-9 * scale
    return expected == got


def check_claim(claim: Claim, fns: dict[str, Callable]) -> tuple[bool, str]:
    if claim.type == "definitional":
        if not claim.text:
            return False, "definitional claim has no text"
        return True, "definitional"
    if claim.type == "cited" and not claim.fn:
        if not claim.source:
            return False, "cited claim needs a source"
        return True, "cited"
    if not claim.fn:
        return False, "missing fn"
    if claim.fn not in fns:
        return False, f"unknown fn {claim.fn}"
    got = fns[claim.fn](*claim.args)
    if not _close(claim.value, got):
        return False, f"expected {claim.value} but computed {got}"
    return True, "ok"


def lint_text(text: str, claim_ids: list[str], book: ClaimBook, where: str) -> list[str]:
    allowed = {item.lower() for item in book.whitelist}
    for claim_id in claim_ids:
        allowed |= book.get(claim_id).tokens()
    errors = []
    for token in digit_sequences(text):
        if token.lower() not in allowed:
            errors.append(f"{where}: {token!r} has no claim")
    return errors


def evaluate(book: ClaimBook) -> list[tuple[str, bool, str]]:
    return [(claim.id, *check_claim(claim, book.fns)) for claim in book.claims.values()]


def render_report(book: ClaimBook, results: list[tuple[str, bool, str]], blurb: str) -> str:
    lines = ["# Claims report", "", blurb, ""]
    for claim_id, ok, detail in results:
        claim = book.get(claim_id)
        used = ", ".join(claim.used_in) or "unused"
        lines.append(f"- {'PASS' if ok else 'FAIL'} `{claim_id}` ({claim.type}): {detail}. Used in {used}.")
    lines.append("")
    return "\n".join(lines)
