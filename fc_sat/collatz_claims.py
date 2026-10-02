"""Claims: recompute, lint spoken numbers, and re-check the verification bound."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from fc_sat.collatz_bound import PAGE_URL, fetch_page, latest_bound
from fc_sat.collatz_format import format_int
from fc_sat.collatz_math import (
    first_step_above,
    first_step_below_after_peak,
    highest_peak_below,
    longest_below,
    peak,
    peak_index,
    steps,
    at_least_steps_below,
)

ROOT = Path(__file__).resolve().parents[1]
CLAIMS_PATH = ROOT / "configs" / "claims.yaml"

_ONES = "one two three four five six seven eight nine".split()
_TEENS = "ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
_TENS = "twenty thirty forty fifty sixty seventy eighty ninety".split()
_SCALES = "hundred thousand million billion trillion quadrillion quintillion sextillion".split()
_NUMBER_WORDS = set(_ONES + _TEENS + _TENS + _SCALES)
_WORD = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)?")
_DECIMAL = re.compile(r"\d+\.\d+")
_COMMAS = re.compile(r"\d{1,3}(?:,\d{3})+")
_DIGITS = re.compile(r"\d+")

FNS = {
    "steps": steps,
    "peak": peak,
    "peak_index": peak_index,
    "first_above": first_step_above,
    "first_below_after_peak": first_step_below_after_peak,
    "longest_below": longest_below,
    "highest_peak_below": highest_peak_below,
    "at_least": at_least_steps_below,
    "identity": lambda value: value,
    "pow2": lambda exp: 2 ** int(exp),
    "mul_pow2": lambda coef, exp: int(coef) * (2 ** int(exp)),
}


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
    verified: bool = True
    recheck_page: bool = False
    language: str = ""

    def allowed_tokens(self) -> set[str]:
        tokens: set[str] = set()
        for surface in self.surfaces:
            tokens |= {tok.lower() for tok in number_tokens(surface)}
            tokens.add(surface.lower())
        tokens |= {tok.lower() for tok in number_tokens(self.source)}
        tokens |= {tok.lower() for tok in number_tokens(self.text)}
        if isinstance(self.value, int) and not isinstance(self.value, bool):
            tokens |= {tok.lower() for tok in number_tokens(str(self.value))}
            tokens |= {tok.lower() for tok in number_tokens(format_int(self.value))}
        return tokens


@dataclass
class ClaimBook:
    claims: dict[str, Claim]
    whitelist: set[str]
    path: Path

    def get(self, claim_id: str) -> Claim:
        if claim_id not in self.claims:
            raise KeyError(claim_id)
        return self.claims[claim_id]


def _is_number_word(word: str) -> bool:
    parts = word.lower().split("-")
    return all(part in _NUMBER_WORDS for part in parts)


def number_tokens(text: str) -> list[str]:
    """Digit sequences and spelled number phrases, longest match first."""
    occupied = [False] * (len(text) + 1)
    found: list[tuple[int, str]] = []

    def take(pattern: re.Pattern[str], source: str) -> None:
        for match in pattern.finditer(source):
            if any(occupied[match.start() : match.end()]):
                continue
            found.append((match.start(), match.group()))
            for index in range(match.start(), match.end()):
                occupied[index] = True

    take(_DECIMAL, text)
    take(_COMMAS, text)
    take(_DIGITS, text)

    words = [(m.group(), m.start(), m.end()) for m in _WORD.finditer(text)]
    index = 0
    while index < len(words):
        word, start, end = words[index]
        leading = word.lower() in {"a", "an"}
        if not _is_number_word(word) and not leading:
            index += 1
            continue
        if leading:
            if index + 1 >= len(words) or not _is_number_word(words[index + 1][0]):
                index += 1
                continue
            if not re.fullmatch(r"\s+", text[end : words[index + 1][1]]):
                index += 1
                continue
        end_index = index
        cursor = index
        while cursor + 1 < len(words):
            prev = words[cursor]
            nxt = words[cursor + 1]
            between = text[prev[2] : nxt[1]]
            tok = nxt[0].lower()
            if not re.fullmatch(r"\s+", between):
                break
            if tok == "and":
                if cursor + 2 >= len(words) or not _is_number_word(words[cursor + 2][0]):
                    break
                if not re.fullmatch(r"\s+", text[nxt[2] : words[cursor + 2][1]]):
                    break
                cursor += 2
                end_index = cursor
                continue
            if _is_number_word(tok):
                cursor += 1
                end_index = cursor
                continue
            break
        phrase = text[words[index][1] : words[end_index][2]]
        if not any(occupied[words[index][1] : words[end_index][2]]):
            found.append((words[index][1], phrase))
        index = end_index + 1
    found.sort()
    return [phrase for _, phrase in found]


def load_claims(path: Path | None = None) -> ClaimBook:
    path = path or CLAIMS_PATH
    raw = yaml.safe_load(path.read_text())
    claims: dict[str, Claim] = {}
    for claim_id, body in raw["claims"].items():
        claims[claim_id] = Claim(
            id=claim_id,
            type=body["type"],
            text=body.get("text", ""),
            value=body.get("value"),
            source=body.get("source", ""),
            surfaces=list(body.get("surfaces") or []),
            used_in=list(body.get("used_in") or []),
            fn=body.get("fn"),
            args=list(body.get("args") or []),
            verified=bool(body.get("verified", True)),
            recheck_page=bool(body.get("recheck_page", False)),
            language=body.get("language", ""),
        )
    whitelist = {str(item) for item in raw.get("whitelist", [])}
    return ClaimBook(claims, whitelist, path)


def _same(expected, got) -> bool:
    if isinstance(expected, list):
        got_list = [list(item) if isinstance(item, tuple) else item for item in got]
        return got_list == expected
    return got == expected


def check_claim(claim: Claim) -> tuple[bool, str]:
    if claim.type == "cited" and not claim.fn:
        if claim.id == "c_1937" and not claim.verified:
            return False, "c_1937 is not verified, so it must not be spoken"
        return True, "cited"
    if not claim.fn:
        return False, "missing fn"
    got = FNS[claim.fn](*claim.args)
    if not _same(claim.value, got):
        if claim.recheck_page and isinstance(claim.value, int) and claim.value >= 2**71:
            return True, f"page bound {claim.value}; yaml expression still computes {got}"
        return False, f"expected {claim.value} but computed {got}"
    return True, "ok"


def recheck_verified(book: ClaimBook, html: str | None = None, fetch: bool = False) -> str:
    """Compare c_verified with the project page. Update the in-memory value if it moved."""
    claim = book.get("c_verified")
    if html is None and fetch:
        try:
            html = fetch_page()
        except Exception as exc:
            return f"bound recheck failed ({exc}); keeping {claim.value}"
    if html is None:
        return "bound recheck skipped"
    found = latest_bound(html)
    if found is None:
        return f"bound recheck could not parse {PAGE_URL}; keeping {claim.value}"
    if found == claim.value:
        return f"verification bound unchanged: {claim.value}"
    previous = claim.value
    claim.value = found
    claim.surfaces = list(dict.fromkeys([*claim.surfaces, format_int(found), str(found)]))
    return (
        f"bound diff: {previous} ({format_int(previous) if isinstance(previous, int) else previous})"
        f" -> {found} ({format_int(found)})"
    )


def lint_text(text: str, claim_ids: list[str], book: ClaimBook, where: str) -> list[str]:
    allowed = {item.lower() for item in book.whitelist}
    for claim_id in claim_ids:
        allowed |= book.get(claim_id).allowed_tokens()
    errors = []
    for token in number_tokens(text):
        if token.lower() not in allowed and token not in book.whitelist:
            errors.append(f"{where}: {token!r} has no claim")
    return errors


def render_report(book: ClaimBook, results: list[tuple[str, bool, str]]) -> str:
    lines = ["# Claims report", ""]
    for claim_id, ok, detail in results:
        claim = book.get(claim_id)
        flag = "PASS" if ok else "FAIL"
        used = ", ".join(claim.used_in) or "unused"
        lines.append(f"- {flag} `{claim_id}` ({claim.type}): {detail}. Used in {used}.")
    lines.append("")
    return "\n".join(lines)


def evaluate(book: ClaimBook, html: str | None = None, fetch: bool = False) -> tuple[list[tuple[str, bool, str]], str]:
    note = recheck_verified(book, html=html, fetch=fetch)
    results = []
    for claim_id, claim in book.claims.items():
        ok, detail = check_claim(claim)
        results.append((claim_id, ok, detail))
    return results, note
