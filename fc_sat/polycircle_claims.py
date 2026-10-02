"""Every number on screen is recomputed from a claim. A caption with a stray digit fails."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from fc_sat.polycircle_config import load_config
from fc_sat.polycircle_format import format_count, format_gap
from fc_sat.polycircle_geometry import edge_length, first_n_within, gap, sides_for_gap
from fc_sat.polycircle_schedule import build_schedule

ROOT = Path(__file__).resolve().parents[1]
CLAIMS_PATH = ROOT / "configs" / "polycircle_claims.yaml"

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


def _zoom_gaps(radius: float, zoom: float) -> list[float]:
    return [gap(96 * (2 ** k), radius) * zoom for k in range(9)]


def _schedule_n(frame: int) -> int:
    return build_schedule().n(frame)


def _schedule_count(kind: str) -> int:
    schedule = build_schedule()
    if kind == "adds":
        return len(schedule.adds)
    if kind == "doubles":
        return len(schedule.doubles)
    raise KeyError(kind)


def _first_frame(n_sides: int) -> int:
    schedule = build_schedule()
    for index, value in enumerate(schedule.n_at):
        if int(value) == int(n_sides):
            return int(index)
    raise ValueError(f"side count {n_sides} never appears")


def _config_number(name: str) -> float:
    cfg = load_config()
    if name == "radius":
        return float(cfg.radius)
    if name == "width":
        return float(cfg.width)
    if name == "zoom_max":
        return float(cfg.zoom_max)
    if name == "gap_px":
        return float(cfg.gap_px)
    raise KeyError(name)


FNS = {
    "gap": lambda n, radius: gap(int(n), float(radius)),
    "gap_zoom": lambda n, radius, zoom: gap(int(n), float(radius)) * float(zoom),
    "edge": lambda n, radius: edge_length(int(n), float(radius)),
    "edge_zoom": lambda n, radius, zoom: edge_length(int(n), float(radius)) * float(zoom),
    "first_n": lambda radius, tolerance: first_n_within(float(radius), float(tolerance)),
    "sides_for_gap": lambda radius, tolerance: sides_for_gap(float(radius), float(tolerance)),
    "scaled_n": lambda base, exp: int(base) * (2 ** int(exp)),
    "identity": lambda value: value,
    "zoom_gaps": _zoom_gaps,
    "schedule_n": _schedule_n,
    "schedule_count": _schedule_count,
    "first_frame": _first_frame,
    "config_number": _config_number,
    "half": lambda value: float(value) / 2.0,
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

    def tokens(self) -> set[str]:
        allowed = {item.lower() for item in self.surfaces}
        allowed |= {item.lower() for item in digit_sequences(self.text)}
        allowed |= {item.lower() for item in digit_sequences(self.source)}
        if isinstance(self.value, bool):
            return allowed
        if isinstance(self.value, int):
            allowed.add(str(self.value))
            allowed.add(format_count(self.value).lower())
        elif isinstance(self.value, float):
            allowed.add(format_gap(self.value).lower())
        elif isinstance(self.value, list):
            for item in self.value:
                if isinstance(item, int):
                    allowed.add(str(item))
                    allowed.add(format_count(item).lower())
                elif isinstance(item, float):
                    allowed.add(format_gap(item).lower())
        return allowed


@dataclass
class ClaimBook:
    claims: dict[str, Claim]
    whitelist: set[str]
    path: Path

    def get(self, claim_id: str) -> Claim:
        if claim_id not in self.claims:
            raise KeyError(claim_id)
        return self.claims[claim_id]


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
        )
    whitelist = {str(item) for item in raw.get("whitelist", [])}
    return ClaimBook(claims, whitelist, path)


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


def check_claim(claim: Claim) -> tuple[bool, str]:
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
    got = FNS[claim.fn](*claim.args)
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
    return [(claim.id, *check_claim(claim)) for claim in book.claims.values()]


def render_report(book: ClaimBook, results: list[tuple[str, bool, str]]) -> str:
    lines = ["# Claims report", "", "Polycircle claims. Each computed value was recomputed for this build.", ""]
    for claim_id, ok, detail in results:
        claim = book.get(claim_id)
        used = ", ".join(claim.used_in) or "unused"
        lines.append(f"- {'PASS' if ok else 'FAIL'} `{claim_id}` ({claim.type}): {detail}. Used in {used}.")
    lines.append("")
    return "\n".join(lines)
