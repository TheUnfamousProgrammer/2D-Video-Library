"""Every number on screen comes from a cited size. The friendly line under the counter is
checked against that size, so a typo in 'display' cannot ship."""

from __future__ import annotations

import math
import re
from pathlib import Path

from fc_sat.beatkit.claims import (
    Claim,
    ClaimBook,
    digit_sequences,
    evaluate,
    lint_text,
    load_claims as load_claim_book,
    render_report as render_claim_report,
)
from fc_sat.scale_format import metres

ROOT = Path(__file__).resolve().parents[1]
CLAIMS_PATH = ROOT / "configs" / "scale_claims.yaml"
# IAU nominal solar radius is 695,700 km; 1 light-year is 9,460,730,472,580,800 m (IAU, Julian year).
SUN_DIAMETER_M = 2.0 * 695_700_000.0
LIGHT_YEAR_M = 9_460_730_472_580_800.0
UNITS = {
    "FEMTOMETRES": 1e-15,
    "FEMTOMETRE": 1e-15,
    "NANOMETRES": 1e-9,
    "NANOMETRE": 1e-9,
    "MICROMETRES": 1e-6,
    "MICROMETRE": 1e-6,
    "MICRONS": 1e-6,
    "MILLIMETRES": 1e-3,
    "MILLIMETRE": 1e-3,
    "MM": 1e-3,
    "CENTIMETRES": 1e-2,
    "CM": 1e-2,
    "METRES": 1.0,
    "METRE": 1.0,
    "KM": 1e3,
    "KILOMETRES": 1e3,
    "LIGHT-YEARS": LIGHT_YEAR_M,
    "LIGHT-YEAR": LIGHT_YEAR_M,
    "SUNS WIDE": SUN_DIAMETER_M,
    "SUNS": SUN_DIAMETER_M,
}
WORDS = {"THOUSAND": 1e3, "MILLION": 1e6, "BILLION": 1e9, "TRILLION": 1e12}
# A friendly line rounds harder than the counter. It may differ from the cited value by this much.
DISPLAY_TOLERANCE = 0.08
_NUMBER = re.compile(r"(HALF A|A|\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)\s+(?:(THOUSAND|MILLION|BILLION|TRILLION)\s+)?(.+)$")

__all__ = [
    "Claim",
    "ClaimBook",
    "digit_sequences",
    "display_metres",
    "display_errors",
    "evaluate",
    "lint_text",
    "load_claims",
    "render_report",
]


def display_metres(display: str) -> float | None:
    """'12,742 KM' -> 12742000.0. A leading label ending in ':' and a leading ABOUT are skipped."""
    text = display.upper().strip()
    if ":" in text:
        text = text.split(":", 1)[1].strip()
    for lead in ("ABOUT ", "UP TO "):
        if text.startswith(lead):
            text = text[len(lead):]
    match = _NUMBER.match(text)
    if match is None:
        return None
    number, word, unit = match.groups()
    unit = unit.strip().rstrip(".")
    if unit not in UNITS:
        return None
    if number == "HALF A":
        value = 0.5
    elif number == "A":
        value = 1.0
    else:
        value = float(number.replace(",", ""))
    if word:
        value *= WORDS[word]
    return value * UNITS[unit]


def display_errors(sizes: dict[str, dict]) -> list[str]:
    errors = []
    for object_id, body in sizes.items():
        shown = display_metres(str(body.get("display", "")))
        if shown is None:
            errors.append(f"{object_id}: cannot read display {body.get('display')!r}")
            continue
        value = float(body["value"])
        if abs(math.log(shown / value)) > math.log(1.0 + DISPLAY_TOLERANCE):
            errors.append(f"{object_id}: display {body['display']!r} is {shown:.4g} m, the cited size is {value:.4g} m")
    return errors


def load_claims(path: Path | None = None) -> ClaimBook:
    """The claim book, with each object's counter and friendly line added to what it may show."""
    book = load_claim_book(path or CLAIMS_PATH, {})
    for claim in book.claims.values():
        if claim.type != "cited":
            continue
        if isinstance(claim.value, bool) or not isinstance(claim.value, (int, float)):
            # YAML 1.1 reads 9.0e12 as a string; it must be written 9.0e+12.
            raise SystemExit(f"{claim.id}: value {claim.value!r} is not a number")
        digits = int(_raw_digits(book, claim.id))
        counter = metres(float(claim.value), digits)
        display = _raw_display(book, claim.id)
        claim.surfaces = sorted(set(claim.surfaces) | set(digit_sequences(counter)) | set(digit_sequences(display)))
    return book


def _raw(book: ClaimBook) -> dict:
    import yaml

    return yaml.safe_load(book.path.read_text())["claims"]


def _raw_digits(book: ClaimBook, claim_id: str) -> int:
    return int(_raw(book)[claim_id].get("digits", 2))


def _raw_display(book: ClaimBook, claim_id: str) -> str:
    return str(_raw(book)[claim_id].get("display", ""))


def render_report(book: ClaimBook, results: list[tuple[str, bool, str]]) -> str:
    return render_claim_report(
        book,
        results,
        "Scale claims. Every size is cited; each friendly line was read back and compared to its size.",
    )
