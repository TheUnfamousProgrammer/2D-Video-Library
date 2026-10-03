"""Paperfold claims. Heights and lengths are recomputed; cited numbers name a source."""

from __future__ import annotations

from pathlib import Path

from fc_sat.beatkit.claims import ClaimBook, digit_sequences, evaluate, lint_text, load_claims as load_claim_book
from fc_sat.beatkit.claims import render_report as render_claim_report
from fc_sat.paperfold_format import rounded_au_times, rounded_km, rounded_light_years
from fc_sat.paperfold_math import (
    AU_M,
    BURJ_M,
    EARTH_CIRC_M,
    EARTH_RADIUS_M,
    EVEREST_M,
    KARMAN_M,
    LIGHT_YEAR_M,
    MILKY_WAY_LY,
    MOON_M,
    PERSON_M,
    RECORD_FOLDS,
    RECORD_LENGTH_M,
    THICKNESS_M,
    first_fold_past,
    height_m,
    length_km,
    length_m,
    milestone_fold,
)

ROOT = Path(__file__).resolve().parents[1]
CLAIMS_PATH = ROOT / "configs" / "paperfold_claims.yaml"

CONST = {
    "thickness_mm": THICKNESS_M * 1000.0,
    "moon_km": MOON_M / 1000.0,
    "everest_m": EVEREST_M,
    "burj_m": BURJ_M,
    "karman_km": KARMAN_M / 1000.0,
    "earth_circ_km": EARTH_CIRC_M / 1000.0,
    "au_km": AU_M / 1000.0,
    "light_year_m": LIGHT_YEAR_M,
    "milky_way_ly": MILKY_WAY_LY,
    "person_m": PERSON_M,
    "earth_radius_km": EARTH_RADIUS_M / 1000.0,
    "record_folds": RECORD_FOLDS,
    "record_m": RECORD_LENGTH_M,
    "record_year": 2002,
}


def _const(name: str):
    if name not in CONST:
        raise KeyError(name)
    value = CONST[name]
    if isinstance(value, float) and value.is_integer() and abs(value) < 1e15:
        return int(value)
    return value


def _pass_length(name: str) -> int:
    if name != "earth":
        raise KeyError(name)
    n = 1
    while length_m(n) <= EARTH_CIRC_M:
        n += 1
    return n


def _pass_distance(name: str) -> int:
    if name != "sun":
        raise KeyError(name)
    return first_fold_past(AU_M)


def _cm(n: int) -> int:
    return int(round(height_m(int(n)) * 100.0))


def _cm_tenth(n: int) -> float:
    return round(height_m(int(n)) * 100.0, 2)


def _km_decimal(n: int) -> float:
    return round(length_km(int(n)), 1)


FNS = {
    "const": _const,
    "height": lambda n: height_m(int(n)),
    "length": lambda n: length_m(int(n)),
    "pass_fold": lambda name: milestone_fold(str(name)),
    "pass_length": _pass_length,
    "pass_distance": _pass_distance,
    "au_times": lambda n: rounded_au_times(int(n)),
    "ly_round": lambda n: rounded_light_years(int(n)),
    "km_round": lambda n: rounded_km(int(n)),
    "cm_round": _cm,
    "cm_tenth": _cm_tenth,
    "km_decimal": _km_decimal,
}


def load_claims(path: Path | None = None) -> ClaimBook:
    return load_claim_book(path or CLAIMS_PATH, FNS)


def render_report(book: ClaimBook, results: list[tuple[str, bool, str]]) -> str:
    return render_claim_report(
        book,
        results,
        "Paperfold claims. Each computed value was recomputed for this build. "
        "In theory, where the sheet is 0.1 mm and every fold is in one direction.",
    )


__all__ = [
    "digit_sequences",
    "evaluate",
    "lint_text",
    "load_claims",
    "render_report",
]
