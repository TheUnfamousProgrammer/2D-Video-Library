"""Every number on screen is recomputed or cited from a claim."""

from __future__ import annotations

import math
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
from fc_sat.coinspin_math import (
    carry_spins,
    count_spins,
    count_spins_inside,
    film_lap_spins,
    road_spins,
    sat_lap_spins,
    segment_spins,
)
from fc_sat.coinspin_schedule import screen_counts
from fc_sat.polycircle_format import format_count

ROOT = Path(__file__).resolve().parents[1]
CLAIMS_PATH = ROOT / "configs" / "coinspin_claims.yaml"

__all__ = [
    "Claim",
    "ClaimBook",
    "digit_sequences",
    "evaluate",
    "lint_text",
    "load_claims",
    "render_report",
]


def _rolling_part(ratio: int) -> int:
    """Spins that come from rolling: all spins minus the one the trip around adds."""
    return count_spins(ratio) - carry_spins()


FNS = {
    "segment_spins": lambda start, end: int(round(segment_spins(int(start), int(end)))),
    "film_lap_spins": film_lap_spins,
    "road_spins": road_spins,
    "carry_spins": carry_spins,
    "sat_lap_spins": sat_lap_spins,
    "rolling_part": _rolling_part,
    "screen_counts": screen_counts,
    "count_spins": lambda ratio: count_spins(ratio),
    "count_spins_inside": lambda ratio: count_spins_inside(ratio),
    "whole_turns": lambda days: int(math.floor(float(days) + 1.0)),
    "identity": lambda value: value,
}


def load_claims(path: Path | None = None) -> ClaimBook:
    return load_claim_book(path or CLAIMS_PATH, FNS, format_int=format_count)


def render_report(book: ClaimBook, results: list[tuple[str, bool, str]]) -> str:
    return render_claim_report(
        book,
        results,
        "Coinspin claims. Each computed value was recomputed for this build.",
    )
