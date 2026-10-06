"""Every number on screen is recomputed or cited from a claim."""

from __future__ import annotations

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
from fc_sat.coinspin_config import load_config
from fc_sat.coinspin_math import (
    DOUBLING_RATIOS,
    LAPS,
    count_spins,
    count_spins_inside,
    rolling_part,
    rotations_per_year,
    sidereal_day,
    trip_bonus,
    whole_rotations,
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


def _config_number(name: str) -> float:
    cfg = load_config()
    if name == "stage_radius":
        return float(cfg.stage_radius)
    if name == "width":
        return float(cfg.width)
    raise KeyError(name)


def _format_float(value: float) -> str:
    return f"{float(value):.2f}"


FNS = {
    "count_spins": lambda ratio: count_spins(ratio),
    "count_spins_inside": lambda ratio: count_spins_inside(ratio),
    "lap_ratio": lambda index: LAPS[int(index)].ratio,
    "rolling_part": lambda ratio: rolling_part(ratio),
    "trip_bonus": trip_bonus,
    "doubling_ratios": lambda: list(DOUBLING_RATIOS),
    "doubling_spins": lambda: [count_spins(ratio) for ratio in DOUBLING_RATIOS],
    "screen_counts": screen_counts,
    "rotations_per_year": lambda days: rotations_per_year(days),
    "whole_rotations": lambda days: whole_rotations(days),
    "sidereal_day": lambda days: sidereal_day(days),
    "identity": lambda value: value,
    "config_number": _config_number,
}


def load_claims(path: Path | None = None) -> ClaimBook:
    return load_claim_book(
        path or CLAIMS_PATH,
        FNS,
        format_int=format_count,
        format_float=_format_float,
    )


def render_report(book: ClaimBook, results: list[tuple[str, bool, str]]) -> str:
    return render_claim_report(
        book,
        results,
        "Coinspin claims. Each computed value was recomputed for this build.",
    )
