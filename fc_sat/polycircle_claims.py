"""Every number on screen is recomputed from a claim. A caption with a stray digit fails.

The book, the linter, and the report live in ``fc_sat.beatkit.claims``.
The functions below are the polycircle math those claims call.
"""

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
from fc_sat.polycircle_config import load_config
from fc_sat.polycircle_format import format_count, format_gap
from fc_sat.polycircle_geometry import edge_length, first_n_within, gap, sides_for_gap
from fc_sat.polycircle_schedule import build_schedule

ROOT = Path(__file__).resolve().parents[1]
CLAIMS_PATH = ROOT / "configs" / "polycircle_claims.yaml"

__all__ = [
    "Claim",
    "ClaimBook",
    "digit_sequences",
    "evaluate",
    "lint_text",
    "load_claims",
    "render_report",
]


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


def load_claims(path: Path | None = None) -> ClaimBook:
    return load_claim_book(
        path or CLAIMS_PATH,
        FNS,
        format_int=format_count,
        format_float=format_gap,
    )


def render_report(book: ClaimBook, results: list[tuple[str, bool, str]]) -> str:
    return render_claim_report(
        book,
        results,
        "Polycircle claims. Each computed value was recomputed for this build.",
    )
