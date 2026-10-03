"""Every number on screen is recomputed from a claim."""

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
from fc_sat.circlesquare_config import load_config
from fc_sat.circlesquare_math import (
    A,
    ASYMPTOTE,
    CORNER,
    CX,
    CY,
    R1,
    ZMAX,
    asymptotic_gap,
    circles_for_tolerance,
    first_k_within,
    gap,
)
from fc_sat.circlesquare_schedule import FINAL_K, build_schedule, counter_value
from fc_sat.polycircle_format import format_count, format_gap

ROOT = Path(__file__).resolve().parents[1]
CLAIMS_PATH = ROOT / "configs" / "circlesquare_claims.yaml"

__all__ = [
    "Claim",
    "ClaimBook",
    "digit_sequences",
    "evaluate",
    "lint_text",
    "load_claims",
    "render_report",
]


def _gap_at(circles: int) -> float:
    circles = int(circles)
    if circles > 2976:
        return asymptotic_gap(circles)
    return gap(circles)


def _screen_counts() -> list[int]:
    schedule = build_schedule()
    found = set()
    for frame in range(schedule.n_frames):
        value = counter_value(schedule, frame)
        if value is not None:
            found.add(int(value))
    return sorted(found)


def _screen_gaps() -> list[float]:
    """Every gap float the sub-line formats, world-size and at 36x."""
    schedule = build_schedule()
    values: set[float] = set()
    for frame in range(schedule.n_frames):
        if 948 <= frame < 1344:
            values.add(_gap_at(schedule.k(frame)) * ZMAX)
        elif frame < 948 or 1344 <= frame < 1440 or frame >= 1806:
            values.add(_gap_at(schedule.k(frame)))
    return sorted(values)


def _doubling_ks() -> list[int]:
    return [93 * (2 ** index) for index in range(17)]


def _schedule_k(frame: int) -> int:
    return build_schedule().k(int(frame))


def _schedule_count(kind: str) -> int:
    schedule = build_schedule()
    if kind == "adds":
        return len(schedule.adds)
    if kind == "doubles":
        return len(schedule.doubles)
    raise KeyError(kind)


def _config_number(name: str) -> float:
    cfg = load_config()
    if name == "half_side":
        return float(cfg.half_side)
    if name == "width":
        return float(cfg.width)
    if name == "height":
        return float(cfg.height)
    if name == "zoom_max":
        return float(cfg.zoom_max)
    if name == "gap_px":
        return float(cfg.gap_px)
    raise KeyError(name)


FNS = {
    "gap": lambda circles: _gap_at(int(circles)),
    "gap_zoom": lambda circles, zoom: _gap_at(int(circles)) * float(zoom),
    "first_k": lambda tolerance: first_k_within(float(tolerance)),
    "identity": lambda value: value,
    "r1": lambda: R1,
    "corner": lambda: CORNER,
    "asymptote": lambda: ASYMPTOTE,
    "side": lambda: 2.0 * A,
    "center_x": lambda: CX,
    "center_y": lambda: CY,
    "metre_k": lambda: circles_for_tolerance(5.0, 0.001),
    "final_k": lambda: FINAL_K,
    "doubling_ks": _doubling_ks,
    "screen_counts": _screen_counts,
    "screen_gaps": _screen_gaps,
    "schedule_k": _schedule_k,
    "schedule_count": _schedule_count,
    "config_number": _config_number,
    "scaled": lambda base, exp: int(base) * (2 ** int(exp)),
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
        "Circlesquare claims. Each computed value was recomputed for this build.",
    )
