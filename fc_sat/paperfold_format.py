"""On-screen numbers, built from the formulas. Nothing here is a typed constant."""

from __future__ import annotations

from fc_sat.paperfold_doctor import times_glyph
from fc_sat.paperfold_math import (
    EARTH_CIRC_M,
    au_multiple,
    earth_multiple,
    format_sig,
    height_km,
    height_m,
    length_km,
    length_m,
    light_years,
    sigfigs,
)


def times_mark() -> str:
    return times_glyph()


def rounded_au_times(n: int) -> int:
    """2 significant figures. 403.5 AU becomes 400."""
    return int(round(sigfigs(au_multiple(n), 2)))


def earth_sun_needed(n: int) -> str:
    """Paper needed, in Earth-Sun distances.

    The figure is the claims `au_times` value (two significant figures), with a
    thousands separator. The mark is the ASCII x used in the posted copy.
    """
    return f"PAPER NEEDED {format_sig(au_multiple(n), 2)}x EARTH TO SUN"


def rounded_light_years(n: int) -> int:
    """3 significant figures. 107,052 light-years becomes 107,000."""
    return int(round(sigfigs(light_years(n), 3)))


def rounded_km(n: int, digits: int = 3) -> int:
    return int(round(sigfigs(height_km(n), digits)))


def paper_km_label(n: int) -> str:
    """Readable length. Under 1 km, one decimal so 0.879 km is 0.9 km."""
    km = length_km(n)
    if km < 1:
        return f"{km:.1f} KM"
    if km < 1000:
        return f"{format_sig(km, 3)} KM"
    return f"{format_sig(km, 3)} KM"


def stamp_lines(n: int) -> tuple[str, ...]:
    """Top-slot lines for a paper-needed marker. Every number comes from L(n).

    The brief's phrases are kept. Lines stay at or under 14 characters, so
    "400× EARTH TO SUN" is two lines under the fold count, and "0.9 KM OF PAPER"
    drops "OF" to fit.
    """
    head = f"{n} FOLDS"
    meters = length_m(n)
    if au_multiple(n) >= 10:
        mark = f"{format_sig(au_multiple(n), 2)}{times_mark()}"
        lines = (head, mark, "EARTH TO SUN")
    elif meters >= EARTH_CIRC_M and earth_multiple(n) < 100:
        lines = (head, "WRAPS THE", "EARTH")
    else:
        lines = (head, f"{paper_km_label(n)} PAPER")
    for line in lines:
        if len(line) > 14:
            raise ValueError(f"stamp line {line!r} is {len(line)} characters")
    if len(lines) > 3:
        raise ValueError(f"stamp has {len(lines)} lines")
    return lines


def space_km_label(n: int = 30) -> str:
    return f"{rounded_km(n)} KM"


def record_cm_label() -> str:
    cm = int(round(height_m(12) * 100.0))
    return f"{cm} CM"


def milky_way_label(n: int = 42) -> str:
    return f"{format_sig(light_years(n), 3)} LIGHT-YEARS"
