"""Integer display. Numbers under 10 digits get separators. Larger ones get a short-scale name."""

from __future__ import annotations

SCALES = (
    "",
    "thousand",
    "million",
    "billion",
    "trillion",
    "quadrillion",
    "quintillion",
    "sextillion",
)


def format_int(n: int) -> str:
    if isinstance(n, bool) or not isinstance(n, int):
        raise TypeError(f"format_int expects an int, got {n!r}")
    if n < 0:
        raise ValueError(f"format_int expects a non-negative int, got {n}")
    text = str(n)
    if len(text) < 10:
        return f"{n:,}"
    exponent = len(text) - 1
    group = exponent // 3
    if group >= len(SCALES):
        raise ValueError(f"{n} is larger than the named scales")
    leading = int(text[:3])
    if len(text) > 3 and text[3] >= "5":
        leading += 1
    if leading == 1000:
        leading = 100
        exponent += 1
        group = exponent // 3
        if group >= len(SCALES):
            raise ValueError(f"{n} is larger than the named scales")
    whole = leading // 100
    frac = leading % 100
    name = SCALES[group]
    if not name:
        return f"{n:,}"
    return f"{whole}.{frac:02d} {name}"


def display_hundredths(n: int) -> int:
    """Hundredths of the short-scale prefix. 2^71 -> 236, meaning 2.36."""
    text = format_int(n)
    if " " not in text:
        raise ValueError(f"{n} has no short-scale prefix")
    prefix = text.split()[0]
    whole, frac = prefix.split(".")
    return int(whole) * 100 + int(frac)


def format_hundredths(hundredths: int) -> str:
    if hundredths < 0:
        raise ValueError("hundredths must be >= 0")
    return f"{hundredths // 100}.{hundredths % 100:02d}"
