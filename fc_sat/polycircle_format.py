"""On-screen numbers. Every formatter is pure so a caption cannot drift from the math."""

from __future__ import annotations

import math


def format_count(n: int) -> str:
    """Integers with thousands separators. 6291456 -> '6,291,456'."""
    if isinstance(n, bool) or not isinstance(n, int):
        raise TypeError(f"format_count expects an int, got {n!r}")
    if n < 0:
        raise ValueError(f"format_count expects a non-negative int, got {n}")
    return f"{n:,}"


def format_gap(value: float) -> str:
    """Two significant digits, no scientific notation, no binary float tails.

    Values under 0.0001 are fixed-point, so 4.6e-11 is printed as 0.000000000046.
    """
    value = float(value)
    if value < 0:
        raise ValueError(f"gap is negative: {value}")
    if value == 0:
        return "0"
    exp = math.floor(math.log10(value))
    mant = value / (10.0 ** exp)
    mant_r = round(mant, 1)
    if mant_r >= 10:
        mant_r = 1.0
        exp += 1
    if value < 0.0001:
        digits = int(round(mant_r * 10))
        if digits == 100:
            digits = 10
            exp += 1
        zeros = -exp - 1
        if zeros < 0:
            raise ValueError(f"cannot format {value} in fixed decimals")
        return "0." + ("0" * zeros) + str(digits)
    rounded = mant_r * (10.0 ** exp)
    decimals = max(0, 1 - exp)
    return f"{rounded:.{decimals}f}"
