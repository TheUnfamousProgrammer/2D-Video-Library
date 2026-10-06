"""Sizes on screen. Everything is written in plain metres so the zeros grow from both ends.

Small sizes keep two significant digits after their leading zeros (0.0000000000000017 m).
Large sizes are whole metres rounded to the cited value's precision (12,742,000 m).
"""

from __future__ import annotations

import math
from decimal import Decimal


def significant(value: float, digits: int) -> Decimal:
    """``value`` rounded to ``digits`` significant digits, as an exact decimal."""
    if value <= 0:
        raise ValueError(f"size must be positive, got {value}")
    exponent = math.floor(math.log10(value))
    quantum = Decimal(1).scaleb(exponent - digits + 1)
    return Decimal(repr(value)).quantize(quantum)


def metres(value: float, digits: int = 2) -> str:
    """Plain metres with no scientific notation: 0.0000000000000017 m, 1.7 m, 12,742,000 m."""
    rounded = significant(value, digits)
    if rounded >= 1000:
        return f"{int(rounded):,} m"
    text = format(rounded, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".") if rounded >= 1 else text
    return f"{text} m"


def zero_count(text: str) -> int:
    return text.count("0")
