"""Timing curves shared by generators that need them.

ease_out_back overshoots by at most 8 percent and returns to 1.
"""

from __future__ import annotations

import math


def clamp01(t: float) -> float:
    if t <= 0.0:
        return 0.0
    if t >= 1.0:
        return 1.0
    return float(t)


def ease_out_cubic(t: float) -> float:
    u = 1.0 - clamp01(t)
    return 1.0 - u * u * u


def ease_in_cubic(t: float) -> float:
    u = clamp01(t)
    return u * u * u


def ease_in_out_cubic(t: float) -> float:
    u = clamp01(t)
    if u < 0.5:
        return 4.0 * u * u * u
    v = -2.0 * u + 2.0
    return 1.0 - (v * v * v) / 2.0


def ease_out_back(t: float, overshoot: float = 0.08) -> float:
    """Cubic ease-out plus a return from a small overshoot.

    The extra term is zero at both ends. Its peak is scaled so the curve
    never rises above 1 + overshoot.
    """
    u = clamp01(t)
    base = 1.0 - (1.0 - u) ** 3
    bump = math.sin(math.pi * u) * (1.0 - u)
    # max of sin(pi u) * (1-u) on [0, 1] is about 0.287 (near u = 0.65).
    peak = 0.287
    return base + (overshoot / peak) * bump


def monotonic(samples: list[float], tol: float = 1e-9) -> bool:
    return all(b + tol >= a for a, b in zip(samples, samples[1:]))
