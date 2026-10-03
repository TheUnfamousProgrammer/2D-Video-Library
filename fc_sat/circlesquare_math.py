"""Fourier circles that trace a square.

Math coordinates have y up. Screen coordinates have y down. The closed curve is

    z(t) = sum_j w_j * c_j * exp(i * k_j * t)

with one revolution per bar, so frame 1824 has the same phase as frame 0.
Every coefficient and every gap is computed. Nothing in this file is typed in
as a measured constant except the square's center and half-side.
"""

from __future__ import annotations

import math

import numpy as np

from fc_sat.easing import ease_in_out_cubic

A = 340.0
CX = 540.0
CY = 905.0
ZMAX = 36.0
EXACT_LIMIT = 2976
DRAW_CAP = 23808
GAP_SAMPLES = 65536

R1 = 8.0 * math.sqrt(2.0) * A / (math.pi ** 2)
CORNER = A * math.sqrt(2.0)
ASYMPTOTE = (2.0 * A) / (math.pi ** 2)  # gap(K) ~ ASYMPTOTE / K

_CURVES: dict[tuple[int, int], np.ndarray] = {}
_GAPS: dict[int, float] = {}
_PHASE = np.exp(1j * np.pi / 4.0)


def harmonic(index: int) -> int:
    """1-based circle index -> frequency 1, -3, 5, -7, ..."""
    return ((-1) ** (index + 1)) * (2 * index - 1)


def coefficient(index: int) -> complex:
    frequency = harmonic(index)
    return complex(_PHASE * (R1 / float(frequency * frequency)))


def asymptotic_gap(circles: int) -> float:
    return ASYMPTOTE / float(circles)


def _choose_n(circles: int, zoom: float, floor: int = 4096) -> int:
    """Power-of-two sample count: 8 per highest period, and 4 per screen pixel."""
    highest = max(1, 2 * int(circles) - 1)
    need = max(floor, 8 * highest, int(math.ceil(32.0 * A * max(float(zoom), 1.0))))
    n = 1 << (need - 1).bit_length()
    while n <= 2 * highest:
        n *= 2
    return int(n)


def full_curve(circles: int, n: int | None = None, zoom: float = 1.0) -> np.ndarray:
    """Complex samples of the settled K-circle curve. Cached per (K, n)."""
    circles = int(circles)
    if circles < 1:
        raise ValueError("need at least one circle")
    count = int(n or _choose_n(circles, zoom))
    key = (circles, count)
    cached = _CURVES.get(key)
    if cached is not None:
        return cached
    spec = np.zeros(count, dtype=np.complex128)
    indices = np.arange(1, circles + 1, dtype=np.int64)
    freqs = ((-1) ** (indices + 1) * (2 * indices - 1)).astype(np.int64)
    mags = R1 / (freqs.astype(np.float64) ** 2)
    spec[freqs % count] = _PHASE * mags
    samples = np.fft.ifft(spec) * count
    _CURVES[key] = samples
    return samples


def curve_last_weight(circles: int, weight: float, n: int) -> np.ndarray:
    """Circles 1..K-1 settled, circle K multiplied by ``weight``."""
    weight = float(weight)
    if circles <= 1:
        return full_curve(1, n) * weight
    if weight >= 1.0:
        return full_curve(circles, n)
    if weight <= 0.0:
        return full_curve(circles - 1, n)
    base = full_curve(circles - 1, n)
    freq = harmonic(circles)
    gain = coefficient(circles) * weight
    theta = (2.0 * np.pi) * (np.arange(n, dtype=np.float64) * (freq / float(n)))
    return base + gain * (np.cos(theta) + 1j * np.sin(theta))


def blend_doubling(before: int, weight: float, n: int) -> np.ndarray:
    """Blend the settled ``before`` curve toward ``2 * before``."""
    low = full_curve(int(before), n)
    high = full_curve(int(before) * 2, n)
    return (1.0 - float(weight)) * low + float(weight) * high


def square_complex(n: int) -> np.ndarray:
    """Axis-aligned square, corner (a, a) at t = 0, uniform in arc length."""
    theta = (2.0 * np.pi) * (np.arange(int(n), dtype=np.float64) / float(n))
    wrapped = np.mod(theta, 2.0 * np.pi)
    side = np.floor(wrapped / (np.pi / 2.0)).astype(np.int64)
    along = (wrapped - side * (np.pi / 2.0)) / (np.pi / 2.0)
    x = np.empty(n, dtype=np.float64)
    y = np.empty(n, dtype=np.float64)
    mask = side == 0
    x[mask] = A - 2.0 * A * along[mask]
    y[mask] = A
    mask = side == 1
    x[mask] = -A
    y[mask] = A - 2.0 * A * along[mask]
    mask = side == 2
    x[mask] = -A + 2.0 * A * along[mask]
    y[mask] = -A
    mask = side == 3
    x[mask] = A
    y[mask] = -A + 2.0 * A * along[mask]
    return x + 1j * y


def distance_to_boundary(samples: np.ndarray, half: float = A) -> np.ndarray:
    """Distance from each complex sample to the square's perimeter."""
    x = np.real(samples)
    y = np.imag(samples)
    ax = np.abs(x)
    ay = np.abs(y)
    inside = (ax <= half) & (ay <= half)
    inside_d = np.minimum(half - ax, half - ay)
    outside_d = np.hypot(x - np.clip(x, -half, half), y - np.clip(y, -half, half))
    return np.where(inside, inside_d, outside_d)


def exact_gap(circles: int, samples: int = GAP_SAMPLES) -> float:
    circles = int(circles)
    n = max(int(samples), _choose_n(circles, zoom=1.0, floor=samples))
    return float(np.max(distance_to_boundary(full_curve(circles, n))))


def gap(circles: int) -> float:
    """Max distance from the K-circle curve to the square.

    Exact on a dense sample through K = 2976. Beyond that the closed form
    ``2a / (pi^2 K)`` matches the exact value to well under 0.2%.
    """
    circles = int(circles)
    if circles < 1:
        raise ValueError("gap needs K >= 1")
    if circles > EXACT_LIMIT:
        return asymptotic_gap(circles)
    cached = _GAPS.get(circles)
    if cached is None:
        cached = exact_gap(circles)
        _GAPS[circles] = cached
    return cached


def first_k_within(tolerance: float = 0.5) -> int:
    """Smallest K whose gap is at or below ``tolerance`` pixels."""
    guess = max(1, int(math.ceil(ASYMPTOTE / float(tolerance) - 1e-9)))
    k = max(1, guess - 2)
    while gap(k) > tolerance:
        k += 1
        if k > guess + 8:
            raise RuntimeError(f"no K near {guess} reaches gap {tolerance}")
    return k


def circles_for_tolerance(half_side: float, tolerance: float) -> int:
    """How many circles a square of half-side ``half_side`` needs to be within ``tolerance``.

    The gap scales with the square, so the asymptotic inverts directly.
    Both arguments share a unit (metres, or millimetres).
    """
    return int(round((2.0 * float(half_side)) / (math.pi ** 2 * float(tolerance))))


def use_exact_square(circles: int, zoom: float) -> bool:
    """K above 2976 at zoom <= 2 is drawn as the square itself.

    The world-space gap at K = 2976 is about 0.023 px (under a twentieth of a
    pixel on screen at zoom 2). By K = 11904 it is under 0.01 px.
    """
    return int(circles) > EXACT_LIMIT and float(zoom) <= 2.0


def capped_circles(circles: int) -> int:
    """Doublings past 23808 reuse that curve. The counter still shows the true K."""
    return int(circles) if int(circles) <= DRAW_CAP else DRAW_CAP


def tip(circles: int, theta: float, weights: np.ndarray | None = None) -> complex:
    """Point on the curve at parameter ``theta``. ``weights`` aligns with circles 1..K."""
    total = 0j
    limit = int(circles)
    for index in range(1, limit + 1):
        gain = 1.0 if weights is None else float(weights[index - 1])
        total += coefficient(index) * gain * np.exp(1j * harmonic(index) * float(theta))
    return complex(total)


def partial_centers(count: int, theta: float, weights: np.ndarray) -> list[complex]:
    """Center of each of the first ``count`` circles, then the running tip is the last sum."""
    point = 0j
    centers = []
    for index in range(1, int(count) + 1):
        centers.append(point)
        point += coefficient(index) * float(weights[index - 1]) * np.exp(1j * harmonic(index) * float(theta))
    return centers


def zoom_at(frame: int) -> float:
    """Log-space zoom. Identity until 876, 36 by 948, hold, back to 1 from 1152 to 1280."""
    frame = int(frame)
    if frame < 876:
        return 1.0
    if frame <= 948:
        return math.exp(math.log(ZMAX) * ease_in_out_cubic((frame - 876) / (948 - 876)))
    if frame < 1152:
        return ZMAX
    if frame <= 1280:
        return math.exp(math.log(ZMAX) * (1.0 - ease_in_out_cubic((frame - 1152) / (1280 - 1152))))
    return 1.0


def corner_screen() -> np.ndarray:
    """Square corner (a, a) in math coordinates, as a screen point (y down)."""
    return np.array([CX + A, CY - A], dtype=np.float64)


def center_screen() -> np.ndarray:
    return np.array([CX, CY], dtype=np.float64)


def camera_anchor(zoom: float, zoom_max: float = ZMAX) -> np.ndarray:
    """World point that the zoom holds fixed. Zoom 1 leaves the anchor at the center."""
    center = center_screen()
    if zoom <= 1.0:
        return center.copy()
    alpha = (1.0 - 1.0 / float(zoom)) / (1.0 - 1.0 / float(zoom_max))
    return center + (corner_screen() - center) * alpha


def project_points(points: np.ndarray, zoom: float, zoom_max: float = ZMAX) -> np.ndarray:
    """screen = center + zoom * (world_screen - anchor). Zoom 1 is the identity."""
    anchor = camera_anchor(zoom, zoom_max)
    return center_screen() + float(zoom) * (np.asarray(points, dtype=np.float64) - anchor)


def math_to_screen(samples: np.ndarray) -> np.ndarray:
    """Complex math samples -> Nx2 screen points, y down, before the camera."""
    values = np.asarray(samples)
    return np.column_stack((CX + np.real(values), CY - np.imag(values))).astype(np.float64)


def square_corners_screen() -> np.ndarray:
    """Four corners, counterclockwise from (a, a), in screen space."""
    return math_to_screen(np.array([A + 1j * A, -A + 1j * A, -A - 1j * A, A - 1j * A], dtype=np.complex128))


def theta_at(frame: int) -> float:
    """One revolution per bar. The tip meets the corner ray on every bar line."""
    return (2.0 * math.pi) * (float(frame) / 96.0)


def fourier_compare(n: int = 2 ** 20, terms: int = 40) -> tuple[float, float]:
    """Relative closed-form error, and the largest FFT bin that is not k ≡ 1 (mod 4)."""
    samples = square_complex(n)
    spec = np.fft.fft(samples) / n
    worst = 0.0
    for index in range(1, terms + 1):
        freq = harmonic(index)
        closed = coefficient(index)
        got = spec[freq % n]
        worst = max(worst, abs(got - closed) / abs(closed))
    signed = np.arange(n)
    signed = np.where(signed <= n // 2, signed, signed - n)
    other = (signed != 0) & (np.mod(signed, 4) != 1)
    stray = float(np.max(np.abs(spec[other]))) if np.any(other) else 0.0
    return worst, stray
