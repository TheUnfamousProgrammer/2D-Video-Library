"""Analytic Pixel Morph timeline. Every pose is a pure function of time."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PHASES = ("hold_a_start", "morph_ab", "hold_b", "morph_ba", "hold_a_end")


def smootherstep(u: np.ndarray | float) -> np.ndarray:
    """Perlin smootherstep, ``6u^5 - 15u^4 + 10u^3``, clamped to 0..1."""
    x = np.clip(np.asarray(u, dtype=np.float64), 0.0, 1.0)
    return x * x * x * (x * (x * 6.0 - 15.0) + 10.0)


def phase_frames(seconds: dict[str, float], fps: int = 60) -> list[tuple[str, int]]:
    frames = []
    for name in PHASES:
        count = int(round(float(seconds[name]) * fps))
        if count < 1:
            raise ValueError(f"phase '{name}' needs at least one frame")
        frames.append((name, count))
    return frames


def total_frames(plan: list[tuple[str, int]]) -> int:
    return int(sum(count for _, count in plan))


def phase_bounds(seconds: dict[str, float]) -> dict[str, tuple[float, float]]:
    """Map each phase name to ``(start_seconds, duration_seconds)``."""
    cursor = 0.0
    bounds: dict[str, tuple[float, float]] = {}
    for name in PHASES:
        duration = float(seconds[name])
        bounds[name] = (cursor, duration)
        cursor += duration
    return bounds


def phase_at(index: int, plan: list[tuple[str, int]]) -> tuple[str, int, float]:
    """Return ``(name, local_index, tau)`` with ``tau = i / (n - 1)``."""
    cursor = 0
    for name, count in plan:
        if index < cursor + count:
            local = index - cursor
            tau = 0.0 if count <= 1 else local / (count - 1)
            return name, local, float(tau)
        cursor += count
    name, count = plan[-1]
    return name, count - 1, 1.0


def rank_delays(cols: int, rows: int, delay_frac: float) -> np.ndarray:
    """``delay_frac`` times the rank of each cell's distance from the center.

    Rank 0 is the cell nearest the center, so motion ripples outward.
    Ties keep mergesort order. Values lie in ``[0, delay_frac]``.
    """
    ys, xs = np.mgrid[0:rows, 0:cols]
    cx = (cols - 1) / 2.0
    cy = (rows - 1) / 2.0
    dist = np.hypot(xs.ravel() - cx, ys.ravel() - cy)
    order = np.argsort(dist, kind="mergesort")
    ranks = np.empty(dist.size, dtype=np.float64)
    denom = max(dist.size - 1, 1)
    ranks[order] = np.arange(dist.size, dtype=np.float64) / denom
    return delay_frac * ranks


def arc_scales(n: int, seed: int) -> np.ndarray:
    """Signed scales in ``[-1, -0.5] U [0.5, 1]``, fixed per particle.

    Each particle uses ``numpy.random.default_rng([seed, i])``.
    """
    scales = np.empty(n, dtype=np.float64)
    for index in range(n):
        rng = np.random.default_rng([int(seed), int(index)])
        magnitude = float(rng.uniform(0.5, 1.0))
        sign = -1.0 if rng.random() < 0.5 else 1.0
        scales[index] = sign * magnitude
    return scales


def cell_centers(cols: int, rows: int, cell: float, origin_x: float, origin_y: float) -> np.ndarray:
    ys, xs = np.mgrid[0:rows, 0:cols]
    points = np.empty((rows * cols, 2), dtype=np.float64)
    points[:, 0] = origin_x + (xs.ravel() + 0.5) * cell
    points[:, 1] = origin_y + (ys.ravel() + 0.5) * cell
    return points


@dataclass(frozen=True)
class Pose:
    position: np.ndarray
    lab: np.ndarray
    ease: np.ndarray
    progress: np.ndarray
    settled: bool
    showing: str


def leg_pose(
    tau: float,
    start: np.ndarray,
    end: np.ndarray,
    start_lab: np.ndarray,
    end_lab: np.ndarray,
    delay: np.ndarray,
    arc_sign: np.ndarray,
    *,
    delay_frac: float,
    arc_amp: float,
    recolor_strength: float,
    box: tuple[float, float, float, float],
    inset: float,
) -> Pose:
    """One morph leg. ``arc_sign`` is flipped by the caller on the return leg."""
    span = 1.0 - float(delay_frac)
    if span <= 1e-9:
        raise ValueError("delay_frac must be below 1")
    u = np.clip((float(tau) - delay) / span, 0.0, 1.0)
    ease = smootherstep(u)
    delta = end - start
    length = np.hypot(delta[:, 0], delta[:, 1])
    safe = np.maximum(length, 1e-12)
    normal = np.stack((-delta[:, 1] / safe, delta[:, 0] / safe), axis=1)
    normal[length < 1e-9] = 0.0
    amplitude = float(arc_amp) * length * arc_sign
    offset = normal * (amplitude * np.sin(np.pi * ease))[:, None]
    position = start + delta * ease[:, None] + offset
    # Rest centers stay on the cell grid so a settled frame matches the
    # nearest-neighbor picture. In flight the center is held inside the
    # inset; the renderer also clips the square to the box.
    flying = (ease > 1e-8) & (ease < 1.0 - 1e-8)
    if np.any(flying):
        x0, y0, x1, y1 = box
        position[flying, 0] = np.clip(position[flying, 0], x0 + inset, x1 - inset)
        position[flying, 1] = np.clip(position[flying, 1], y0 + inset, y1 - inset)
    blend = float(recolor_strength) * ease
    lab = start_lab * (1.0 - blend)[:, None] + end_lab * blend[:, None]
    settled = bool(np.all((u <= 0.0) | (u >= 1.0)))
    return Pose(position, lab, ease, u, settled, "flight")


def rest_pose(
    position: np.ndarray,
    lab: np.ndarray,
    showing: str,
) -> Pose:
    n = position.shape[0]
    ease = np.ones(n, dtype=np.float64) if showing == "b" else np.zeros(n, dtype=np.float64)
    return Pose(position.copy(), lab.copy(), ease, ease.copy(), True, showing)


def arrival_tau(delay: np.ndarray, delay_frac: float) -> np.ndarray:
    """Normalized leg time at which each particle first reaches ease 1.

    ``tau = 1 - delay_frac + delay``, which is 1 when ``delay == delay_frac``.
    """
    return 1.0 - float(delay_frac) + delay
