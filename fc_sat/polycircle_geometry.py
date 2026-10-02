"""Regular polygons, the sagitta gap, the camera, and the rotation that lands on the freeze angle.

Angles are mathematical: 0 is +x, positive angles run clockwise because y grows downward.
A polygon whose vertices are in strictly increasing angle around an interior point, with
each step under pi, is convex. Morphs interpolate angles, never cartesian positions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from fc_sat.easing import ease_in_out_cubic

PHI_F = -math.pi / 2 - math.pi / 96
GOLDEN_TURN = 0.3819660112501051  # one golden-angle step, as a fraction of the circle
DRAW_CAP = 2048


def gap(n: int, radius: float) -> float:
    """Maximum distance from an edge of a regular n-gon to its circumcircle."""
    n = int(n)
    if n < 3:
        raise ValueError(f"n must be at least 3, got {n}")
    return float(radius) * (1.0 - math.cos(math.pi / n))


def edge_length(n: int, radius: float) -> float:
    return 2.0 * float(radius) * math.sin(math.pi / int(n))


def first_n_within(radius: float, tolerance: float) -> int:
    """Smallest n whose sagitta is at most ``tolerance``."""
    if tolerance <= 0 or radius <= 0:
        raise ValueError("radius and tolerance must be positive")
    ratio = 1.0 - (tolerance / radius)
    if ratio <= -1:
        return 3
    n = math.pi / math.acos(min(1.0, max(-1.0, ratio)))
    candidate = max(3, int(math.ceil(n - 1e-9)))
    while gap(candidate, radius) > tolerance:
        candidate += 1
    if candidate > 3 and gap(candidate - 1, radius) <= tolerance:
        candidate -= 1
    return candidate


def sides_for_gap(radius: float, tolerance: float) -> int:
    """Alias used by the claims engine. ``radius`` and ``tolerance`` share a unit."""
    return first_n_within(radius, tolerance)


def insertion_edge(index: int, n_before: int) -> int:
    """Edge split by add ``index``. The step is the golden angle, so splits do not pile up."""
    frac = (index * GOLDEN_TURN) % 1.0
    return int(math.floor(frac * n_before)) % n_before


def regular_angles(n: int, phi: float) -> np.ndarray:
    return np.asarray(phi, dtype=np.float64) + (2.0 * math.pi * np.arange(n, dtype=np.float64) / n)


def _ease(frame: int, start: int, duration: int) -> float:
    if duration <= 1:
        return 1.0
    raw = (frame - start) / (duration - 1)
    return ease_in_out_cubic(raw)


def add_morph(n_before: int, phi: float, radius: float, edge: int, u: float) -> tuple[np.ndarray, np.ndarray]:
    """+1 morph. The new vertex starts on the chosen edge and moves out to the circle."""
    n = int(n_before)
    old = regular_angles(n, phi)
    step_old = math.pi / n
    start_a = np.empty(n + 1, dtype=np.float64)
    end_a = np.empty(n + 1, dtype=np.float64)
    start_r = np.empty(n + 1, dtype=np.float64)
    end_r = np.full(n + 1, radius, dtype=np.float64)
    cursor = 0
    for k in range(edge + 1):
        start_a[cursor] = old[k]
        end_a[cursor] = phi + 2.0 * math.pi * k / (n + 1)
        start_r[cursor] = radius
        cursor += 1
    start_a[cursor] = old[edge] + step_old
    end_a[cursor] = phi + 2.0 * math.pi * (edge + 1) / (n + 1)
    start_r[cursor] = radius * math.cos(step_old)
    cursor += 1
    for k in range(edge + 1, n):
        start_a[cursor] = old[k]
        end_a[cursor] = phi + 2.0 * math.pi * (k + 1) / (n + 1)
        start_r[cursor] = radius
        cursor += 1
    angles = (1.0 - u) * start_a + u * end_a
    radii = (1.0 - u) * start_r + u * end_r
    return angles, radii


def double_morph(n_before: int, phi: float, radius: float, u: float) -> tuple[np.ndarray, np.ndarray]:
    """Archimedes doubling: old vertices stay, new ones grow from the edge midpoints."""
    n = int(n_before)
    old = regular_angles(n, phi)
    mids = phi + (2.0 * math.pi * (np.arange(n, dtype=np.float64) + 0.5) / n)
    r_mid = (1.0 - u) * radius * math.cos(math.pi / n) + u * radius
    angles = np.empty(2 * n, dtype=np.float64)
    radii = np.empty(2 * n, dtype=np.float64)
    angles[0::2] = old
    angles[1::2] = mids
    radii[0::2] = radius
    radii[1::2] = r_mid
    return angles, radii


def points_from(center: np.ndarray, angles: np.ndarray, radii: np.ndarray) -> np.ndarray:
    pts = np.empty((len(angles), 2), dtype=np.float64)
    pts[:, 0] = center[0] + radii * np.cos(angles)
    pts[:, 1] = center[1] + radii * np.sin(angles)
    return pts


def monotone_angles(angles: np.ndarray) -> bool:
    if len(angles) < 3:
        return False
    steps = np.diff(angles)
    if np.any(steps <= 1e-9) or np.any(steps >= math.pi):
        return False
    wrap = (angles[0] + 2.0 * math.pi) - angles[-1]
    return bool(1e-9 < wrap < math.pi)


def is_convex(points: np.ndarray) -> bool:
    n = len(points)
    if n < 3:
        return False
    signs = np.empty(n, dtype=np.float64)
    for i in range(n):
        a = points[i]
        b = points[(i + 1) % n]
        c = points[(i + 2) % n]
        signs[i] = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
    # A new vertex starts on an edge, so one turn may be zero. That polygon is still convex.
    return bool(np.all(signs >= -1e-4) or np.all(signs <= 1e-4))


def _speed_shape(frames: np.ndarray) -> np.ndarray:
    """Unscaled deg/s. Linear 30 -> 150 by frame 700, then easeInOutCubic down to 0 at frame 864."""
    f = np.asarray(frames, dtype=np.float64)
    out = np.empty(f.shape, dtype=np.float64)
    rise = f <= 700.0
    out[rise] = 30.0 + 120.0 * (f[rise] / 700.0)
    fall = (f > 700.0) & (f < 864.0)
    u = (f[fall] - 700.0) / 164.0
    eased = np.where(u < 0.5, 4.0 * u ** 3, 1.0 - ((-2.0 * u + 2.0) ** 3) / 2.0)
    out[fall] = 150.0 * (1.0 - eased)
    out[f >= 864.0] = 0.0
    return out


@dataclass(frozen=True)
class RotationSolution:
    scale: float
    turns: int
    phi0: float
    phi_f: float
    speed0_deg_s: float
    speed700_deg_s: float
    total_deg: float


def solve_rotation(fps: int = 60) -> RotationSolution:
    """Scale the speed shape so the spin from frame 0 to 864 is an integer number of quarter-turns.

    The unscaled shape is the brief's 30 deg/s to 150 deg/s to 0. The nearest quarter-turn
    count keeps the scale near 1. Frame 0 then equals the freeze angle modulo pi/2, which
    is the square's symmetry, and frame 864 is the freeze angle exactly.
    """
    frames = np.arange(0, 864, dtype=np.float64)
    unscaled = float(np.sum(_speed_shape(frames) * (math.pi / 180.0) / fps))
    quarter = math.pi / 2.0
    turns = max(1, int(round(unscaled / quarter)))
    scale = (turns * quarter) / unscaled
    phi0 = PHI_F - turns * quarter
    return RotationSolution(
        scale=scale,
        turns=turns,
        phi0=phi0,
        phi_f=PHI_F,
        speed0_deg_s=scale * 30.0,
        speed700_deg_s=scale * 150.0,
        total_deg=turns * 90.0,
    )


def rotation_angles(solution: RotationSolution, n_frames: int, fps: int = 60) -> np.ndarray:
    """Angle at each frame. Index 864 is exactly the freeze angle; later frames hold it."""
    frames = np.arange(0, 864, dtype=np.float64)
    per_frame = solution.scale * _speed_shape(frames) * (math.pi / 180.0) / fps
    relative = np.cumsum(per_frame)
    angles = np.empty(n_frames, dtype=np.float64)
    angles[0] = solution.phi0
    last = min(n_frames - 1, 864)
    if last >= 1:
        angles[1 : last + 1] = solution.phi0 + relative[:last]
    if n_frames > 864:
        angles[864:] = solution.phi_f
    return angles


def zoom_at(frame: int) -> float:
    """Log-space zoom. 1 until 876, ease to 36 by 948, hold, ease back to 1 from 1152 to 1280."""
    if frame < 876:
        return 1.0
    if frame <= 948:
        u = ease_in_out_cubic((frame - 876) / (948 - 876))
        return math.exp(math.log(36.0) * u)
    if frame < 1152:
        return 36.0
    if frame <= 1280:
        u = ease_in_out_cubic((frame - 1152) / (1280 - 1152))
        return math.exp(math.log(36.0) * (1.0 - u))
    return 1.0


def camera_anchor(center: np.ndarray, top: np.ndarray, zoom: float, zoom_max: float = 36.0) -> np.ndarray:
    if zoom <= 1.0:
        return np.array(center, dtype=np.float64, copy=True)
    alpha = (1.0 - 1.0 / zoom) / (1.0 - 1.0 / zoom_max)
    return np.asarray(center, dtype=np.float64) + (np.asarray(top, dtype=np.float64) - center) * alpha


def project(points: np.ndarray, center: np.ndarray, top: np.ndarray, zoom: float, zoom_max: float = 36.0) -> np.ndarray:
    anchor = camera_anchor(center, top, zoom, zoom_max)
    return np.asarray(center, dtype=np.float64) + zoom * (np.asarray(points, dtype=np.float64) - anchor)


def angular_window(zoom: float, radius: float, width: float, margin: float) -> tuple[float, float] | None:
    """Top-arc angles visible on screen, or None when the whole circle is drawn."""
    if zoom <= 1.001:
        return None
    reach = zoom * radius
    half = width * 0.5 + margin
    if reach <= half:
        return None
    delta = math.asin(min(1.0, half / reach))
    pad = max(0.02, 12.0 / reach)
    return (-math.pi / 2.0 - delta - pad, -math.pi / 2.0 + delta + pad)


def cap_error_px(radius: float, n_true: int, cap: int = DRAW_CAP) -> float:
    """How far a ``cap``-gon sits off the true n-gon, at zoom 1. Under 0.001 px for n > cap."""
    if n_true <= cap:
        return 0.0
    return gap(cap, radius) - gap(n_true, radius)


def vertices_in_window(n: int, phi: float, radius: float, window: tuple[float, float] | None, cap: int = DRAW_CAP) -> tuple[np.ndarray, np.ndarray]:
    """Regular n-gon vertices. At zoom 1, n above ``cap`` is drawn as a cap-gon."""
    if window is None:
        draw_n = n if n <= cap else cap
        if n > cap:
            err = cap_error_px(radius, n, cap)
            if err >= 0.001:
                raise AssertionError(f"cap error {err:.6f} px is not under 0.001")
        angles = regular_angles(draw_n, phi)
        return angles, np.full(draw_n, radius, dtype=np.float64)
    t0, t1 = window
    step = 2.0 * math.pi / n
    k0 = int(math.ceil((t0 - phi) / step - 1e-9))
    k1 = int(math.floor((t1 - phi) / step + 1e-9))
    count = k1 - k0 + 1
    if count <= 1:
        k0 -= 1
        k1 += 1
        count = k1 - k0 + 1
    stride = 1 if count <= cap else int(math.ceil(count / cap))
    ks = np.arange(k0, k1 + 1, stride, dtype=np.float64)
    angles = phi + step * ks
    return angles, np.full(len(ks), radius, dtype=np.float64)


def culled_matches_full(n: int, phi: float, radius: float, center: np.ndarray, window: tuple[float, float] | None) -> bool:
    """Culling is a filter: every kept vertex is a real vertex, and a full window keeps them all."""
    full_a, full_r = vertices_in_window(n, phi, radius, None, cap=max(n, 8))
    if window is None:
        got_a, _ = vertices_in_window(n, phi, radius, None, cap=max(n, 8))
        return bool(np.allclose(full_a, got_a))
    got_a, _ = vertices_in_window(n, phi, radius, window, cap=max(n, 8))
    if len(got_a) == 0:
        return False
    # Each culled angle equals some full angle modulo 2 pi.
    for angle in got_a:
        delta = np.abs((full_a - angle + math.pi) % (2 * math.pi) - math.pi)
        if float(delta.min()) > 1e-6:
            return False
    return True


@dataclass
class MorphSpec:
    kind: str
    frame: int
    duration: int
    n_before: int
    edge: int = 0


def _double_in_window(
    n_before: int,
    phi: float,
    radius: float,
    u: float,
    window: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Doubling vertices inside the visible arc. Old vertices stay; new ones grow outward."""
    n = int(n_before)
    t0, t1 = window
    step = 2.0 * math.pi / n
    k0 = int(math.floor((t0 - phi) / step)) - 1
    k1 = int(math.ceil((t1 - phi) / step)) + 1
    r_mid = (1.0 - u) * radius * math.cos(math.pi / n) + u * radius
    angles = []
    radii = []
    for k in range(k0, k1 + 1):
        old = phi + step * k
        mid = old + step * 0.5
        if t0 <= old <= t1:
            angles.append(old)
            radii.append(radius)
        if t0 <= mid <= t1:
            angles.append(mid)
            radii.append(r_mid)
    if len(angles) < 2:
        return double_morph(min(n, 64), phi, radius, u)
    order = np.argsort(np.asarray(angles, dtype=np.float64))
    got_a = np.asarray(angles, dtype=np.float64)[order]
    got_r = np.asarray(radii, dtype=np.float64)[order]
    if len(got_a) > DRAW_CAP:
        stride = int(math.ceil(len(got_a) / DRAW_CAP))
        got_a = got_a[::stride]
        got_r = got_r[::stride]
    return got_a, got_r


def polygon_arrays(
    n_settled: int,
    phi: float,
    radius: float,
    morph: MorphSpec | None,
    frame: int,
    window: tuple[float, float] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Vertices for this frame. Large settled polygons are capped or windowed."""
    if morph is not None and morph.kind == "add":
        u = _ease(frame, morph.frame, morph.duration)
        return add_morph(morph.n_before, phi, radius, morph.edge, u)
    if morph is not None and morph.kind == "double":
        u = _ease(frame, morph.frame, morph.duration)
        if window is None and morph.n_before * 2 <= DRAW_CAP:
            return double_morph(morph.n_before, phi, radius, u)
        if window is None:
            err = cap_error_px(radius, morph.n_before * 2, DRAW_CAP)
            if err >= 0.001:
                raise AssertionError(f"cap error {err:.6f} px is not under 0.001")
            return vertices_in_window(DRAW_CAP, phi, radius, None, cap=DRAW_CAP)
        return _double_in_window(morph.n_before, phi, radius, u, window)
    return vertices_in_window(n_settled, phi, radius, window)


def circle_samples(phi_unused: float, radius: float, zoom: float, window: tuple[float, float] | None) -> np.ndarray:
    """Angles for the reference circle. Chords stay at or under 4 screen pixels."""
    del phi_unused
    reach = max(zoom * radius, 1.0)
    step = 4.0 / reach
    if window is None:
        count = max(32, int(math.ceil(2.0 * math.pi / step)))
        return -math.pi + (2.0 * math.pi * np.arange(count, dtype=np.float64) / count)
    t0, t1 = window
    count = max(8, int(math.ceil((t1 - t0) / step)))
    return np.linspace(t0, t1, count, dtype=np.float64)
