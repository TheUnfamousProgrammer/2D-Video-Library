"""Rolling without slipping, and where every body sits on each frame.

Screen y grows downward. Angles run clockwise from straight up, so a direction
is d(a) = (sin a, -cos a). In one lap the rolling coin's center goes once around
the fixed coin (phi from 0 to 2 pi). With no slipping the arrow on the rolling
coin turns (R + r) / r times as far, so a coin k times wider gives k + 1 spins.
The claims count spins from ``rolling_pose``, the same function the renderer draws.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from fc_sat.easing import ease_in_out_cubic, ease_out_cubic

TAU = 2.0 * math.pi
CX = 540.0
CY = 915.0
STAGE = 400.0
N_FRAMES = 1824
# A rolling coin smaller than this many pixels is drawn at this size inside a ring.
MIN_COIN_PX = 7.0


@dataclass(frozen=True)
class Lap:
    start: int
    end: int
    ratio: int
    counted: bool

    def contains(self, frame: int) -> bool:
        return self.start <= frame < self.end


LAPS = (
    Lap(0, 192, 1, True),
    Lap(216, 360, 2, True),
    Lap(384, 768, 3, True),
    Lap(768, 960, 3, False),
)
# (first frame, landing frame, ratio before, ratio after). The rolling coin sits on top.
GROWS = ((192, 216, 1.0, 2.0), (360, 384, 2.0, 3.0))
DOUBLING_FRAMES = tuple(960 + 24 * j for j in range(16))
DOUBLING_RATIOS = tuple(3 * 2 ** (j + 1) for j in range(16))
DOUBLING_EASE = 8
DOUBLING_LAP = 192
DOUBLING_END = 1344
EARTH_MORPH = (1344, 1392)
YEAR = (1392, 1488)
TRAIL = (1632, 1800)
SNAP = (1800, 1806)
SUN_PX = 150.0
ORBIT_PX = 330.0
EARTH_PX = 34.0
TROPICAL_YEAR = 365.24219


def direction(angle: float) -> np.ndarray:
    return np.array([math.sin(angle), -math.cos(angle)], dtype=np.float64)


def scale_for(ratio: float) -> float:
    """Pixels per rolling-coin radius. The whole pair fits inside the stage radius."""
    return STAGE / (float(ratio) + 2.0)


def rolling_pose(ratio: float, phi: float, theta: float, scale: float | None = None) -> dict:
    """Screen geometry of a rolling coin whose center is at orbit angle phi."""
    scale = scale_for(ratio) if scale is None else scale
    center = np.array([CX, CY], dtype=np.float64)
    rolling = center + (ratio + 1.0) * scale * direction(phi)
    return {
        "fixed_center": center,
        "fixed_px": ratio * scale,
        "rolling_center": rolling,
        "rolling_px": scale,
        "arrow": theta,
        "mark": rolling + scale * direction(theta),
        "contact": center + ratio * scale * direction(phi),
    }


def lap_theta(lap: Lap, frame: float) -> float:
    """Arrow angle inside a lap. No slipping: theta = (R + r) / r * phi."""
    phi = lap_phi(lap, frame)
    return (lap.ratio + 1.0) * phi


def lap_phi(lap: Lap, frame: float) -> float:
    u = (float(frame) - lap.start) / float(lap.end - lap.start)
    return TAU * min(1.0, max(0.0, u))


def lap_at(frame: int) -> Lap | None:
    for lap in LAPS:
        if lap.contains(int(frame)):
            return lap
    return None


def doubling_ratio(frame: float) -> float:
    """Big-coin ratio while it doubles. Each doubling eases over 8 frames from its beat."""
    frame = float(frame)
    ratio = 3.0
    for landing, after in zip(DOUBLING_FRAMES, DOUBLING_RATIOS):
        if frame < landing:
            break
        before = after / 2.0
        u = (frame - landing) / DOUBLING_EASE
        if u >= 1.0:
            ratio = float(after)
        else:
            ratio = math.exp(math.log(before) + (math.log(after) - math.log(before)) * ease_out_cubic(u))
    return ratio


def landed_ratio(frame: int) -> int:
    """The doubling ratio named on screen: the last beat that has landed."""
    shown = 3
    for landing, after in zip(DOUBLING_FRAMES, DOUBLING_RATIOS):
        if frame >= landing:
            shown = after
    return shown


def ratio_at(frame: int) -> float:
    frame = int(frame)
    if frame >= SNAP[0]:
        return 1.0
    for first, landing, before, after in GROWS:
        if first <= frame < landing:
            return before + (after - before) * ease_in_out_cubic((frame - first) / float(landing - first))
    if frame < 192:
        return 1.0
    if frame < 360:
        return 2.0
    if frame < DOUBLING_FRAMES[0]:
        return 3.0
    return doubling_ratio(min(frame, DOUBLING_END - 1))


@lru_cache(maxsize=1)
def _doubling_theta() -> np.ndarray:
    """Arrow angle while the big coin doubles, integrated frame by frame from phi."""
    frames = DOUBLING_END - DOUBLING_FRAMES[0] + 1
    theta = np.zeros(frames, dtype=np.float64)
    steps = 16
    dphi = TAU / DOUBLING_LAP / steps
    angle = 0.0
    for index in range(1, frames):
        base = DOUBLING_FRAMES[0] + index - 1
        for step in range(steps):
            ratio = doubling_ratio(base + (step + 0.5) / steps)
            angle += (ratio + 1.0) * dphi
        theta[index] = angle
    return theta


def phi_at(frame: int) -> float:
    frame = int(frame)
    lap = lap_at(frame)
    if lap is not None:
        return lap_phi(lap, frame)
    if DOUBLING_FRAMES[0] <= frame < DOUBLING_END:
        return TAU * (frame - DOUBLING_FRAMES[0]) / DOUBLING_LAP
    return 0.0


def theta_at(frame: int) -> float:
    frame = int(frame)
    lap = lap_at(frame)
    if lap is not None:
        return lap_theta(lap, frame)
    if DOUBLING_FRAMES[0] <= frame < DOUBLING_END:
        return float(_doubling_theta()[frame - DOUBLING_FRAMES[0]])
    return 0.0


@dataclass(frozen=True)
class Scene:
    """Bodies on one frame. In the sky the fixed body is the Sun and the rolling one is Earth."""

    fixed_center: tuple[float, float]
    fixed_px: float
    rolling_center: tuple[float, float]
    rolling_px: float
    arrow: float
    ratio: float
    phi: float
    sky: float
    year: float


def _mix(a: float, b: float, u: float) -> float:
    return a + (b - a) * u


def _coin_scene(frame: int) -> Scene:
    ratio = ratio_at(frame)
    pose = rolling_pose(ratio, phi_at(frame), theta_at(frame))
    return Scene(
        fixed_center=(float(pose["fixed_center"][0]), float(pose["fixed_center"][1])),
        fixed_px=float(pose["fixed_px"]),
        rolling_center=(float(pose["rolling_center"][0]), float(pose["rolling_center"][1])),
        rolling_px=float(pose["rolling_px"]),
        arrow=float(pose["arrow"]),
        ratio=ratio,
        phi=phi_at(frame),
        sky=0.0,
        year=0.0,
    )


def year_progress(frame: int) -> float:
    """Sine ease: the fastest Earth step is pi/2 times the average, about 34 px a frame."""
    if frame < YEAR[0]:
        return 0.0
    if frame >= YEAR[1]:
        return 1.0
    u = (frame - YEAR[0]) / float(YEAR[1] - YEAR[0])
    return 0.5 - 0.5 * math.cos(math.pi * u)


def earth_angle(frame: int) -> float:
    if YEAR[0] <= frame < YEAR[1]:
        return TAU * year_progress(frame)
    return 0.0


def _sky_scene(frame: int, year_angle: float) -> Scene:
    earth = np.array([CX, CY]) + ORBIT_PX * direction(year_angle)
    return Scene(
        fixed_center=(CX, CY),
        fixed_px=SUN_PX,
        rolling_center=(float(earth[0]), float(earth[1])),
        rolling_px=EARTH_PX,
        arrow=0.0,
        ratio=float(DOUBLING_RATIOS[-1]),
        phi=year_angle,
        sky=1.0,
        year=year_progress(frame),
    )


def _blend(a: Scene, b: Scene, u: float, frame: int) -> Scene:
    return Scene(
        fixed_center=a.fixed_center,
        fixed_px=_mix(a.fixed_px, b.fixed_px, u),
        rolling_center=(
            _mix(a.rolling_center[0], b.rolling_center[0], u),
            _mix(a.rolling_center[1], b.rolling_center[1], u),
        ),
        rolling_px=_mix(a.rolling_px, b.rolling_px, u),
        arrow=b.arrow if u >= 0.5 else a.arrow,
        ratio=b.ratio if u >= 0.5 else a.ratio,
        phi=b.phi if u >= 0.5 else a.phi,
        sky=_mix(a.sky, b.sky, u),
        year=year_progress(frame),
    )


def scene_at(frame: int) -> Scene:
    frame = int(frame)
    if frame >= SNAP[1]:
        return _coin_scene(0)
    if frame >= SNAP[0]:
        u = ease_out_cubic((frame - SNAP[0]) / float(SNAP[1] - SNAP[0]))
        return _blend(_sky_scene(frame, 0.0), _coin_scene(0), u, frame)
    if frame >= EARTH_MORPH[1]:
        return _sky_scene(frame, earth_angle(frame))
    if frame >= EARTH_MORPH[0]:
        start = _coin_scene(EARTH_MORPH[0] - 1)
        start = Scene(
            fixed_center=start.fixed_center,
            fixed_px=start.fixed_px,
            rolling_center=start.rolling_center,
            rolling_px=max(start.rolling_px, MIN_COIN_PX),
            arrow=start.arrow,
            ratio=start.ratio,
            phi=start.phi,
            sky=0.0,
            year=0.0,
        )
        u = ease_in_out_cubic((frame - EARTH_MORPH[0]) / float(EARTH_MORPH[1] - EARTH_MORPH[0]))
        return _blend(start, _sky_scene(frame, 0.0), u, frame)
    return _coin_scene(frame)


def trace_points(lap: Lap, upto: float, per_turn: int = 720) -> np.ndarray:
    """Screen path of the arrow tip's rim point from the lap start to ``upto`` (a frame)."""
    end_phi = lap_phi(lap, upto)
    count = max(2, int(math.ceil(per_turn * (lap.ratio + 1) * end_phi / TAU)) + 1)
    phis = np.linspace(0.0, end_phi, count)
    scale = scale_for(lap.ratio)
    radius = (lap.ratio + 1.0) * scale
    centers_x = CX + radius * np.sin(phis)
    centers_y = CY - radius * np.cos(phis)
    thetas = (lap.ratio + 1.0) * phis
    return np.stack([centers_x + scale * np.sin(thetas), centers_y - scale * np.cos(thetas)], axis=1)


def _check_rolling(ratio: float, samples: int, inside: bool) -> float:
    """Walk one lap through ``rolling_pose`` and return the arrow's total turn in radians.

    Every sample must touch the fixed coin and roll without slipping: the arc the
    contact sweeps on the fixed rim equals the arc it sweeps on the rolling rim.
    """
    scale = 1.0
    previous = None
    total = 0.0
    for index in range(samples + 1):
        phi = TAU * index / samples
        if inside:
            pose = inside_pose(ratio, phi, scale)
        else:
            pose = rolling_pose(ratio, phi, (ratio + 1.0) * phi, scale)
        gap = float(np.hypot(*(pose["rolling_center"] - pose["fixed_center"])))
        want = (ratio - 1.0 if inside else ratio + 1.0) * scale
        if abs(gap - want) > 1e-9:
            raise ValueError(f"rolling coin left the rim at phi {phi:.4f}")
        if previous is not None:
            prev_phi, prev_theta = previous
            fixed_arc = ratio * scale * (phi - prev_phi)
            contact_dir = phi if inside else phi + math.pi
            prev_dir = prev_phi if inside else prev_phi + math.pi
            body_turn = (contact_dir - pose["arrow"]) - (prev_dir - prev_theta)
            rolling_arc = abs(body_turn) * scale
            if abs(rolling_arc - fixed_arc) > 1e-9 * max(1.0, fixed_arc):
                raise ValueError(f"rolling coin slipped at phi {phi:.4f}")
            total += pose["arrow"] - prev_theta
        previous = (phi, pose["arrow"])
    return total


def inside_pose(ratio: float, phi: float, scale: float | None = None) -> dict:
    """A coin rolling around the inside of a ring ``ratio`` times wider. It turns the other way."""
    scale = scale_for(ratio) if scale is None else scale
    center = np.array([CX, CY], dtype=np.float64)
    rolling = center + (ratio - 1.0) * scale * direction(phi)
    theta = -(ratio - 1.0) * phi
    return {
        "fixed_center": center,
        "fixed_px": ratio * scale,
        "rolling_center": rolling,
        "rolling_px": scale,
        "arrow": theta,
    }


def count_spins(ratio: float, samples: int = 4096) -> int:
    turn = _check_rolling(float(ratio), samples, inside=False)
    return int(math.floor(abs(turn) / TAU + 1e-9))


def count_spins_inside(ratio: float, samples: int = 4096) -> int:
    turn = _check_rolling(float(ratio), samples, inside=True)
    return int(math.floor(abs(turn) / TAU + 1e-9))


def rolling_part(ratio: float) -> int:
    """Spins measured against the rim: rim length over the rolling coin's circumference."""
    rim = TAU * float(ratio)
    return int(round(rim / TAU))


def trip_bonus() -> int:
    bonuses = {count_spins(ratio) - ratio for ratio in (1, 2, 3, 10, *DOUBLING_RATIOS)}
    if len(bonuses) != 1:
        raise ValueError(f"the trip bonus is not constant: {sorted(bonuses)}")
    return int(bonuses.pop())


def rotations_per_year(solar_days: float) -> float:
    """Spins against the stars: each solar day is one turn against the Sun, plus one orbit."""
    turn_against_sun = TAU * float(solar_days)
    orbit = TAU
    return (turn_against_sun + orbit) / TAU


def whole_rotations(solar_days: float) -> int:
    return int(math.floor(rotations_per_year(solar_days)))


def sidereal_day(solar_days: float) -> float:
    return 86400.0 * float(solar_days) / rotations_per_year(solar_days)


def format_hms(seconds: float) -> str:
    total = int(math.floor(seconds))
    return f"{total // 3600}H {(total % 3600) // 60}M {total % 60}S"
