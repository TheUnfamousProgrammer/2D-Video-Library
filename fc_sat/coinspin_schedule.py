"""What the counter shows on each frame, and the frames where a spin lands."""

from __future__ import annotations

import math
from dataclasses import dataclass

from fc_sat.coinspin_math import (
    DOUBLING_FRAMES,
    LAPS,
    SNAP,
    TAU,
    TROPICAL_YEAR,
    YEAR,
    landed_ratio,
    lap_theta,
    rotations_per_year,
    year_progress,
)

N_FRAMES = 1824
# Frames where a finished lap's count is held in gold before the next coin.
HOLDS = ((192, 216, 2), (360, 384, 3), (768, 876, 4))
FORMULA_SAT = (876, 960)
SKY_ROWS = (1344, 1632)
FORMULA_YEAR = (1632, 1800)


@dataclass(frozen=True)
class Counter:
    """``live`` and ``hold`` show one number, ``formula`` shows value + plus, ``sky`` shows two rows."""

    mode: str
    value: int | None = None
    plus: int | None = None
    days: int | None = None
    spins: int | None = None
    gold: bool = False


def live_spins(frame: int) -> int | None:
    for lap in LAPS:
        if lap.counted and lap.start <= frame < lap.end:
            return int(math.floor(lap_theta(lap, frame) / TAU + 1e-9))
    return None


def sky_counts(frame: int) -> tuple[int, int]:
    progress = year_progress(frame)
    days = int(math.floor(TROPICAL_YEAR * progress + 1e-9))
    spins = int(math.floor(rotations_per_year(TROPICAL_YEAR) * progress + 1e-9))
    return days, spins


def counter_at(frame: int) -> Counter:
    frame = int(frame)
    if frame >= SNAP[0]:
        return Counter("live", value=0)
    for first, end, value in HOLDS:
        if first <= frame < end:
            return Counter("hold", value=value, gold=True)
    if FORMULA_SAT[0] <= frame < FORMULA_SAT[1]:
        return Counter("formula", value=LAPS[2].ratio, plus=1)
    if DOUBLING_FRAMES[0] <= frame < SKY_ROWS[0]:
        return Counter("formula", value=landed_ratio(frame), plus=1)
    if SKY_ROWS[0] <= frame < SKY_ROWS[1]:
        days, spins = sky_counts(frame)
        return Counter("sky", days=days, spins=spins, gold=frame >= YEAR[1])
    if FORMULA_YEAR[0] <= frame < FORMULA_YEAR[1]:
        days, _spins = sky_counts(YEAR[1])
        return Counter("formula", value=days, plus=1)
    value = live_spins(frame)
    return Counter("live", value=0 if value is None else value)


def spin_frames() -> list[int]:
    """Frames where a counted lap finishes a spin: the arrow points straight up again."""
    frames = []
    for lap in LAPS:
        if not lap.counted:
            continue
        span = lap.end - lap.start
        for spin in range(1, lap.ratio + 2):
            frames.append(lap.start + span * spin // (lap.ratio + 1))
    return frames


def uncounted_spin_frames() -> list[int]:
    frames = []
    for lap in LAPS:
        if lap.counted:
            continue
        span = lap.end - lap.start
        for spin in range(1, lap.ratio + 2):
            frames.append(lap.start + span * spin // (lap.ratio + 1))
    return frames


def quarter_frames() -> list[int]:
    """Frames where the arrow passes a quarter turn that is not a full spin."""
    frames = []
    for lap in LAPS:
        span = lap.end - lap.start
        quarters = 4 * (lap.ratio + 1)
        for index in range(1, quarters):
            if index % 4 == 0:
                continue
            frames.append(lap.start + span * index // quarters)
    return frames


def screen_counts() -> list[int]:
    found: set[int] = set()
    for frame in range(N_FRAMES):
        counter = counter_at(frame)
        for value in (counter.value, counter.plus, counter.days, counter.spins):
            if value is not None:
                found.add(int(value))
    return sorted(found)
