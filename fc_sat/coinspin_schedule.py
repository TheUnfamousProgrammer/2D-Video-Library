"""What the counter area shows on each frame: one number, an equation, or the SAT chip.

A count only changes on a frame where the gold coin's face comes back upright, so the
viewer can check every number by eye. The counter never shows 0; before a count it shows ?.
"""

from __future__ import annotations

from dataclasses import dataclass

from fc_sat.coinspin_math import CARRY, DROP, ROAD, SAT_END, SNAP, upright_frames

N_FRAMES = 1824
UNIT = "1 SPIN = FACE UPRIGHT AGAIN"
EQUATION = (876, 1344)
SAT_COUNT = (1344, 1632)
SAT_HOLD = (1632, 1680)
SAT_SPLIT = (1680, SNAP[0])
CHIP = (1368, 1680)
CHIP_FADE = 1668
CHIP_SLIDE = 12
STRIKE = (SAT_END, SAT_END + 12)
PULSES = {
    0: (1272, 1680),
    1: (1296, 1740),
    2: (1320,),
}
PULSE_FRAMES = 12


@dataclass(frozen=True)
class Term:
    text: str
    color: str
    label: str = ""
    pulse: float = 0.0


@dataclass(frozen=True)
class Chip:
    label: str
    value: str
    alpha: float
    slide: float
    struck: float
    pulse: float


@dataclass(frozen=True)
class Counter:
    """``single`` shows one term under the SPINS label; ``equation`` shows a + b = result."""

    mode: str
    terms: tuple[Term, ...]
    label: str = ""
    sub: str = ""
    chip: Chip | None = None


def _pulse(frame: int, events: tuple[int, ...]) -> float:
    for event in events:
        if event <= frame < event + PULSE_FRAMES:
            return 1.0 - (frame - event) / float(PULSE_FRAMES)
    return 0.0


def _sat_count(frame: int) -> int:
    return sum(1 for tick in upright_frames() if SAT_COUNT[0] <= tick <= frame and tick < SAT_END)


def chip_at(frame: int) -> Chip | None:
    if not CHIP[0] <= frame < CHIP[1]:
        return None
    slide = min(1.0, (frame - CHIP[0] + 1) / float(CHIP_SLIDE))
    alpha = slide
    if frame >= CHIP_FADE:
        alpha *= 1.0 - (frame - CHIP_FADE) / float(CHIP[1] - CHIP_FADE)
    struck = 0.0 if frame < STRIKE[0] else min(1.0, (frame - STRIKE[0] + 1) / float(STRIKE[1] - STRIKE[0]))
    return Chip("SAT ANSWER", "3", alpha, slide, struck, _pulse(frame, (1572,)))


def counter_at(frame: int) -> Counter:
    frame = int(frame)
    if frame >= SNAP[0] or frame < 288:
        return Counter("single", (Term("?", "text"),), "SPINS", UNIT)
    if frame < DROP:
        return Counter("single", (Term("1", "text"),), "SPINS", UNIT)
    if frame < EQUATION[0]:
        return Counter("single", (Term("2", "gold"),), "SPINS", UNIT)
    if frame < EQUATION[1]:
        result = Term("2", "gold", pulse=_pulse(frame, PULSES[2]))
        if frame < ROAD[1]:
            return Counter("equation", (Term("?", "muted"), Term("?", "muted"), result))
        rolled = Term("1", "rolling", "ROLLING", _pulse(frame, PULSES[0]))
        if frame < CARRY[1]:
            return Counter("equation", (rolled, Term("?", "muted"), result))
        return Counter("equation", (rolled, Term("1", "trip", "TRIP AROUND", _pulse(frame, PULSES[1])), result))
    if frame < SAT_COUNT[1]:
        count = _sat_count(frame)
        return Counter("single", (Term(str(count) if count else "?", "text"),), "SPINS", chip=chip_at(frame))
    if frame < SAT_HOLD[1]:
        return Counter("single", (Term("4", "gold"),), "", chip=chip_at(frame))
    return Counter(
        "equation",
        (
            Term("3", "rolling", "ROLLING", _pulse(frame, PULSES[0])),
            Term("1", "trip", "TRIP AROUND", _pulse(frame, PULSES[1])),
            Term("4", "gold"),
        ),
        chip=chip_at(frame),
    )


def count_changes() -> list[int]:
    """Frames where a counted number first appears. Each must be an upright frame."""
    frames = []
    previous = counter_at(0)
    for frame in range(1, N_FRAMES):
        current = counter_at(frame)
        before = [term.text for term in previous.terms if term.text != "?"]
        after = [term.text for term in current.terms if term.text != "?"]
        if current.mode == previous.mode and after != before and len(after) >= len(before) and after:
            frames.append(frame)
        elif current.mode != previous.mode and after and frame in upright_frames():
            frames.append(frame)
        previous = current
    return frames


def screen_counts() -> list[int]:
    found: set[int] = set()
    for frame in range(N_FRAMES):
        counter = counter_at(frame)
        for term in counter.terms:
            if term.text.isdigit():
                found.add(int(term.text))
        if counter.chip is not None:
            found.add(int(counter.chip.value))
    return sorted(found)
