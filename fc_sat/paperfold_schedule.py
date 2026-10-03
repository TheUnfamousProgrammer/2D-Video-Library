"""When each fold lands. Frames are exact at 150 bpm and 60 fps."""

from __future__ import annotations

from dataclasses import dataclass

from fc_sat.beatkit.grid import FRAMES_PER_BAR, FRAMES_PER_BEAT, N_FRAMES, snap_frame


@dataclass(frozen=True)
class FoldEvent:
    n: int
    frame: int


def fold_frame(n: int) -> int:
    """Frame where fold n finishes. n is 1..42."""
    if n < 1 or n > 42:
        raise ValueError(f"fold {n} is outside 1..42")
    if n <= 14:
        return 12 * n
    if n <= 30:
        return 216 + 24 * (n - 15)
    return 864 + 24 * (n - 31)


def build_folds() -> list[FoldEvent]:
    return [FoldEvent(n, fold_frame(n)) for n in range(1, 43)]


def max_snap_error(bpm: float = 150.0, fps: int = 60) -> float:
    """The grid is integer at 150 bpm. Other tempos still have to stay under half a frame."""
    error = 0.0
    for beat in range(0, 76):
        _, err = snap_frame(float(beat), bpm, fps)
        error = max(error, err)
    return error


# Fold 6 lands at frame 72. The camera swings from the sheet to the tower across
# the next 12 frames. Fold 7 lands at frame 84: the ghost copy has doubled the
# stack, and the tower view is fully up.
TRANSITION_START = 72
TRANSITION_END = 84


def camera_at(frame: int) -> str:
    if frame < TRANSITION_START:
        return "topdown"
    if frame < TRANSITION_END:
        return "transition"
    if frame < 1248:
        return "tower"
    if frame < 1440:
        return "reality"
    if frame < 1632:
        return "answer"
    if frame < 1800:
        return "outro"
    return "collapse"


SEGMENTS = (
    ("hook", 0, 192),
    ("rising", 192, 576),
    ("drop", 576, 672),
    ("twist", 672, 864),
    ("rising2", 864, 1152),
    ("climax", 1152, 1248),
    ("reality", 1248, 1440),
    ("answer", 1440, 1632),
    ("outro", 1632, 1824),
)


def segment_at(frame: int) -> str:
    for name, start, end in SEGMENTS:
        if start <= frame < end:
            return name
    raise ValueError(f"frame {frame} is outside 0..{N_FRAMES - 1}")


def bar_start(bar_1based: int) -> int:
    return (bar_1based - 1) * FRAMES_PER_BAR


def beats_in(start: int, end: int) -> list[int]:
    first = start if start % FRAMES_PER_BEAT == 0 else start + (FRAMES_PER_BEAT - start % FRAMES_PER_BEAT)
    return list(range(first, end, FRAMES_PER_BEAT))
