"""Frame-snapped side counts. The musical grid is in beats; frames are the nearest 60 fps slot.

At 150 bpm every beat is an integer 24 frames, so the snap error is 0.
Other tempos still snap, and the build refuses a snap error above half a frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from fc_sat.polycircle_geometry import insertion_edge


def snap_frame(beats: float, bpm: float, fps: int) -> tuple[int, float]:
    exact = float(beats) * (60.0 / float(bpm)) * int(fps)
    frame = int(round(exact))
    return frame, abs(exact - frame)


@dataclass(frozen=True)
class AddEvent:
    frame: int
    n_after: int
    interval: int
    edge: int
    index: int


@dataclass(frozen=True)
class DoubleEvent:
    frame: int
    n_before: int
    n_after: int
    index: int


@dataclass
class Schedule:
    bpm: float
    fps: int
    n_frames: int
    max_snap_error: float
    adds: list[AddEvent]
    doubles: list[DoubleEvent]
    n_at: np.ndarray
    bar_starts: list[int] = field(default_factory=list)

    def n(self, frame: int) -> int:
        frame = min(max(int(frame), 0), self.n_frames - 1)
        return int(self.n_at[frame])


def _beat_groups(bpm: float, fps: int) -> tuple[list[float], list[float], float]:
    adds: list[float] = []
    for k in range(1, 9):
        adds.append(float(k))
    for k in range(1, 17):
        adds.append(8.0 + 0.5 * k)
    for k in range(1, 33):
        adds.append(16.0 + 0.25 * k)
    for k in range(1, 33):
        adds.append(24.0 + 0.125 * k)
    for beat in (29.0, 30.0, 31.0, 32.0):
        adds.append(beat)
    doubles = [40.0 + j for j in range(8)] + [48.0 + j for j in range(8)]
    error = 0.0
    for beat in adds + doubles + [0.0]:
        _, err = snap_frame(beat, bpm, fps)
        error = max(error, err)
    return adds, doubles, error


def build_schedule(bpm: float = 150.0, fps: int = 60, n_frames: int = 1824) -> Schedule:
    add_beats, double_beats, error = _beat_groups(bpm, fps)
    if error > 0.5:
        raise SystemExit(f"maximum snap error is {error:.3f} frames; the limit is 0.5")
    add_frames = [snap_frame(beat, bpm, fps)[0] for beat in add_beats]
    double_frames = [snap_frame(beat, bpm, fps)[0] for beat in double_beats]
    if len(add_frames) != 92:
        raise SystemExit(f"expected 92 adds, got {len(add_frames)}")
    if len(double_frames) != 16:
        raise SystemExit(f"expected 16 doublings, got {len(double_frames)}")

    intervals = [add_frames[i + 1] - add_frames[i] for i in range(len(add_frames) - 1)]
    intervals.append(max(1, snap_frame(1.0, bpm, fps)[0]))
    adds: list[AddEvent] = []
    n = 4
    for index, (frame, interval) in enumerate(zip(add_frames, intervals)):
        edge = insertion_edge(index, n)
        n += 1
        adds.append(AddEvent(frame=frame, n_after=n, interval=interval, edge=edge, index=index))
    if n != 96:
        raise SystemExit(f"adds should finish at n=96, got {n}")

    doubles: list[DoubleEvent] = []
    for index, frame in enumerate(double_frames):
        before = n
        n *= 2
        doubles.append(DoubleEvent(frame=frame, n_before=before, n_after=n, index=index))

    n_at = np.full(n_frames, 4, dtype=np.int64)
    cursor = 4
    events = [(item.frame, "add", item.n_after) for item in adds] + [
        (item.frame, "double", item.n_after) for item in doubles
    ]
    events.sort(key=lambda item: (item[0], 0 if item[1] == "add" else 1))
    position = 0
    for frame, _kind, n_after in events:
        frame = min(frame, n_frames)
        if frame > position:
            n_at[position:frame] = cursor
        cursor = n_after
        position = frame
    if position < n_frames:
        n_at[position:] = cursor

    # The snap back to a square occupies frames 1800-1805 and then holds.
    for frame in range(min(n_frames, 1800), n_frames):
        if frame >= 1806:
            n_at[frame] = 4
        else:
            u = (frame - 1800) / 5.0
            log_n = (1.0 - u) * np.log(6_291_456.0) + u * np.log(4.0)
            n_at[frame] = max(4, int(round(math_exp(log_n))))

    bar_starts = [snap_frame(4.0 * bar, bpm, fps)[0] for bar in range(19)]
    return Schedule(
        bpm=bpm,
        fps=fps,
        n_frames=n_frames,
        max_snap_error=error,
        adds=adds,
        doubles=doubles,
        n_at=n_at,
        bar_starts=bar_starts,
    )


def math_exp(value: float) -> float:
    import math

    return math.exp(value)


def active_morph(schedule: Schedule, frame: int):
    """The morph that owns this frame, if an add or doubling is still easing."""
    for item in schedule.adds:
        duration = min(6, item.interval)
        if item.frame <= frame <= item.frame + duration - 1:
            return ("add", item, duration)
    for item in schedule.doubles:
        duration = 10
        if item.frame <= frame <= item.frame + duration - 1:
            return ("double", item, duration)
    return None
