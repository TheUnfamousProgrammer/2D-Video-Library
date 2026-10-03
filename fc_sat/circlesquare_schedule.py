"""When each circle arrives. Frames are exact at 150 bpm and 60 fps."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from fc_sat.beatkit.grid import snap_frame
from fc_sat.easing import ease_out_cubic

FINAL_K = 93 * (2 ** 16)  # 6,094,848


@dataclass(frozen=True)
class AddEvent:
    frame: int
    k_after: int
    interval: int
    index: int


@dataclass(frozen=True)
class DoubleEvent:
    frame: int
    k_before: int
    k_after: int
    index: int


@dataclass
class Schedule:
    bpm: float
    fps: int
    n_frames: int
    max_snap_error: float
    adds: list[AddEvent]
    doubles: list[DoubleEvent]
    k_at: np.ndarray
    bar_starts: list[int] = field(default_factory=list)

    def k(self, frame: int) -> int:
        frame = min(max(int(frame), 0), self.n_frames - 1)
        return int(self.k_at[frame])


def add_frames() -> list[int]:
    frames = [24 * k for k in range(1, 9)]
    frames += [192 + 12 * k for k in range(1, 17)]
    frames += [384 + 6 * k for k in range(1, 33)]
    frames += [576 + 3 * k for k in range(1, 33)]
    frames += [696, 720, 744, 768]
    return frames


def double_frames() -> list[int]:
    return [960 + 24 * j for j in range(8)] + [1152 + 24 * j for j in range(8)]


def build_schedule(bpm: float = 150.0, fps: int = 60, n_frames: int = 1824) -> Schedule:
    adds_at = add_frames()
    doubles_at = double_frames()
    if len(adds_at) != 92:
        raise SystemExit(f"expected 92 adds, got {len(adds_at)}")
    if len(doubles_at) != 16:
        raise SystemExit(f"expected 16 doublings, got {len(doubles_at)}")
    error = 0.0
    for beat in [frame * bpm / (60.0 * fps) for frame in adds_at + doubles_at]:
        _, err = snap_frame(beat, bpm, fps)
        error = max(error, err)
    if error > 0.5:
        raise SystemExit(f"maximum snap error is {error:.3f} frames")

    intervals = [adds_at[i + 1] - adds_at[i] for i in range(len(adds_at) - 1)]
    intervals.append(doubles_at[0] - adds_at[-1])
    adds: list[AddEvent] = []
    k = 1
    for index, (frame, interval) in enumerate(zip(adds_at, intervals)):
        k += 1
        adds.append(AddEvent(frame=frame, k_after=k, interval=interval, index=index))
    if k != 93:
        raise SystemExit(f"adds should finish at K=93, got {k}")

    doubles: list[DoubleEvent] = []
    for index, frame in enumerate(doubles_at):
        before = k
        k *= 2
        doubles.append(DoubleEvent(frame=frame, k_before=before, k_after=k, index=index))
    if k != FINAL_K:
        raise SystemExit(f"doublings should finish at {FINAL_K}, got {k}")

    k_at = np.ones(n_frames, dtype=np.int64)
    cursor = 1
    events = [(item.frame, item.k_after) for item in adds] + [(item.frame, item.k_after) for item in doubles]
    events.sort()
    position = 0
    for frame, k_after in events:
        frame = min(int(frame), n_frames)
        if frame > position:
            k_at[position:frame] = cursor
        cursor = k_after
        position = frame
    if position < n_frames:
        k_at[position:] = cursor
    # The collapse is a blend in the renderer. From frame 1806 the curve is one circle.
    if n_frames > 1806:
        k_at[1806:] = 1

    bar_starts = [snap_frame(4.0 * bar, bpm, fps)[0] for bar in range(19)]
    return Schedule(
        bpm=bpm,
        fps=fps,
        n_frames=n_frames,
        max_snap_error=error,
        adds=adds,
        doubles=doubles,
        k_at=k_at,
        bar_starts=bar_starts,
    )


def add_weight(schedule: Schedule, frame: int, circles: int) -> float:
    """Weight of circle ``circles`` (1-based). Fully present circles stay at 1.

    The newest circle eases from 0 to 1 over min(8, interval) frames with no jump
    at either end. Circle 1 is present on frame 0.
    """
    if circles <= 1:
        return 1.0
    if circles > len(schedule.adds) + 1:
        return 1.0
    item = schedule.adds[circles - 2]
    if item.k_after != circles:
        raise RuntimeError(f"add {circles} is out of order")
    if frame < item.frame:
        return 0.0
    duration = min(8, item.interval)
    if duration <= 1 or frame >= item.frame + duration - 1:
        return 1.0
    u = (frame - item.frame) / float(duration - 1)
    return ease_out_cubic(u)


def _blend_start(item: DoubleEvent) -> int:
    """Eight frames before the landing, except the first.

    Frame 954 is the published zoom measurement of the settled 93-circle
    curve. An 8-frame blend into frame 960 would already have moved that
    corner, so the first doubling eases across the five frames after it.
    """
    if item.frame == 960:
        return 955
    return item.frame - 8


def doubling_blend(schedule: Schedule, frame: int) -> tuple[DoubleEvent, float] | None:
    """Blend that owns this frame, completing on the landing frame."""
    for item in schedule.doubles:
        start = _blend_start(item)
        if start <= frame <= item.frame:
            span = max(1, item.frame - start)
            return item, ease_out_cubic((frame - start) / span)
    return None


def collapse_mix(frame: int) -> float:
    """0 on frame 1800 (still the square), 1 from frame 1806 (one circle)."""
    if frame < 1800:
        return 0.0
    if frame >= 1806:
        return 1.0
    return ease_out_cubic((frame - 1800) / 6.0)


def counter_value(schedule: Schedule, frame: int) -> int | None:
    """Integer on the counter, or None when the counter is the infinity sign."""
    frame = int(frame)
    if 1632 <= frame < 1800:
        return None
    if 1440 <= frame < 1488:
        return _rewind(frame, FINAL_K, 138, 1440, 1488)
    if 1488 <= frame < 1632:
        return 138
    if 1800 <= frame < 1806:
        return _collapse_count(frame)
    return schedule.k(frame)


def _rewind(frame: int, start: int, end: int, frame0: int, frame1: int) -> int:
    log0 = math.log10(start)
    log1 = math.log10(end)
    span = log0 - log1
    u = (frame - frame0) / float(frame1 - frame0)
    step = math.floor((u * span) / 0.1 + 1e-12)
    shown = int(round(10 ** (log0 - step * 0.1)))
    return max(end, shown)


def _collapse_count(frame: int) -> int:
    if frame >= 1805:
        return 1
    u = (frame - 1800) / 5.0
    log_n = (1.0 - u) * math.log(float(FINAL_K))
    return max(1, int(round(math.exp(log_n))))


def rewind_tick_frames() -> list[int]:
    frames = []
    previous = 0
    for frame in range(1440, 1488):
        value = _rewind(frame, FINAL_K, 138, 1440, 1488)
        step = value
        if step != previous:
            frames.append(frame)
            previous = step
    return frames
