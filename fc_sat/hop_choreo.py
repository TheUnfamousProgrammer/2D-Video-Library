"""Analytic hop from one pad to the next.

Time is snapped so every landing is a video frame and an integer audio sample.
``T0 = beats * 60 / bpm``, ``N = round(T0 * 60)``, ``T = N / 60``, and the
seconds-per-beat used for onsets is ``spb = T / beats``. That snap is what
makes the loop an exact number of frames. Ode to Joy at 120 bpm is already
exact, so the snap does not move it.

Position is a pure function of ``t mod T``:

    x = x_a + (x_b - x_a) * u
    y = y_c - 4 * h * u * (1 - u)
    h = clamp(h_ref * (D / D_median) ** exponent, h_min, h_max)

Squash keeps the ball bottom fixed. Pad press is 0 on the contact frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from fc_sat.song import Song

SAMPLES_PER_FRAME = 800


class GridError(ValueError):
    def __init__(self, index: int, message: str) -> None:
        self.index = index
        super().__init__(f"token {index}: {message}")


@dataclass(frozen=True)
class Grid:
    """Frame-snapped onsets. ``spb`` is the snapped seconds per beat."""

    bpm: float
    total_beats: float
    t0: float
    n_frames: int
    duration: float
    spb: float
    onset_times: tuple[float, ...]
    onset_frames: tuple[int, ...]
    onset_samples: tuple[int, ...]
    flight_seconds: tuple[float, ...]
    flight_frames: tuple[int, ...]


@dataclass(frozen=True)
class Pad:
    name: str
    midi: int
    index: int
    x: float
    width: float
    top: float
    thickness: float
    corner: float

    @property
    def left(self) -> float:
        return self.x - self.width / 2.0

    @property
    def right(self) -> float:
        return self.x + self.width / 2.0


@dataclass(frozen=True)
class Layout:
    pads: tuple[Pad, ...]
    span: float
    ball_r: float
    y_contact: float

    @property
    def width(self) -> float:
        return self.pads[0].width

    @property
    def centers(self) -> tuple[float, ...]:
        return tuple(pad.x for pad in self.pads)


@dataclass(frozen=True)
class Pose:
    x: float
    y: float
    y_arc: float
    rx: float
    ry: float
    age: float
    src: int
    dst: int
    press: float
    flash: float
    bottom: float
    pad_top: float
    u: float
    height: float
    duration: float


def build_grid(song: Song, bpm: float | None = None) -> Grid:
    """Snap onsets onto frames. Reject a flight shorter than 6 frames."""
    used = float(song.bpm if bpm is None else bpm)
    if used <= 0:
        raise GridError(song.notes[-1].token_index, "bpm must be positive")
    total = float(song.total_beats)
    t0 = total * 60.0 / used
    if t0 > 60.0:
        raise GridError(song.notes[-1].token_index, f"loop is {t0:.3f}s, longer than 60s")
    n_frames = int(round(t0 * 60.0))
    if n_frames < 1:
        raise GridError(song.notes[0].token_index, "loop is shorter than one frame")
    duration = n_frames / 60.0
    spb = duration / total
    frames: list[int] = []
    for note, beat in zip(song.notes, song.onset_beats):
        frame = int(round(beat * spb * 60.0))
        if frames and frame <= frames[-1]:
            raise GridError(note.token_index, f"onset frame {frame} is not strictly increasing")
        if frame < 0 or frame >= n_frames:
            raise GridError(note.token_index, f"onset frame {frame} is outside 0..{n_frames - 1}")
        frames.append(frame)
    if frames[0] != 0:
        raise GridError(song.notes[0].token_index, "onset 0 is not frame 0")
    flight_frames: list[int] = []
    for index, frame in enumerate(frames):
        nxt = frames[index + 1] if index + 1 < len(frames) else n_frames
        span = nxt - frame
        if span < 6:
            raise GridError(
                song.notes[index].token_index,
                f"flight {index} is {span} frames, shorter than 6",
            )
        flight_frames.append(span)
    onset_times = tuple(frame / 60.0 for frame in frames)
    return Grid(
        bpm=used,
        total_beats=total,
        t0=t0,
        n_frames=n_frames,
        duration=duration,
        spb=spb,
        onset_times=onset_times,
        onset_frames=tuple(frames),
        onset_samples=tuple(frame * SAMPLES_PER_FRAME for frame in frames),
        flight_seconds=tuple(span / 60.0 for span in flight_frames),
        flight_frames=tuple(flight_frames),
    )


def build_layout(
    song: Song,
    *,
    span: float = 700.0,
    pad_max_width: float = 112.0,
    pad_top_y: float = 1180.0,
    pad_thickness: float = 36.0,
    corner: float = 14.0,
    ball_radius: float = 34.0,
) -> Layout:
    """Pitches sorted low to high, left to right. Centers are spaced by ``span / P``."""
    order: dict[int, str] = {}
    for note in song.notes:
        order.setdefault(note.midi, note.name)
    midis = sorted(order)
    count = len(midis)
    if count < 2 or count > 10:
        raise GridError(song.notes[-1].token_index, f"layout needs 2 to 10 pitches, got {count}")
    step = span / count
    width = min(pad_max_width, 0.8 * step)
    radius = min(ball_radius, 0.42 * width)
    pads = []
    for index, midi in enumerate(midis):
        x = 540.0 + (index - (count - 1) / 2.0) * step
        pads.append(
            Pad(
                name=order[midi],
                midi=midi,
                index=index,
                x=x,
                width=width,
                top=pad_top_y,
                thickness=pad_thickness,
                corner=corner,
            )
        )
    return Layout(pads=tuple(pads), span=span, ball_r=radius, y_contact=pad_top_y - radius)


def pad_press(age: float) -> float:
    """0 at the contact frame, about 10 px near 30 ms."""
    if age <= 0.0:
        return 0.0
    return 14.0 * (1.0 - math.exp(-age / 0.012)) * math.exp(-age / 0.10)


def flash_envelope(age: float) -> float:
    if age <= 0.0:
        return 1.0
    return math.exp(-age / 0.12)


def squash(age: float, radius: float) -> tuple[float, float, float]:
    """Return ``(rx, ry, scale)``. Bottom stays fixed when the center shifts by ``r - ry``."""
    amount = 0.30 * math.exp(-max(age, 0.0) / 0.06)
    rx = radius * (1.0 + 0.5 * amount)
    ry = radius * (1.0 - amount)
    return rx, ry, amount


class Choreography:
    def __init__(
        self,
        song: Song,
        grid: Grid,
        layout: Layout,
        *,
        h_ref: float = 220.0,
        h_min: float = 60.0,
        h_max: float = 600.0,
        exponent: float = 1.2,
    ) -> None:
        self.song = song
        self.grid = grid
        self.layout = layout
        self.h_ref = h_ref
        self.h_min = h_min
        self.h_max = h_max
        self.exponent = exponent
        durations = np.array(grid.flight_seconds, dtype=np.float64)
        self.d_median = float(np.median(durations))
        self.heights = tuple(
            _clamp(h_ref * (duration / self.d_median) ** exponent, h_min, h_max)
            for duration in grid.flight_seconds
        )
        self._pad_of = {pad.midi: pad.index for pad in layout.pads}

    def pad_index(self, note_index: int) -> int:
        return self._pad_of[self.song.notes[note_index].midi]

    def takeoff_speed(self, note_index: int) -> float:
        """Peak vertical speed of the flight that leaves this landing, ``4 * h / D``."""
        return 4.0 * self.heights[note_index] / self.grid.flight_seconds[note_index]

    def pose(self, t: float) -> Pose:
        wrapped = _wrap(t, self.grid.duration)
        flight = self._flight(wrapped)
        t0 = self.grid.onset_times[flight]
        duration = self.grid.flight_seconds[flight]
        u = 0.0 if duration <= 0 else (wrapped - t0) / duration
        u = min(1.0, max(0.0, u))
        src = self.pad_index(flight)
        dst = self.pad_index(0 if flight + 1 == len(self.song.notes) else flight + 1)
        x_a = self.layout.pads[src].x
        x_b = self.layout.pads[dst].x
        height = self.heights[flight]
        y_arc = self.layout.y_contact - 4.0 * height * u * (1.0 - u)
        age = wrapped - t0
        rx, ry, _scale = squash(age, self.layout.ball_r)
        y = y_arc + (self.layout.ball_r - ry)
        press = pad_press(age)
        return Pose(
            x=x_a + (x_b - x_a) * u,
            y=y,
            y_arc=y_arc,
            rx=rx,
            ry=ry,
            age=age,
            src=src,
            dst=dst,
            press=press,
            flash=flash_envelope(age),
            bottom=y + ry,
            pad_top=self.layout.pads[src].top + press,
            u=u,
            height=height,
            duration=duration,
        )

    def _flight(self, t: float) -> int:
        frames = self.grid.onset_times
        # side='right' then -1 selects the latest onset at or before t.
        index = int(np.searchsorted(frames, t, side="right") - 1)
        return max(0, min(len(frames) - 1, index))


def build_choreography(song: Song, bpm: float | None = None, **layout) -> Choreography:
    hop = {key: layout.pop(key) for key in ("h_ref", "h_min", "h_max", "exponent") if key in layout}
    grid = build_grid(song, bpm=bpm)
    pads = build_layout(song, **layout)
    return Choreography(song, grid, pads, **hop)


def _wrap(t: float, duration: float) -> float:
    """Map ``t`` into ``[0, duration)``. A time exactly on the loop point is 0."""
    if duration <= 0:
        return 0.0
    wrapped = math.fmod(t, duration)
    if wrapped < 0.0:
        wrapped += duration
    if wrapped >= duration - 1e-12:
        return 0.0
    return wrapped


def _clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))
