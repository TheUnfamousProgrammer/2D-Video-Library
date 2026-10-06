"""Where every stop sits, in real metres from sea level, and what the camera shows on each frame.

The dive (act "down") puts sea level near the top of the screen and depth below it; the climb
(act "up") puts sea level near the bottom and height above it. On a stop's landing frame the
camera's span is that stop's depth or height, so the stop sits on the anchor line. Between
stops the span grows on a log scale, so everything passed sinks toward sea level and shrinks.
The drop cuts from the bottom of the dive to sea level for the climb.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import yaml

from fc_sat.easing import ease_in_out_cubic

ROOT = Path(__file__).resolve().parents[1]
STOPS_PATH = ROOT / "configs" / "dive_objects.yaml"
CLAIMS_PATH = ROOT / "configs" / "dive_claims.yaml"

WIDTH = 1080
HEIGHT = 1920
N_FRAMES = 1824
SNAP = (1800, 1806)
SEA = {"down": 640.0, "up": 1250.0}
ANCHOR = {"down": 1130.0, "up": 660.0}
# The surface stop has no depth; the dive opens on this many metres of water.
SURFACE_SPAN = 12.0
DRIFT = 0.04
MOVE_MAX = 40
PUNCH_FRAMES = 10


@dataclass(frozen=True)
class Stop:
    id: str
    name: str
    art: str
    act: str
    kind: str
    land: int
    realm: str
    label: str
    value_m: float
    aspect: float  # width / height of the cutout

    @property
    def span(self) -> float:
        return self.value_m if self.value_m > 0 else SURFACE_SPAN


@dataclass(frozen=True)
class Camera:
    act: str
    span: float
    focus: int
    blend: float
    x_m: float = 0.0  # the metre mark on screen center, for structures standing side by side

    @property
    def sea_y(self) -> float:
        return SEA[self.act]

    @property
    def px_per_m(self) -> float:
        return abs(ANCHOR[self.act] - SEA[self.act]) / self.span

    def x_of(self, x_m: float) -> float:
        return WIDTH / 2.0 + (x_m - self.x_m) * self.px_per_m

    def y_of(self, value_m: float) -> float:
        step = value_m * self.px_per_m
        return self.sea_y + step if self.act == "down" else self.sea_y - step


def load_stops(path: Path | None = None) -> list[dict]:
    return list(yaml.safe_load((path or STOPS_PATH).read_text())["stops"])


def check_order(stops: list[Stop]) -> list[str]:
    errors = []
    for a, b in zip(stops, stops[1:]):
        if b.land <= a.land:
            errors.append(f"{b.id} lands at {b.land}, not after {a.id}")
        if a.act == b.act and b.span <= a.span:
            errors.append(f"{b.id} ({b.value_m:g} m) is not farther than {a.id} ({a.value_m:g} m)")
    if [s.act for s in stops] != sorted((s.act for s in stops), key=lambda act: act != "down"):
        errors.append("every down stop must come before every up stop")
    return errors


def landed_index(stops: list[Stop], frame: float) -> int:
    index = 0
    for position, stop in enumerate(stops):
        if frame >= stop.land:
            index = position
    return index


def structure_x(stops: list[Stop]) -> dict[int, float]:
    """Center of each structure in metres: they stand side by side on sea level, left to right."""
    out: dict[int, float] = {}
    previous = None
    for index, stop in enumerate(stops):
        if stop.kind != "structure":
            continue
        width = stop.value_m * stop.aspect
        if previous is None:
            out[index] = 0.0
        else:
            last = stops[previous]
            out[index] = out[previous] + last.value_m * last.aspect / 2.0 + 0.08 * width + width / 2.0
        previous = index
    return out


def _held_x(stops: list[Stop], xs: dict[int, float], index: int) -> float:
    known = [i for i in xs if i <= index]
    return xs[max(known)] if known else 0.0


def camera_at(stops: list[Stop], frame: float) -> Camera:
    frame = float(frame)
    if frame >= SNAP[0]:
        frame = 0.0
    xs = structure_x(stops)
    index = landed_index(stops, frame)
    current = stops[index]
    x_now = _held_x(stops, xs, index)
    following = stops[index + 1] if index + 1 < len(stops) else None
    if following is None or following.act != current.act:
        end = following.land if following else SNAP[0]
        hold = (frame - current.land) / float(end - current.land)
        return Camera(current.act, current.span * (1.0 + DRIFT * hold), index, 0.0, x_now)
    move = int(min(MOVE_MAX, round(0.6 * (following.land - current.land))))
    start = following.land - move
    if frame < start:
        hold = (frame - current.land) / float(start - current.land)
        return Camera(current.act, current.span * (1.0 + DRIFT * hold), index, 0.0, x_now)
    begin = current.span * (1.0 + DRIFT)
    u = ease_in_out_cubic((frame - start) / float(move))
    span = math.exp(math.log(begin) + (math.log(following.span) - math.log(begin)) * u)
    x = x_now
    if index + 1 in xs:
        # Pan in step with the zoom, as the scale short does.
        share = (span - begin) / (following.span - begin)
        x = x_now + (xs[index + 1] - x_now) * share
    return Camera(current.act, span, index, u, x)


def punch(stops: list[Stop], frame: int) -> float:
    """A 3% push-in on every landing after the first."""
    for stop in stops[1:]:
        if stop.land <= frame < stop.land + PUNCH_FRAMES:
            return 1.0 + 0.03 * math.sin(math.pi * (frame - stop.land) / PUNCH_FRAMES)
    return 1.0


def focus_index(stops: list[Stop], frame: int) -> int:
    return 0 if frame >= SNAP[0] else landed_index(stops, frame)
