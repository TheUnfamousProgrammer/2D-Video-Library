"""Where every object stands, in real metres, and where the camera looks on each frame.

Objects stand on one baseline, left to right, each to the right of the one before, at their
true relative sizes. The camera frames one object at a time. Between landings it pans right
and zooms out on a log scale, arriving on the landing frame (a beat). While it waits it
drifts out by a few percent so the picture never freezes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import yaml

from fc_sat.easing import ease_in_out_cubic

ROOT = Path(__file__).resolve().parents[1]
OBJECTS_PATH = ROOT / "configs" / "scale_objects.yaml"

WIDTH = 1080
HEIGHT = 1920
N_FRAMES = 1824
# The stage: the featured object fits inside this box, standing on the baseline.
BASELINE_Y = 1250.0
MAX_W = 860.0
MAX_H = 690.0
GAP = 0.08
DRIFT = 0.04
MOVE_MAX = 40
SNAP = (1800, 1806)
PUNCH_FRAMES = 10


@dataclass(frozen=True)
class Item:
    id: str
    name: str
    art: str
    measure: str
    land: int
    realm: str
    shape: str
    color: str
    size_m: float
    aspect: float  # width / height of the drawn object

    @property
    def width_m(self) -> float:
        return self.size_m if self.measure == "width" else self.size_m * self.aspect

    @property
    def height_m(self) -> float:
        return self.size_m if self.measure == "height" else self.size_m / self.aspect


@dataclass(frozen=True)
class Placed:
    item: Item
    x_m: float  # center
    scale: float  # px per metre that frames this item


@dataclass(frozen=True)
class Camera:
    x_m: float
    scale: float
    focus: int
    blend: float  # 0 = still on ``focus``, rising to 1 as it arrives on ``focus + 1``

    def to_screen(self, x_m: float, y_m: float) -> tuple[float, float]:
        return WIDTH / 2.0 + (x_m - self.x_m) * self.scale, BASELINE_Y - y_m * self.scale


def load_catalog(path: Path | None = None) -> list[dict]:
    raw = yaml.safe_load((path or OBJECTS_PATH).read_text())
    return list(raw["objects"])


def framing_scale(item: Item) -> float:
    return min(MAX_W / item.width_m, MAX_H / item.height_m)


def place(items: list[Item]) -> list[Placed]:
    placed: list[Placed] = []
    x = 0.0
    for index, item in enumerate(items):
        if index:
            previous = items[index - 1]
            x += previous.width_m / 2.0 + GAP * item.width_m + item.width_m / 2.0
        placed.append(Placed(item, x, framing_scale(item)))
    return placed


def check_order(items: list[Item]) -> list[str]:
    errors = []
    for a, b in zip(items, items[1:]):
        if b.land <= a.land:
            errors.append(f"{b.id} lands at {b.land}, not after {a.id} at {a.land}")
        if b.size_m <= a.size_m:
            errors.append(f"{b.id} ({b.size_m:g} m) is not bigger than {a.id} ({a.size_m:g} m)")
    return errors


def move_frames(previous_land: int, land: int) -> int:
    return int(min(MOVE_MAX, round(0.6 * (land - previous_land))))


def camera_at(placed: list[Placed], frame: float) -> Camera:
    """The camera on any frame, including fractional frames for sub-steps."""
    frame = float(frame)
    if frame >= SNAP[0]:
        frame = 0.0
    lands = [p.item.land for p in placed]
    index = 0
    for position, land in enumerate(lands):
        if frame >= land:
            index = position
    current = placed[index]
    if index + 1 >= len(placed):
        hold = (frame - current.item.land) / float(SNAP[0] - current.item.land)
        return Camera(current.x_m, current.scale * (1.0 - DRIFT * hold), index, 0.0)
    following = placed[index + 1]
    move = move_frames(current.item.land, following.item.land)
    start = following.item.land - move
    if frame < start:
        hold = (frame - current.item.land) / float(start - current.item.land)
        return Camera(current.x_m, current.scale * (1.0 - DRIFT * hold), index, 0.0)
    begin_scale = current.scale * (1.0 - DRIFT)
    u = ease_in_out_cubic((frame - start) / float(move))
    scale = math.exp(math.log(begin_scale) + (math.log(following.scale) - math.log(begin_scale)) * u)
    # Pan in step with the zoom: the share of the pan done equals the share of the view growth done.
    view_begin = 1.0 / begin_scale
    view_end = 1.0 / following.scale
    share = (1.0 / scale - view_begin) / (view_end - view_begin) if view_end != view_begin else u
    x = current.x_m + (following.x_m - current.x_m) * share
    return Camera(x, scale, index, u)


def punch(placed: list[Placed], frame: int) -> float:
    """A 3% scale bump on every landing after the first."""
    for p in placed[1:]:
        if p.item.land <= frame < p.item.land + PUNCH_FRAMES:
            return 1.0 + 0.03 * math.sin(math.pi * (frame - p.item.land) / PUNCH_FRAMES)
    return 1.0


def focus_index(placed: list[Placed], frame: int) -> int:
    """The object the words are about: the one the camera last landed on."""
    if frame >= SNAP[0]:
        return 0
    index = 0
    for position, p in enumerate(placed):
        if frame >= p.item.land:
            index = position
    return index
