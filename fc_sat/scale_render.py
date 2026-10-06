"""One frame of the scale short: realm-tinted background, the row of objects, flat type."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from fc_sat.circlesquare_draw import bgr, make_canvas, mix_hex
from fc_sat.polycircle_draw import measure
from fc_sat.scale_art import load_cutout
from fc_sat.scale_scene import Scene, build_scene
from fc_sat.scale_world import BASELINE_Y, HEIGHT, SNAP, WIDTH, Camera, camera_at, focus_index, punch

try:
    import skia
except ImportError:  # pragma: no cover
    skia = None

ROOT = Path(__file__).resolve().parents[1]
TEXT_PATH = ROOT / "configs" / "scale_text.yaml"
SAFE = (130.0, 950.0, 200.0, 1536.0)
COUNTER_TOP = 1296.0
# Below this size a counter wraps onto two lines at a comma.
WRAP_BELOW = 64.0
# Two rows must still leave the friendly line inside the safe zone.
TWO_ROW_MAX = 72.0
REALMS = {
    "quantum": "#150F26",
    "micro": "#0C1C1E",
    "life": "#11181F",
    "human": "#0E1117",
    "city": "#0F1626",
    "sky": "#0C1A33",
    "space": "#07080D",
    "deep": "#050509",
}
SPACE = {"space": 1.0, "deep": 1.0, "sky": 0.25}
GROUND = {"life": 1.0, "human": 1.0, "city": 1.0, "sky": 1.0}
TEXT = "#F2F4F8"
MUTED = "#8A93A6"
GOLD = "#FFD166"
PLATE = "#1B2030"


@dataclass
class TextBox:
    text: str
    x: float
    y: float
    w: float
    h: float
    size: float
    kind: str
    color: str
    alpha: float

    def right(self) -> float:
        return self.x + self.w

    def bottom(self) -> float:
        return self.y + self.h


def overlaps(left: TextBox, right: TextBox) -> bool:
    return left.x < right.right() - 0.5 and right.x < left.right() - 0.5 and left.y < right.bottom() - 0.5 and right.y < left.bottom() - 0.5


def inside_safe(box: TextBox, scale: float) -> bool:
    x0, x1, y0, y1 = (edge * scale for edge in SAFE)
    return box.x >= x0 - 1.0 and box.right() <= x1 + 1.0 and box.y >= y0 - 1.0 and box.bottom() <= y1 + 1.0


def _fit(kind: str, text: str, max_size: float, min_size: float, max_width: float) -> float:
    size = max_size
    while size > min_size + 0.5:
        if measure(kind, size, text).width <= max_width:
            return size
        size -= 2.0
    return min_size


def load_text(path: Path | None = None) -> dict:
    return yaml.safe_load((path or TEXT_PATH).read_text())


def _stars(count: int = 160, seed: int = 5) -> np.ndarray:
    rng = np.random.Generator(np.random.PCG64(seed))
    xs = rng.uniform(0, WIDTH, count)
    ys = rng.uniform(0, HEIGHT, count)
    sizes = rng.uniform(1.0, 3.2, count)
    glints = rng.uniform(0.25, 1.0, count)
    return np.stack([xs, ys, sizes, glints], axis=1)


class ScaleRenderer:
    def __init__(self, scene: Scene | None = None, width: int = WIDTH, height: int = HEIGHT, hook: str = "A") -> None:
        self.scene = scene or build_scene()
        self.width = int(width)
        self.height = int(height)
        self.k = self.width / float(WIDTH)
        self.hook = hook
        self.text = load_text()
        self.placed = list(self.scene.placed)
        self.stars = _stars()
        self._images: dict[str, object] = {}

    def px(self, value: float) -> float:
        return float(value) * self.k

    # ----- words -----

    def top_lines(self, frame: int) -> tuple[tuple[str, ...], bool]:
        """(lines, is the object's name)."""
        if frame < self.placed[1].item.land - 24 or frame >= SNAP[0]:
            return tuple(self.text["hooks"][self.hook]), False
        for card in self.text["cards"]:
            if int(card["start"]) <= frame < int(card["end"]):
                return tuple(card["lines"]), False
        name = self.placed[focus_index(self.placed, frame)].item.name
        return _wrap(name), True

    def plan(self, frame: int) -> list[TextBox]:
        frame = int(frame) % 1824
        boxes: list[TextBox] = []
        limit = (SAFE[1] - SAFE[0]) * self.k
        lines, is_name = self.top_lines(frame)
        size = min(_fit("word", line, self.px(96), self.px(64), limit) for line in lines)
        step = size * 1.06
        for index, line in enumerate(lines):
            ink = measure("word", size, line)
            boxes.append(TextBox(line, (self.width - ink.width) / 2.0, self.px(200) + index * step, ink.width, ink.height, size, "word", TEXT, 1.0))
        index = focus_index(self.placed, frame)
        label = "SIZE" if is_name or frame >= SNAP[0] or frame < self.placed[1].item.land - 24 else self.placed[index].item.name
        label_size = _fit("word", label, self.px(40), self.px(28), limit)
        ink = measure("word", label_size, label)
        cursor = self.px(COUNTER_TOP)
        boxes.append(TextBox(label, (self.width - ink.width) / 2.0, cursor, ink.width, ink.height, label_size, "word", MUTED, 1.0))
        cursor += measure("word", self.px(40), "SIZE").height + self.px(16)
        rows = counter_rows(self.scene.counters[index], limit, self.px(110), self.px(WRAP_BELOW), self.px(30))
        top_size = self.px(110 if len(rows) == 1 else TWO_ROW_MAX)
        value_size = min(_fit("mono", row, top_size, self.px(30), limit) for row in rows)
        # Rows share a baseline grid; a comma's tail hangs below it, so each row reserves room for one.
        digit = measure("mono", value_size, "8")
        row_height = measure("mono", value_size, "8,").height
        for row in rows:
            ink = measure("mono", value_size, row)
            baseline = cursor - digit.top
            boxes.append(TextBox(row, (self.width - ink.width) / 2.0, baseline + ink.top, ink.width, ink.height, value_size, "mono", TEXT, 1.0))
            cursor += row_height + self.px(8)
        cursor += self.px(4)
        sub = self.scene.displays[index]
        sub_size = _fit("word", sub, self.px(40), self.px(26), limit)
        ink = measure("word", sub_size, sub)
        boxes.append(TextBox(sub, (self.width - ink.width) / 2.0, cursor, ink.width, ink.height, sub_size, "word", GOLD, 1.0))
        return boxes

    # ----- picture -----

    def _background(self, camera: Camera) -> tuple[str, float, float]:
        a = self.placed[camera.focus].item.realm
        b = self.placed[min(camera.focus + 1, len(self.placed) - 1)].item.realm
        color = mix_hex(REALMS[a], REALMS[b], camera.blend)
        space = SPACE.get(a, 0.0) * (1 - camera.blend) + SPACE.get(b, 0.0) * camera.blend
        ground = GROUND.get(a, 0.0) * (1 - camera.blend) + GROUND.get(b, 0.0) * camera.blend
        return color, space, ground

    def _image(self, art: str):
        if art not in self._images:
            cut = load_cutout(art)
            if cut is None or skia is None:
                self._images[art] = None
            else:
                rgba = np.ascontiguousarray(cut[:, :, [2, 1, 0, 3]])
                image = skia.Image.fromarray(rgba, colorType=skia.kRGBA_8888_ColorType, alphaType=skia.kUnpremul_AlphaType)
                self._images[art] = image.withDefaultMipmaps()
        return self._images[art]

    def render(self, frame: int) -> np.ndarray:
        frame = int(frame) % 1824
        camera = camera_at(self.placed, frame)
        bump = punch(self.placed, frame) if frame < SNAP[0] else 1.0
        canvas = make_canvas(self.width, self.height)
        color, space, ground = self._background(camera)
        canvas.fill(color)
        if space > 0.01:
            for x, y, size, glint in self.stars:
                canvas.dot(self.px(x), self.px(y), self.px(size) * 0.5, TEXT, 0.55 * space * glint)
        if ground > 0.01:
            canvas.stroke([(0.0, self.px(BASELINE_Y)), (float(self.width), self.px(BASELINE_Y))], MUTED, self.px(3), 0.35 * ground)
        for placed in self.placed:
            self._draw_item(canvas, placed, camera, bump)
        self._draw_text(canvas, frame)
        return bgr(canvas)

    def _rect(self, placed, camera: Camera, bump: float) -> tuple[float, float, float, float]:
        item = placed.item
        cx, base = camera.to_screen(placed.x_m, 0.0)
        width = item.width_m * camera.scale
        height = item.height_m * camera.scale
        cx = WIDTH / 2.0 + (cx - WIDTH / 2.0) * bump
        base = BASELINE_Y + (base - BASELINE_Y) * bump
        return cx - width * bump / 2.0, base - height * bump, width * bump, height * bump

    def _draw_item(self, canvas, placed, camera: Camera, bump: float) -> None:
        x, y, w, h = self._rect(placed, camera, bump)
        if max(w, h) < 1.5 or x > WIDTH + 4 or x + w < -4 or y > HEIGHT + 4 or y + h < -4:
            return
        x, y, w, h = self.px(x), self.px(y), self.px(w), self.px(h)
        image = self._image(placed.item.art)
        if image is not None and hasattr(canvas, "canvas"):
            sampling = skia.SamplingOptions(skia.FilterMode.kLinear, skia.MipmapMode.kLinear)
            canvas.canvas.drawImageRect(image, skia.Rect.MakeXYWH(x, y, w, h), sampling, skia.Paint(AntiAlias=True))
            return
        self._placeholder(canvas, placed.item, x, y, w, h)

    def _placeholder(self, canvas, item, x: float, y: float, w: float, h: float) -> None:
        dark = mix_hex(item.color, "#000000", 0.35)
        cx, cy = x + w / 2.0, y + h / 2.0
        if item.shape in ("circle", "galaxy"):
            r = min(w, h) / 2.0
            canvas.dot(cx, cy, r, item.color, 1.0)
            canvas.circle_stroke(cx, cy, r * 0.94, dark, max(1.0, r * 0.08), 1.0)
            if item.shape == "galaxy" and r > 6:
                for arm in range(2):
                    pts = [(cx + r * t * math.cos(arm * math.pi + 3.2 * t), cy + r * 0.9 * t * math.sin(arm * math.pi + 3.2 * t)) for t in np.linspace(0.08, 0.95, 40)]
                    canvas.stroke(pts, dark, max(1.0, r * 0.07), 0.9)
        elif item.shape == "peak":
            canvas.polygon([(x, y + h), (cx, y), (x + w, y + h)], item.color, 1.0)
            canvas.polygon([(cx - w * 0.12, y + h * 0.24), (cx, y), (cx + w * 0.12, y + h * 0.24)], "#FFFFFF", 0.9)
        else:
            canvas.round_rect(x, y, w, h, min(w, h) * 0.25, item.color, 1.0)
            canvas.round_rect(x + w * 0.12, y + h * 0.12, w * 0.76, h * 0.76, min(w, h) * 0.2, dark, 0.35)
        if w > 120:
            label = item.id.split("_", 1)[1].replace("_", " ").upper()
            size = max(10.0, min(28.0, w / 10.0))
            ink = measure("word", size, label)
            canvas.text(label, cx - ink.width / 2.0, cy - ink.height / 2.0, "word", size, "#0E1117", 0.8)

    def _draw_text(self, canvas, frame: int) -> None:
        boxes = self.plan(frame)
        tops = [box for box in boxes if box.y < self.px(700)]
        bottoms = [box for box in boxes if box.y >= self.px(700)]
        for group, alpha in ((tops, 0.92), (bottoms, 0.92)):
            if not group:
                continue
            pad = self.px(22)
            left = min(b.x for b in group) - pad
            right = max(b.right() for b in group) + pad
            top = min(b.y for b in group) - pad
            bottom = max(b.bottom() for b in group) + pad
            canvas.round_rect(left, top, right - left, bottom - top, self.px(16), PLATE, alpha)
        for box in boxes:
            canvas.text(box.text, box.x, box.y, box.kind, box.size, box.color, box.alpha)

    def profile_ms(self) -> float:
        started = time.perf_counter()
        self.render(1000)
        return (time.perf_counter() - started) * 1000.0


def counter_rows(value: str, limit: float, top: float, wrap_below: float, floor: float) -> tuple[str, ...]:
    """One line if it fits big enough, else two lines split at the comma nearest the middle."""
    if _fit("mono", value, top, floor, limit) >= wrap_below or value.count(",") < 2:
        return (value,)
    commas = [index for index, char in enumerate(value) if char == ","]
    middle = len(value) / 2.0
    cut = min(commas, key=lambda index: abs(index + 1 - middle))
    return value[: cut + 1], value[cut + 1 :]


def _wrap(name: str, limit: int = 16) -> tuple[str, ...]:
    words = name.split()
    lines: list[str] = []
    for word in words:
        if lines and len(lines[-1]) + 1 + len(word) <= limit:
            lines[-1] += " " + word
        else:
            lines.append(word)
    return tuple(lines)
