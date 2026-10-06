"""One frame of the deepest-to-highest short: water or sky, the stops, the depth gauge, flat type."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import yaml

from fc_sat.circlesquare_draw import bgr, make_canvas, mix_hex
from fc_sat.dive_scene import DiveScene, build_scene
from fc_sat.dive_world import ANCHOR, HEIGHT, SEA, SNAP, WIDTH, Camera, camera_at, focus_index, punch, structure_x
from fc_sat.polycircle_draw import measure
from fc_sat.scale_art import load_cutout
from fc_sat.scale_render import (
    COUNTER_TOP,
    GOLD,
    MUTED,
    PLATE,
    SAFE,
    TEXT,
    TWO_ROW_MAX,
    WRAP_BELOW,
    TextBox,
    _fit,
    _stars,
    _wrap,
    counter_rows,
)

try:
    import skia
except ImportError:  # pragma: no cover
    skia = None

ROOT = Path(__file__).resolve().parents[1]
TEXT_PATH = ROOT / "configs" / "dive_text.yaml"
REALMS = {
    "shallow": "#0E3346",
    "twilight": "#0B2536",
    "midnight": "#081826",
    "abyss": "#060F19",
    "trench": "#040A11",
    "rock": "#1B140F",
    "ground": "#0E1726",
    "sky": "#0C1A33",
    "high": "#0A1229",
    "space": "#07080D",
    "deep": "#050509",
}
AIR = "#101826"
WATER = "#0E3346"
SEA_LINE = "#7FB3C8"
SPACE = {"space": 1.0, "deep": 1.0, "high": 0.35}
ICON = 250.0
SURFACE_ICON = 190.0
GAUGE_X = 70.0
# Frames the icon just passed takes to fade out after the next landing.
FADE = 14


def _entry_alpha(act: str, cy: float) -> float:
    if act == "up":
        return max(0.0, min(1.0, (cy - 520.0) / 120.0))
    return max(0.0, min(1.0, (1260.0 - cy) / 120.0))


def load_text(path: Path | None = None) -> dict:
    return yaml.safe_load((path or TEXT_PATH).read_text())


class DiveRenderer:
    def __init__(self, scene: DiveScene | None = None, width: int = WIDTH, height: int = HEIGHT, hook: str = "A") -> None:
        self.scene = scene or build_scene()
        self.stops = list(self.scene.stops)
        self.width = int(width)
        self.height = int(height)
        self.k = self.width / float(WIDTH)
        self.hook = hook
        self.text = load_text()
        self.stars = _stars()
        self._images: dict[str, object] = {}

    def px(self, value: float) -> float:
        return float(value) * self.k

    # ----- words -----

    def top_lines(self, frame: int) -> tuple[tuple[str, ...], bool]:
        if frame < self.stops[1].land - 24 or frame >= SNAP[0]:
            return tuple(self.text["hooks"][self.hook]), False
        for card in self.text["cards"]:
            if int(card["start"]) <= frame < int(card["end"]):
                return tuple(card["lines"]), False
        return _wrap(self.stops[focus_index(self.stops, frame)].name), True

    def plan(self, frame: int) -> list[TextBox]:
        frame = int(frame) % 1824
        boxes: list[TextBox] = []
        limit = (SAFE[1] - SAFE[0]) * self.k
        lines, is_name = self.top_lines(frame)
        size = min(_fit("word", line, self.px(92), self.px(60), limit) for line in lines)
        step = size * 1.06
        for index, line in enumerate(lines):
            ink = measure("word", size, line)
            boxes.append(TextBox(line, (self.width - ink.width) / 2.0, self.px(200) + index * step, ink.width, ink.height, size, "word", TEXT, 1.0))
        index = focus_index(self.stops, frame)
        stop = self.stops[index]
        label = stop.label if is_name or frame >= SNAP[0] or frame < self.stops[1].land - 24 else stop.name
        label_size = _fit("word", label, self.px(40), self.px(26), limit)
        ink = measure("word", label_size, label)
        cursor = self.px(COUNTER_TOP)
        boxes.append(TextBox(label, (self.width - ink.width) / 2.0, cursor, ink.width, ink.height, label_size, "word", MUTED, 1.0))
        cursor += measure("word", self.px(40), "SIZE").height + self.px(16)
        rows = counter_rows(self.scene.counters[index], limit, self.px(110), self.px(WRAP_BELOW), self.px(30))
        top_size = self.px(110 if len(rows) == 1 else TWO_ROW_MAX)
        value_size = min(_fit("mono", row, top_size, self.px(30), limit) for row in rows)
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

    def _realm(self, camera: Camera) -> tuple[str, float]:
        a = self.stops[camera.focus].realm
        following = self.stops[min(camera.focus + 1, len(self.stops) - 1)]
        b = following.realm if following.act == camera.act else a
        color = mix_hex(REALMS[a], REALMS[b], camera.blend)
        space = SPACE.get(a, 0.0) * (1 - camera.blend) + SPACE.get(b, 0.0) * camera.blend
        return color, space

    def boxes_at(self, frame: int) -> list[tuple[int, float, float, float, float, float]]:
        """(stop index, x, y, w, h, alpha) in design pixels for every stop drawn on this frame."""
        frame = int(frame) % 1824
        camera = camera_at(self.stops, frame)
        bump = punch(self.stops, frame) if frame < SNAP[0] else 1.0
        camera = Camera(camera.act, camera.span / bump, camera.focus, camera.blend, camera.x_m)
        xs = structure_x(self.stops)
        out = []
        for i in sorted(xs, key=lambda i: -self.stops[i].value_m):
            stop = self.stops[i]
            if stop.act != camera.act or i > camera.focus + 1:
                continue
            h = stop.value_m * camera.px_per_m
            w = h * stop.aspect
            if h >= 1.5:
                out.append((i, camera.x_of(xs[i]) - w / 2.0, camera.sea_y - h, w, h, 1.0))
        since = frame - self.stops[camera.focus].land
        for i, stop in enumerate(self.stops):
            if stop.act != camera.act or stop.kind != "icon":
                continue
            alpha = 1.0
            if stop.value_m <= 0:
                size, cy = SURFACE_ICON, camera.sea_y - SURFACE_ICON * 0.18
            else:
                if i == camera.focus - 1 and 0 <= since < FADE:
                    alpha = 1.0 - since / float(FADE)
                elif i not in (camera.focus, camera.focus + 1):
                    continue
                size = ICON * min(1.0, stop.value_m / camera.span)
                cy = camera.y_of(stop.value_m)
                if i == camera.focus + 1:
                    # The next stop arrives from past the anchor; it fades in only once clear of the cards.
                    alpha = _entry_alpha(camera.act, cy)
            if size < 4.0 or alpha <= 0.0:
                continue
            w, h = (size, size / stop.aspect) if stop.aspect >= 1.0 else (size * stop.aspect, size)
            out.append((i, WIDTH / 2.0 - w / 2.0, cy - h / 2.0, w, h, alpha))
        return out

    def render(self, frame: int) -> np.ndarray:
        frame = int(frame) % 1824
        camera = camera_at(self.stops, frame)
        canvas = make_canvas(self.width, self.height)
        color, space = self._realm(camera)
        sea = self.px(camera.sea_y)
        if camera.act == "down":
            canvas.fill(color)
            canvas.round_rect(0.0, 0.0, float(self.width), sea, 0.0, AIR, 1.0)
        else:
            canvas.fill(color)
            canvas.round_rect(0.0, sea, float(self.width), float(self.height) - sea, 0.0, WATER, 1.0)
        if space > 0.01:
            for x, y, size, glint in self.stars:
                if self.px(y) < sea:
                    canvas.dot(self.px(x), self.px(y), self.px(size) * 0.5, TEXT, 0.55 * space * glint)
        canvas.stroke([(0.0, sea), (float(self.width), sea)], SEA_LINE, self.px(4), 0.8)
        self._gauge(canvas, camera)
        for i, x, y, w, h, alpha in self.boxes_at(frame):
            self._draw(canvas, self.stops[i], self.px(x), self.px(y), self.px(w), self.px(h), alpha)
        self._draw_text(canvas, frame)
        return bgr(canvas)

    def _gauge(self, canvas, camera: Camera) -> None:
        anchor = self.px(ANCHOR[camera.act])
        sea = self.px(camera.sea_y)
        x = self.px(GAUGE_X)
        canvas.stroke([(x, sea), (x, anchor)], MUTED, self.px(3), 0.5)
        for i, stop in enumerate(self.stops):
            if stop.act != camera.act or i > camera.focus + 1 or stop.value_m <= 0:
                continue
            y = self.px(camera.y_of(stop.value_m))
            if (camera.act == "down" and y > anchor + 2) or (camera.act == "up" and y < anchor - 2):
                continue
            focused = i == camera.focus
            canvas.stroke([(x - self.px(14), y), (x + self.px(14), y)], GOLD if focused else MUTED, self.px(3), 0.9 if focused else 0.5)

    def _draw(self, canvas, stop, x: float, y: float, w: float, h: float, alpha: float = 1.0) -> None:
        if x > self.width + 4 or x + w < -4 or y > self.height + 4 or y + h < -4:
            return
        image = self._image(stop.art)
        if image is not None and hasattr(canvas, "canvas"):
            sampling = skia.SamplingOptions(skia.FilterMode.kLinear, skia.MipmapMode.kLinear)
            paint = skia.Paint(AntiAlias=True)
            paint.setAlphaf(float(alpha))
            canvas.canvas.drawImageRect(image, skia.Rect.MakeXYWH(x, y, w, h), sampling, paint)
            return
        canvas.round_rect(x, y, w, h, min(w, h) * 0.2, "#5F7C8A", alpha)
        if w > 120:
            label = stop.art.split("_", 1)[1].replace("_", " ").upper()
            size = max(10.0, min(26.0, w / 9.0))
            ink = measure("word", size, label)
            canvas.text(label, x + w / 2.0 - ink.width / 2.0, y + h / 2.0 - ink.height / 2.0, "word", size, "#0E1117", 0.85)

    def _draw_text(self, canvas, frame: int) -> None:
        boxes = self.plan(frame)
        for group in ([b for b in boxes if b.y < self.px(700)], [b for b in boxes if b.y >= self.px(700)]):
            if not group:
                continue
            pad = self.px(22)
            left = min(b.x for b in group) - pad
            right = max(b.right() for b in group) + pad
            top = min(b.y for b in group) - pad
            bottom = max(b.bottom() for b in group) + pad
            canvas.round_rect(left, top, right - left, bottom - top, self.px(16), PLATE, 0.92)
        for box in boxes:
            canvas.text(box.text, box.x, box.y, box.kind, box.size, box.color, box.alpha)

    def profile_ms(self) -> float:
        started = time.perf_counter()
        self.render(1000)
        return (time.perf_counter() - started) * 1000.0
