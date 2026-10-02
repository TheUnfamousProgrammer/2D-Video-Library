"""Flat vector canvas. Round joins, subpixel coverage, no filters.

skia is preferred. pycairo is next. Pillow draws at 2x and box-downsamples.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from fc_sat.fonts import font_file, glyph_present, load_font

try:
    import skia
except ImportError:  # pragma: no cover
    skia = None

try:
    import cairo
except ImportError:  # pragma: no cover
    cairo = None


_FACES: dict[str, object] = {}


def backend_name() -> str:
    if skia is not None:
        return "skia"
    if cairo is not None:
        return "cairo"
    return "pillow"


def _face(kind: str):
    if kind not in _FACES:
        _FACES[kind] = skia.Typeface.MakeFromFile(str(font_file(kind)))
    return _FACES[kind]


@dataclass(frozen=True)
class Ink:
    width: float
    height: float
    left: float
    top: float


def advance_width(kind: str, size: float, text: str) -> float:
    """Pen travel, including spaces. Tight ink bounds drop those."""
    if not text:
        return 0.0
    if skia is not None:
        return float(skia.Font(_face(kind), size).measureText(text))
    return float(load_font(kind, max(1, int(round(size)))).getlength(text))


def measure(kind: str, size: float, text: str) -> Ink:
    if not text:
        return Ink(0.0, 0.0, 0.0, 0.0)
    if skia is not None:
        font = skia.Font(_face(kind), size)
        bounds = skia.Rect()
        font.measureText(text, bounds=bounds)
        return Ink(
            max(1.0, float(bounds.width())),
            max(1.0, float(bounds.height())),
            float(bounds.left()),
            float(bounds.top()),
        )
    font = load_font(kind, max(1, int(round(size))))
    box = font.getbbox(text)
    return Ink(float(box[2] - box[0]), float(max(1, box[3] - box[1])), float(box[0]), float(box[1]))


def infinity_in_font() -> bool:
    return glyph_present(load_font("mono", 64), "\u221e")


def lemniscate(cx: float, cy: float, scale: float, samples: int = 96) -> list[tuple[float, float]]:
    """Infinity sign as a figure-eight, used only when the mono font has no glyph."""
    pts = []
    for index in range(samples + 1):
        theta = 2.0 * math.pi * index / samples
        denom = 1.0 + math.sin(theta) ** 2
        pts.append((cx + scale * math.cos(theta) / denom, cy + scale * math.sin(theta) * math.cos(theta) / denom))
    return pts


def _rgba(hex_color: str, alpha: float) -> tuple[int, int, int, int]:
    text = hex_color.removeprefix("#")
    red, green, blue = int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16)
    return red, green, blue, max(0, min(255, int(round(alpha * 255))))


class SkiaCanvas:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.surface = skia.Surface(width, height)
        self.canvas = self.surface.getCanvas()

    def fill(self, hex_color: str) -> None:
        red, green, blue, _ = _rgba(hex_color, 1)
        self.canvas.clear(skia.Color4f(red / 255, green / 255, blue / 255, 1))

    def stroke(self, pts: list[tuple[float, float]], hex_color: str, width: float, alpha: float = 1.0, closed: bool = False) -> None:
        if len(pts) < 2 or alpha <= 0 or width <= 0:
            return
        path = skia.Path()
        path.moveTo(pts[0][0], pts[0][1])
        for x, y in pts[1:]:
            path.lineTo(x, y)
        if closed:
            path.close()
        paint = skia.Paint(
            AntiAlias=True,
            Color=skia.Color(*_rgba(hex_color, alpha)),
            Style=skia.Paint.kStroke_Style,
            StrokeWidth=width,
            StrokeCap=skia.Paint.kRound_Cap,
            StrokeJoin=skia.Paint.kRound_Join,
        )
        self.canvas.drawPath(path, paint)

    def dot(self, x: float, y: float, radius: float, hex_color: str, alpha: float = 1.0) -> None:
        if alpha <= 0 or radius <= 0:
            return
        paint = skia.Paint(AntiAlias=True, Color=skia.Color(*_rgba(hex_color, alpha)))
        self.canvas.drawCircle(x, y, radius, paint)

    def text(self, text: str, x: float, y: float, kind: str, size: float, hex_color: str, alpha: float = 1.0) -> None:
        if not text or alpha <= 0:
            return
        ink = measure(kind, size, text)
        font = skia.Font(_face(kind), size)
        blob = skia.TextBlob.MakeFromString(text, font)
        paint = skia.Paint(AntiAlias=True, Color=skia.Color(*_rgba(hex_color, alpha)))
        self.canvas.drawTextBlob(blob, x - ink.left, y - ink.top, paint)

    def rgb(self) -> np.ndarray:
        image = self.surface.makeImageSnapshot().toarray()
        return np.ascontiguousarray(image[:, :, :3])


class CairoCanvas:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
        self.ctx = cairo.Context(self.surface)
        self.ctx.set_line_join(cairo.LINE_JOIN_ROUND)
        self.ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        self.ctx.set_antialias(cairo.ANTIALIAS_BEST)

    def fill(self, hex_color: str) -> None:
        red, green, blue, _ = _rgba(hex_color, 1)
        self.ctx.set_source_rgb(red / 255, green / 255, blue / 255)
        self.ctx.paint()

    def stroke(self, pts, hex_color, width, alpha=1.0, closed=False) -> None:
        if len(pts) < 2 or alpha <= 0:
            return
        red, green, blue, _ = _rgba(hex_color, 1)
        self.ctx.new_path()
        self.ctx.move_to(*pts[0])
        for point in pts[1:]:
            self.ctx.line_to(*point)
        if closed:
            self.ctx.close_path()
        self.ctx.set_source_rgba(red / 255, green / 255, blue / 255, alpha)
        self.ctx.set_line_width(width)
        self.ctx.stroke()

    def dot(self, x, y, radius, hex_color, alpha=1.0) -> None:
        if alpha <= 0:
            return
        red, green, blue, _ = _rgba(hex_color, 1)
        self.ctx.arc(x, y, radius, 0, 2 * math.pi)
        self.ctx.set_source_rgba(red / 255, green / 255, blue / 255, alpha)
        self.ctx.fill()

    def text(self, text, x, y, kind, size, hex_color, alpha=1.0) -> None:
        if not text or alpha <= 0:
            return
        ink = measure(kind, size, text)
        red, green, blue, _ = _rgba(hex_color, 1)
        self.ctx.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
        self.ctx.set_font_size(size)
        self.ctx.set_source_rgba(red / 255, green / 255, blue / 255, alpha)
        self.ctx.move_to(x - ink.left, y - ink.top)
        self.ctx.show_text(text)

    def rgb(self) -> np.ndarray:
        raw = np.frombuffer(self.surface.get_data(), dtype=np.uint8).reshape(self.height, self.width, 4)
        return np.ascontiguousarray(raw[:, :, 2::-1])


class PillowCanvas:
    def __init__(self, width: int, height: int) -> None:
        from PIL import Image, ImageDraw

        self.width = width
        self.height = height
        self.factor = 2
        self.image = Image.new("RGB", (width * 2, height * 2))
        self.draw = ImageDraw.Draw(self.image)

    def _s(self, value: float) -> int:
        return int(round(value * self.factor))

    def fill(self, hex_color: str) -> None:
        self.draw.rectangle((0, 0, self.image.width, self.image.height), fill=_rgba(hex_color, 1)[:3])

    def stroke(self, pts, hex_color, width, alpha=1.0, closed=False) -> None:
        if len(pts) < 2 or alpha <= 0:
            return
        color = _rgba(hex_color, 1)[:3]
        pen = max(1, self._s(width))
        xy = [(self._s(x), self._s(y)) for x, y in pts]
        if closed:
            xy = [*xy, xy[0]]
        self.draw.line(xy, fill=color, width=pen, joint="curve")
        radius = pen / 2
        for x, y in xy:
            self.draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)

    def dot(self, x, y, radius, hex_color, alpha=1.0) -> None:
        if alpha <= 0:
            return
        color = _rgba(hex_color, 1)[:3]
        cx, cy, r = self._s(x), self._s(y), self._s(radius)
        self.draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=color)

    def text(self, text, x, y, kind, size, hex_color, alpha=1.0) -> None:
        if not text or alpha <= 0:
            return
        ink = measure(kind, size, text)
        font = load_font(kind, max(1, self._s(size)))
        color = _rgba(hex_color, 1)[:3]
        self.draw.text((self._s(x - ink.left), self._s(y - ink.top)), text, font=font, fill=color)

    def rgb(self) -> np.ndarray:
        from PIL import Image

        small = self.image.resize((self.width, self.height), resample=Image.Resampling.BOX)
        return np.asarray(small)


def make_canvas(width: int, height: int):
    if skia is not None:
        return SkiaCanvas(width, height)
    if cairo is not None:
        return CairoCanvas(width, height)
    return PillowCanvas(width, height)


def bgr(canvas) -> np.ndarray:
    image = canvas.rgb()
    return np.ascontiguousarray(image[:, :, ::-1])
