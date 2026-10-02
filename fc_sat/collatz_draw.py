"""Vector canvas. Strokes use round caps and joins. No filters."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fc_sat.collatz_fonts import font_file

try:
    import skia
except ImportError:  # pragma: no cover
    skia = None


_FACES: dict[str, object] = {}


def _face(kind: str):
    if kind not in _FACES:
        _FACES[kind] = skia.Typeface.MakeFromFile(str(font_file(kind)))
    return _FACES[kind]


@dataclass
class Ink:
    width: float
    height: float
    left: float
    top: float


def measure(kind: str, size: float, text: str) -> Ink:
    """Ink box. `left` and `top` are relative to the pen and the baseline."""
    if not text:
        return Ink(0.0, 0.0, 0.0, 0.0)
    if skia is not None:
        font = skia.Font(_face(kind), size)
        bounds = skia.Rect()
        font.measureText(text, bounds=bounds)
        return Ink(
            max(1.0, float(bounds.right() - bounds.left())),
            max(1.0, float(bounds.bottom() - bounds.top())),
            float(bounds.left()),
            float(bounds.top()),
        )
    from PIL import ImageFont

    font = ImageFont.truetype(str(font_file(kind)), size=max(1, int(round(size))))
    box = font.getbbox(text)
    return Ink(float(box[2] - box[0]), float(max(1, box[3] - box[1])), float(box[0]), float(box[1]))


def _rgba(hex_color: str, alpha: float) -> tuple[int, int, int, int]:
    text = hex_color.removeprefix("#")
    red, green, blue = int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16)
    return red, green, blue, max(0, min(255, int(round(alpha * 255))))


@dataclass
class Stroke:
    pts: list[tuple[float, float]]
    color: str
    width: float
    alpha: float = 1.0


@dataclass
class Dot:
    x: float
    y: float
    r: float
    color: str
    width: float = 0.0
    alpha: float = 1.0


@dataclass
class Plate:
    x: float
    y: float
    w: float
    h: float
    color: str
    alpha: float
    radius: float = 16.0


class SkiaCanvas:
    def __init__(self, width: int, height: int, *, transparent: bool = False) -> None:
        self.width = width
        self.height = height
        self.surface = skia.Surface(width, height)
        self.canvas = self.surface.getCanvas()
        if transparent:
            self.canvas.clear(skia.Color4f(0, 0, 0, 0))

    def fill(self, hex_color: str) -> None:
        red, green, blue, _ = _rgba(hex_color, 1)
        self.canvas.clear(skia.Color4f(red / 255, green / 255, blue / 255, 1))

    def plate(self, plate: Plate) -> None:
        paint = skia.Paint(AntiAlias=True, Color=skia.Color(*_rgba(plate.color, plate.alpha)))
        rect = skia.Rect.MakeXYWH(plate.x, plate.y, plate.w, plate.h)
        self.canvas.drawRoundRect(rect, plate.radius, plate.radius, paint)

    def stroke(self, stroke: Stroke) -> None:
        if len(stroke.pts) < 2:
            return
        path = skia.Path()
        path.moveTo(*stroke.pts[0])
        for point in stroke.pts[1:]:
            path.lineTo(*point)
        paint = skia.Paint(
            AntiAlias=True,
            Color=skia.Color(*_rgba(stroke.color, stroke.alpha)),
            Style=skia.Paint.kStroke_Style,
            StrokeWidth=stroke.width,
            StrokeCap=skia.Paint.kRound_Cap,
            StrokeJoin=skia.Paint.kRound_Join,
        )
        self.canvas.drawPath(path, paint)

    def dot(self, dot: Dot) -> None:
        paint = skia.Paint(AntiAlias=True, Color=skia.Color(*_rgba(dot.color, dot.alpha)))
        if dot.width > 0:
            paint.setStyle(skia.Paint.kStroke_Style)
            paint.setStrokeWidth(dot.width)
            paint.setStrokeCap(skia.Paint.kRound_Cap)
        self.canvas.drawCircle(dot.x, dot.y, dot.r, paint)

    def text(self, text: str, x: float, y: float, kind: str, size: float, color: str, alpha: float = 1) -> None:
        ink = measure(kind, size, text)
        font = skia.Font(_face(kind), size)
        blob = skia.TextBlob.MakeFromString(text, font)
        paint = skia.Paint(AntiAlias=True, Color=skia.Color(*_rgba(color, alpha)))
        self.canvas.drawTextBlob(blob, x - ink.left, y - ink.top, paint)

    def rgb(self) -> np.ndarray:
        array = self.surface.makeImageSnapshot().toarray()
        return np.ascontiguousarray(array[:, :, :3])

    def rgba(self) -> np.ndarray:
        return np.ascontiguousarray(self.surface.makeImageSnapshot().toarray())


class PillowCanvas:
    """2x supersample, then a box downsample. Used only when skia is absent."""

    def __init__(self, width: int, height: int) -> None:
        from PIL import Image, ImageDraw

        self.width = width
        self.height = height
        self.scale = 2
        self.image = Image.new("RGB", (width * 2, height * 2))
        self.draw = ImageDraw.Draw(self.image)

    def _s(self, value: float) -> int:
        return int(round(value * self.scale))

    def fill(self, hex_color: str) -> None:
        self.draw.rectangle((0, 0, self.image.width, self.image.height), fill=_rgba(hex_color, 1)[:3])

    def plate(self, plate: Plate) -> None:
        red, green, blue, alpha = _rgba(plate.color, plate.alpha)
        overlay = self.image.copy()
        pen = __import__("PIL").ImageDraw.Draw(overlay)
        box = [self._s(plate.x), self._s(plate.y), self._s(plate.x + plate.w), self._s(plate.y + plate.h)]
        pen.rounded_rectangle(box, radius=self._s(plate.radius), fill=(red, green, blue))
        self.image = __import__("PIL").Image.blend(self.image, overlay, alpha / 255)
        self.draw = __import__("PIL").ImageDraw.Draw(self.image)

    def stroke(self, stroke: Stroke) -> None:
        if len(stroke.pts) < 2:
            return
        red, green, blue, _ = _rgba(stroke.color, stroke.alpha)
        width = max(1, self._s(stroke.width))
        radius = width / 2
        xy = [(self._s(x), self._s(y)) for x, y in stroke.pts]
        self.draw.line(xy, fill=(red, green, blue), width=width, joint="curve")
        for x, y in xy:
            self.draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(red, green, blue))

    def dot(self, dot: Dot) -> None:
        red, green, blue, _ = _rgba(dot.color, dot.alpha)
        cx, cy, r = self._s(dot.x), self._s(dot.y), self._s(dot.r)
        if dot.width > 0:
            pen = max(1, self._s(dot.width))
            self.draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(red, green, blue), width=pen)
        else:
            self.draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(red, green, blue))

    def text(self, text: str, x: float, y: float, kind: str, size: float, color: str, alpha: float = 1) -> None:
        from PIL import ImageFont

        font = ImageFont.truetype(str(font_file(kind)), size=max(1, self._s(size)))
        ink = measure(kind, size, text)
        red, green, blue, _ = _rgba(color, alpha)
        self.draw.text((self._s(x - ink.left), self._s(y - ink.top)), text, font=font, fill=(red, green, blue))

    def rgb(self) -> np.ndarray:
        small = self.image.resize((self.width, self.height), resample=__import__("PIL").Image.Resampling.BOX)
        return np.asarray(small)


def make_canvas(width: int, height: int, transparent: bool = False):
    if skia is not None:
        return SkiaCanvas(width, height, transparent=transparent)
    return PillowCanvas(width, height)


def backend_name() -> str:
    return "skia" if skia is not None else "pillow"
