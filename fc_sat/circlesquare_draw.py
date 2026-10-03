"""Flat canvas for the circlesquare short. Round joins, coverage antialiasing.

skia is preferred. pycairo is next. Pillow draws at 2x and box-downsamples.
"""

from __future__ import annotations

import numpy as np

from fc_sat.fonts import font_file
from fc_sat.polycircle_draw import advance_width, measure

try:
    import skia
except ImportError:  # pragma: no cover
    skia = None

try:
    import cairo
except ImportError:  # pragma: no cover
    cairo = None


def mix_hex(start: str, end: str, amount: float) -> str:
    amount = 0.0 if amount < 0 else 1.0 if amount > 1 else amount

    def channel(color: str, index: int) -> int:
        return int(color.removeprefix("#")[index * 2 : index * 2 + 2], 16)

    parts = [int(round(channel(start, i) * (1.0 - amount) + channel(end, i) * amount)) for i in range(3)]
    return "#" + "".join(f"{part:02X}" for part in parts)


def _rgba(hex_color: str, alpha: float) -> tuple[int, int, int, int]:
    text = hex_color.removeprefix("#")
    red, green, blue = int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16)
    return red, green, blue, max(0, min(255, int(round(alpha * 255))))


def _pairs(pts) -> list[tuple[float, float]]:
    return [(float(x), float(y)) for x, y in pts]


class SkiaCanvas:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.surface = skia.Surface(width, height)
        self.canvas = self.surface.getCanvas()
        self._face: dict[str, object] = {}

    def _typeface(self, kind: str):
        if kind not in self._face:
            self._face[kind] = skia.Typeface.MakeFromFile(str(font_file(kind)))
        return self._face[kind]

    def fill(self, hex_color: str) -> None:
        red, green, blue, _ = _rgba(hex_color, 1)
        self.canvas.clear(skia.Color4f(red / 255, green / 255, blue / 255, 1))

    def _paint(self, hex_color: str, alpha: float):
        return skia.Paint(AntiAlias=True, Color=skia.Color(*_rgba(hex_color, alpha)))

    def polygon(self, pts, hex_color: str, alpha: float = 1.0) -> None:
        if alpha <= 0 or len(pts) < 3:
            return
        path = skia.Path()
        path.moveTo(float(pts[0][0]), float(pts[0][1]))
        for x, y in pts[1:]:
            path.lineTo(float(x), float(y))
        path.close()
        self.canvas.drawPath(path, self._paint(hex_color, alpha))

    def stroke(self, pts, hex_color: str, width: float, alpha: float = 1.0, closed: bool = False) -> None:
        if len(pts) < 2 or alpha <= 0 or width <= 0:
            return
        path = skia.Path()
        path.moveTo(float(pts[0][0]), float(pts[0][1]))
        for x, y in pts[1:]:
            path.lineTo(float(x), float(y))
        if closed:
            path.close()
        paint = self._paint(hex_color, alpha)
        paint.setStyle(skia.Paint.kStroke_Style)
        paint.setStrokeWidth(float(width))
        paint.setStrokeCap(skia.Paint.kRound_Cap)
        paint.setStrokeJoin(skia.Paint.kRound_Join)
        self.canvas.drawPath(path, paint)

    def circle_stroke(self, x: float, y: float, radius: float, hex_color: str, width: float, alpha: float = 1.0) -> None:
        if alpha <= 0 or radius <= 0 or width <= 0:
            return
        paint = self._paint(hex_color, alpha)
        paint.setStyle(skia.Paint.kStroke_Style)
        paint.setStrokeWidth(float(width))
        self.canvas.drawCircle(float(x), float(y), float(radius), paint)

    def dot(self, x: float, y: float, radius: float, hex_color: str, alpha: float = 1.0) -> None:
        if alpha <= 0 or radius <= 0:
            return
        self.canvas.drawCircle(float(x), float(y), float(radius), self._paint(hex_color, alpha))

    def round_rect(self, x: float, y: float, w: float, h: float, radius: float, hex_color: str, alpha: float = 1.0) -> None:
        if alpha <= 0 or w <= 0 or h <= 0:
            return
        rect = skia.Rect.MakeXYWH(float(x), float(y), float(w), float(h))
        self.canvas.drawRoundRect(rect, float(radius), float(radius), self._paint(hex_color, alpha))

    def text(self, text: str, x: float, y: float, kind: str, size: float, hex_color: str, alpha: float = 1.0) -> None:
        if not text or alpha <= 0 or size <= 0:
            return
        ink = measure(kind, size, text)
        font = skia.Font(self._typeface(kind), float(size))
        blob = skia.TextBlob.MakeFromString(text, font)
        self.canvas.drawTextBlob(blob, float(x) - ink.left, float(y) - ink.top, self._paint(hex_color, alpha))

    def rgb(self) -> np.ndarray:
        image = self.surface.makeImageSnapshot().toarray()
        return np.ascontiguousarray(image[:, :, :3])


class CairoCanvas:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
        self.ctx = cairo.Context(self.surface)
        self.ctx.set_antialias(cairo.ANTIALIAS_BEST)
        self.ctx.set_line_join(cairo.LINE_JOIN_ROUND)
        self.ctx.set_line_cap(cairo.LINE_CAP_ROUND)

    def fill(self, hex_color: str) -> None:
        red, green, blue, _ = _rgba(hex_color, 1)
        self.ctx.set_source_rgb(red / 255, green / 255, blue / 255)
        self.ctx.paint()

    def _source(self, hex_color: str, alpha: float) -> None:
        red, green, blue, _ = _rgba(hex_color, 1)
        self.ctx.set_source_rgba(red / 255, green / 255, blue / 255, alpha)

    def polygon(self, pts, hex_color, alpha=1.0) -> None:
        if alpha <= 0 or len(pts) < 3:
            return
        self.ctx.new_path()
        self.ctx.move_to(*pts[0])
        for point in pts[1:]:
            self.ctx.line_to(*point)
        self.ctx.close_path()
        self._source(hex_color, alpha)
        self.ctx.fill()

    def stroke(self, pts, hex_color, width, alpha=1.0, closed=False) -> None:
        if len(pts) < 2 or alpha <= 0 or width <= 0:
            return
        self.ctx.new_path()
        self.ctx.move_to(*pts[0])
        for point in pts[1:]:
            self.ctx.line_to(*point)
        if closed:
            self.ctx.close_path()
        self._source(hex_color, alpha)
        self.ctx.set_line_width(width)
        self.ctx.stroke()

    def circle_stroke(self, x, y, radius, hex_color, width, alpha=1.0) -> None:
        if alpha <= 0 or radius <= 0:
            return
        self.ctx.new_path()
        self.ctx.arc(x, y, radius, 0, 6.283185307179586)
        self._source(hex_color, alpha)
        self.ctx.set_line_width(width)
        self.ctx.stroke()

    def dot(self, x, y, radius, hex_color, alpha=1.0) -> None:
        if alpha <= 0 or radius <= 0:
            return
        self.ctx.new_path()
        self.ctx.arc(x, y, radius, 0, 6.283185307179586)
        self._source(hex_color, alpha)
        self.ctx.fill()

    def round_rect(self, x, y, w, h, radius, hex_color, alpha=1.0) -> None:
        if alpha <= 0 or w <= 0 or h <= 0:
            return
        self.ctx.new_path()
        self.ctx.rectangle(x, y, w, h)
        self._source(hex_color, alpha)
        self.ctx.fill()

    def text(self, text, x, y, kind, size, hex_color, alpha=1.0) -> None:
        if not text or alpha <= 0:
            return
        ink = measure(kind, size, text)
        self._source(hex_color, alpha)
        self.ctx.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
        self.ctx.set_font_size(size)
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

    def polygon(self, pts, hex_color, alpha=1.0) -> None:
        if alpha <= 0 or len(pts) < 3:
            return
        self.draw.polygon([(self._s(x), self._s(y)) for x, y in pts], fill=_rgba(hex_color, 1)[:3])

    def stroke(self, pts, hex_color, width, alpha=1.0, closed=False) -> None:
        if len(pts) < 2 or alpha <= 0:
            return
        color = _rgba(hex_color, 1)[:3]
        xy = [(self._s(x), self._s(y)) for x, y in pts]
        if closed:
            xy = [*xy, xy[0]]
        pen = max(1, self._s(width))
        self.draw.line(xy, fill=color, width=pen, joint="curve")

    def circle_stroke(self, x, y, radius, hex_color, width, alpha=1.0) -> None:
        if alpha <= 0 or radius <= 0:
            return
        color = _rgba(hex_color, 1)[:3]
        cx, cy, r = self._s(x), self._s(y), self._s(radius)
        pen = max(1, self._s(width))
        self.draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=color, width=pen)

    def dot(self, x, y, radius, hex_color, alpha=1.0) -> None:
        if alpha <= 0 or radius <= 0:
            return
        color = _rgba(hex_color, 1)[:3]
        cx, cy, r = self._s(x), self._s(y), self._s(radius)
        self.draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=color)

    def round_rect(self, x, y, w, h, radius, hex_color, alpha=1.0) -> None:
        if alpha <= 0 or w <= 0 or h <= 0:
            return
        box = (self._s(x), self._s(y), self._s(x + w), self._s(y + h))
        self.draw.rounded_rectangle(box, radius=self._s(radius), fill=_rgba(hex_color, 1)[:3])

    def text(self, text, x, y, kind, size, hex_color, alpha=1.0) -> None:
        if not text or alpha <= 0:
            return
        from fc_sat.fonts import load_font

        ink = measure(kind, size, text)
        font = load_font(kind, max(1, self._s(size)))
        self.draw.text((self._s(x - ink.left), self._s(y - ink.top)), text, font=font, fill=_rgba(hex_color, 1)[:3])

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


def text_width(kind: str, size: float, text: str) -> float:
    return advance_width(kind, size, text)
