"""Filled shapes for the paperfold short. No filters.

skia is preferred. The polycircle canvas has no filled polygon, and that file
stays untouched so its frame hash does not move.
"""

from __future__ import annotations

import numpy as np

from fc_sat.fonts import font_file
from fc_sat.polycircle_draw import advance_width, measure

try:
    import skia
except ImportError:  # pragma: no cover
    skia = None


def _rgba(hex_color: str, alpha: float) -> tuple[int, int, int, int]:
    text = hex_color.removeprefix("#")
    red, green, blue = int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16)
    return red, green, blue, max(0, min(255, int(round(alpha * 255))))


def mix_hex(a: str, b: str, t: float) -> str:
    """One flat colour between two flat colours. A solid fill, not a ramp across the frame."""
    t = 0.0 if t < 0 else 1.0 if t > 1 else t

    def channel(color: str, index: int) -> int:
        return int(color.removeprefix("#")[index * 2 : index * 2 + 2], 16)

    parts = [int(round(channel(a, i) * (1.0 - t) + channel(b, i) * t)) for i in range(3)]
    return "#" + "".join(f"{part:02X}" for part in parts)


class PaperCanvas:
    def __init__(self, width: int, height: int) -> None:
        if skia is None:
            raise SystemExit("paperfold pictures need skia-python")
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

    def rect(self, x: float, y: float, w: float, h: float, hex_color: str, alpha: float = 1.0) -> None:
        if alpha <= 0 or w <= 0 or h <= 0:
            return
        self.canvas.drawRect(skia.Rect.MakeXYWH(x, y, w, h), self._paint(hex_color, alpha))

    def polygon(self, pts: list[tuple[float, float]], hex_color: str, alpha: float = 1.0) -> None:
        if alpha <= 0 or len(pts) < 3:
            return
        path = skia.Path()
        path.moveTo(pts[0][0], pts[0][1])
        for x, y in pts[1:]:
            path.lineTo(x, y)
        path.close()
        self.canvas.drawPath(path, self._paint(hex_color, alpha))

    def circle(self, x: float, y: float, radius: float, hex_color: str, alpha: float = 1.0) -> None:
        if alpha <= 0 or radius <= 0:
            return
        self.canvas.drawCircle(x, y, radius, self._paint(hex_color, alpha))

    def image(self, rgba: np.ndarray, x: float, y: float) -> None:
        """Draw an RGB or RGBA uint8 image. RGB is treated as opaque."""
        if rgba.size == 0:
            return
        array = np.ascontiguousarray(rgba)
        if array.ndim != 3 or array.shape[2] not in (3, 4):
            raise ValueError(f"image must be HxWx3 or HxWx4, got {array.shape}")
        if array.shape[2] == 3:
            array = np.dstack([array, np.full(array.shape[:2], 255, np.uint8)])
        self.canvas.drawImage(skia.Image.fromarray(array), x, y)

    def soft_rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        hex_color: str,
        alpha: float,
        blur: float,
        dx: float = 0.0,
        dy: float = 0.0,
    ) -> None:
        """A blurred rectangle. Used for the paper shadow and the stack's contact shadow."""
        if alpha <= 0 or w <= 0 or h <= 0 or blur <= 0:
            return
        paint = self._paint(hex_color, alpha)
        paint.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, blur))
        self.canvas.drawRect(skia.Rect.MakeXYWH(x + dx, y + dy, w, h), paint)

    def round_rect(
        self, x: float, y: float, w: float, h: float, radius: float, hex_color: str, alpha: float = 1.0
    ) -> None:
        if alpha <= 0 or w <= 0 or h <= 0:
            return
        rect = skia.Rect.MakeXYWH(x, y, w, h)
        self.canvas.drawRoundRect(rect, radius, radius, self._paint(hex_color, alpha))

    def soft_polygon(
        self,
        pts: list[tuple[float, float]],
        hex_color: str,
        alpha: float,
        blur: float,
        dx: float = 0.0,
        dy: float = 0.0,
    ) -> None:
        if alpha <= 0 or len(pts) < 3 or blur <= 0:
            return
        path = skia.Path()
        path.moveTo(pts[0][0] + dx, pts[0][1] + dy)
        for x, y in pts[1:]:
            path.lineTo(x + dx, y + dy)
        path.close()
        paint = self._paint(hex_color, alpha)
        paint.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, blur))
        self.canvas.drawPath(path, paint)

    def stroke_polygon(self, pts: list[tuple[float, float]], hex_color: str, width: float) -> None:
        if len(pts) < 3 or width <= 0:
            return
        path = skia.Path()
        path.moveTo(pts[0][0], pts[0][1])
        for x, y in pts[1:]:
            path.lineTo(x, y)
        path.close()
        paint = self._paint(hex_color, 1.0)
        paint.setStyle(skia.Paint.kStroke_Style)
        paint.setStrokeWidth(width)
        paint.setStrokeJoin(skia.Paint.kMiter_Join)
        self.canvas.drawPath(path, paint)

    def text(self, text: str, x: float, y: float, kind: str, size: float, hex_color: str, alpha: float = 1.0) -> None:
        if not text or alpha <= 0 or size <= 0:
            return
        ink = measure(kind, size, text)
        font = skia.Font(self._typeface(kind), size)
        blob = skia.TextBlob.MakeFromString(text, font)
        self.canvas.drawTextBlob(blob, x - ink.left, y - ink.top, self._paint(hex_color, alpha))

    def rgb(self) -> np.ndarray:
        image = self.surface.makeImageSnapshot().toarray()
        return np.ascontiguousarray(image[:, :, :3])


def bgr(canvas: PaperCanvas) -> np.ndarray:
    image = canvas.rgb()
    return np.ascontiguousarray(image[:, :, ::-1])


def text_width(kind: str, size: float, text: str) -> float:
    return advance_width(kind, size, text)
