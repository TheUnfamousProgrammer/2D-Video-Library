"""Float32 line layer. Adding lines one frame at a time matches redrawing them."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw


class Accum:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.color = np.zeros((height, width, 3), dtype=np.float32)
        self.cover = np.zeros((height, width), dtype=np.float32)

    def reset(self) -> None:
        self.color.fill(0)
        self.cover.fill(0)

    def add_polyline(self, pts: list[tuple[float, float]], rgb: tuple[int, int, int], width: float, alpha: float) -> None:
        if len(pts) < 2 or alpha <= 0:
            return
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        pad = int(width) + 2
        x0 = max(0, int(min(xs)) - pad)
        y0 = max(0, int(min(ys)) - pad)
        x1 = min(self.width, int(max(xs)) + pad + 1)
        y1 = min(self.height, int(max(ys)) + pad + 1)
        if x1 <= x0 or y1 <= y0:
            return
        mask_image = Image.new("L", (x1 - x0, y1 - y0), 0)
        pen = ImageDraw.Draw(mask_image)
        local = [(p[0] - x0, p[1] - y0) for p in pts]
        pen.line(local, fill=255, width=max(1, int(round(width))), joint="curve")
        mask = np.asarray(mask_image, dtype=np.float32) / 255.0
        src_a = mask * float(alpha)
        region_cover = self.cover[y0:y1, x0:x1]
        added = src_a * (1.0 - region_cover)
        color = np.array(rgb, dtype=np.float32)
        self.color[y0:y1, x0:x1] += added[:, :, None] * color
        self.cover[y0:y1, x0:x1] = region_cover + added

    def composite(self, image: np.ndarray) -> np.ndarray:
        cover = self.cover[:, :, None]
        return np.clip(image.astype(np.float32) * (1.0 - cover) + self.color, 0, 255).astype(np.uint8)
