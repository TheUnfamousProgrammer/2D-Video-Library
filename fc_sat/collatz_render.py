"""Paint a frame plan and keep the funnel layer in step with the clock."""

from __future__ import annotations

import numpy as np

from fc_sat.collatz_accum import Accum
from fc_sat.collatz_draw import make_canvas
from fc_sat.collatz_layout import Film


class CollatzRenderer:
    def __init__(self, film: Film, *, width: int, height: int, fps: int) -> None:
        self.film = film
        self.width = width
        self.height = height
        self.fps = fps
        self.scale_x = width / film.timeline.width
        self.scale_y = height / film.timeline.height
        self.accum = Accum(width, height)
        self.drawn = 0
        self.last_index = -1

    @property
    def n_frames(self) -> int:
        full = self.film.timeline.n_frames
        if self.fps == self.film.timeline.fps:
            return full
        return int(round(self.film.timeline.duration * self.fps))

    def source_index(self, index: int) -> int:
        if self.fps == self.film.timeline.fps:
            return index
        return min(self.film.timeline.n_frames - 1, int(round(index * self.film.timeline.fps / self.fps)))

    def _sync_funnel(self, count: int) -> None:
        lines = self.film.funnel_polylines()
        # count is the raw 0..1000 clock. Line i is start n in 1..1000 excluding 27,
        # so the visible lines are the first min(count, 999) paths. When count passes
        # 27 the raw clock is one ahead of the line list; cap at the line count.
        visible = min(len(lines), count if count < 27 else count - 1)
        if visible < self.drawn:
            self.accum.reset()
            self.drawn = 0
        color = (0xC9, 0xD1, 0xE3)
        while self.drawn < visible:
            self.accum.add_polyline(self._scale_pts(lines[self.drawn]), color, max(1.0, 1.5 * self.scale_x), 0.10)
            self.drawn += 1

    def _scale_pts(self, pts):
        return [(x * self.scale_x, y * self.scale_y) for x, y in pts]

    def render(self, index: int) -> np.ndarray:
        source = self.source_index(index)
        plan = self.film.plan(source)
        self._sync_funnel(plan.funnel_count)
        sx, sy = self.scale_x, self.scale_y
        use_layer = bool(plan.funnel_count and self.drawn)
        canvas = make_canvas(self.width, self.height, transparent=use_layer)
        if not use_layer:
            canvas.fill(plan.bg)
        self._paint(canvas, plan, sx, sy)
        if use_layer:
            bg = plan.bg.removeprefix("#")
            base = np.zeros((self.height, self.width, 3), dtype=np.float32)
            base[:, :] = (int(bg[0:2], 16), int(bg[2:4], 16), int(bg[4:6], 16))
            layer = self.accum.composite(base.astype(np.uint8)).astype(np.float32)
            if plan.alpha_lines < 0.5:
                layer = base * 0.94 + layer * 0.06
            over = canvas.rgba().astype(np.float32)
            alpha = over[:, :, 3:4] / 255.0
            image = layer * (1.0 - alpha) + over[:, :, :3] * alpha
            image = np.clip(image, 0, 255).astype(np.uint8)
        else:
            image = canvas.rgb()
        self.last_index = index
        return np.ascontiguousarray(image[:, :, ::-1])

    def _paint(self, canvas, plan, sx: float, sy: float) -> None:
        for plate in plan.plates:
            plate.x *= sx
            plate.y *= sy
            plate.w *= sx
            plate.h *= sy
            plate.radius *= (sx + sy) / 2
            canvas.plate(plate)
        for stroke in plan.strokes:
            stroke.pts = self._scale_pts(stroke.pts)
            stroke.width *= (sx + sy) / 2
            canvas.stroke(stroke)
        for dot in plan.dots:
            dot.x *= sx
            dot.y *= sy
            dot.r *= (sx + sy) / 2
            dot.width *= (sx + sy) / 2
            canvas.dot(dot)
        for text in plan.texts:
            canvas.text(text.text, text.x * sx, text.y * sy, text.font, max(1.0, text.size * min(sx, sy)), text.color)

    def profile_ms(self) -> float:
        import time

        ride = min(self.n_frames - 1, int(round(8.0 * self.fps)))
        started = time.perf_counter()
        self.render(ride)
        self.paint_ms = (time.perf_counter() - started) * 1000.0
        self.accum.reset()
        self.drawn = 0
        index = min(self.n_frames - 1, int(round(26.4 * self.fps)))
        started = time.perf_counter()
        self.render(index)
        self.heavy_ms = (time.perf_counter() - started) * 1000.0
        self.accum.reset()
        self.drawn = 0
        return self.heavy_ms
