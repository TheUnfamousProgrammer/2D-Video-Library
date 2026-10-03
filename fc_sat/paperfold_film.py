"""The diorama film. Tower frames come from the still renderer. The twist,
reality, answer, outro, and the loop collapse are drawn here.

Frame 1823 is frame 0. The stack collapses across frames 1800-1805. The hook
text fades in across 1800-1823.
"""

from __future__ import annotations

import cv2
import numpy as np

from fc_sat.paperfold_draw import PaperCanvas, bgr
from fc_sat.paperfold_format import record_cm_label, stamp_lines
from fc_sat.paperfold_render import folds_done
from fc_sat.paperfold_scenes import FRAME_H, FRAME_W, px_per_m
from fc_sat.paperfold_schedule import segment_at
from fc_sat.paperfold_stills import DioramaStill, SAFE_L
from fc_sat.paperfold_text import top_lines

GOLD = "#E7C36A"
MARKERS = (
    (0.12, "EARTH AROUND", "word"),
    (0.38, "EARTH TO SUN", "word"),
    (0.62, "1 LIGHT-YEAR", "mono"),
    (0.86, "THE MILKY WAY", "word"),
)


class Film:
    def __init__(self, hook: str = "A") -> None:
        self.hook = hook
        self.still = DioramaStill(hook)
        self._opening: np.ndarray | None = None
        self._bare: np.ndarray | None = None
        self._outro: np.ndarray | None = None

    def scene(self, name: str):
        for item in self.still.scenes:
            if item.name == name:
                return item
        raise KeyError(name)

    def render(self, frame: int) -> np.ndarray:
        if self.hook == "D" and frame < 24:
            return self._tease()
        if frame <= 0 or frame >= 1823:
            return self.opening()
        if frame >= 1800:
            return self._collapse(frame)
        name = segment_at(frame)
        if name == "twist":
            return self._twist(frame)
        if name == "reality":
            return self._reality(frame)
        if name == "answer":
            return self._answer(frame)
        if name == "outro":
            return self._outro_frame(frame)
        return self.still.render_animated(frame)

    def opening(self) -> np.ndarray:
        if self._opening is None:
            saved = self.still.hook
            self.still.hook = "A"
            self._opening = self.still.render_animated(0)
            self.still.hook = saved
        return self._opening

    def bare_opening(self) -> np.ndarray:
        """Frame 0 with the hook plate removed. The card does not reach y 420."""
        if self._bare is None:
            image = self.opening().copy()
            plate = self.still.plate(self.scene("desk").plate)
            image[180:420] = plate[180:420]
            self._bare = image
        return self._bare

    def _tease(self) -> np.ndarray:
        """Hook D. Frames 0-23 are the Moon still, then the film cuts to the card."""
        image = self.still.render_animated(1128)
        return image

    def _collapse(self, frame: int) -> np.ndarray:
        if self._outro is None:
            self._outro = self._outro_frame(1799)
        collapse = min(1.0, (frame - 1800) / 6.0)
        base = self._blend(self._outro, self.bare_opening(), collapse)
        hook = (frame - 1800) / 23.0
        band = slice(180, 420)
        full = self.opening()
        mixed = base.copy()
        src = full[band].astype(np.float32)
        dst = mixed[band].astype(np.float32)
        mixed[band] = np.clip(dst * (1.0 - hook) + src * hook, 0, 255).astype(np.uint8)
        return mixed

    def _twist(self, frame: int) -> np.ndarray:
        canvas = PaperCanvas(FRAME_W, FRAME_H)
        plate = cv2.cvtColor(self.still.plate(self.scene("deep_space").plate), cv2.COLOR_BGR2RGB)
        travel = (frame - 672) / 191.0
        canvas.image(plate, -80.0 * travel, 0)
        strip_w = 720.0 + travel * 1200.0
        top = 900.0
        canvas.round_rect(80, top, min(strip_w, 980.0), 88, 8, "#F2F4F8", 1.0)
        canvas.rect(80, top, strip_w, 4, "#1B2030")
        marker = MARKERS[min(len(MARKERS) - 1, int(travel * len(MARKERS)))]
        self._marker(canvas, marker[1], 540.0, top - 120.0, marker[2])
        shown = False
        for stamp_frame, fold in ((696, 12), (744, 20), (792, 30)):
            if stamp_frame <= frame < stamp_frame + 24:
                self.still._centered_title(canvas, stamp_lines(fold), 216.0)
                shown = True
                break
        if not shown:
            lines = top_lines(self.still.script, frame, "A")
            if lines:
                self.still._centered_title(canvas, lines, 216.0)
        self.still._bottom(canvas, frame, folds_done(frame))
        return bgr(canvas)

    def _marker(self, canvas: PaperCanvas, text: str, x: float, y: float, kind: str) -> None:
        from fc_sat.paperfold_draw import text_width

        from fc_sat.paperfold_stills import SAFE_R

        size = 70.0
        width = text_width(kind, size, text)
        while width > 740 and size > 48:
            size -= 2.0
            width = text_width(kind, size, text)
        left = x - (width + 48.0) / 2.0
        left = min(max(left, SAFE_L), SAFE_R - width - 48.0)
        canvas.round_rect(left, y - 8, width + 48, size + 28, 16, "#1B2030", 0.92)
        canvas.text(text, left + 24, y, kind, size, "#F2F4F8")

    def _reality(self, frame: int) -> np.ndarray:
        canvas = PaperCanvas(FRAME_W, FRAME_H)
        scene = self.scene("desk")
        canvas.image(cv2.cvtColor(self.still.plate(scene.plate), cv2.COLOR_BGR2RGB), 0, 0)
        scale = px_per_m(scene.world_m, screen_ground=scene.screen_ground)
        self.still._objects(canvas, scene, scale, [])
        self.still._stack(canvas, 12, scale, scene, label=False)
        lines = top_lines(self.still.script, frame, "A")
        if lines:
            self.still._centered_title(canvas, lines, 216.0)
        cm = record_cm_label()
        self.still._chip(canvas, f"12 FOLDS = {cm}", FRAME_W / 2.0, 548.0, "mono")
        self.still._bottom(canvas, frame, 12)
        return bgr(canvas)

    def _answer(self, frame: int) -> np.ndarray:
        canvas = PaperCanvas(FRAME_W, FRAME_H)
        if frame < 1536:
            scene = self.scene("deep_space")
            words = ("IN MATH:",)
            number = "42"
            ink = GOLD
        else:
            scene = self.scene("desk")
            words = ("IN REAL LIFE:",)
            number = "12"
            ink = "#F2F4F8"
        canvas.image(cv2.cvtColor(self.still.plate(scene.plate), cv2.COLOR_BGR2RGB), 0, 0)
        self.still._centered_title(canvas, words, 280.0)
        self._big_number(canvas, number, ink)
        self.still._bottom(canvas, frame, folds_done(frame))
        return bgr(canvas)

    def _big_number(self, canvas: PaperCanvas, number: str, ink: str) -> None:
        from fc_sat.paperfold_draw import text_width

        size = 280.0
        width = text_width("mono", size, number)
        x = FRAME_W / 2.0 - width / 2.0
        y = 620.0
        canvas.round_rect(x - 24, y - 16, width + 48, size + 40, 16, "#1B2030", 0.92)
        canvas.text(number, x, y, "mono", size, ink)

    def _outro_frame(self, frame: int) -> np.ndarray:
        canvas = PaperCanvas(FRAME_W, FRAME_H)
        scene = self.scene("desk")
        canvas.image(cv2.cvtColor(self.still.plate(scene.plate), cv2.COLOR_BGR2RGB), 0, 0)
        scale = px_per_m(scene.world_m, screen_ground=scene.screen_ground)
        self.still._stack(canvas, 12, scale, scene)
        lines = top_lines(self.still.script, min(frame, 1799), "A")
        if lines:
            self.still._centered_title(canvas, lines, 240.0)
        self.still._bottom(canvas, min(frame, 1799), 12)
        return bgr(canvas)

    @staticmethod
    def _blend(a: np.ndarray, b: np.ndarray, u: float) -> np.ndarray:
        mixed = a.astype(np.float32) * (1.0 - u) + b.astype(np.float32) * u
        return np.clip(mixed, 0, 255).astype(np.uint8)


def downscale(frame: np.ndarray) -> np.ndarray:
    return cv2.resize(frame, (540, 960), interpolation=cv2.INTER_AREA)
