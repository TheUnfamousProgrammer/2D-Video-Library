"""Draw the paper, the throws, the zoom, the comparison, and the type."""

from __future__ import annotations

import math

import cv2
import numpy as np

from fc_sat.fonts import font_file
from fc_sat.linedraw.choreo import (
    HOOK_LINES,
    THROW_LINES,
    caption_alpha,
    caption_lines,
    flyoff_progress,
    hook_progress,
    loop_progress,
    show_thrown_counter,
    throw_endpoints,
    zoom_at,
)
from fc_sat.linedraw.optimize import INK, replay
from fc_sat.linedraw.tone import LUT, paper_rgb
from skimage.draw import line_aa

try:
    import skia
except ImportError:  # pragma: no cover
    skia = None

try:
    from numba import njit
except ImportError:  # pragma: no cover
    njit = None


def _paint_segments(image: np.ndarray, x0: np.ndarray, y0: np.ndarray, x1: np.ndarray, y1: np.ndarray, ink: np.ndarray) -> bool:
    if njit is None or len(x0) == 0:
        return False
    _paint_numba(image, np.asarray(x0, np.float64), np.asarray(y0, np.float64), np.asarray(x1, np.float64), np.asarray(y1, np.float64), np.asarray(ink, np.int8))
    return True


if njit is not None:

    @njit(cache=True)
    def _paint_numba(image, x0, y0, x1, y1, ink):
        height, width, _ = image.shape
        for index in range(x0.shape[0]):
            steps = int(max(abs(x1[index] - x0[index]), abs(y1[index] - y0[index]))) + 1
            red = 0x1C if ink[index] < 0 else 0xF3
            green = 0x17 if ink[index] < 0 else 0xE9
            blue = 0x12 if ink[index] < 0 else 0xD2
            for step in range(steps):
                t = 0.0 if steps == 1 else step / (steps - 1)
                x = int(x0[index] + (x1[index] - x0[index]) * t)
                y = int(y0[index] + (y1[index] - y0[index]) * t)
                for dx in range(2):
                    xx = x + dx
                    if 0 <= xx < width and 0 <= y < height:
                        image[y, xx, 0] = red
                        image[y, xx, 1] = green
                        image[y, xx, 2] = blue

else:

    def _paint_numba(*_args):
        return None


PAPER = (159, 420, 762, 1104)
PLATE = (124, 190, 832, 140)
SAFE = (130, 200, 950, 1536)


def _hex(value: str) -> tuple[int, int, int]:
    text = value.removeprefix("#")
    return int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16)


def _ease_out(t: float) -> float:
    t = min(1.0, max(0.0, t))
    return 1.0 - (1.0 - t) ** 3


class LineRenderer:
    def __init__(self, fit, picture: dict, plan, job: dict, *, width: int = 1080, height: int = 1920, hook: str = "A") -> None:
        self.fit = fit
        self.picture = picture
        self.plan = plan
        self.job = job
        self.hook = hook
        self.width = int(width)
        self.height = int(height)
        self.scale = self.width / 1080.0
        self.n = int(len(fit.x0))
        self.appear = np.searchsorted(plan.counts[:1800], np.arange(1, self.n + 1), side="left").astype(np.int32)
        self.canvas = np.full(picture["target"].shape, 0.5, dtype=np.float64)
        self.committed = 0
        self._paper_cache: np.ndarray | None = None
        self._then: np.ndarray | None = None
        self._now: np.ndarray | None = None
        self._original = self._prepare_original()
        self._surface = None
        self._dark_path = None
        self._light_path = None
        self._path_count = 0
        self.last_caption_px = 0.0
        if skia is not None:
            self._surface = skia.Surface(self.width, self.height)
            self._word = skia.Typeface.MakeFromFile(str(font_file("word")))
            self._mono = skia.Typeface.MakeFromFile(str(font_file("mono")))

    def _s(self, value: float) -> float:
        return float(value) * self.scale

    def _prepare_original(self) -> np.ndarray:
        color = self.picture["color"]
        paper_w = max(1, int(round(self._s(PAPER[2]))))
        paper_h = max(1, int(round(self._s(PAPER[3]))))
        return cv2.resize(color, (paper_w, paper_h), interpolation=cv2.INTER_LANCZOS4)

    def commit(self, count: int) -> np.ndarray:
        count = max(0, min(int(count), self.n))
        if count < self.committed:
            self.canvas.fill(0.5)
            self.committed = 0
            self._paper_cache = None
        height, width = self.canvas.shape
        while self.committed < count:
            index = self.committed
            rr, cc, val = line_aa(
                int(round(float(self.fit.y0[index]))),
                int(round(float(self.fit.x0[index]))),
                int(round(float(self.fit.y1[index]))),
                int(round(float(self.fit.x1[index]))),
            )
            mask = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
            if np.any(mask):
                np.add.at(self.canvas, (rr[mask], cc[mask]), int(self.fit.ink[index]) * INK * val[mask])
            self.committed += 1
            self._paper_cache = None
        return self.canvas

    def paper_image(self, count: int) -> np.ndarray:
        self.commit(count)
        if self._paper_cache is None:
            self._paper_cache = paper_rgb(self.canvas, int(round(self._s(PAPER[2]))), int(round(self._s(PAPER[3]))))
        return self._paper_cache

    def redraw(self, count: int) -> np.ndarray:
        return replay(self.fit.x0, self.fit.y0, self.fit.x1, self.fit.y1, self.fit.ink, count, self.picture["target"].shape)

    def likeness_percent(self, count: int) -> int:
        counts = self.fit.likeness_counts
        scores = self.fit.likeness
        if count <= 0 or len(counts) == 0:
            return 0
        index = int(np.searchsorted(counts, count, side="right") - 1)
        if index < 0:
            return 0
        return int(round(float(scores[index]) * 100.0))

    def thrown_at(self, count: int) -> int:
        if count <= 0 or self.n == 0:
            return 0
        return int(self.fit.thrown[min(count, self.n) - 1])

    def render(self, frame: int) -> np.ndarray:
        frame = int(frame)
        if self._surface is None:
            return self._render_cv(frame)
        canvas = self._surface.getCanvas()
        canvas.clear(skia.Color4f(14 / 255, 17 / 255, 23 / 255, 1))
        self._draw_picture(canvas, frame)
        self._draw_type(canvas, frame)
        image = self._surface.makeImageSnapshot().toarray()
        rgb = np.ascontiguousarray(image[:, :, :3])
        return rgb[:, :, ::-1]

    def _draw_picture(self, canvas, frame: int) -> None:
        if frame == 0 or frame >= 1819:
            progress = loop_progress(frame) if frame >= 1819 else hook_progress(0, 0)
            self._blit_paper(canvas, self._blank())
            self._stroke_grid(canvas, self._flight(0, progress), 1.0)
            return
        if frame >= 1800:
            self._draw_flyoff(canvas, frame)
            return
        count = int(self.plan.counts[frame])
        zoom = zoom_at(frame)
        wipe = _wipe(frame)
        if wipe is not None:
            self._blit_paper(canvas, self._comparison(wipe))
            return
        if zoom != 1.0:
            self._draw_zoom(canvas, frame, count, zoom)
            return
        self._blit_paper(canvas, self.paper_image(count))
        self._draw_motion(canvas, frame, count)

    def _draw_motion(self, canvas, frame: int, count: int) -> None:
        overlay = _overlay_alpha(frame)
        if overlay > 0 and count > 0:
            last = min(count, THROW_LINES)
            for index in range(last):
                progress = hook_progress(frame, index) if index < HOOK_LINES else (frame - int(self.appear[index]) + 6) / 10.0
                if index < HOOK_LINES and progress <= 0:
                    continue
                if progress < 1.2:
                    pose = self._flight(index, progress)
                else:
                    pose = self._final(index)
                self._stroke_grid(canvas, pose, overlay)
        if frame <= 575 and frame > 0 and count > 0:
            self._draw_rejects(canvas, frame, count)
        if count <= THROW_LINES or frame > 575:
            return
        newcomers = range(max(THROW_LINES, int(self.plan.counts[frame - 1]) if frame else 0), count)
        shown = 0
        for index in newcomers:
            if shown >= 8:
                break
            age = frame - int(self.appear[index])
            if age < 0 or age > 2:
                continue
            self._stroke_grid(canvas, self._partial(index, (age + 1) / 3.0), 0.9)
            shown += 1

    def _draw_rejects(self, canvas, frame: int, count: int) -> None:
        rejects = self.fit.reject_x0
        if len(rejects) < 3:
            return
        line_index = min(count - 1, len(rejects) // 3 - 1)
        if line_index < 0 or int(self.appear[line_index]) != frame:
            return
        base = line_index * 3
        for offset in range(3):
            index = base + offset
            if index >= len(rejects):
                return
            pose = (
                float(self.fit.reject_x0[index]),
                float(self.fit.reject_y0[index]),
                float(self.fit.reject_x1[index]),
                float(self.fit.reject_y1[index]),
                -1,
            )
            self._stroke_grid(canvas, pose, 0.10)

    def _draw_zoom(self, canvas, frame: int, count: int, zoom: float) -> None:
        x, y, w, h = self._paper_box()
        mouth_x, mouth_y = self._grid_to_screen(self.picture["mouth"][0], self.picture["mouth"][1])
        canvas.save()
        canvas.clipRRect(self._paper_rrect(), True)
        canvas.translate(mouth_x, mouth_y)
        canvas.scale(zoom, zoom)
        canvas.translate(-mouth_x, -mouth_y)
        paint = skia.Paint(AntiAlias=True, Color=skia.Color(*_hex("#8F826D")))
        canvas.drawRect(skia.Rect.MakeXYWH(x, y, w, h), paint)
        self._ensure_paths(count)
        stroke = max(1.0, self._s(2.4))
        dark = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=stroke, StrokeCap=skia.Paint.kRound_Cap, Color=skia.Color(*_hex("#1C1712")))
        light = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=stroke, StrokeCap=skia.Paint.kRound_Cap, Color=skia.Color(*_hex("#F3E9D2")))
        canvas.drawPath(self._dark_path, dark)
        canvas.drawPath(self._light_path, light)
        canvas.restore()
        self.commit(count)

    def _draw_flyoff(self, canvas, frame: int) -> None:
        blend = min(1.0, max(0.0, (frame - 1800) / 3.0))
        vectors = self._vector_paper(frame)
        if blend < 1:
            tone = self.paper_image(self.n).astype(np.float32)
            mixed = tone * (1.0 - blend) + vectors.astype(np.float32) * blend
            self._blit_paper(canvas, np.clip(np.rint(mixed), 0, 255).astype(np.uint8))
        else:
            self._blit_paper(canvas, vectors)

    def _vector_paper(self, frame: int) -> np.ndarray:
        width = int(round(self._s(PAPER[2])))
        height = int(round(self._s(PAPER[3])))
        image = np.empty((height, width, 3), dtype=np.uint8)
        image[:] = _hex("#8F826D")
        scale_x = width / self.picture["target"].shape[1]
        scale_y = height / self.picture["target"].shape[0]
        xs0, ys0, xs1, ys1, inks = [], [], [], [], []
        for index in range(self.n):
            progress = flyoff_progress(frame, index, self.n)
            if progress >= 1.0:
                continue
            pose = self._final(index) if progress <= 0.0 else self._flight(index, 1.0 - _ease_out(progress))
            xs0.append(pose[0] * scale_x)
            ys0.append(pose[1] * scale_y)
            xs1.append(pose[2] * scale_x)
            ys1.append(pose[3] * scale_y)
            inks.append(pose[4])
        if xs0 and not _paint_segments(image, np.array(xs0), np.array(ys0), np.array(xs1), np.array(ys1), np.array(inks)):
            for x_a, y_a, x_b, y_b, ink in zip(xs0, ys0, xs1, ys1, inks):
                color = (0x1C, 0x17, 0x12) if ink < 0 else (0xF3, 0xE9, 0xD2)
                cv2.line(image, (int(x_a), int(y_a)), (int(x_b), int(y_b)), color, 2, cv2.LINE_AA)
        return image

    def _comparison(self, amount: float) -> np.ndarray:
        if self.job["show_original"]:
            left = self._original
        else:
            if self._then is None:
                self._then = paper_rgb(self.redraw(int(self.plan.counts[600])), self._original.shape[1], self._original.shape[0])
            left = self._then
        if self._now is None:
            self._now = self.paper_image(self.n)
        cut = int(round(amount * left.shape[1]))
        image = self._now.copy()
        if cut > 0:
            image[:, :cut] = left[:, :cut]
        if 0 < cut < image.shape[1]:
            image[:, cut : cut + max(1, int(self._s(3)))] = (243, 233, 210)
        return image

    def _blank(self) -> np.ndarray:
        image = np.empty((int(round(self._s(PAPER[3]))), int(round(self._s(PAPER[2]))), 3), dtype=np.uint8)
        image[:] = LUT[len(LUT) // 2]
        return image

    def _flight(self, index: int, progress: float) -> tuple[float, float, float, float, int]:
        pose = throw_endpoints(
            float(self.fit.x0[index]),
            float(self.fit.y0[index]),
            float(self.fit.x1[index]),
            float(self.fit.y1[index]),
            progress,
            self.picture["target"].shape[1],
            self.picture["target"].shape[0],
        )
        return (*pose, int(self.fit.ink[index]))

    def _final(self, index: int) -> tuple[float, float, float, float, int]:
        return (
            float(self.fit.x0[index]),
            float(self.fit.y0[index]),
            float(self.fit.x1[index]),
            float(self.fit.y1[index]),
            int(self.fit.ink[index]),
        )

    def _partial(self, index: int, amount: float) -> tuple[float, float, float, float, int]:
        x0, y0, x1, y1, ink = self._final(index)
        return x0, y0, x0 + (x1 - x0) * amount, y0 + (y1 - y0) * amount, ink

    def _paper_box(self) -> tuple[float, float, float, float]:
        return self._s(PAPER[0]), self._s(PAPER[1]), self._s(PAPER[2]), self._s(PAPER[3])

    def _paper_rrect(self):
        x, y, w, h = self._paper_box()
        return skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(x, y, w, h), self._s(8), self._s(8))

    def _blit_paper(self, canvas, rgb: np.ndarray) -> None:
        x, y, w, h = self._paper_box()
        if rgb.shape[1] != int(round(w)) or rgb.shape[0] != int(round(h)):
            rgb = cv2.resize(rgb, (int(round(w)), int(round(h))), interpolation=cv2.INTER_LINEAR)
        rgba = np.concatenate([rgb, np.full((*rgb.shape[:2], 1), 255, np.uint8)], axis=2)
        image = skia.Image.fromarray(np.ascontiguousarray(rgba))
        canvas.save()
        canvas.clipRRect(self._paper_rrect(), True)
        canvas.drawImage(image, x, y)
        canvas.restore()

    def _stroke_grid(self, canvas, pose, alpha: float) -> None:
        if alpha <= 0:
            return
        x0, y0 = self._grid_to_screen(pose[0], pose[1])
        x1, y1 = self._grid_to_screen(pose[2], pose[3])
        color = _hex("#1C1712") if pose[4] < 0 else _hex("#F3E9D2")
        paint = skia.Paint(
            AntiAlias=True,
            Style=skia.Paint.kStroke_Style,
            StrokeWidth=max(1.0, self._s(3.0)),
            StrokeCap=skia.Paint.kRound_Cap,
            Color=skia.Color(color[0], color[1], color[2], int(round(alpha * 255))),
        )
        canvas.save()
        canvas.clipRect(skia.Rect.MakeXYWH(0, self._s(416), self.width, self.height - self._s(416)))
        canvas.drawLine(x0, y0, x1, y1, paint)
        canvas.restore()

    def _grid_to_screen(self, x: float, y: float) -> tuple[float, float]:
        height, width = self.picture["target"].shape
        return self._s(PAPER[0]) + x / width * self._s(PAPER[2]), self._s(PAPER[1]) + y / height * self._s(PAPER[3])

    def _ensure_paths(self, count: int) -> None:
        count = max(0, min(int(count), self.n))
        if self._dark_path is None or count < self._path_count:
            self._dark_path = skia.Path()
            self._light_path = skia.Path()
            self._path_count = 0
        while self._path_count < count:
            x0, y0 = self._grid_to_screen(float(self.fit.x0[self._path_count]), float(self.fit.y0[self._path_count]))
            x1, y1 = self._grid_to_screen(float(self.fit.x1[self._path_count]), float(self.fit.y1[self._path_count]))
            path = self._dark_path if int(self.fit.ink[self._path_count]) < 0 else self._light_path
            path.moveTo(x0, y0)
            path.lineTo(x1, y1)
            self._path_count += 1

    def _draw_type(self, canvas, frame: int) -> None:
        count = self.plan.count_at(frame)
        lines = caption_lines(
            frame,
            self.hook,
            self.job["subject"],
            bool(self.job["show_original"]),
            self.thrown_at(self.n),
            self.n,
        )
        alpha = caption_alpha(frame) if lines else 0.0
        plate = skia.RRect.MakeRectXY(
            skia.Rect.MakeXYWH(self._s(PLATE[0]), self._s(PLATE[1]), self._s(PLATE[2]), self._s(PLATE[3])),
            self._s(16),
            self._s(16),
        )
        plate_paint = skia.Paint(AntiAlias=True, Color=skia.Color(27, 32, 48, int(round(0.92 * 255))))
        canvas.drawRRect(plate, plate_paint)
        if lines and alpha > 0:
            self._draw_caption(canvas, lines, alpha)
        kept = f"LINES KEPT {count:,}"
        if show_thrown_counter(frame) and frame < 1800:
            right = f"THROWN {self.thrown_at(count):,}"
        else:
            right = f"LIKENESS {self.likeness_percent(count)}%"
        self._draw_counters(canvas, kept, right)

    def _draw_caption(self, canvas, lines: list[str], alpha: float) -> None:
        max_w = self._s(SAFE[2] - SAFE[0] - 48)
        max_h = self._s(PLATE[3] - 48)
        floor = self._s(64 if max(len(line) for line in lines) <= 14 else 52)
        size = self._fit(self._word, lines, max_w, max_h, self._s(90), floor)
        self.last_caption_px = size / max(self.scale, 1e-6)
        gap = self._s(6)
        heights = [self._measure(self._word, size, line)[1] for line in lines]
        total = sum(heights) + gap * (len(lines) - 1)
        top = self._s(PLATE[1] + PLATE[3] / 2) - total / 2
        paint = skia.Paint(AntiAlias=True, Color=skia.Color(244, 241, 234, int(round(alpha * 255))))
        font = skia.Font(self._word, size)
        y = top
        for line, height in zip(lines, heights):
            width = self._measure(self._word, size, line)[0]
            x = self._s(540) - width / 2
            blob = skia.TextBlob.MakeFromString(line, font)
            canvas.drawTextBlob(blob, x, y + height * 0.82, paint)
            y += height + gap

    def _draw_counters(self, canvas, left: str, right: str) -> None:
        size = self._s(40)
        while size >= self._s(32):
            if self._measure(self._mono, size, left)[0] + self._measure(self._mono, size, right)[0] < self._s(760):
                break
            size -= self._s(1)
        font = skia.Font(self._mono, size)
        paint = skia.Paint(AntiAlias=True, Color=skia.Color(243, 233, 210, 255))
        baseline = self._s(382)
        canvas.drawTextBlob(skia.TextBlob.MakeFromString(left, font), self._s(150), baseline, paint)
        right_w = self._measure(self._mono, size, right)[0]
        canvas.drawTextBlob(skia.TextBlob.MakeFromString(right, font), self._s(930) - right_w, baseline, paint)

    def _fit(self, face, lines, max_w, max_h, start, floor) -> float:
        size = start
        while size >= floor:
            gap = self._s(6)
            widths = []
            heights = []
            for line in lines:
                width, height = self._measure(face, size, line)
                widths.append(width)
                heights.append(height)
            if max(widths) <= max_w and sum(heights) + gap * (len(lines) - 1) <= max_h:
                return size
            size -= self._s(1)
        return floor

    def _measure(self, face, size: float, text: str) -> tuple[float, float]:
        font = skia.Font(face, size)
        bounds = skia.Rect()
        font.measureText(text, bounds=bounds)
        return max(1.0, float(bounds.width())), max(1.0, float(bounds.height()))

    def _render_cv(self, frame: int) -> np.ndarray:
        image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        image[:] = (23, 17, 14)
        paper = self.paper_image(int(self.plan.counts[min(frame, 1799)])) if 0 < frame < 1800 else self._blank()
        x, y = int(round(self._s(PAPER[0]))), int(round(self._s(PAPER[1])))
        image[y : y + paper.shape[0], x : x + paper.shape[1]] = paper[:, :, ::-1]
        return image


def _wipe(frame: int) -> float | None:
    if 1440 <= frame <= 1535:
        return (frame - 1440) / (1535 - 1440)
    if 1536 <= frame <= 1631:
        return 1.0 - (frame - 1536) / (1631 - 1536)
    return None


def _overlay_alpha(frame: int) -> float:
    """Keep the first strokes readable until the page is dense, then let the ink take over."""
    if frame < 700:
        return 1.0
    if frame < 768:
        return 1.0 - (frame - 700) / 68.0
    return 0.0


def panel_fraction() -> float:
    return (PAPER[2] * PAPER[3]) / (1080 * 1920)


def text_boxes(lines: list[str], size: float, scale: float = 1.0) -> list[tuple[float, float, float, float]]:
    """Approximate caption boxes inside the plate, used by layout tests."""
    gap = 6 * scale
    height = size * 0.78
    total = height * len(lines) + gap * max(0, len(lines) - 1)
    top = (PLATE[1] + PLATE[3] / 2) * scale - total / 2
    boxes = []
    for line in lines:
        width = min((SAFE[2] - SAFE[0] - 48) * scale, max(8.0, len(line) * size * 0.62))
        x = 540 * scale - width / 2
        boxes.append((x, top, width, height))
        top += height + gap
    return boxes
