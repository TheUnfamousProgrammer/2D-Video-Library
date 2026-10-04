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

    @njit(cache=True)
    def _composite_clipped(image, x0, y0, x1, y1, dark, alpha, radius):
        """Alpha-composite straight strokes in order. Coordinates are already in image pixels."""
        height, width, _ = image.shape
        dark_rgb = (0x1C / 255.0, 0x17 / 255.0, 0x12 / 255.0)
        light_rgb = (0xF3 / 255.0, 0xE9 / 255.0, 0xD2 / 255.0)
        span = radius * 2 + 1
        for index in range(x0.shape[0]):
            coverage = alpha[index]
            if coverage <= 0.0:
                continue
            color = dark_rgb if dark[index] else light_rgb
            x_a = x0[index]
            y_a = y0[index]
            x_b = x1[index]
            y_b = y1[index]
            dx = x_b - x_a
            dy = y_b - y_a
            p0 = -dx
            p1 = dx
            p2 = -dy
            p3 = dy
            q0 = x_a
            q1 = (width - 1.0) - x_a
            q2 = y_a
            q3 = (height - 1.0) - y_a
            u0 = 0.0
            u1 = 1.0
            keep = True
            for edge in range(4):
                if edge == 0:
                    edge_p, edge_q = p0, q0
                elif edge == 1:
                    edge_p, edge_q = p1, q1
                elif edge == 2:
                    edge_p, edge_q = p2, q2
                else:
                    edge_p, edge_q = p3, q3
                if edge_p == 0.0:
                    if edge_q < 0.0:
                        keep = False
                        break
                    continue
                edge_t = edge_q / edge_p
                if edge_p < 0.0:
                    if edge_t > u1:
                        keep = False
                        break
                    if edge_t > u0:
                        u0 = edge_t
                else:
                    if edge_t < u0:
                        keep = False
                        break
                    if edge_t < u1:
                        u1 = edge_t
            if not keep or u1 < u0:
                continue
            cx0 = x_a + dx * u0
            cy0 = y_a + dy * u0
            cx1 = x_a + dx * u1
            cy1 = y_a + dy * u1
            steps = int(max(abs(cx1 - cx0), abs(cy1 - cy0))) + 1
            for step in range(steps):
                t = 0.0 if steps == 1 else step / (steps - 1)
                x = int(cx0 + (cx1 - cx0) * t)
                y = int(cy0 + (cy1 - cy0) * t)
                for oy in range(span):
                    yy = y + oy - radius
                    if yy < 0 or yy >= height:
                        continue
                    row = image[yy]
                    for ox in range(span):
                        xx = x + ox - radius
                        if xx < 0 or xx >= width:
                            continue
                        row[xx, 0] = row[xx, 0] * (1.0 - coverage) + color[0] * coverage
                        row[xx, 1] = row[xx, 1] * (1.0 - coverage) + color[1] * coverage
                        row[xx, 2] = row[xx, 2] * (1.0 - coverage) + color[2] * coverage

else:

    def _paint_numba(*_args):
        return None

    def _composite_clipped(*_args):
        return None


PAPER = (159, 420, 762, 1104)
_PAPER_RGB = np.array((0x8F, 0x82, 0x6D), dtype=np.float32) / 255.0
_DARK_RGB = np.array((0x1C, 0x17, 0x12), dtype=np.float32) / 255.0
_LIGHT_RGB = np.array((0xF3, 0xE9, 0xD2), dtype=np.float32) / 255.0


def _stroke_alpha(index: int) -> float:
    """Early strokes stay readable. Later ones add detail without burying them."""
    return max(0.02, 0.55 * math.exp(-index / 1600.0))
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
        self._ink = np.empty((*picture["target"].shape, 3), dtype=np.float32)
        self._ink[:] = _PAPER_RGB
        self._ink_count = 0
        self._alpha = np.array([_stroke_alpha(index) for index in range(self.n)], dtype=np.float32)
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
            self._blit_paper(canvas, self._ink_image(frame, self.n, 1.0))
            return
        count = int(self.plan.counts[frame])
        zoom = zoom_at(frame)
        wipe = _wipe(frame)
        if wipe is not None:
            self._blit_paper(canvas, self._comparison(wipe))
            return
        self._blit_paper(canvas, self._ink_image(frame, count, zoom))
        if zoom == 1.0 and frame <= 180 and count > 0:
            self._draw_rejects(canvas, frame, count)

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
        self._draw_arrivals(canvas, frame, count)

    def _draw_arrivals(self, canvas, frame: int, count: int) -> None:
        """Stroke each new line on from one end, so a burst is a pour and not a cut."""
        if frame <= 0 or count <= THROW_LINES:
            return
        window = 5
        cap = 90 if 768 <= frame <= 840 else 16
        shown = 0
        for index in range(count - 1, THROW_LINES - 1, -1):
            age = frame - int(self.appear[index])
            if age > window:
                break
            if age < 0:
                continue
            self._stroke_grid(canvas, self._partial(index, (age + 1) / (window + 1)), 1.0, clip_paper=True)
            shown += 1
            if shown >= cap:
                break

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

    def _commit_ink(self, count: int) -> None:
        count = max(0, min(int(count), self.n))
        if count < self._ink_count:
            self._ink[:] = _PAPER_RGB
            self._ink_count = 0
        height, width = self._ink.shape[:2]
        while self._ink_count < count:
            index = self._ink_count
            rr, cc, val = line_aa(
                int(round(float(self.fit.y0[index]))),
                int(round(float(self.fit.x0[index]))),
                int(round(float(self.fit.y1[index]))),
                int(round(float(self.fit.x1[index]))),
            )
            mask = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
            if np.any(mask):
                alpha = (_stroke_alpha(index) * val[mask])[:, None]
                color = _DARK_RGB if int(self.fit.ink[index]) < 0 else _LIGHT_RGB
                self._ink[rr[mask], cc[mask]] = self._ink[rr[mask], cc[mask]] * (1.0 - alpha) + color * alpha
            self._ink_count += 1

    def _ink_image(self, frame: int, count: int, zoom: float) -> np.ndarray:
        if frame >= 1800:
            return self._flyoff_ink(frame)
        self._commit_ink(count)
        paper_w = int(round(self._s(PAPER[2])))
        paper_h = int(round(self._s(PAPER[3])))
        rgb = (np.clip(self._ink, 0.0, 1.0) * 255.0).astype(np.uint8)
        if zoom == 1.0:
            return cv2.resize(rgb, (paper_w, paper_h), interpolation=cv2.INTER_NEAREST)
        return self._zoom_strokes(count, zoom, paper_w, paper_h)

    def _zoom_strokes(self, count: int, zoom: float, paper_w: int, paper_h: int) -> np.ndarray:
        """Redraw the mouth crop from the same strokes, so the zoom stays lines instead of blocks."""
        height, width = self.picture["target"].shape
        half_w = width / (2.0 * zoom)
        half_h = height / (2.0 * zoom)
        cx, cy = float(self.picture["mouth"][0]), float(self.picture["mouth"][1])
        limit = max(0, min(int(count), self.n))
        x0 = self.fit.x0[:limit]
        y0 = self.fit.y0[:limit]
        x1 = self.fit.x1[:limit]
        y1 = self.fit.y1[:limit]
        hit = (
            (np.maximum(x0, x1) >= cx - half_w)
            & (np.minimum(x0, x1) <= cx + half_w)
            & (np.maximum(y0, y1) >= cy - half_h)
            & (np.minimum(y0, y1) <= cy + half_h)
        )
        scale_x = paper_w / (2.0 * half_w)
        scale_y = paper_h / (2.0 * half_h)
        origin_x = cx - half_w
        origin_y = cy - half_h
        image = np.empty((paper_h, paper_w, 3), dtype=np.float32)
        image[:] = _PAPER_RGB
        if np.any(hit) and _composite_clipped is not None:
            chosen = np.flatnonzero(hit)
            _composite_clipped(
                image,
                (x0[chosen] - origin_x).astype(np.float64) * scale_x,
                (y0[chosen] - origin_y).astype(np.float64) * scale_y,
                (x1[chosen] - origin_x).astype(np.float64) * scale_x,
                (y1[chosen] - origin_y).astype(np.float64) * scale_y,
                self.fit.ink[:limit][chosen] < 0,
                self._alpha[:limit][chosen],
                1,
            )
        return (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)

    def _flyoff_ink(self, frame: int) -> np.ndarray:
        height, width = self.picture["target"].shape
        image = np.empty((height, width, 3), dtype=np.float32)
        image[:] = _PAPER_RGB
        paper_w = int(round(self._s(PAPER[2])))
        paper_h = int(round(self._s(PAPER[3])))
        for index in range(self.n):
            progress = flyoff_progress(frame, index, self.n)
            if progress >= 1.0:
                continue
            # Square the progress so the first frame barely moves. A linear map
            # throws the face-building strokes sideways on the first fly-off frame.
            pose = self._final(index) if progress <= 0.0 else self._flight(index, 1.0 - progress * progress)
            rr, cc, val = line_aa(
                int(round(pose[1])),
                int(round(pose[0])),
                int(round(pose[3])),
                int(round(pose[2])),
            )
            mask = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
            if not np.any(mask):
                continue
            alpha = (_stroke_alpha(index) * val[mask])[:, None]
            color = _DARK_RGB if pose[4] < 0 else _LIGHT_RGB
            image[rr[mask], cc[mask]] = image[rr[mask], cc[mask]] * (1.0 - alpha) + color * alpha
        rgb = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
        return cv2.resize(rgb, (paper_w, paper_h), interpolation=cv2.INTER_NEAREST)

    def _draw_sheet(self, canvas, frame: int, count: int, stroke: float | None = None) -> None:
        canvas.save()
        canvas.clipRRect(self._paper_rrect(), True)
        self._paint_lines(canvas, frame, count, stroke)
        canvas.restore()

    def _paint_lines(self, canvas, frame: int, count: int, stroke: float | None = None) -> None:
        """Every kept line is a straight stroke. New ones grow on from one end."""
        x, y, w, h = self._paper_box()
        canvas.drawRect(skia.Rect.MakeXYWH(x, y, w, h), skia.Paint(AntiAlias=True, Color=skia.Color(*_hex("#8F826D"))))
        width = stroke if stroke is not None else max(1.0, self._s(1.35))
        settled = self._settled_count(frame, count)
        self._ensure_paths(settled)
        light = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=width, StrokeCap=skia.Paint.kRound_Cap, Color=skia.Color(*_hex("#F3E9D2")))
        dark = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=width, StrokeCap=skia.Paint.kRound_Cap, Color=skia.Color(*_hex("#1C1712")))
        if self._light_path is not None:
            canvas.drawPath(self._light_path, light)
            canvas.drawPath(self._dark_path, dark)
        window = 4
        ink = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=width, StrokeCap=skia.Paint.kRound_Cap, Color=skia.Color(*_hex("#1C1712")))
        chalk = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=width, StrokeCap=skia.Paint.kRound_Cap, Color=skia.Color(*_hex("#F3E9D2")))
        for index in range(settled, count):
            age = frame - int(self.appear[index])
            amount = 1.0 if age >= window else max(0.08, (age + 1) / (window + 1))
            pose = self._partial(index, amount)
            x0, y0 = self._grid_to_screen(pose[0], pose[1])
            x1, y1 = self._grid_to_screen(pose[2], pose[3])
            canvas.drawLine(x0, y0, x1, y1, ink if pose[4] < 0 else chalk)

    def _settled_count(self, frame: int, count: int) -> int:
        if count <= 0 or frame <= 0 or len(self.appear) == 0:
            return 0
        ready = int(np.searchsorted(self.appear, frame - 4, side="right"))
        return max(0, min(count, ready))

    def _draw_zoom(self, canvas, frame: int, count: int, zoom: float) -> None:
        mouth_x, mouth_y = self._grid_to_screen(self.picture["mouth"][0], self.picture["mouth"][1])
        canvas.save()
        canvas.clipRRect(self._paper_rrect(), True)
        canvas.translate(mouth_x, mouth_y)
        canvas.scale(zoom, zoom)
        canvas.translate(-mouth_x, -mouth_y)
        screen_px = self._s(1.6 + 5.0 * min(1.0, (zoom - 1.0) / 11.0))
        stroke = max(0.12, screen_px / max(zoom, 1.0))
        self._paint_lines(canvas, frame, count, stroke)
        canvas.restore()

    def _draw_flyoff(self, canvas, frame: int) -> None:
        x, y, w, h = self._paper_box()
        canvas.save()
        canvas.clipRRect(self._paper_rrect(), True)
        canvas.drawRect(skia.Rect.MakeXYWH(x, y, w, h), skia.Paint(AntiAlias=True, Color=skia.Color(*_hex("#8F826D"))))
        dark = skia.Path()
        light = skia.Path()
        for index in range(self.n):
            progress = flyoff_progress(frame, index, self.n)
            if progress >= 1.0:
                continue
            # Square the progress so the first frame barely moves. A linear map
            # throws the face-building strokes sideways on the first fly-off frame.
            pose = self._final(index) if progress <= 0.0 else self._flight(index, 1.0 - progress * progress)
            x0, y0 = self._grid_to_screen(pose[0], pose[1])
            x1, y1 = self._grid_to_screen(pose[2], pose[3])
            path = dark if pose[4] < 0 else light
            path.moveTo(x0, y0)
            path.lineTo(x1, y1)
        width = max(1.0, self._s(1.35))
        canvas.drawPath(light, skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=width, StrokeCap=skia.Paint.kRound_Cap, Color=skia.Color(*_hex("#F3E9D2"))))
        canvas.drawPath(dark, skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=width, StrokeCap=skia.Paint.kRound_Cap, Color=skia.Color(*_hex("#1C1712"))))
        canvas.restore()

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
            # Square the progress so the first frame barely moves. A linear map
            # throws the face-building strokes sideways on the first fly-off frame.
            pose = self._final(index) if progress <= 0.0 else self._flight(index, 1.0 - progress * progress)
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

    def _snapshot_ink(self, count: int) -> np.ndarray:
        saved_ink = self._ink
        saved_count = self._ink_count
        self._ink = np.empty_like(saved_ink)
        self._ink[:] = _PAPER_RGB
        self._ink_count = 0
        image = self._ink_image(1320, count, 1.0)
        self._ink = saved_ink
        self._ink_count = saved_count
        return image

    def _raster_vectors(self, count: int) -> np.ndarray:
        width = int(round(self._s(PAPER[2])))
        height = int(round(self._s(PAPER[3])))
        surface = skia.Surface(width, height)
        canvas = surface.getCanvas()
        red, green, blue = _hex("#8F826D")
        canvas.clear(skia.Color4f(red / 255, green / 255, blue / 255, 1))
        grid_h, grid_w = self.picture["target"].shape
        scale_x = width / grid_w
        scale_y = height / grid_h
        dark = skia.Path()
        light = skia.Path()
        limit = min(int(count), self.n)
        for index in range(limit):
            path = dark if int(self.fit.ink[index]) < 0 else light
            path.moveTo(float(self.fit.x0[index]) * scale_x, float(self.fit.y0[index]) * scale_y)
            path.lineTo(float(self.fit.x1[index]) * scale_x, float(self.fit.y1[index]) * scale_y)
        stroke = max(1.0, self._s(1.35))
        canvas.drawPath(light, skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=stroke, StrokeCap=skia.Paint.kRound_Cap, Color=skia.Color(*_hex("#F3E9D2"))))
        canvas.drawPath(dark, skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=stroke, StrokeCap=skia.Paint.kRound_Cap, Color=skia.Color(*_hex("#1C1712"))))
        image = surface.makeImageSnapshot().toarray()
        return np.ascontiguousarray(image[:, :, :3])

    def _comparison(self, amount: float) -> np.ndarray:
        if self.job["show_original"]:
            left = self._original
        else:
            if self._then is None:
                self._then = self._snapshot_ink(int(self.plan.counts[600]))
            left = self._then
        if self._now is None:
            self._now = self._snapshot_ink(self.n)
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

    def _stroke_grid(self, canvas, pose, alpha: float, clip_paper: bool = False) -> None:
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
        if clip_paper:
            canvas.clipRRect(self._paper_rrect(), True)
        else:
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
        if lines and alpha > 0:
            plate_paint = skia.Paint(AntiAlias=True, Color=skia.Color(27, 32, 48, int(round(0.92 * 255))))
            canvas.drawRRect(plate, plate_paint)
            self._draw_caption(canvas, lines, alpha)
        kept = f"LINES KEPT {count:,}"
        if show_thrown_counter(frame) and frame < 1800:
            right = f"THROWN {self.thrown_at(count):,}"
        else:
            right = f"LIKENESS {self.likeness_percent(count)}%"
        self._draw_counters(canvas, kept, right)

    def _draw_caption(self, canvas, lines: list[str], alpha: float) -> None:
        max_w = self._s(PLATE[2] - 32)
        max_h = self._s(PLATE[3] - 24)
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
