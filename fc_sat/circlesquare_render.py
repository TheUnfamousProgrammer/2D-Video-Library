"""One frame of the circlesquare short. One background, one curve, flat type."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from fc_sat.circlesquare_claims import ClaimBook, load_claims
from fc_sat.circlesquare_draw import bgr, make_canvas, mix_hex
from fc_sat.circlesquare_math import (
    A,
    DRAW_CAP,
    ZMAX,
    _choose_n,
    blend_doubling,
    camera_anchor,
    capped_circles,
    coefficient,
    curve_last_weight,
    full_curve,
    gap,
    math_to_screen,
    project_points,
    square_complex,
    square_corners_screen,
    theta_at,
    use_exact_square,
    zoom_at,
)
from fc_sat.circlesquare_schedule import (
    add_weight,
    collapse_mix,
    counter_value,
    doubling_blend,
)
from fc_sat.circlesquare_text import Script, load_script, top_lines
from fc_sat.circlesquare_timeline import Timeline, build_timeline
from fc_sat.easing import ease_out_cubic
from fc_sat.polycircle_draw import infinity_in_font, lemniscate, measure
from fc_sat.polycircle_format import format_count, format_gap

SAFE = (130.0, 950.0, 200.0, 1536.0)


@dataclass
class TextBox:
    text: str
    x: float
    y: float
    w: float
    h: float
    size: float
    kind: str
    color: str
    alpha: float

    def right(self) -> float:
        return self.x + self.w

    def bottom(self) -> float:
        return self.y + self.h


def overlaps(left: TextBox, right: TextBox) -> bool:
    return (
        left.x < right.right() - 0.5
        and right.x < left.right() - 0.5
        and left.y < right.bottom() - 0.5
        and right.y < left.bottom() - 0.5
    )


def inside_safe(box: TextBox, scale: float) -> bool:
    x0, x1, y0, y1 = (edge * scale for edge in SAFE)
    return box.x >= x0 - 1.0 and box.right() <= x1 + 1.0 and box.y >= y0 - 1.0 and box.bottom() <= y1 + 1.0


def hook_alpha(frame: int) -> float:
    if frame <= 179:
        return 1.0
    if frame <= 191:
        return (192 - frame) / 12.0
    if frame < 1800:
        return 0.0
    return min(1.0, (frame - 1800) / 23.0)


def curve_color(frame: int, colors: tuple[str, ...]) -> str:
    if frame >= 1812:
        return colors[0]
    if frame >= 1800:
        return mix_hex(colors[(1800 // 96) % 8], colors[0], (frame - 1800) / 12.0)
    bar = frame // 96
    local = frame % 96
    if local < 4 and bar > 0:
        return mix_hex(colors[(bar - 1) % 8], colors[bar % 8], local / 4.0)
    return colors[bar % 8]


def machinery_alpha(frame: int) -> float:
    if frame < 852:
        return 1.0
    if frame < 876:
        return 1.0 - (frame - 852) / 24.0
    if frame < 1800:
        return 0.0
    return collapse_mix(frame)


def reference_alpha(frame: int) -> float:
    if frame < 1800 or frame >= 1812:
        return 1.0
    if frame < 1806:
        return 1.0 - (frame - 1800) / 6.0
    return (frame - 1806) / 6.0


def _fit(kind: str, text: str, max_size: float, min_size: float, max_width: float) -> float:
    size = max_size
    while size > min_size + 0.5:
        if measure(kind, size, text).width <= max_width:
            return size
        size -= 2.0
    return min_size


def _dash_segments(pts: list[tuple[float, float]], on: float, off: float) -> list[list[tuple[float, float]]]:
    seq = [*pts, pts[0]]
    segments: list[list[tuple[float, float]]] = []
    drawing = True
    remain = on
    current: list[tuple[float, float]] = []
    for (ax, ay), (bx, by) in zip(seq, seq[1:]):
        length = math.hypot(bx - ax, by - ay)
        if length < 1e-6:
            continue
        traveled = 0.0
        while traveled < length - 1e-4:
            step = min(remain, length - traveled)
            t0 = traveled / length
            t1 = (traveled + step) / length
            start = (ax + (bx - ax) * t0, ay + (by - ay) * t0)
            end = (ax + (bx - ax) * t1, ay + (by - ay) * t1)
            if drawing:
                if not current:
                    current.append(start)
                current.append(end)
            traveled += step
            remain -= step
            if remain <= 1e-4:
                if drawing and len(current) >= 2:
                    segments.append(current)
                current = []
                drawing = not drawing
                remain = on if drawing else off
    if drawing and len(current) >= 2:
        segments.append(current)
    return segments


def _clip_segment(p0, p1, width: float, height: float, margin: float = 8.0):
    x0, y0 = p0
    x1, y1 = p1
    dx, dy = x1 - x0, y1 - y0
    u0, u1 = 0.0, 1.0
    for pi, qi in ((-dx, x0 + margin), (dx, width + margin - x0), (-dy, y0 + margin), (dy, height + margin - y0)):
        if abs(pi) < 1e-9:
            if qi < 0:
                return None
            continue
        t = qi / pi
        if pi < 0:
            if t > u1:
                return None
            u0 = max(u0, t)
        else:
            if t < u0:
                return None
            u1 = min(u1, t)
    if u0 > u1:
        return None
    return (x0 + u0 * dx, y0 + u0 * dy), (x0 + u1 * dx, y0 + u1 * dy)


def _window(points: np.ndarray, width: int, height: int, margin: float) -> np.ndarray:
    if len(points) < 8:
        return points
    inside = (
        (points[:, 0] >= -margin)
        & (points[:, 0] <= width + margin)
        & (points[:, 1] >= -margin)
        & (points[:, 1] <= height + margin)
    )
    if float(inside.mean()) > 0.8 or not np.any(inside):
        return points
    padded = np.concatenate(([False], inside, [False]))
    changes = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    if starts.size == 0:
        return points
    runs = [(int(a), int(b)) for a, b in zip(starts, ends)]
    if len(runs) >= 2 and runs[0][0] == 0 and runs[-1][1] == len(points):
        wrapped = np.concatenate((points[runs[-1][0] : runs[-1][1]], points[runs[0][0] : runs[0][1]]))
        return wrapped
    longest = max(runs, key=lambda item: item[1] - item[0])
    return points[longest[0] : longest[1]]


def _stride(points: np.ndarray, limit: int) -> np.ndarray:
    if len(points) <= limit:
        return points
    step = int(math.ceil(len(points) / limit))
    return points[::step]


def _as_tuples(points: np.ndarray) -> list[tuple[float, float]]:
    return [(float(x), float(y)) for x, y in points]


class CircleRenderer:
    def __init__(
        self,
        timeline: Timeline | None = None,
        script: Script | None = None,
        book: ClaimBook | None = None,
        width: int = 1080,
        height: int = 1920,
        hook: str = "A",
    ) -> None:
        self.timeline = timeline or build_timeline()
        self.book = book or load_claims()
        self.script = script or load_script(book=self.book)
        self.width = int(width)
        self.height = int(height)
        self.hook = hook
        self.scale = self.width / 1080.0
        self._infinity_font = infinity_in_font()

    def px(self, design: float) -> float:
        return float(design) * self.scale

    def stroke_px(self, design: float) -> float:
        """Output pixels. A stroke specified at 10 or more never lands thinner than 10."""
        width = float(design) * self.scale
        if design >= 10.0:
            return max(10.0, width)
        return width

    def _tease(self, frame: int) -> bool:
        return self.hook == "D" and int(frame) < 24

    def _shown_k(self, frame: int) -> int:
        if self._tease(frame):
            return 93
        return self.timeline.schedule.k(frame)

    def counter_text(self, frame: int) -> tuple[str, str]:
        if self._tease(frame):
            return format_count(93), self.timeline.config.text
        value = counter_value(self.timeline.schedule, frame)
        if value is None:
            glyph = "\u221e" if self._infinity_font else ""
            return glyph, self.timeline.config.text
        color = self.timeline.config.gold if value == 138 and 1488 <= frame < 1632 else self.timeline.config.text
        return format_count(value), color

    def sub_text(self, frame: int) -> str:
        frame = int(frame)
        if 1632 <= frame < 1806:
            return ""
        if 1440 <= frame < 1632:
            return "GAP UNDER 0.5 PX ON THIS SCREEN"
        circles = self._shown_k(frame)
        if 948 <= frame < 1344 and not self._tease(frame):
            return f"GAP {format_gap(gap(circles) * ZMAX)} PX AT {int(ZMAX)}x"
        return f"GAP {format_gap(gap(circles))} PX"

    def design_text_size(self, frame: int) -> float:
        if self._tease(frame):
            lines = self.script.hooks["A"]
        else:
            lines = top_lines(self.script, frame, self.hook)
        if not lines:
            return 0.0
        if (frame <= 191 or frame >= 1800) and hook_alpha(frame) <= 0.0 and not self._tease(frame):
            return 0.0
        limit = SAFE[1] - SAFE[0]
        return min(_fit("word", line, 100.0, 72.0, limit) for line in lines)

    def design_stroke(self, frame: int) -> float:
        return 10.0 if zoom_at(frame) > 1.05 and not self._tease(frame) else 14.0

    def plan(self, frame: int) -> list[TextBox]:
        frame = int(frame)
        cfg = self.timeline.config
        scale = self.scale
        boxes: list[TextBox] = []
        if self._tease(frame):
            lines = self.script.hooks["A"]
            alpha = 1.0
        elif frame <= 191 or frame >= 1800:
            lines = top_lines(self.script, frame, self.hook)
            alpha = hook_alpha(frame)
        else:
            lines = top_lines(self.script, frame, self.hook)
            alpha = 1.0 if lines else 0.0
        if lines and alpha > 0.01:
            limit = (SAFE[1] - SAFE[0]) * scale
            size = min(_fit("word", line, self.px(100), self.px(72), limit) for line in lines)
            ceiling = self.px(400) - self.px(12)
            floor = self.px(200)
            while size > self.px(72):
                step = size * 1.02
                block = step * (len(lines) - 1) + max(measure("word", size, line).height for line in lines)
                if floor + block <= ceiling:
                    break
                size -= 2.0
            step = size * 1.02
            block = step * (len(lines) - 1) + max(measure("word", size, line).height for line in lines)
            top = floor
            if top + block > ceiling:
                top = max(self.px(190), ceiling - block)
            for index, line in enumerate(lines):
                ink = measure("word", size, line)
                x = (self.width - ink.width) / 2.0
                y = top + index * step
                color = cfg.gold if (1440 <= frame < 1632 and "138" in line) else cfg.text
                boxes.append(TextBox(line, x, y, ink.width, ink.height, size, "word", color, alpha))

        counter, counter_color = self.counter_text(frame)
        sub = self.sub_text(frame)
        label = "CIRCLES"
        label_size = self.px(36)
        counter_size = self.px(130) * self._pop(frame)
        sub_size = self.px(34)
        if counter:
            counter_size = _fit("mono", counter, counter_size, self.px(72), (SAFE[1] - SAFE[0]) * scale)
        if sub:
            sub_size = _fit("word", sub, sub_size, self.px(22), (SAFE[1] - SAFE[0]) * scale)
        label_ink = measure("word", label_size, label)
        counter_ink = measure("mono", counter_size, counter or "0")
        sub_ink = measure("word", sub_size, sub) if sub else None
        gap_y = self.px(8)
        pad = self.px(16)
        stack = label_ink.height + gap_y + counter_ink.height
        if sub_ink is not None:
            stack += gap_y + sub_ink.height
        bottom_limit = SAFE[3] * scale
        top_min = self.px(1306) + pad
        top = min(top_min, bottom_limit - pad - stack)
        top = max(top, self.px(1306))
        cursor = top
        boxes.append(
            TextBox(
                label,
                (self.width - label_ink.width) / 2.0,
                cursor,
                label_ink.width,
                label_ink.height,
                label_size,
                "word",
                cfg.muted,
                1.0,
            )
        )
        cursor += label_ink.height + gap_y
        if counter:
            boxes.append(
                TextBox(
                    counter,
                    (self.width - counter_ink.width) / 2.0,
                    cursor,
                    counter_ink.width,
                    counter_ink.height,
                    counter_size,
                    "mono",
                    counter_color,
                    1.0,
                )
            )
            cursor += counter_ink.height + gap_y
        elif not self._infinity_font:
            # The lemniscate is drawn in render(); the plan still reserves its box.
            width = self.px(150)
            boxes.append(
                TextBox(
                    "",
                    (self.width - width) / 2.0,
                    cursor,
                    width,
                    counter_ink.height,
                    counter_size,
                    "mono",
                    cfg.text,
                    1.0,
                )
            )
            cursor += counter_ink.height + gap_y
        if sub_ink is not None:
            boxes.append(
                TextBox(
                    sub,
                    (self.width - sub_ink.width) / 2.0,
                    cursor,
                    sub_ink.width,
                    sub_ink.height,
                    sub_size,
                    "word",
                    cfg.muted,
                    1.0,
                )
            )
        return boxes

    def _pop(self, frame: int) -> float:
        if self._tease(frame):
            return 1.0
        schedule = self.timeline.schedule
        for item in schedule.adds:
            if item.interval < 6:
                continue
            duration = min(6, item.interval)
            if item.frame <= frame <= item.frame + duration - 1:
                return 1.0 + 0.08 * math.sin(math.pi * (frame - item.frame) / max(1, duration - 1))
        for item in schedule.doubles:
            if item.frame <= frame <= item.frame + 5:
                return 1.0 + 0.08 * math.sin(math.pi * (frame - item.frame) / 5.0)
        return 1.0

    def _shape(self, frame: int, zoom: float) -> np.ndarray | None:
        """Complex math samples, or None when the frame is the exact square."""
        frame = int(frame)
        if self._tease(frame):
            return full_curve(93, _choose_n(93, 1.0))
        mix = collapse_mix(frame)
        if frame >= 1800 and mix < 1.0:
            n = _choose_n(1, 1.0)
            return (1.0 - mix) * square_complex(n) + mix * full_curve(1, n)
        if frame >= 1806:
            return full_curve(1, _choose_n(1, 1.0))
        schedule = self.timeline.schedule
        blend = doubling_blend(schedule, frame)
        if blend is not None:
            item, weight = blend
            if use_exact_square(item.k_after, zoom) and weight > 0.999:
                return None
            after = capped_circles(item.k_after)
            before = capped_circles(item.k_before)
            n = _choose_n(after, zoom)
            if before >= DRAW_CAP and after >= DRAW_CAP:
                if use_exact_square(item.k_after, zoom):
                    return None
                return full_curve(DRAW_CAP, n)
            if use_exact_square(item.k_before, zoom) and weight < 0.001:
                return None
            return blend_doubling(before, weight, n)
        circles = schedule.k(frame)
        if use_exact_square(circles, zoom):
            return None
        draw = capped_circles(circles)
        n = _choose_n(draw, zoom)
        if draw >= DRAW_CAP:
            return full_curve(DRAW_CAP, n)
        if circles <= 93:
            weight = add_weight(schedule, frame, circles)
            if weight < 1.0:
                return curve_last_weight(circles, weight, n)
        return full_curve(draw, n)

    def _project_shape(self, samples: np.ndarray | None, zoom: float) -> np.ndarray:
        if samples is None:
            corners = square_corners_screen()
            return project_points(corners, zoom)
        screen = math_to_screen(samples)
        projected = project_points(screen, zoom)
        if zoom <= 1.05:
            return _stride(projected, 9000)
        window = _window(projected, self.width, self.height, self.px(120))
        return _stride(window, 9000)

    def _tip(self, frame: int, samples: np.ndarray | None) -> np.ndarray:
        theta = 0.0 if frame >= 1823 else theta_at(frame)
        if self._tease(frame):
            samples = full_curve(93, _choose_n(93, 1.0))
        if samples is None:
            samples = square_complex(4096)
        frac = (theta / (2.0 * math.pi)) % 1.0
        index = int(round(frac * len(samples))) % len(samples)
        point = math_to_screen(np.array([samples[index]]))
        return project_points(point, zoom_at(frame) if not self._tease(frame) else 1.0)[0]

    def render(self, frame: int) -> np.ndarray:
        frame = int(frame) % self.timeline.n_frames
        cfg = self.timeline.config
        zoom = 1.0 if self._tease(frame) else zoom_at(frame)
        canvas = make_canvas(self.width, self.height)
        canvas.fill(cfg.background)
        samples = self._shape(frame, zoom)
        color = curve_color(0 if self._tease(frame) else frame, cfg.bar_colors)
        points = self._project_shape(samples, zoom)
        tuples = _as_tuples(points)
        closed = zoom <= 1.05 or samples is None
        if len(tuples) >= 3:
            fill_pts = tuples if len(tuples) <= 1800 else tuples[:: max(1, len(tuples) // 1800)]
            if not closed:
                center = project_points(np.array([[cfg.center_x, cfg.center_y]]), zoom)[0]
                fill_pts = [*fill_pts, (float(center[0]), float(center[1]))]
            canvas.polygon(fill_pts, color, 0.25)
        self._reference(canvas, frame, zoom)
        stroke = self.stroke_px(self.design_stroke(frame))
        if len(tuples) >= 2:
            canvas.stroke(tuples, color, stroke, 1.0, closed=closed and samples is not None or samples is None)
        self._machinery(canvas, frame, zoom, color)
        tip = self._tip(frame, samples)
        canvas.dot(float(tip[0]), float(tip[1]), self.px(16), cfg.gold, 1.0)
        self._draw_text(canvas, frame)
        return bgr(canvas)

    def _reference(self, canvas, frame: int, zoom: float) -> None:
        alpha = reference_alpha(frame)
        if alpha <= 0.01:
            return
        corners = _as_tuples(project_points(square_corners_screen(), zoom))
        if zoom > 2.0:
            # The two edges that meet at the corner (first corner, then its neighbors).
            for other in (corners[1], corners[3]):
                clipped = _clip_segment(corners[0], other, self.width, self.height, self.px(8))
                if clipped is not None:
                    canvas.stroke([clipped[0], clipped[1]], "#FFFFFF", self.stroke_px(4), 0.80 * alpha, closed=False)
            return
        for segment in _dash_segments(corners, self.px(6), self.px(6)):
            canvas.stroke(segment, "#FFFFFF", self.stroke_px(4), 0.55 * alpha, closed=False)

    def _machinery(self, canvas, frame: int, zoom: float, color: str) -> None:
        alpha = 0.0 if self._tease(frame) else machinery_alpha(frame)
        if alpha <= 0.02:
            return
        schedule = self.timeline.schedule
        if frame >= 1800:
            circles = 1
            weights = np.ones(1, dtype=np.float64)
        else:
            circles = min(9, schedule.k(frame))
            weights = np.array([add_weight(schedule, frame, index) for index in range(1, circles + 1)], dtype=np.float64)
        theta = 0.0 if frame >= 1823 else theta_at(frame)
        point = 0j
        cfg = self.timeline.config
        for index in range(1, circles + 1):
            center = point
            gain = coefficient(index) * float(weights[index - 1]) * np.exp(1j * ((-1) ** (index + 1) * (2 * index - 1)) * theta)
            point += gain
            pop = 1.0
            flash = 0.0
            if circles <= 9 and index == circles and index > 1:
                item = schedule.adds[index - 2]
                if 0 <= frame - item.frame <= 7:
                    u = (frame - item.frame) / 7.0
                    pop = 0.6 + 0.4 * ease_out_cubic(u)
                    flash = 1.0 - u
            radius = abs(coefficient(index)) * pop
            center_s = project_points(math_to_screen(np.array([center])), zoom)[0]
            rim = project_points(math_to_screen(np.array([center + radius + 0j])), zoom)[0]
            screen_r = abs(float(rim[0] - center_s[0]))
            end_s = project_points(math_to_screen(np.array([point if weights[index - 1] > 0 else center])), zoom)[0]
            canvas.stroke(
                [(float(center_s[0]), float(center_s[1])), (float(end_s[0]), float(end_s[1]))],
                "#F2F4F8",
                self.stroke_px(6),
                0.60 * alpha,
            )
            canvas.circle_stroke(float(center_s[0]), float(center_s[1]), screen_r, "#F2F4F8", self.stroke_px(4), 0.35 * alpha)
            if flash > 0:
                canvas.circle_stroke(
                    float(center_s[0]), float(center_s[1]), screen_r, cfg.gold, self.stroke_px(4), flash * alpha
                )

    def _draw_text(self, canvas, frame: int) -> None:
        boxes = self.plan(frame)
        if not boxes:
            return
        cfg = self.timeline.config
        tops = [box for box in boxes if box.y < self.px(500)]
        bottoms = [box for box in boxes if box.y >= self.px(500)]
        self._plate(canvas, tops, 0.92)
        self._plate(canvas, bottoms, 1.0)
        for box in boxes:
            if not box.text:
                self._lemniscate(canvas, box)
                continue
            canvas.text(box.text, box.x, box.y, box.kind, box.size, box.color, box.alpha)
        if any(not box.text for box in boxes):
            return

    def _plate(self, canvas, boxes: list[TextBox], alpha: float) -> None:
        if not boxes or all(box.alpha <= 0.01 for box in boxes):
            return
        pad = self.px(24 if boxes[0].y < self.px(500) else 16)
        left = min(box.x for box in boxes) - pad
        right = max(box.right() for box in boxes) + pad
        top = min(box.y for box in boxes) - pad
        bottom = max(box.bottom() for box in boxes) + pad
        opacity = alpha * max(box.alpha for box in boxes)
        canvas.round_rect(left, top, right - left, bottom - top, self.px(16), self.timeline.config.plate, opacity)

    def _lemniscate(self, canvas, box: TextBox) -> None:
        cx = box.x + box.w / 2.0
        cy = box.y + box.h / 2.0
        path = lemniscate(cx, cy, box.h * 0.42)
        canvas.stroke(path, box.color, max(6.0, self.px(8)), box.alpha)

    def bbox_fraction(self, frame: int) -> float:
        zoom = 1.0 if self._tease(frame) else zoom_at(frame)
        samples = self._shape(frame, zoom)
        points = self._project_shape(samples, zoom)
        if len(points) < 2:
            return 0.0
        width = float(points[:, 0].max() - points[:, 0].min())
        height = float(points[:, 1].max() - points[:, 1].min())
        return (width * height) / float(self.width * self.height)

    def profile_ms(self) -> float:
        started = time.perf_counter()
        self.render(768)
        return (time.perf_counter() - started) * 1000.0
