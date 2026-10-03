"""One frame of the short. Text stays in the safe zone. The polygon is a flat stroke."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from fc_sat.polycircle_claims import ClaimBook, load_claims
from fc_sat.polycircle_draw import advance_width, bgr, infinity_in_font, lemniscate, make_canvas, measure
from fc_sat.polycircle_format import format_count, format_gap
from fc_sat.polycircle_geometry import (
    MorphSpec,
    angular_window,
    circle_samples,
    gap,
    points_from,
    polygon_arrays,
    project,
    zoom_at,
)
from fc_sat.polycircle_names import shape_labels
from fc_sat.polycircle_schedule import active_morph
from fc_sat.polycircle_text import Script, load_script, sub_template, top_lines
from fc_sat.polycircle_timeline import Timeline, build_timeline, counter_number

SAFE = (130.0, 950.0, 200.0, 1536.0)


def blend_hex(start: str, end: str, amount: float) -> str:
    amount = min(1.0, max(0.0, amount))

    def channels(value: str) -> np.ndarray:
        text = value.removeprefix("#")
        return np.array([int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16)], dtype=np.float64)

    mixed = (1.0 - amount) * channels(start) + amount * channels(end)
    red, green, blue = (int(round(channel)) for channel in mixed)
    return f"#{red:02X}{green:02X}{blue:02X}"


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
    gold: str = ""

    def right(self) -> float:
        return self.x + self.w

    def bottom(self) -> float:
        return self.y + self.h


def overlaps(left: TextBox, right: TextBox) -> bool:
    return left.x < right.right() - 0.5 and right.x < left.right() - 0.5 and left.y < right.bottom() - 0.5 and right.y < left.bottom() - 0.5


def inside_safe(box: TextBox, scale: float) -> bool:
    x0, x1, y0, y1 = (edge * scale for edge in SAFE)
    return box.x >= x0 - 0.6 and box.right() <= x1 + 0.6 and box.y >= y0 - 0.6 and box.bottom() <= y1 + 0.6


def hook_alpha(frame: int) -> float:
    if frame <= 179:
        return 1.0
    if frame <= 191:
        return (192 - frame) / 12.0
    if frame < 1800:
        return 0.0
    return min(1.0, (frame - 1800) / 23.0)


def kick_scale(frame: int, kicks: list[int]) -> float:
    for kick in kicks:
        if kick <= frame <= kick + 7:
            return 1.0 + 0.03 * math.sin(math.pi * (frame - kick) / 7.0)
    return 1.0


def counter_scale(frame: int, timeline: Timeline) -> float:
    for item in timeline.schedule.adds:
        if item.interval < 6:
            continue
        duration = min(6, item.interval)
        if item.frame <= frame <= item.frame + duration - 1:
            span = max(1, duration - 1)
            return 1.0 + 0.08 * math.sin(math.pi * (frame - item.frame) / span)
    return 1.0


class PolyRenderer:
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
        self.width = width
        self.height = height
        self.hook = hook
        self.scale = width / 1080.0
        self.center = np.array([540.0 * self.scale, 910.0 * self.scale], dtype=np.float64)
        self.radius = 370.0 * self.scale
        self.top = np.array([self.center[0], self.center[1] - self.radius], dtype=np.float64)
        self._infinity_font = infinity_in_font()
        self._labels = shape_labels(self.timeline)

    def _px(self, design: float) -> float:
        return design * self.scale

    def _fit_size(self, kind: str, text: str, max_size: float, max_width: float) -> float:
        size = max_size
        while size > 16:
            if measure(kind, size, text).width <= max_width:
                return size
            size -= 2
        return size

    def _box(self, text: str, cx: float, top: float, size: float, kind: str, color: str, alpha: float, gold: str = "") -> TextBox:
        ink = measure(kind, size, text)
        return TextBox(text, cx - ink.width / 2.0, top, ink.width, ink.height, size, kind, color, alpha, gold)

    def plan(self, frame: int) -> list[TextBox]:
        frame = int(frame)
        tops: list[TextBox] = []
        headline = top_lines(self.script, frame, self.hook)
        alpha = 1.0
        gold_bits: tuple[str, ...] = ()
        if frame <= 191 or frame >= 1800:
            alpha = hook_alpha(frame)
        else:
            for card in self.script.cards:
                if card.start <= frame < card.end:
                    gold_bits = card.gold
                    break
        if headline and alpha > 0.01:
            size = self._px(70)
            widest = max(measure("word", size, line).width for line in headline)
            limit = self._px(820)
            if widest > limit:
                size = min(self._fit_size("word", line, size, limit) for line in headline)
            # Three lines sit on the top of the safe area so the last line clears the shape.
            if len(headline) >= 3:
                start = self._px(200)
                step = self._px(100)
            else:
                start = self._px({1: 292, 2: 220}[len(headline)])
                step = self._px(112)
            for index, line in enumerate(headline):
                gold = next((bit for bit in gold_bits if bit and bit in line), "")
                tops.append(self._box(line, self.center[0], start + index * step, size, "word", self.timeline.config.text, alpha, gold))
            circle_top = float(self.center[1] - self.radius)
            overflow = tops[-1].bottom() - (circle_top - self._px(72))
            if overflow > 0:
                room = tops[0].y - self._px(200)
                shift = min(overflow, max(0.0, room))
                if shift > 0:
                    for box in tops:
                        box.y -= shift

        sub_text, sub_alpha = self._sub(frame)
        count = counter_number(self.timeline, frame)
        if count is None:
            counter_text = "\u221e" if self._infinity_font else "oo"
            counter_kind = "mono"
        else:
            counter_text = format_count(count)
            counter_kind = "mono"
        pop = counter_scale(frame, self.timeline)
        counter_size = self._fit_size(counter_kind, counter_text, self._px(118) * pop, self._px(800))
        counter_color = self.timeline.config.gold if 1488 <= frame < 1632 else self.timeline.config.text
        # Sub line stays on the bottom of the safe area. The name sits clear of the circle, then the number follows it.
        sub_top = self._px(1508)
        label = self._labels[frame] or "SIDES"
        label_size = self._fit_size("word", label, self._px(42 if label != "SIDES" else 36), self._px(800))
        ink = measure("word", label_size, label)
        circle_bottom = float(self.center[1] + self.radius)
        label_top = circle_bottom + self._px(64)
        counter_top = label_top + ink.height + self._px(28)
        counter = self._box(counter_text, self.center[0], counter_top, counter_size, counter_kind, counter_color, 1.0)
        while counter.bottom() > sub_top - self._px(22) and counter_size > self._px(72):
            counter_size -= self._px(4)
            counter = self._box(counter_text, self.center[0], counter_top, counter_size, counter_kind, counter_color, 1.0)
        # If the number still crowds the sub line, lift the name and the number together, but not into the circle.
        if counter.bottom() > sub_top - self._px(22):
            nudge = counter.bottom() - (sub_top - self._px(22))
            floor = circle_bottom + self._px(36)
            nudge = min(nudge, max(0.0, label_top - floor))
            label_top -= nudge
            counter_top -= nudge
            counter = self._box(counter_text, self.center[0], counter_top, counter_size, counter_kind, counter_color, 1.0)

        bottom: list[TextBox] = []
        label_box = self._box(label, self.center[0], label_top, label_size, "word", self.timeline.config.text if label != "SIDES" else self.timeline.config.muted, 1.0)
        bottom.append(label_box)
        bottom.append(counter)
        if sub_text and sub_alpha > 0.01:
            sub_size = self._fit_size("word", sub_text, self._px(32), self._px(800))
            sub = self._box(sub_text, self.center[0], sub_top, sub_size, "word", self.timeline.config.muted, sub_alpha)
            limit = self._px(1534)
            if sub.bottom() > limit:
                sub = self._box(sub_text, self.center[0], limit - sub.h, sub_size, "word", self.timeline.config.muted, sub_alpha)
            bottom.append(sub)
        # The name takes the SIDES slot. Five elements is the hook (three lines) plus the name plus the count.
        while len(tops) + len(bottom) > 5 and any(item.text == label for item in bottom):
            bottom = [item for item in bottom if item.text != label]
        return [*tops, *bottom]

    def _sub(self, frame: int) -> tuple[str, float]:
        sub = sub_template(self.script, frame)
        if sub is None:
            return "", 0.0
        alpha = 1.0
        if sub.fade and sub.fade[0] <= frame < sub.fade[1]:
            alpha = (sub.fade[1] - frame) / (sub.fade[1] - sub.fade[0])
        if sub.dynamic == "zoom_gap":
            n_sides = max(3, self.timeline.schedule.n(frame))
            label = format_gap(gap(n_sides, 370.0) * 36.0)
            return f"GAP {label} PX AT 36x", alpha
        if sub.id == "gap_end":
            return f"GAP {format_gap(float(self.book.get('gap_end').value))} PX", alpha
        if sub.id == "gap_61":
            shown = format_gap(float(self.book.get("gap_61").value))
            width = int(self.book.get("screen_px").value)
            return f"GAP {shown} PX ON THIS {width} PX SCREEN", alpha
        return "", 0.0

    def _color(self, frame: int) -> str:
        colors = self.timeline.config.bar_colors
        if frame >= 1805:
            return colors[0]
        index = 0
        for cursor, start in enumerate(self.timeline.schedule.bar_starts):
            if start <= frame:
                index = cursor
        if frame >= 1800:
            return blend_hex(colors[index % 8], colors[0], (frame - 1800) / 5.0)
        if index == 0 or frame - self.timeline.schedule.bar_starts[index] >= 4:
            return colors[index % 8]
        amount = (frame - self.timeline.schedule.bar_starts[index]) / 3.0
        return blend_hex(colors[(index - 1) % 8], colors[index % 8], amount)

    def _geometry(self, frame: int):
        zoom = zoom_at(frame)
        phi = float(self.timeline.angles[frame])
        n_sides = int(self.timeline.schedule.n(frame))
        window = angular_window(zoom, self.radius, self.width, self._px(64))
        morph = None
        found = active_morph(self.timeline.schedule, frame) if frame < 1800 else None
        if found is not None:
            kind, item, duration = found
            before = item.n_before if kind == "double" else item.n_after - 1
            morph = MorphSpec(kind, item.frame, duration, before, getattr(item, "edge", 0))
        angles, radii = polygon_arrays(max(3, n_sides), phi, self.radius, morph, frame, window)
        return zoom, window, morph, angles, radii

    def render(self, frame: int) -> np.ndarray:
        frame = int(min(self.timeline.n_frames - 1, max(0, frame)))
        canvas = make_canvas(self.width, self.height)
        cfg = self.timeline.config
        canvas.fill(cfg.background)
        zoom, window, morph, angles, radii = self._geometry(frame)
        pulse = kick_scale(frame, self.timeline.kicks)
        world = points_from(self.center, angles, radii)
        pulsed = self.center + pulse * (world - self.center)
        screen = project(pulsed, self.center, self.top, zoom, cfg.zoom_max)
        circle_alpha = self._circle_alpha(frame)
        if circle_alpha > 0:
            samples = circle_samples(0.0, self.radius, zoom, window)
            ring = points_from(self.center, samples, np.full(len(samples), self.radius))
            ring_screen = project(ring, self.center, self.top, zoom, cfg.zoom_max)
            closed = window is None
            canvas.stroke([(float(p[0]), float(p[1])) for p in ring_screen], "#FFFFFF", self._px(3), circle_alpha, closed=closed)
        closed_poly = window is None and len(angles) >= 3
        canvas.stroke(
            [(float(p[0]), float(p[1])) for p in screen],
            self._color(frame),
            self._px(3),
            1.0,
            closed=closed_poly,
        )
        self._vertex_dot(canvas, frame, morph, angles, radii, zoom, pulse)
        self._gap_marker(canvas, frame, angles, radii, zoom)
        for box in self.plan(frame):
            self._draw_text(canvas, box)
        if not self._infinity_font:
            count = counter_number(self.timeline, frame)
            if count is None:
                self._draw_infinity(canvas, frame)
        return bgr(canvas)

    def _circle_alpha(self, frame: int) -> float:
        if frame < 768 or frame > 1799:
            return 0.0
        if frame < 780:
            return 0.35 * (frame - 768) / 12.0
        return 0.35

    def _vertex_dot(self, canvas, frame, morph, angles, radii, zoom, pulse) -> None:
        if morph is None or morph.kind != "add":
            return
        event = next(add for add in self.timeline.schedule.adds if add.frame == morph.frame)
        if event.interval < 12 or frame >= event.frame + 6:
            return
        index = min(len(angles) - 1, event.edge + 1)
        point = points_from(self.center, angles[index : index + 1], radii[index : index + 1])[0]
        point = self.center + pulse * (point - self.center)
        screen = project(point.reshape(1, 2), self.center, self.top, zoom, self.timeline.config.zoom_max)[0]
        alpha = 1.0 - (frame - event.frame) / 6.0
        canvas.dot(float(screen[0]), float(screen[1]), self._px(9), self.timeline.config.gold, alpha)

    def _gap_marker(self, canvas, frame, angles, radii, zoom) -> None:
        if not 948 <= frame < 1176 or len(angles) < 2:
            return
        alpha = 1.0
        if frame >= 1152:
            alpha = (1176 - frame) / 24.0
        best = None
        best_score = 1e9
        for index in range(len(angles) - 1):
            p0 = points_from(self.center, angles[index : index + 1], radii[index : index + 1])[0]
            p1 = points_from(self.center, angles[index + 1 : index + 2], radii[index + 1 : index + 2])[0]
            mid = (p0 + p1) * 0.5
            direction = mid - self.center
            length = float(np.hypot(direction[0], direction[1]))
            if length < 1e-6:
                continue
            angle = math.atan2(float(direction[1]), float(direction[0]))
            score = abs((angle + math.pi / 2 + math.pi) % (2 * math.pi) - math.pi)
            if score < best_score:
                best_score = score
                circle = self.center + direction / length * self.radius
                best = (mid, circle)
        if best is None or best_score > 0.35:
            return
        mid_s = project(best[0].reshape(1, 2), self.center, self.top, zoom, 36.0)[0]
        circ_s = project(best[1].reshape(1, 2), self.center, self.top, zoom, 36.0)[0]
        # Shift the dimension line off the center column so the gap itself stays readable.
        shift = self._px(36)
        x = float(mid_s[0]) + shift
        y0, y1 = float(circ_s[1]), float(mid_s[1])
        gold = self.timeline.config.gold
        width = max(1.0, self._px(2))
        canvas.stroke([(x, y0), (x, y1)], gold, width, alpha)
        tick = self._px(10)
        canvas.stroke([(x - tick, y0), (x + tick, y0)], gold, width, alpha)
        canvas.stroke([(x - tick, y1), (x + tick, y1)], gold, width, alpha)

    def _draw_text(self, canvas, box: TextBox) -> None:
        if box.gold and box.gold in box.text:
            before, _, after = box.text.partition(box.gold)
            pen = box.x - measure(box.kind, box.size, box.text).left
            for piece, color in ((before, box.color), (box.gold, self.timeline.config.gold), (after, box.color)):
                if not piece:
                    continue
                ink = measure(box.kind, box.size, piece)
                canvas.text(piece, pen + ink.left, box.y, box.kind, box.size, color, box.alpha)
                pen += advance_width(box.kind, box.size, piece)
            return
        canvas.text(box.text, box.x, box.y, box.kind, box.size, box.color, box.alpha)

    def _draw_infinity(self, canvas, frame: int) -> None:
        boxes = [box for box in self.plan(frame) if box.text == "oo"]
        if not boxes:
            return
        box = boxes[0]
        canvas.stroke(
            lemniscate(box.x + box.w / 2, box.y + box.h / 2, box.h * 0.42),
            box.color,
            max(2.0, self._px(6)),
            1.0,
        )

    def profile_ms(self) -> float:
        import time

        started = time.perf_counter()
        self.render(1128)
        return (time.perf_counter() - started) * 1000.0
