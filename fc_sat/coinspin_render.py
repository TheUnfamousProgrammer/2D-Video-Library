"""One frame of the coinspin short. Flat coins, one arrow, flat type."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from fc_sat.circlesquare_draw import bgr, make_canvas, mix_hex
from fc_sat.coinspin_claims import ClaimBook, load_claims
from fc_sat.coinspin_math import (
    CX,
    CY,
    DOUBLING_FRAMES,
    LAPS,
    MIN_COIN_PX,
    ORBIT_PX,
    TRAIL,
    TAU,
    Lap,
    Scene,
    direction,
    landed_ratio,
    scene_at,
    trace_points,
)
from fc_sat.coinspin_schedule import Counter, counter_at, spin_frames, uncounted_spin_frames
from fc_sat.coinspin_text import Script, card_at, load_script, sub_text, top_lines
from fc_sat.coinspin_timeline import Timeline, build_timeline
from fc_sat.easing import ease_in_out_cubic, ease_out_cubic
from fc_sat.polycircle_draw import advance_width, measure
from fc_sat.polycircle_format import format_count

SAFE = (130.0, 950.0, 200.0, 1536.0)
# Each counted lap keeps its own trace color index so a trace never recolors mid-lap.
TRACE_INDEX = {0: 4, 1: 3, 2: 7, 3: 7}
CHIPS = ("3/2", "3", "6", "9/2", "9")
CHIP_TEST_ANSWER = 1
CHIP_IN = 384
CHIP_PICK = 672
CHIP_STRIKE = 768
CHIP_OUT = (868, 876)
CHIP_Y = 420.0
CHIP_H = 72.0
FLASH_FRAMES = 12
# Rim ticks: one per rolling-coin circumference along the big coin's rim, so they count
# the spins that come from rolling. They light up during 3 FROM ROLLING, then the orbit
# path flashes for +1 FROM THE TRIP.
TICKS = (768, 1344)
TICK_LIGHTS = (876, 888, 900)
TRIP_FLASH = (912, 960)
TICK_MIN_SPACING = 6.0


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


def chip_alpha(frame: int, index: int) -> float:
    start = CHIP_IN + 12 * index
    if frame < start or frame >= CHIP_OUT[1]:
        return 0.0
    alpha = min(1.0, (frame - start + 1) / 4.0)
    if frame >= CHIP_OUT[0]:
        alpha *= 1.0 - (frame - CHIP_OUT[0]) / float(CHIP_OUT[1] - CHIP_OUT[0])
    return alpha


def chip_scale(frame: int, index: int) -> float:
    start = CHIP_IN + 12 * index
    if frame < start or frame > start + 8:
        return 1.0
    return 0.7 + 0.3 * ease_out_cubic((frame - start) / 8.0)


def trace_alpha(frame: int, lap_index: int) -> float:
    """How strongly a lap's trace shows. It draws during its lap, then holds or fades."""
    lap = LAPS[lap_index]
    if lap_index == 0:
        if frame < 192:
            return 1.0
        return max(0.0, 1.0 - (frame - 192) / 24.0) if frame < 216 else 0.0
    if lap_index == 1:
        if 216 <= frame < 360:
            return 1.0
        return max(0.0, 1.0 - (frame - 360) / 24.0) if 360 <= frame < 384 else 0.0
    if lap_index == 2:
        if lap.start <= frame < 948:
            return 1.0
        return max(0.0, 1.0 - (frame - 948) / 12.0) if 948 <= frame < 960 else 0.0
    return 0.0


def tick_count(frame: int) -> int:
    if not TICKS[0] <= frame < TICKS[1]:
        return 0
    if frame < DOUBLING_FRAMES[0]:
        return LAPS[2].ratio
    return landed_ratio(frame)


def tick_lit(frame: int, index: int) -> bool:
    return frame < DOUBLING_FRAMES[0] and index < len(TICK_LIGHTS) and frame >= TICK_LIGHTS[index]


def trip_alpha(frame: int) -> float:
    if not TRIP_FLASH[0] <= frame < TRIP_FLASH[1]:
        return 0.0
    rise = min(1.0, (frame - TRIP_FLASH[0] + 1) / 6.0)
    fall = min(1.0, (TRIP_FLASH[1] - frame) / 8.0)
    return min(rise, fall)


def mark_lap(frame: int) -> int:
    """The lap whose trace the rim dot is drawing, so the dot and its line share a color."""
    if frame < 216:
        return 0
    if frame < 384:
        return 1
    return 2


def sky_trail(frame: int) -> float:
    """Fraction of the orbit traced in gold during IT'S THE TRIP AROUND."""
    if frame < TRAIL[0]:
        return 0.0
    if frame >= TRAIL[1]:
        return 1.0
    return ease_in_out_cubic((frame - TRAIL[0]) / float(TRAIL[1] - TRAIL[0] - 12))


def _flash(frame: int, events: list[int]) -> float:
    for event in events:
        if event <= frame < event + FLASH_FRAMES:
            return 1.0 - (frame - event) / float(FLASH_FRAMES)
    return 0.0


def _fit(kind: str, text: str, max_size: float, min_size: float, max_width: float) -> float:
    size = max_size
    while size > min_size + 0.5:
        if measure(kind, size, text).width <= max_width:
            return size
        size -= 2.0
    return min_size


class CoinRenderer:
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
        self.cfg = self.timeline.config
        self._spins = spin_frames() + [frame for frame in uncounted_spin_frames() if frame < DOUBLING_FRAMES[0]]
        self._pops = sorted(set(spin_frames() + list(DOUBLING_FRAMES) + [768, 876, 1488, 1632]))
        self._traces: dict[int, np.ndarray] = {}

    def px(self, design: float) -> float:
        return float(design) * self.scale

    def stroke_px(self, design: float) -> float:
        """Output pixels. A stroke specified at 6 or more never lands thinner than 3."""
        width = float(design) * self.scale
        if design >= 6.0:
            return max(3.0, width)
        return width

    def _pt(self, x: float, y: float) -> tuple[float, float]:
        return float(x) * self.scale, float(y) * self.scale

    # ----- text -----

    def _pop(self, frame: int) -> float:
        for event in self._pops:
            if event <= frame <= event + 5:
                return 1.0 + 0.08 * math.sin(math.pi * (frame - event) / 5.0)
        return 1.0

    def _top_boxes(self, frame: int) -> list[TextBox]:
        cfg = self.cfg
        lines = top_lines(self.script, frame, self.hook)
        if frame <= 191 or frame >= 1800:
            alpha = hook_alpha(frame)
            gold_bits: tuple[str, ...] = ()
        else:
            alpha = 1.0 if lines else 0.0
            card = card_at(self.script, frame)
            gold_bits = card.gold if card is not None else ()
        if not lines or alpha <= 0.01:
            return []
        limit = (SAFE[1] - SAFE[0]) * self.scale
        size = min(_fit("word", line, self.px(96), self.px(72), limit) for line in lines)
        step = size * 1.02
        top = self.px(200)
        boxes = []
        for index, line in enumerate(lines):
            ink = measure("word", size, line)
            gold = next((bit for bit in gold_bits if bit and bit in line), "")
            boxes.append(
                TextBox(line, (self.width - ink.width) / 2.0, top + index * step, ink.width, ink.height, size, "word", cfg.text, alpha, gold)
            )
        return boxes

    def _chip_layout(self, frame: int) -> tuple[list[tuple[float, float, float]], float, float]:
        """Pill rectangles (x, width, alpha), the text size, and the row's top edge."""
        size = self.px(46)
        pad = self.px(26)
        gap = self.px(18)
        widths = [measure("mono", size, chip).width + 2 * pad for chip in CHIPS]
        total = sum(widths) + gap * (len(CHIPS) - 1)
        x = (self.width - total) / 2.0
        pills = []
        for index, width in enumerate(widths):
            pills.append((x, width, chip_alpha(frame, index)))
            x += width + gap
        return pills, size, self.px(CHIP_Y)

    def chip_box(self, frame: int) -> TextBox | None:
        pills, size, top = self._chip_layout(frame)
        alpha = max(alpha for _x, _w, alpha in pills)
        if alpha <= 0.01:
            return None
        left = pills[0][0]
        right = pills[-1][0] + pills[-1][1]
        return TextBox("  ".join(CHIPS), left, top, right - left, self.px(CHIP_H), size, "chips", self.cfg.text, alpha)

    def _counter_boxes(self, frame: int, counter: Counter) -> list[TextBox]:
        cfg = self.cfg
        limit = (SAFE[1] - SAFE[0]) * self.scale
        boxes: list[TextBox] = []
        sub = sub_text(self.script, frame)
        sub_size = _fit("word", sub, self.px(34), self.px(22), limit) if sub else 0.0
        gap_y = self.px(10)
        if counter.mode == "sky":
            row_size = self.px(84)
            days = f"DAYS  {counter.days}"
            spins = f"SPINS {counter.spins}"
            rows = [(days, cfg.text), (spins, cfg.gold if counter.gold else cfg.text)]
            inks = [measure("mono", row_size, text) for text, _ in rows]
            width = max(ink.width for ink in inks)
            cursor = self.px(1340)
            x = (self.width - width) / 2.0
            for (text, color), ink in zip(rows, inks):
                boxes.append(TextBox(text, x, cursor, ink.width, ink.height, row_size, "mono", color, 1.0))
                cursor += ink.height + self.px(18)
            if sub:
                ink = measure("word", sub_size, sub)
                boxes.append(TextBox(sub, (self.width - ink.width) / 2.0, cursor, ink.width, ink.height, sub_size, "word", cfg.muted, 1.0))
            return boxes
        label = "SPINS"
        label_size = self.px(36)
        label_ink = measure("word", label_size, label)
        if counter.mode == "formula":
            pieces = [(f"{format_count(int(counter.value))}", cfg.text), (f"+ {counter.plus}", cfg.gold)]
        else:
            pieces = [(format_count(int(counter.value)), cfg.gold if counter.gold else cfg.text)]
        joined = "  ".join(text for text, _ in pieces) if len(pieces) > 1 else pieces[0][0]
        # Lay the block out at rest size. The pop scales the number around its own center,
        # so the label and the sub-line never move.
        rest = _fit("mono", joined, self.px(130), self.px(72), limit)
        rest_height = max(measure("mono", rest, text).height for text, _ in pieces)
        size = rest * self._pop(frame)
        inks = [measure("mono", size, text) for text, _ in pieces]
        space = advance_width("mono", size, " ")
        width = sum(ink.width for ink in inks) + space * (len(pieces) - 1)
        height = max(ink.height for ink in inks)
        cursor = self.px(1340)
        boxes.append(TextBox(label, (self.width - label_ink.width) / 2.0, cursor, label_ink.width, label_ink.height, label_size, "word", cfg.muted, 1.0))
        cursor += label_ink.height + gap_y
        x = (self.width - width) / 2.0
        y = cursor + (rest_height - height) / 2.0
        for (text, color), ink in zip(pieces, inks):
            boxes.append(TextBox(text, x, y, ink.width, height, size, "mono", color, 1.0))
            x += ink.width + space
        cursor += rest_height + gap_y
        if sub:
            ink = measure("word", sub_size, sub)
            boxes.append(TextBox(sub, (self.width - ink.width) / 2.0, cursor, ink.width, ink.height, sub_size, "word", cfg.muted, 1.0))
        return boxes

    def plan(self, frame: int) -> list[TextBox]:
        frame = int(frame) % self.timeline.n_frames
        boxes = self._top_boxes(frame)
        chips = self.chip_box(frame)
        if chips is not None:
            boxes.append(chips)
        boxes.extend(self._counter_boxes(frame, counter_at(frame)))
        return boxes

    def design_text_size(self, frame: int) -> float:
        boxes = self._top_boxes(frame)
        return min((box.size for box in boxes), default=0.0) / self.scale

    # ----- drawing -----

    def render(self, frame: int) -> np.ndarray:
        frame = int(frame) % self.timeline.n_frames
        cfg = self.cfg
        canvas = make_canvas(self.width, self.height)
        canvas.fill(cfg.background)
        scene = scene_at(frame)
        if scene.sky > 0.0:
            self._orbit(canvas, frame, scene)
        self._traces_draw(canvas, frame)
        self._fixed(canvas, frame, scene)
        self._rolling(canvas, frame, scene)
        self._chips(canvas, frame)
        self._text(canvas, frame)
        return bgr(canvas)

    def _trace(self, lap_index: int) -> np.ndarray:
        if lap_index not in self._traces:
            lap = LAPS[lap_index]
            self._traces[lap_index] = trace_points(lap, lap.end) * self.scale
        return self._traces[lap_index]

    def _traces_draw(self, canvas, frame: int) -> None:
        for lap_index in range(3):
            alpha = trace_alpha(frame, lap_index)
            if alpha <= 0.01:
                continue
            lap: Lap = LAPS[lap_index]
            full = self._trace(lap_index)
            if frame < lap.end:
                fraction = (frame - lap.start) / float(lap.end - lap.start)
                count = max(2, int(round(fraction * (len(full) - 1))) + 1)
                points = full[:count]
            else:
                points = full
            if len(points) < 2:
                continue
            color = self.cfg.trace_colors[TRACE_INDEX[lap_index]]
            canvas.stroke([(float(x), float(y)) for x, y in points], color, self.stroke_px(6), 0.9 * alpha)

    def _fixed(self, canvas, frame: int, scene: Scene) -> None:
        cfg = self.cfg
        x, y = self._pt(*scene.fixed_center)
        radius = self.px(scene.fixed_px)
        body = mix_hex(cfg.fixed_coin, cfg.sun, scene.sky)
        rim = mix_hex(cfg.fixed_rim, cfg.sun, scene.sky)
        canvas.dot(x, y, radius, body, 1.0)
        rim_width = min(self.px(10), max(self.px(4), radius * 0.06))
        canvas.circle_stroke(x, y, radius - rim_width / 2.0, rim, rim_width, 1.0 - scene.sky)
        canvas.circle_stroke(x, y, radius * 0.84, rim, max(self.px(2), rim_width * 0.4), 0.5 * (1.0 - scene.sky))
        pulse = _flash(frame, list(DOUBLING_FRAMES)) * (1.0 - scene.sky)
        if pulse > 0:
            canvas.circle_stroke(x, y, radius + self.px(10), cfg.gold, self.stroke_px(4), 0.6 * pulse)
        self._ticks(canvas, frame, x, y, radius, rim_width, scene)
        trip = trip_alpha(frame)
        if trip > 0.0:
            path = self.px(scene.fixed_px + scene.rolling_px)
            self._dashed_circle(canvas, x, y, path, cfg.gold, self.stroke_px(6), trip, 72)
        if scene.sky > 0.0:
            beat = (frame % 24) / 24.0
            ring = 1.0 - beat
            canvas.circle_stroke(x, y, radius + self.px(14) + self.px(10) * beat, cfg.sun, self.stroke_px(4), 0.35 * ring * scene.sky)

    def _ticks(self, canvas, frame: int, x: float, y: float, radius: float, rim_width: float, scene: Scene) -> None:
        count = tick_count(frame)
        if count <= 0 or scene.sky > 0.0:
            return
        cfg = self.cfg
        outer = radius - rim_width
        length = min(self.px(26), 0.16 * radius)
        if TAU * outer / count < self.px(TICK_MIN_SPACING):
            canvas.circle_stroke(x, y, outer - length / 2.0, cfg.fixed_rim, length, 0.55)
            return
        width = max(self.px(3), min(self.px(7), 0.35 * TAU * outer / count))
        for index in range(count):
            d = direction(TAU * index / count)
            start = (x + outer * d[0], y + outer * d[1])
            end = (x + (outer - length) * d[0], y + (outer - length) * d[1])
            if tick_lit(frame, index):
                deep = (x + (outer - 1.4 * length) * d[0], y + (outer - 1.4 * length) * d[1])
                canvas.stroke([start, deep], cfg.gold, max(width, self.px(11)), 1.0)
            else:
                canvas.stroke([start, end], cfg.fixed_rim, width, 1.0)

    def _dashed_circle(self, canvas, x: float, y: float, radius: float, color: str, width: float, alpha: float, dashes: int) -> None:
        step = TAU / (2 * dashes)
        for index in range(dashes):
            a0 = 2 * index * step
            pts = [(x + radius * math.sin(a0 + step * k / 4.0), y - radius * math.cos(a0 + step * k / 4.0)) for k in range(5)]
            canvas.stroke(pts, color, width, alpha)

    def _rolling(self, canvas, frame: int, scene: Scene) -> None:
        cfg = self.cfg
        x, y = self._pt(*scene.rolling_center)
        design_radius = scene.rolling_px
        tiny = design_radius < MIN_COIN_PX and scene.sky < 0.5
        radius = self.px(max(design_radius, MIN_COIN_PX))
        body = mix_hex(cfg.rolling_coin, cfg.earth, scene.sky)
        rim = mix_hex(cfg.rolling_rim, cfg.earth, scene.sky)
        canvas.dot(x, y, radius, body, 1.0)
        if scene.sky < 1.0 and design_radius >= 14.0:
            rim_width = min(self.px(8), max(self.px(3), radius * 0.07))
            canvas.circle_stroke(x, y, radius - rim_width / 2.0, rim, rim_width, 1.0 - scene.sky)
        if tiny or (design_radius < 20.0 and scene.sky < 0.5):
            canvas.circle_stroke(x, y, radius + self.px(12), cfg.gold, self.stroke_px(4), 0.85 * (1.0 - scene.sky))
        if scene.sky > 0.0:
            self._night(canvas, scene, x, y, radius)
        if design_radius >= 20.0 and scene.sky < 0.5:
            color = self.cfg.trace_colors[TRACE_INDEX[mark_lap(frame)]]
            self._arrow(canvas, x, y, radius, scene.arrow, color, 1.0 - 2.0 * scene.sky)
        flash = _flash(frame, self._spins)
        if flash > 0 and scene.sky < 0.5:
            grow = 1.0 + 0.35 * (1.0 - flash)
            canvas.circle_stroke(x, y, radius * grow, cfg.gold, self.stroke_px(6), flash)

    def _arrow(self, canvas, x: float, y: float, radius: float, theta: float, mark: str, alpha: float) -> None:
        if alpha <= 0.01:
            return
        ink = self.cfg.background
        d = direction(theta)
        side = np.array([-d[1], d[0]])
        center = np.array([x, y])
        tail = center - 0.30 * radius * d
        neck = center + 0.42 * radius * d
        tip = center + 0.80 * radius * d
        width = max(self.px(3), 0.15 * radius)
        canvas.stroke([(float(tail[0]), float(tail[1])), (float(neck[0]), float(neck[1]))], ink, width, alpha)
        head = [tip, neck + 0.26 * radius * side, neck - 0.26 * radius * side]
        canvas.polygon([(float(p[0]), float(p[1])) for p in head], ink, alpha)
        canvas.dot(float(center[0]), float(center[1]), max(self.px(3), 0.09 * radius), ink, alpha)
        pen = center + radius * d
        canvas.dot(float(pen[0]), float(pen[1]), max(self.px(4), 0.1 * radius), mark, alpha)

    def _night(self, canvas, scene: Scene, x: float, y: float, radius: float) -> None:
        sun = np.array(self._pt(*scene.fixed_center))
        earth = np.array([x, y])
        toward = sun - earth
        length = float(np.hypot(*toward)) or 1.0
        u = toward / length
        v = np.array([-u[1], u[0]])
        points = []
        for step in range(49):
            t = -math.pi / 2.0 + math.pi * step / 48.0
            p = earth + radius * (math.cos(t) * (-u) + math.sin(t) * v)
            points.append((float(p[0]), float(p[1])))
        canvas.polygon(points, self.cfg.night, scene.sky)

    def _orbit(self, canvas, frame: int, scene: Scene) -> None:
        cfg = self.cfg
        cx, cy = self._pt(CX, CY)
        radius = self.px(ORBIT_PX)
        dash = TAU / 120.0
        for index in range(120):
            if index % 2:
                continue
            a0 = index * dash
            pts = []
            for k in range(5):
                a = a0 + dash * k / 4.0
                pts.append((cx + radius * math.sin(a), cy - radius * math.cos(a)))
            canvas.stroke(pts, cfg.muted, self.stroke_px(3), 0.45 * scene.sky)
        trail = sky_trail(frame)
        if trail > 0.0:
            count = max(2, int(240 * trail) + 1)
            pts = [
                (cx + radius * math.sin(TAU * trail * k / (count - 1)), cy - radius * math.cos(TAU * trail * k / (count - 1)))
                for k in range(count)
            ]
            canvas.stroke(pts, cfg.gold, self.stroke_px(8), scene.sky)
            if trail < 1.0:
                head = pts[-1]
                canvas.dot(head[0], head[1], self.px(12), cfg.gold, scene.sky)

    def _chips(self, canvas, frame: int) -> None:
        cfg = self.cfg
        pills, size, top = self._chip_layout(frame)
        height = self.px(CHIP_H)
        struck = frame >= CHIP_STRIKE
        for index, ((x, width, alpha), chip) in enumerate(zip(pills, CHIPS)):
            if alpha <= 0.01:
                continue
            scale = chip_scale(frame, index)
            w = width * scale
            h = height * scale
            px = x + (width - w) / 2.0
            py = top + (height - h) / 2.0
            picked = index == CHIP_TEST_ANSWER and frame >= CHIP_PICK
            fade = 0.4 if struck else 1.0
            fill = cfg.gold if picked else cfg.plate
            canvas.round_rect(px, py, w, h, h / 2.0, fill, alpha * (1.0 if not struck else 0.55))
            ink = measure("mono", size * scale, chip)
            color = cfg.background if picked else cfg.text
            canvas.text(chip, px + (w - ink.width) / 2.0, py + (h - ink.height) / 2.0, "mono", size * scale, color, alpha * fade)
        if struck and frame < CHIP_OUT[1]:
            u = min(1.0, (frame - CHIP_STRIKE) / 8.0)
            left = pills[0][0] - self.px(12)
            right = pills[-1][0] + pills[-1][1] + self.px(12)
            mid = top + height / 2.0
            end = left + (right - left) * ease_out_cubic(u)
            alpha = max(alpha for _x, _w, alpha in pills)
            canvas.stroke([(left, mid), (end, mid)], cfg.strike, self.stroke_px(8), alpha)

    def _text(self, canvas, frame: int) -> None:
        boxes = [box for box in self.plan(frame) if box.kind != "chips"]
        if not boxes:
            return
        tops = [box for box in boxes if box.y < self.px(600)]
        bottoms = [box for box in boxes if box.y >= self.px(600)]
        self._plate(canvas, tops, 0.92)
        self._plate(canvas, bottoms, 1.0)
        for box in boxes:
            self._draw_box(canvas, box)

    def _draw_box(self, canvas, box: TextBox) -> None:
        if box.gold and box.gold in box.text:
            full = measure(box.kind, box.size, box.text)
            baseline = box.y - full.top
            pen = box.x - full.left
            before, _, after = box.text.partition(box.gold)
            for piece, color in ((before, box.color), (box.gold, self.cfg.gold), (after, box.color)):
                if not piece:
                    continue
                ink = measure(box.kind, box.size, piece)
                if piece.strip():
                    canvas.text(piece, pen + ink.left, baseline + ink.top, box.kind, box.size, color, box.alpha)
                pen += advance_width(box.kind, box.size, piece)
            return
        canvas.text(box.text, box.x, box.y, box.kind, box.size, box.color, box.alpha)

    def _plate(self, canvas, boxes: list[TextBox], alpha: float) -> None:
        if not boxes or all(box.alpha <= 0.01 for box in boxes):
            return
        pad = self.px(24 if boxes[0].y < self.px(600) else 16)
        left = min(box.x for box in boxes) - pad
        right = max(box.right() for box in boxes) + pad
        top = min(box.y for box in boxes) - pad
        bottom = max(box.bottom() for box in boxes) + pad
        opacity = alpha * max(box.alpha for box in boxes)
        canvas.round_rect(left, top, right - left, bottom - top, self.px(16), self.cfg.plate, opacity)

    # ----- probes for hooks and verify -----

    def stage_fraction(self, frame: int) -> float:
        scene = scene_at(frame)
        extent = max(
            scene.fixed_px,
            float(np.hypot(scene.rolling_center[0] - CX, scene.rolling_center[1] - CY)) + scene.rolling_px,
        )
        return (math.pi * extent * extent) / float(1080 * 1920)

    def profile_ms(self) -> float:
        started = time.perf_counter()
        self.render(700)
        return (time.perf_counter() - started) * 1000.0
