"""One frame of the coinspin short. Flat coins, a face to watch, flat type.

Draw order: path circles, grey coin, rim paint or road, stamps and ghosts, the gold coin,
the rod, marks (arrows, chevrons, strokes), then the text plates and text.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from fc_sat.circlesquare_draw import bgr, make_canvas, mix_hex
from fc_sat.coinspin_claims import ClaimBook, load_claims
from fc_sat.coinspin_math import (
    BOLTS,
    CARRY,
    CX,
    CY,
    DROP,
    GLIDE_OUT,
    HALF,
    R1,
    R_BIG,
    R_SMALL,
    RECOLOR,
    RESIZE,
    RETURN,
    RESUME,
    ROAD_X0,
    ROAD_X1,
    ROAD_Y,
    ROD_GROW,
    SAT_END,
    SNAP,
    TAU,
    UNROLL,
    Pose,
    direction,
    gold_pose,
    grey_radius,
    paint_arc,
    rod_angle,
    stage_scale,
    unroll_points,
    unroll_progress,
    upright_frames,
)
from fc_sat.coinspin_schedule import Chip, Counter, Term, counter_at
from fc_sat.coinspin_text import Script, card_at, load_script, top_lines
from fc_sat.coinspin_timeline import Timeline, build_timeline
from fc_sat.easing import ease_in_out_cubic, ease_out_cubic
from fc_sat.polycircle_draw import advance_width, measure

SAFE = (130.0, 950.0, 200.0, 1536.0)
COUNTER_TOP = 1332.0
GHOST = (144, DROP)
LAP_STAMP = (RESUME, 900)
SAT_STAMPS = (1452, 1512, 1572)
ARROWS = ((292, 322, 384, 396), (780, 810, 876, 888))
STROKES = (DROP, SAT_END)
OUTLINES = ((1368, 400.0), (1376, 540.0), (1384, 680.0))
OUTLINE_OUT = (1400, 1412)
CARRY_PATH_OUT = (1344, 1356)
SAT_PATH = (1704, 1740)
RIM_FLASHES = (1272, 1680)
TRIP_FLASHES = (1296, 1740)
QUARTER_PULSES = (1580, 1604)
POP_FRAMES = 10


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
    spans: list = field(default_factory=list)

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


def _fit(kind: str, text: str, max_size: float, min_size: float, max_width: float) -> float:
    size = max_size
    while size > min_size + 0.5:
        if measure(kind, size, text).width <= max_width:
            return size
        size -= 2.0
    return min_size


def _fade(frame: int, start: int, end: int) -> float:
    """1 before ``start``, 0 from ``end``, linear between."""
    if frame < start:
        return 1.0
    if frame >= end:
        return 0.0
    return 1.0 - (frame - start) / float(end - start)


def _rise(frame: int, start: int, end: int) -> float:
    """0 before ``start``, 1 from ``end``, linear between."""
    return 1.0 - _fade(frame, start, end)


def _since(frame: int, events, span: int) -> float:
    """1 on an event frame, fading to 0 over ``span`` frames."""
    for event in events:
        if event <= frame < event + span:
            return 1.0 - (frame - event) / float(span)
    return 0.0


def grey_alpha(frame: int) -> float:
    if frame < GLIDE_OUT[0] or frame >= RETURN[1]:
        return 1.0
    if frame < GLIDE_OUT[1]:
        return 1.0 - 0.8 * ease_in_out_cubic((frame - GLIDE_OUT[0]) / float(GLIDE_OUT[1] - GLIDE_OUT[0]))
    if frame < RETURN[0]:
        return 0.2
    return 0.2 + 0.8 * ease_in_out_cubic((frame - RETURN[0]) / float(RETURN[1] - RETURN[0]))


def arrow_progress(frame: int) -> tuple[float, float]:
    """(how much of the circular arrow is drawn, its opacity)."""
    for start, done, fade, gone in ARROWS:
        if start <= frame < gone:
            drawn = ease_out_cubic(min(1.0, (frame - start) / float(done - start)))
            return drawn, _fade(frame, fade, gone)
    return 0.0, 0.0


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
        self.ticks = upright_frames()
        self.pops = sorted(set(self.ticks))

    # ----- coordinates -----

    def px(self, design: float) -> float:
        return float(design) * self.scale

    def _stage(self, frame: int, x: float, y: float) -> tuple[float, float]:
        """Stage point to output pixels, with the punch and snap scale around the grey center."""
        s = stage_scale(frame)
        return self.px(CX + s * (x - CX)), self.px(CY + s * (y - CY))

    def _len(self, frame: int, design: float) -> float:
        return self.px(design * stage_scale(frame))

    def color(self, key: str) -> str:
        return getattr(self.cfg, key)

    # ----- text plan -----

    def _pop(self, frame: int) -> float:
        for event in self.pops:
            if event <= frame < event + POP_FRAMES:
                return 1.0 + 0.18 * (1.0 - ease_out_cubic((frame - event) / float(POP_FRAMES)))
        return 1.0

    def _top_boxes(self, frame: int) -> list[TextBox]:
        lines = top_lines(self.script, frame, self.hook)
        card = card_at(self.script, frame)
        gold_bits = card.gold if card is not None else ()
        limit = (SAFE[1] - SAFE[0]) * self.scale
        size = min(_fit("word", line, self.px(92), self.px(64), limit) for line in lines)
        step = size * 1.06
        top = self.px(200)
        boxes = []
        for index, line in enumerate(lines):
            ink = measure("word", size, line)
            gold = next((bit for bit in gold_bits if bit and bit in line), "")
            boxes.append(
                TextBox(line, (self.width - ink.width) / 2.0, top + index * step, ink.width, ink.height, size, "word", self.cfg.text, 1.0, gold)
            )
        return boxes

    def _row(self, pieces: list[tuple[str, str]], size: float, y: float, alpha: float) -> TextBox:
        """One line of mono text made of colored pieces, centered."""
        text = "".join(piece for piece, _ in pieces)
        ink = measure("mono", size, text)
        box = TextBox(text, (self.width - ink.width) / 2.0, y, ink.width, ink.height, size, "row", self.cfg.text, alpha)
        box.spans = pieces
        return box

    def _counter_boxes(self, frame: int, counter: Counter) -> list[TextBox]:
        cfg = self.cfg
        boxes: list[TextBox] = []
        limit = (SAFE[1] - SAFE[0]) * self.scale
        cursor = self.px(COUNTER_TOP)
        gap = self.px(10)
        if counter.mode == "single":
            if counter.label:
                size = self.px(40)
                ink = measure("word", size, counter.label)
                boxes.append(TextBox(counter.label, (self.width - ink.width) / 2.0, cursor, ink.width, ink.height, size, "word", cfg.muted, 1.0))
            cursor += measure("word", self.px(40), "SPINS").height + self.px(16)
            term = counter.terms[0]
            rest = self.px(140)
            rest_ink = measure("mono", rest, "8")
            size = rest * self._pop(frame)
            ink = measure("mono", size, term.text)
            y = cursor + (rest_ink.height - ink.height) / 2.0
            boxes.append(TextBox(term.text, (self.width - ink.width) / 2.0, y, ink.width, ink.height, size, "mono", self.color(term.color), 1.0))
            cursor += rest_ink.height + self.px(14)
            if counter.sub:
                sub_size = _fit("word", counter.sub, self.px(36), self.px(24), limit)
                ink = measure("word", sub_size, counter.sub)
                boxes.append(TextBox(counter.sub, (self.width - ink.width) / 2.0, cursor, ink.width, ink.height, sub_size, "word", cfg.muted, 1.0))
        else:
            a, b, result = counter.terms
            pieces = [(a.text, a.color), (" + ", "muted"), (b.text, b.color), (" = ", "muted"), (result.text, result.color)]
            size = _fit("mono", "".join(p for p, _ in pieces), self.px(124), self.px(80), limit)
            row = self._row(pieces, size, cursor, 1.0)
            boxes.append(row)
            label_y = row.bottom() + self.px(16)
            for index, term in ((0, a), (2, b)):
                if not term.label:
                    continue
                center = self._span_center(row, index)
                label_size = self.px(38)
                ink = measure("word", label_size, term.label)
                left = min(max(center - ink.width / 2.0, self.px(SAFE[0])), self.px(SAFE[1]) - ink.width)
                boxes.append(TextBox(term.label, left, label_y, ink.width, ink.height, label_size, "word", self.color(term.color), 1.0))
        if counter.chip is not None and counter.chip.alpha > 0.0:
            boxes.append(self._chip_box(counter.chip))
        return boxes

    def _span_center(self, row: TextBox, index: int) -> float:
        pen = row.x - measure("mono", row.size, row.text).left
        for position, (piece, _color) in enumerate(row.spans):
            width = advance_width("mono", row.size, piece)
            if position == index:
                return pen + width / 2.0
            pen += width
        return row.x + row.w / 2.0

    def _chip_box(self, chip: Chip) -> TextBox:
        width = self.px(210)
        height = self.px(150)
        x = self.px(150) - self.px(20) * (1.0 - chip.slide)
        return TextBox(f"{chip.label} {chip.value}", x, self.px(COUNTER_TOP), width, height, self.px(30), "chip", self.cfg.text, chip.alpha)

    def plan(self, frame: int) -> list[TextBox]:
        frame = int(frame) % self.timeline.n_frames
        return self._top_boxes(frame) + self._counter_boxes(frame, counter_at(frame))

    def design_text_size(self, frame: int) -> float:
        boxes = self._top_boxes(frame)
        return min((box.size for box in boxes), default=0.0) / self.scale

    # ----- drawing -----

    def render(self, frame: int) -> np.ndarray:
        frame = int(frame) % self.timeline.n_frames
        canvas = make_canvas(self.width, self.height)
        canvas.fill(self.cfg.background)
        pose = gold_pose(frame)
        self._paths(canvas, frame)
        self._grey(canvas, frame, pose)
        self._paint(canvas, frame, pose)
        self._ghosts(canvas, frame)
        self._outlines(canvas, frame)
        self._coin(canvas, frame, pose.center, pose.radius, pose.theta, 1.0)
        self._rod(canvas, frame, pose)
        self._marks(canvas, frame, pose)
        self._text(canvas, frame)
        return bgr(canvas)

    def _dashed_arc(self, canvas, frame: int, radius: float, sweep: float, color: str, width: float, alpha: float) -> None:
        if sweep <= 0 or alpha <= 0.01:
            return
        dash = TAU / 96.0
        count = int(math.ceil(sweep / dash))
        for index in range(count):
            if index % 2:
                continue
            a0 = index * dash
            a1 = min(sweep, a0 + dash)
            pts = [self._stage(frame, CX + radius * math.sin(a), CY - radius * math.cos(a)) for a in np.linspace(a0, a1, 4)]
            canvas.stroke(pts, color, width, alpha)

    def _paths(self, canvas, frame: int) -> None:
        cfg = self.cfg
        width = self._len(frame, 6)
        if CARRY[0] <= frame < CARRY_PATH_OUT[1]:
            sweep = rod_angle(frame)
            boost = _since(frame, TRIP_FLASHES, 12)
            alpha = _fade(frame, *CARRY_PATH_OUT)
            self._dashed_arc(canvas, frame, 2.0 * R1, sweep, cfg.trip, width * (1.0 + 1.2 * boost), alpha)
        if SAT_PATH[0] <= frame < SNAP[0]:
            sweep = TAU * ease_out_cubic(min(1.0, (frame - SAT_PATH[0]) / float(SAT_PATH[1] - SAT_PATH[0])))
            boost = _since(frame, TRIP_FLASHES, 12)
            self._dashed_arc(canvas, frame, R_BIG + R_SMALL, sweep, cfg.trip, width * (1.0 + 1.2 * boost), 1.0)

    def _grey(self, canvas, frame: int, pose: Pose) -> None:
        cfg = self.cfg
        x, y = self._stage(frame, CX, CY)
        radius = self._len(frame, grey_radius(frame))
        alpha = grey_alpha(frame)
        body = mix_hex(cfg.background, cfg.grey_coin, alpha)
        rim = mix_hex(cfg.background, cfg.grey_rim, alpha)
        canvas.dot(x, y, radius, body, 1.0)
        rim_width = self._len(frame, 10)
        canvas.circle_stroke(x, y, radius - rim_width / 2.0, rim, rim_width, 1.0)
        canvas.circle_stroke(x, y, radius * 0.84, rim, self._len(frame, 3), 0.6)
        screw = self._len(frame, 14)
        canvas.dot(x, y, screw, mix_hex(cfg.background, "#3A4152", alpha), 1.0)
        canvas.stroke([(x - screw * 0.6, y), (x + screw * 0.6, y)], mix_hex(cfg.background, cfg.grey_coin, alpha), self._len(frame, 4), 1.0)

    def _paint(self, canvas, frame: int, pose: Pose) -> None:
        cfg = self.cfg
        progress = unroll_progress(frame)
        width = self._len(frame, 10)
        if progress > 0.0:
            right, left = unroll_points(progress)
            for half in (right, left):
                canvas.stroke([self._stage(frame, px, py) for px, py in half], cfg.rolling, width, 1.0)
            if progress >= 1.0:
                for x in (ROAD_X0, ROAD_X1):
                    canvas.stroke([self._stage(frame, x, ROAD_Y - 16), self._stage(frame, x, ROAD_Y + 16)], cfg.rolling, self._len(frame, 6), 1.0)
            return
        radius = grey_radius(frame) - 5.0
        sweep = paint_arc(frame)
        alpha = 1.0
        if RECOLOR[0] <= frame < RECOLOR[1]:
            sweep = TAU
            alpha = _fade(frame, *RECOLOR)
        flash = _since(frame, RIM_FLASHES, 12)
        if flash > 0.0:
            sweep = TAU
            alpha = max(alpha if RESIZE[0] <= frame else 0.0, flash)
            width = width * (1.0 + 0.6 * flash)
        if sweep > 0.0 and alpha > 0.01:
            count = max(2, int(180 * sweep / TAU) + 2)
            pts = [self._stage(frame, CX + radius * math.sin(a), CY - radius * math.cos(a)) for a in np.linspace(0.0, sweep, count)]
            canvas.stroke(pts, cfg.rolling, width, alpha)
        pulse = _since(frame, QUARTER_PULSES, 12)
        if pulse > 0.0:
            pts = [self._stage(frame, CX + radius * math.sin(a), CY - radius * math.cos(a)) for a in np.linspace(sweep, TAU, 40)]
            canvas.stroke(pts, cfg.text, width, pulse)

    def _ghosts(self, canvas, frame: int) -> None:
        if GHOST[0] <= frame < GHOST[1]:
            alpha = 0.3 * _rise(frame, GHOST[0], GHOST[0] + 24)
            self._coin(canvas, frame, (CX, CY - 2 * R1), R1, 0.0, alpha)
        if LAP_STAMP[0] <= frame < LAP_STAMP[1]:
            alpha = 0.3 * _fade(frame, 876, LAP_STAMP[1])
            self._coin(canvas, frame, (CX, CY + 2 * R1), R1, 0.0, alpha)
        for tick in SAT_STAMPS:
            if tick + 6 <= frame < SNAP[0]:
                phi = TAU * (SAT_STAMPS.index(tick) + 1) / 4.0
                center = np.array([CX, CY]) + (R_BIG + R_SMALL) * direction(phi)
                alpha = 0.3 * _rise(frame, tick + 6, tick + 14)
                self._coin(canvas, frame, (float(center[0]), float(center[1])), R_SMALL, 0.0, alpha)

    def _outlines(self, canvas, frame: int) -> None:
        cfg = self.cfg
        for start, x in OUTLINES:
            if not start <= frame < OUTLINE_OUT[1]:
                continue
            pop = 0.6 + 0.4 * ease_out_cubic(min(1.0, (frame - start) / 8.0))
            alpha = _fade(frame, *OUTLINE_OUT)
            sx, sy = self._stage(frame, x, CY)
            canvas.circle_stroke(sx, sy, self._len(frame, R_SMALL * pop), cfg.gold_coin, self._len(frame, 6), alpha)

    def _coin(self, canvas, frame: int, center, radius: float, theta: float, alpha: float) -> None:
        """The gold coin with its face. Theta turns the face clockwise; 0 is upright."""
        if alpha <= 0.01:
            return
        cfg = self.cfg
        x, y = self._stage(frame, *center)
        r = self._len(frame, radius)
        canvas.dot(x, y, r, cfg.gold_coin, alpha)
        canvas.circle_stroke(x, y, r - r * 0.035, cfg.gold_rim, r * 0.07, alpha)
        ex = np.array([math.cos(theta), math.sin(theta)])
        ey = np.array([-math.sin(theta), math.cos(theta)])
        origin = np.array([x, y])

        def at(lx: float, ly: float) -> tuple[float, float]:
            p = origin + r * (lx * ex + ly * ey)
            return float(p[0]), float(p[1])

        face = cfg.face
        for side in (-1.0, 1.0):
            canvas.dot(*at(0.30 * side, -0.16), r * 0.10, face, alpha)
        smile = [at(0.40 * math.cos(a), -0.06 + 0.40 * math.sin(a)) for a in np.linspace(math.radians(25), math.radians(155), 24)]
        canvas.stroke(smile, face, r * 0.09, alpha)
        canvas.polygon([at(-0.16, -0.64), at(0.16, -0.64), at(0.0, -0.93)], face, alpha)

    def _rod(self, canvas, frame: int, pose: Pose) -> None:
        if not ROD_GROW[0] <= frame < CARRY_PATH_OUT[1]:
            return
        cfg = self.cfg
        alpha = _fade(frame, *CARRY_PATH_OUT)
        beta = rod_angle(frame)
        d = direction(beta)
        full = 2.0 * R1 - 0.55 * R1
        grow = ease_out_cubic(min(1.0, (frame - ROD_GROW[0]) / float(ROD_GROW[1] - ROD_GROW[0])))
        end = np.array([CX, CY]) + full * grow * d
        canvas.stroke([self._stage(frame, CX, CY), self._stage(frame, float(end[0]), float(end[1]))], cfg.trip, self._len(frame, 18), alpha)
        for bolt, distance in zip(BOLTS, (2.0 * R1 - 0.88 * R1, 2.0 * R1 - 0.70 * R1)):
            if frame < bolt:
                continue
            pop = 0.6 + 0.4 * ease_out_cubic(min(1.0, (frame - bolt) / 6.0))
            p = np.array([CX, CY]) + distance * d
            canvas.dot(*self._stage(frame, float(p[0]), float(p[1])), self._len(frame, 10 * pop), cfg.text, alpha)

    def _outer(self, pose: Pose) -> np.ndarray:
        if pose.rolling_on == "road" or abs(pose.center[1] - (ROAD_Y - R1)) < 1e-6 and pose.center[0] in (ROAD_X0, ROAD_X1):
            return np.array([0.0, -1.0])
        offset = np.array([pose.center[0] - CX, pose.center[1] - CY])
        length = float(np.hypot(*offset))
        outer = offset / length if length > 1e-6 else np.array([0.0, -1.0])
        # Below the stage the counter plate sits; put the mark beside the coin instead.
        return np.array([1.0, 0.0]) if outer[1] > 0.5 else outer

    def _marks(self, canvas, frame: int, pose: Pose) -> None:
        cfg = self.cfg
        x, y = self._stage(frame, *pose.center)
        r = self._len(frame, pose.radius)
        flash = _since(frame, self.ticks, 8)
        if flash > 0.0:
            canvas.circle_stroke(x, y, r + self.px(3), cfg.text, self.px(8), flash)
        chevron = _since(frame, self.ticks, 18)
        if chevron > 0.0:
            outer = self._outer(pose)
            cx, cy = x + (r + self.px(40)) * outer[0], y + (r + self.px(40)) * outer[1]
            size = self.px(18)
            canvas.stroke([(cx - size, cy + size * 0.6), (cx, cy - size * 0.6), (cx + size, cy + size * 0.6)], cfg.text, self.px(7), chevron)
        drawn, alpha = arrow_progress(frame)
        if drawn > 0.0 and alpha > 0.01:
            radius = r + self.px(24)
            start = math.radians(-60.0)
            sweep = math.radians(300.0) * drawn
            pts = [(x + radius * math.sin(start + t), y - radius * math.cos(start + t)) for t in np.linspace(0.0, sweep, 60)]
            canvas.stroke(pts, cfg.text, self.px(7), alpha)
            tip_angle = start + sweep
            tip = np.array([x + radius * math.sin(tip_angle), y - radius * math.cos(tip_angle)])
            forward = np.array([math.cos(tip_angle), math.sin(tip_angle)])
            outward = np.array([math.sin(tip_angle), -math.cos(tip_angle)])
            head = self.px(20)
            wing_a = tip - head * forward + head * 0.7 * outward
            wing_b = tip - head * forward - head * 0.7 * outward
            canvas.polygon([(float(tip[0]), float(tip[1])), (float(wing_a[0]), float(wing_a[1])), (float(wing_b[0]), float(wing_b[1]))], cfg.text, alpha)
        burst = _since(frame, STROKES, 12)
        if burst > 0.0:
            grow = 1.0 - burst
            for index in range(8):
                d = direction(TAU * index / 8.0 + TAU / 16.0)
                inner = r + self.px(30) + self.px(20) * grow
                outer_r = inner + self.px(28)
                canvas.stroke([(x + inner * d[0], y + inner * d[1]), (x + outer_r * d[0], y + outer_r * d[1])], cfg.text, self.px(7), burst)

    def _text(self, canvas, frame: int) -> None:
        boxes = self.plan(frame)
        tops = [box for box in boxes if box.y < self.px(700)]
        bottoms = [box for box in boxes if box.y >= self.px(700) and box.kind != "chip"]
        self._plate(canvas, tops, 0.92)
        self._plate(canvas, bottoms, 1.0)
        counter = counter_at(frame)
        for box in boxes:
            if box.kind == "chip":
                self._draw_chip(canvas, box, counter.chip)
            elif box.kind == "row":
                self._draw_row(canvas, box, counter)
            else:
                self._draw_box(canvas, box)

    def _draw_box(self, canvas, box: TextBox) -> None:
        if box.gold and box.gold in box.text:
            before, _, after = box.text.partition(box.gold)
            self._draw_spans(canvas, box, [(before, box.color), (box.gold, self.cfg.gold), (after, box.color)])
            return
        canvas.text(box.text, box.x, box.y, box.kind, box.size, box.color, box.alpha)

    def _draw_spans(self, canvas, box: TextBox, spans: list[tuple[str, str]]) -> None:
        kind = "mono" if box.kind == "row" else box.kind
        full = measure(kind, box.size, box.text)
        baseline = box.y - full.top
        pen = box.x - full.left
        for piece, color in spans:
            if piece.strip():
                ink = measure(kind, box.size, piece)
                canvas.text(piece, pen + ink.left, baseline + ink.top, kind, box.size, color, box.alpha)
            pen += advance_width(kind, box.size, piece)

    def _draw_row(self, canvas, box: TextBox, counter: Counter) -> None:
        terms = {0: counter.terms[0], 2: counter.terms[1], 4: counter.terms[2]}
        for index, term in terms.items():
            if term.pulse <= 0.0:
                continue
            center = self._span_center(box, index)
            half = advance_width("mono", box.size, term.text) / 2.0 + self.px(14)
            canvas.round_rect(center - half, box.y - self.px(12), 2 * half, box.h + self.px(24), self.px(14), self.color(term.color), 0.35 * term.pulse)
        self._draw_spans(canvas, box, [(piece, self.color(color)) for piece, color in box.spans])

    def _draw_chip(self, canvas, box: TextBox, chip: Chip | None) -> None:
        if chip is None:
            return
        cfg = self.cfg
        lift = 1.0 + 0.08 * chip.pulse
        canvas.round_rect(box.x, box.y, box.w, box.h, self.px(18), cfg.plate, box.alpha)
        if chip.pulse > 0.0:
            canvas.round_rect(box.x, box.y, box.w, box.h, self.px(18), cfg.text, 0.18 * chip.pulse * box.alpha)
        label_size = self.px(30)
        ink = measure("word", label_size, chip.label)
        canvas.text(chip.label, box.x + (box.w - ink.width) / 2.0, box.y + self.px(18), "word", label_size, cfg.muted, box.alpha)
        value_size = self.px(92) * lift
        ink = measure("mono", value_size, chip.value)
        vx = box.x + (box.w - ink.width) / 2.0
        vy = box.y + self.px(18) + self.px(30) + self.px(14)
        canvas.text(chip.value, vx, vy, "mono", value_size, cfg.text, box.alpha)
        if chip.struck > 0.0:
            x0, y0 = vx - self.px(16), vy + ink.height + self.px(8)
            x1, y1 = vx + ink.width + self.px(16), vy - self.px(8)
            u = ease_out_cubic(chip.struck)
            canvas.stroke([(x0, y0), (x0 + (x1 - x0) * u, y0 + (y1 - y0) * u)], cfg.strike, self.px(10), box.alpha)

    def _plate(self, canvas, boxes: list[TextBox], alpha: float) -> None:
        if not boxes or all(box.alpha <= 0.01 for box in boxes):
            return
        pad = self.px(24 if boxes[0].y < self.px(700) else 16)
        left = min(box.x for box in boxes) - pad
        right = max(box.right() for box in boxes) + pad
        top = min(box.y for box in boxes) - pad
        bottom = max(box.bottom() for box in boxes) + pad
        opacity = alpha * max(box.alpha for box in boxes)
        canvas.round_rect(left, top, right - left, bottom - top, self.px(16), self.cfg.plate, opacity)

    # ----- probes for hooks and verify -----

    def stage_fraction(self, frame: int) -> float:
        pose = gold_pose(frame)
        extent = max(grey_radius(frame), math.hypot(pose.center[0] - CX, pose.center[1] - CY) + pose.radius)
        return (math.pi * extent * extent) / float(1080 * 1920)

    def profile_ms(self) -> float:
        started = time.perf_counter()
        self.render(1200)
        return (time.perf_counter() - started) * 1000.0
