"""Frame plans: text boxes, chart geometry, and the funnel. Drawing is elsewhere."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from fc_sat.collatz_draw import Dot, Ink, Plate, Stroke, measure
from fc_sat.collatz_format import display_hundredths, format_hundredths, format_int
from fc_sat.collatz_math import trajectory
from fc_sat.collatz_script import chunk_caption
from fc_sat.easing import ease_in_cubic, ease_in_out_cubic, ease_out_back, ease_out_cubic

SAFE = (130.0, 950.0, 200.0, 1536.0)
CHART = (110.0, 970.0, 640.0, 1300.0)
FUNNEL_SCALE = 860.0 / 178.0
MAX_TEXTS = 4


@dataclass
class TextEl:
    text: str
    x: float
    y: float
    w: float
    h: float
    size: float
    font: str
    color: str
    role: str
    priority: int


@dataclass
class FramePlan:
    bg: str
    texts: list[TextEl] = field(default_factory=list)
    strokes: list = field(default_factory=list)
    dots: list = field(default_factory=list)
    plates: list = field(default_factory=list)
    funnel_count: int = 0
    hud_step: int | None = None
    hud_value: int | None = None
    alpha_lines: float = 1.0


def funnel_x(steps_to_go: float, x_right: float = CHART[1]) -> float:
    return x_right - steps_to_go * FUNNEL_SCALE


def log_y(value: float, y_top: float, y_bottom: float, vmax: float) -> float:
    base = y_bottom - 28.0
    if value <= 1 or vmax <= 1:
        return base
    span = math.log(max(value, 1.0000001)) / math.log(vmax)
    span = min(1.0, max(0.0, span))
    return base - span * (base - y_top)


def overlaps(a: TextEl, b: TextEl, gap: float = 4.0) -> bool:
    return not (
        a.x + a.w + gap <= b.x
        or b.x + b.w + gap <= a.x
        or a.y + a.h + gap <= b.y
        or b.y + b.h + gap <= a.y
    )


def inside_safe(box: TextEl) -> bool:
    x0, x1, y0, y1 = SAFE
    return box.x >= x0 - 0.6 and box.x + box.w <= x1 + 0.6 and box.y >= y0 - 0.6 and box.y + box.h <= y1 + 0.6


def _fit(text: str, font: str, size: float, max_w: float) -> tuple[float, Ink]:
    ink = measure(font, size, text)
    if ink.width <= max_w or ink.width <= 0:
        return size, ink
    sized = size
    for _ in range(8):
        sized *= max_w / max(ink.width, 1.0)
        sized = max(size * 0.6, sized)
        ink = measure(font, sized, text)
        if ink.width <= max_w:
            break
    return sized, ink


def _center(text: str, font: str, size: float, y: float, color: str, role: str, priority: int) -> TextEl:
    x0, x1, _, _ = SAFE
    sized, ink = _fit(text, font, size, x1 - x0)
    x = x0 + (x1 - x0 - ink.width) / 2
    box = TextEl(text, x, y, ink.width, ink.height, sized, font, color, role, priority)
    if box.y < SAFE[2]:
        box.y = SAFE[2]
    if box.y + box.h > SAFE[3]:
        box.y = SAFE[3] - box.h
    return box


def _keep(candidates: list[TextEl]) -> list[TextEl]:
    kept: list[TextEl] = []
    for box in sorted(candidates, key=lambda item: item.priority):
        if len(kept) >= MAX_TEXTS:
            break
        if not inside_safe(box):
            continue
        if any(overlaps(box, other) for other in kept):
            continue
        kept.append(box)
    return kept


def _hook_rows(title: str) -> list[str]:
    if ". " in title:
        rows: list[str] = []
        parts = [part.strip() for part in title.split(". ") if part.strip()]
        for index, part in enumerate(parts):
            if index < len(parts) - 1 and not part.endswith("."):
                part += "."
            rows.extend(chunk_caption(part))
        return rows[:2]
    return chunk_caption(title)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _pop(frames_since: int, peak_scale: float = 1.06) -> float:
    if frames_since < 0 or frames_since >= 6:
        return 1.0
    if frames_since < 3:
        return 1.0 + (peak_scale - 1.0) * (frames_since / 3.0)
    return peak_scale - (peak_scale - 1.0) * ((frames_since - 3) / 3.0)


class Film:
    def __init__(self, timeline, script, book) -> None:
        self.timeline = timeline
        self.script = script
        self.book = book
        self.palette = {key: f"#{value}" if isinstance(value, str) else value for key, value in timeline.spec["palette"].items()}
        self.traj = {n: trajectory(n) for n in range(1, 1001)}
        self._funnel: list[list[tuple[float, float]]] | None = None
        self.hook_captions: list[tuple[float, str]] | None = None
        self.bound = int(book.get("c_verified").value)
        self.hundredths = display_hundredths(self.bound)

    def funnel_polylines(self) -> list[list[tuple[float, float]]]:
        if self._funnel is None:
            y_top, y_bottom = 330.0, 1300.0
            lines = []
            for n in range(1, 1001):
                if n == 27:
                    continue
                values = self.traj[n]
                total = len(values) - 1
                lines.append(
                    [
                        (funnel_x(total - index), log_y(value, y_top, y_bottom, 250504))
                        for index, value in enumerate(values)
                    ]
                )
            self._funnel = lines
        return self._funnel

    def plan(self, index: int) -> FramePlan:
        fps = self.timeline.fps
        t = index / fps
        scene = self.timeline.at(min(t, self.timeline.duration - 1e-6))
        bg = self.palette["bg_twist"] if scene.id == "S4" else self.palette["bg"]
        plan = FramePlan(bg=bg)
        if scene.id == "S1":
            self._hook(plan, t, index)
        elif scene.id == "S2":
            self._rule(plan, t, index)
        elif scene.id == "S3":
            self._proof(plan, t, index)
        elif scene.id == "S4":
            self._twist(plan, t)
        elif scene.id == "S5":
            self._ride(plan, t, index)
        elif scene.id == "S6":
            self._funnel_scene(plan, t)
        elif scene.id == "S7":
            self._resolve(plan, t)
        elif scene.id == "S8":
            self._punch(plan, t)
        else:
            self._bridge(plan, t)
        plan.texts = _keep(plan.texts)
        return plan

    def _counter(self, hundredths: int) -> str:
        name = format_int(self.bound).split()[-1]
        return f"{format_hundredths(hundredths)} {name}"

    def _caption(self, t: float) -> str | None:
        chosen = None
        for line in self.script:
            start = line.offset - 0.05 + self.timeline.shifted_by * (line.offset >= 2.6)
            end = start + 0.05 + min(line.slot_max_s, 1.8)
            if start <= t < end:
                chosen = line.caption_text
        return chosen

    def _add_caption(self, plan: FramePlan, text: str, priority: int = 2) -> None:
        rows = chunk_caption(text)
        block = "\n".join(rows)
        # One plate, one element per line, still counted separately.
        y = 1340.0
        boxes = []
        for row in rows:
            box = _center(row, "word", 62, y, self.palette["text"], "primary", priority)
            boxes.append(box)
            plan.texts.append(box)
            y += box.h + 6
        if boxes:
            left = min(box.x for box in boxes) - 24
            width = max(box.w for box in boxes) + 48
            plan.plates.append(Plate(max(SAFE[0], left), 1332, min(SAFE[1] - SAFE[0], width), min(110.0, y - 1332), "#000000", 0.55))

    def _hook(self, plan: FramePlan, t: float, index: int) -> None:
        title = "PICK ANY NUMBER" if t < 1.10 else "IT ALWAYS ENDS AT 1"
        if self.hook_captions:
            title = self.hook_captions[0][1]
            for when, text in self.hook_captions:
                if t >= when:
                    title = text
        y = 300.0
        for row in _hook_rows(title):
            box = _center(row, "word", 88, y, self.palette["text"], "primary", 0)
            plan.texts.append(box)
            y += box.h + 8
        bob = math.sin(t * 2.4 + 0.9) * 8.0
        number = _center("6", "mono", 320, 860 + bob, self.palette["text"], "primary", 1)
        plan.texts.append(number)
        if t >= 1.10 and "ENDS AT" in title:
            plan.dots.append(Dot(900, 1240, 18, self.palette["text"], 4))
            ink = measure("mono", 34, "1")
            plan.texts.append(TextEl("1", 820, 1188, ink.width, ink.height, 34, "mono", self.palette["gold"], "primary", 3))
        if not self.hook_captions:
            self._maybe_caption(plan, t)

    def _maybe_caption(self, plan: FramePlan, t: float) -> None:
        caption = self._caption(t)
        if caption and not any(box.text == caption or caption.startswith(box.text) for box in plan.texts):
            self._add_caption(plan, caption, 2)

    def _chart_y(self, value: float, vmax: float, y0: float = CHART[2], y1: float = CHART[3]) -> float:
        return log_y(value, y0, y1, vmax)

    def _plot(self, values: list[int], fired: int, partial: float, vmax: float, x_of, alpha: float = 1.0, node_r: float = 11) -> tuple:
        from fc_sat.collatz_draw import Dot, Stroke

        pts = [(x_of(i), self._chart_y(v, vmax)) for i, v in enumerate(values[: fired + 1])]
        strokes = []
        dots = []
        for index in range(len(pts) - 1):
            color = self.palette["down"] if values[index] % 2 == 0 else self.palette["up"]
            strokes.append(Stroke([pts[index], pts[index + 1]], color, 6, alpha))
            dots.append(Dot(pts[index][0], pts[index][1], node_r, color, 0, alpha))
        head = pts[-1] if pts else (CHART[0], self._chart_y(values[0], vmax))
        if fired < len(values) - 1 and len(pts) >= 1:
            nxt = (x_of(fired + 1), self._chart_y(values[fired + 1], vmax))
            eased = ease_out_cubic(min(1.0, partial / 0.6))
            head = (_lerp(pts[-1][0], nxt[0], eased), _lerp(pts[-1][1], nxt[1], eased))
            color = self.palette["down"] if values[fired] % 2 == 0 else self.palette["up"]
            strokes.append(Stroke([pts[-1], head], color, 6, alpha))
        if pts:
            dots.append(Dot(pts[-1][0], pts[-1][1], node_r, self.palette["text"], 0, alpha))
        dots.append(Dot(head[0], head[1], 18, self.palette["text"], 4, alpha))
        return strokes, dots, head

    def _rule(self, plan: FramePlan, t: float, index: int) -> None:
        from fc_sat.collatz_draw import Dot

        scene = self.timeline.scene("S2")
        values = self.traj[6]
        times = [scene.rel(offset, self.timeline.fps) for offset in scene.spec["step_offsets"]]
        fired = sum(1 for when in times if t >= when)
        fired = min(fired, len(values) - 1)
        if fired == 0:
            partial = 0.0
        else:
            start = times[fired - 1]
            end = times[fired] if fired < len(times) else scene.end
            partial = 0.0 if end <= start else (t - start) / (end - start)
        move = ease_out_cubic(min(1.0, max(0.0, (t - scene.start) / scene.spec["move_s"])))
        vmax = max(values) * 1.15
        width = CHART[1] - CHART[0]

        def x_of(i: int) -> float:
            return CHART[0] + (i / max(1, len(values) - 1)) * width

        strokes, dots, _head = self._plot(values, fired, partial, vmax, x_of)
        if move > 0.02:
            plan.strokes.extend(strokes)
            plan.dots.extend(dots)
        shown = values[min(fired, len(values) - 1)]
        frames_since = 99
        if fired:
            frames_since = int(round((t - times[fired - 1]) * self.timeline.fps))
        scale = _pop(frames_since)
        color = self.palette["text"]
        if 0 <= frames_since < 6 and fired:
            color = self.palette["down"] if values[fired - 1] % 2 == 0 else self.palette["up"]
        number = _center(format_int(shown), "mono", 220 * scale, 360, color, "primary", 0)
        home = _center("6", "mono", 320, 860, self.palette["text"], "primary", 0)
        number.x = _lerp(home.x, number.x, move)
        number.y = _lerp(home.y, number.y, move)
        plan.texts.append(number)
        plan.hud_step = fired
        plan.hud_value = shown
        if move > 0.85 and fired < len(values) - 1:
            plan.texts.append(_center("EVEN: / 2", "word", 36, 1360, self.palette["down"], "primary", 2))
            plan.texts.append(_center("ODD: x 3 + 1", "word", 36, 1410, self.palette["up"], "primary", 3))
        if fired >= len(values) - 1:
            plan.dots.append(Dot(x_of(len(values) - 1), self._chart_y(1, vmax), 26, self.palette["gold"], 4))
            plan.texts.append(_center("ENDS AT 1", "word", 62, 1340, self.palette["text"], "primary", 1))

    def _proof(self, plan: FramePlan, t: float, index: int) -> None:
        scene = self.timeline.scene("S3")
        fps = self.timeline.fps
        runs = []
        for run in scene.spec["runs"]:
            n = int(run["n"])
            start = scene.rel(float(run["offset"]), fps)
            count = len(self.traj[n]) - 1
            step = int(scene.spec["frames_per_step"]) / fps
            runs.append((n, start, count, step))
        vmax = 1.15 * max(max(self.traj[7]), max(self.traj[9]))
        width = CHART[1] - CHART[0]
        active = None
        started = [run for run in runs if t >= run[1]]
        for ordinal, (n, start, count, step) in enumerate(started):
            elapsed = t - start
            fired = min(count, int(elapsed / step + 1e-9))
            partial = 0.0 if fired >= count else (elapsed - fired * step) / step
            total = count

            def x_of(i: int, total=total) -> float:
                return CHART[0] + (i / max(1, total)) * width

            alpha = 0.35 if ordinal < len(started) - 1 else 1.0
            strokes, dots, _ = self._plot(self.traj[n], fired, partial, vmax, x_of, alpha=alpha)
            if t < scene.start + float(scene.spec["wipe_s"]):
                edge = CHART[0] + width * ((t - scene.start) / float(scene.spec["wipe_s"]))
                strokes = [stroke for stroke in strokes if stroke.pts[-1][0] <= edge]
                dots = [dot for dot in dots if dot.x <= edge]
            plan.strokes.extend(strokes)
            plan.dots.extend(dots)
            active = (n, fired, count, start, step)
        if active:
            n, fired, count, start, step = active
            shown = self.traj[n][min(fired, count)]
            frames_since = int(round((t - (start + max(0, fired - 1) * step)) * fps)) if fired else 99
            scale = _pop(frames_since)
            color = self.palette["text"]
            if fired and 0 <= frames_since < 6:
                color = self.palette["down"] if self.traj[n][fired - 1] % 2 == 0 else self.palette["up"]
            if fired >= count and t < start + count * step + 0.5:
                plan.texts.append(_center(f"{count} STEPS", "mono", 120, 360, self.palette["gold"], "primary", 0))
            else:
                plan.texts.append(_center(format_int(shown), "mono", 180 * scale, 360, color, "primary", 0))
            plan.hud_step = fired
            plan.hud_value = shown
        self._maybe_caption(plan, t)

    def _twist(self, plan: FramePlan, t: float) -> None:
        plan.texts.append(_center("NOW TRY", "word", 62, 1340, self.palette["text"], "primary", 0))

    def _ride(self, plan: FramePlan, t: float, index: int) -> None:
        from fc_sat.collatz_draw import Stroke

        scene = self.timeline.scene("S5")
        fps = self.timeline.fps
        values = self.traj[27]
        step = int(scene.spec["frames_per_step"]) / fps
        elapsed = max(0.0, t - scene.start)
        fired = min(len(values) - 1, int(elapsed / step + 1e-9))
        partial = 0.0 if fired >= len(values) - 1 else (elapsed - fired * step) / step
        shown_values = values[: fired + 1]
        vmax = max(shown_values) * 1.15
        width = CHART[1] - CHART[0]
        total = len(values) - 1

        def x_of(i: int) -> float:
            return CHART[0] + (i / total) * width

        spacing = width / total
        strokes, dots, _ = self._plot(values, fired, partial, vmax, x_of, node_r=min(11.0, max(2.5, spacing * 0.42)))
        plan.strokes.extend(strokes)
        plan.dots.extend(dots)
        # Faint log grid.
        for tick in (1, 10, 100, 1000, 10000):
            if tick <= vmax:
                y = self._chart_y(tick, vmax)
                plan.strokes.append(Stroke([(CHART[0], y), (CHART[1], y)], self.palette["muted"], 1, 0.35))
        frames_since = int(round((t - (scene.start + max(0, fired) * step)) * fps)) if fired else int(round(elapsed * fps))
        # The slam owns the first 9 frames.
        slam = int(round(elapsed * fps))
        if slam < 9:
            scale = _lerp(1.2, 1.0, ease_out_cubic(slam / 8 if slam else 0))
            plan.texts.append(_center("27", "mono", 360 * scale, 280, self.palette["text"], "primary", 0))
        else:
            scale = 1.15 if fired == 77 and 0 <= frames_since < 8 else _pop(frames_since if fired else 99)
            color = self.palette["text"]
            if fired and 0 <= frames_since < 6:
                color = self.palette["down"] if values[fired - 1] % 2 == 0 else self.palette["up"]
            if fired >= total:
                plan.texts.append(_center("111 STEPS", "mono", 120, 360, self.palette["gold"], "primary", 0))
            else:
                plan.texts.append(_center(format_int(values[fired]), "mono", 180 * scale, 340, color, "primary", 0))
            if fired and values[fired - 1] % 2 == 0:
                op, op_color = "/ 2", self.palette["down"]
            elif fired:
                op, op_color = "x 3 + 1", self.palette["up"]
            else:
                op, op_color = "", self.palette["muted"]
            if op and fired < total:
                plan.texts.append(_center(op, "mono", 52, 580, op_color, "primary", 3))
        plan.texts.append(_center(f"STEP {fired}", "mono", 48, 214, self.palette["text"], "primary", 1))
        if fired >= 36:
            y = self._chart_y(1000, max(vmax, 1000))
            plan.strokes.append(Stroke([(CHART[0], y), (CHART[1], y)], self.palette["gold"], 2, 1))
        if fired >= 77 and fired < total:
            plan.texts.append(_center(f"PEAK {format_int(max(values))}", "mono", 48, 1260, self.palette["gold"], "primary", 2))
        plan.hud_step = fired
        plan.hud_value = values[fired]
        self._maybe_caption(plan, t)

    def _funnel_scene(self, plan: FramePlan, t: float) -> None:
        scene = self.timeline.scene("S6")
        fps = self.timeline.fps
        relayout_end = scene.rel(float(scene.spec["relayout_s"]), fps)
        u = 0.0 if t <= scene.start else min(1.0, (t - scene.start) / max(1e-6, relayout_end - scene.start))
        u = ease_in_out_cubic(u)
        if t >= relayout_end:
            span = max(1e-6, scene.end - relayout_end)
            raw = int(round(1000 * ease_in_cubic(min(1.0, (t - relayout_end) / span))))
        else:
            raw = 0
        plan.funnel_count = raw
        self._hero_27(plan, u)
        counter_start = scene.rel(float(scene.spec["counter_offset"]), fps)
        if t >= counter_start:
            cue = ease_out_cubic(min(1.0, (t - counter_start) / max(1e-6, scene.end - counter_start)))
            shown = int(round(self.hundredths * cue))
            plan.texts.append(_center(self._counter(shown), "mono", 96, 200, self.palette["gold"], "primary", 0))
            plan.texts.append(_center("numbers checked", "word", 44, 300, self.palette["muted"], "muted", 1))
        plan.hud_value = self.bound if t >= scene.end - 1e-6 else None

    def _hero_27(self, plan: FramePlan, mix: float, alpha: float = 1.0) -> None:
        values = self.traj[27]
        total = len(values) - 1
        pts = []
        for index, value in enumerate(values):
            left = CHART[0] + (index / total) * 860.0
            right = funnel_x(total - index)
            x = _lerp(left, right, mix)
            pts.append((x, log_y(value, 330.0, 1300.0, 250504)))
        plan.strokes.append(Stroke(pts, self.palette["text"], 6, alpha))
        peak_at = values.index(max(values))
        plan.dots.append(Dot(pts[peak_at][0], pts[peak_at][1], 8, self.palette["gold"], 0, 1))

    def _resolve(self, plan: FramePlan, t: float) -> None:
        scene = self.timeline.scene("S7")
        stamp_at = scene.rel(float(scene.spec["stamp_offset"]), self.timeline.fps)
        plan.funnel_count = 1000
        self._hero_27(plan, 1.0)
        plan.texts.append(_center(self._counter(self.hundredths), "mono", 96, 220, self.palette["gold"], "primary", 1))
        if t >= stamp_at:
            plan.texts.append(_center("ALL END AT 1", "word", 96, 360, self.palette["text"], "primary", 0))

    def _punch(self, plan: FramePlan, t: float) -> None:
        scene = self.timeline.scene("S8")
        hit = scene.rel(float(scene.spec["hit_offset"]), self.timeline.fps)
        fade = float(scene.spec["fade_s"])
        plan.funnel_count = 1000
        faded = t >= scene.start + fade
        self._hero_27(plan, 1.0, 0.16 if faded else 1.0)
        plan.alpha_lines = 0.06 if faded else 1.0
        if t >= hit:
            plan.texts.append(_center("STILL UNPROVEN", "word", 110, 280, self.palette["gold"], "primary", 0))
            plan.texts.append(_center("Collatz conjecture", "word", 52, 420, self.palette["muted"], "muted", 1))
            if self.book.get("c_1937").verified:
                plan.texts.append(_center("since 1937", "word", 36, 490, self.palette["muted"], "muted", 2))

    def _bridge(self, plan: FramePlan, t: float) -> None:
        scene = self.timeline.scene("S9")
        fade = float(scene.spec["fade_s"])
        if t >= scene.start + fade:
            plan.texts.append(_center("Pick another number.", "word", 96, 860, self.palette["text"], "primary", 0))
            plan.funnel_count = 0
        else:
            plan.funnel_count = 1000
