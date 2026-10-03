"""Flat pictures for the paperfold short. Design space is 1080x1920.

The tower is 380 px wide. A 150 px tower is about 5 percent of the frame, under
the 12 percent hook rule, so the stack is wide enough to be the subject.
The sheet is 1000 px wide. At the opening pose the lifted half still leaves
more than 18 percent of the frame filled.
"""

from __future__ import annotations

import math
import time

import numpy as np

from fc_sat.easing import ease_in_out_cubic, ease_out_cubic
from fc_sat.paperfold_config import PaperConfig, load_config
from fc_sat.paperfold_copy import credit
from fc_sat.paperfold_draw import PaperCanvas, bgr, mix_hex, text_width
from fc_sat.paperfold_format import paper_km_label
from fc_sat.paperfold_math import (
    EARTH_RADIUS_M,
    MOON_M,
    PERSON_M,
    TOWER_PX,
    height_m,
    length_m,
    log_axis_x,
    milestone_visible,
)
from fc_sat.paperfold_math import MILESTONES
from fc_sat.paperfold_schedule import TRANSITION_END, TRANSITION_START, camera_at, fold_frame
from fc_sat.paperfold_text import Script, load_script, top_lines

GROUND_Y = 1300.0
TOWER_TOP = GROUND_Y - TOWER_PX
TOWER_W = 380.0
SHEET_W = 1000.0
SHEET_H = 620.0
SHEET_BOTTOM = 1260.0
SAFE_L = 130.0
SAFE_R = 950.0


def folds_done(frame: int) -> int:
    done = 0
    for n in range(1, 43):
        if fold_frame(n) <= frame:
            done = n
    return done


def fold_motion(frame: int) -> tuple[int, float, str]:
    """Fold in progress, 0..1, and 'flip' or 'squash'. Settled frames return (0, 0, 'hold')."""
    for n in range(1, 43):
        land = fold_frame(n)
        start = land - 12
        if start <= frame < land:
            local = frame - start
            if n == 1 and local <= 9:
                return n, 0.4 + 0.6 * ease_in_out_cubic(local / 9.0), "flip"
            if local <= 9:
                return n, ease_in_out_cubic(local / 9.0), "flip"
            return n, 1.0, "squash"
    return 0, 0.0, "hold"


def ghost_u(frame: int) -> float:
    """0..1 while a tower fold's copy is landing. 0 when no ghost is in flight."""
    for n in range(7, 43):
        land = fold_frame(n)
        if land - 6 <= frame <= land:
            if frame >= land:
                return 1.0
            return ease_out_cubic((frame - (land - 6)) / 6.0)
    return 0.0


def length_sub(n: int) -> str:
    """Paper needed, in the unit that still fits on one bottom line."""
    from fc_sat.paperfold_format import milky_way_label, rounded_au_times, times_mark
    from fc_sat.paperfold_math import au_multiple, format_sig, light_years

    if light_years(n) >= 1:
        if n >= 42:
            return milky_way_label()
        return f"{format_sig(light_years(n), 3)} LY"
    if au_multiple(n) >= 10:
        return f"PAPER NEEDED {rounded_au_times(n)}{times_mark()} SUN"
    return f"PAPER NEEDED {paper_km_label(n)}"


def axis_label(n: int) -> str:
    from fc_sat.paperfold_format import rounded_au_times, times_mark
    from fc_sat.paperfold_math import EARTH_CIRC_M, au_multiple, length_m

    if au_multiple(n) >= 10:
        return f"{rounded_au_times(n)}{times_mark()}"
    if length_m(n) >= EARTH_CIRC_M:
        return "EARTH"
    return paper_km_label(n)


def height_label(n: int) -> str:
    meters = height_m(max(0, n))
    if meters < 0.1:
        return f"{meters * 100:.2f} CM"
    if meters < 1:
        return f"{meters * 100:.1f} CM"
    if meters < 1000:
        return f"{meters:.1f} M"
    km = meters / 1000.0
    if km < 100:
        return f"{km:.1f} KM"
    return f"{int(round(km)):,} KM"


def sub_line(script: Script, frame: int, done: int) -> str:
    for sub in script.subs:
        if sub.start <= frame < sub.end:
            return sub.text
    if 672 <= frame < 1248 and done > 0:
        return length_sub(done)
    if frame >= TRANSITION_END and done > 0 and frame < 1248:
        return height_label(done)
    return ""


def _clamp(value: float, lo: float, hi: float) -> float:
    return lo if value < lo else hi if value > hi else value


class PaperRenderer:
    def __init__(self, width: int = 1080, height: int = 1920, hook: str = "A", config: PaperConfig | None = None, script: Script | None = None) -> None:
        self.width = width
        self.height = height
        self.hook = hook
        self.config = config or load_config()
        self.script = script or load_script()
        self.s = width / 1080.0
        self.top_size = 0.0
        self.side_label = ""
        rng = np.random.default_rng(self.config.seed)
        picks = rng.random((40, 2))
        self.stars = [(80.0 + float(p[0]) * 920.0, 560.0 + float(p[1]) * 680.0) for p in picks]
        self._credit = credit()

    def sx(self, value: float) -> float:
        return value * self.s

    def render(self, frame: int) -> np.ndarray:
        self.canvas = PaperCanvas(self.width, self.height)
        self.top_size = 0.0
        self.side_label = ""
        self.stamp_label = ""
        frame = int(frame)
        opening = frame == 0 or frame >= 1823
        if self.hook == "D" and (frame < 24 or frame >= 1823):
            self._tease()
        elif opening:
            self._opening(1.0, 1.0)
        elif frame >= 1800:
            u = (frame - 1800) / 23.0
            scale = ease_in_out_cubic(min(1.0, (frame - 1800) / 5.0))
            self._opening(u, 0.35 + 0.65 * scale)
        else:
            self._body(frame, 1.0)
        return bgr(self.canvas)

    def profile_ms(self) -> float:
        start = time.perf_counter()
        for frame in (84, 576, 1128):
            self.render(frame)
        return (time.perf_counter() - start) / 3.0 * 1000.0

    def _opening(self, text_alpha: float, sheet_scale: float) -> None:
        """Frame 0. The last frame calls this with alpha 1 and scale 1, so the loop matches."""
        cfg = self.config
        self.canvas.fill(cfg.background)
        self._accent(0)
        self._ground()
        self._sheet(1, 0.4, "flip", sheet_scale)
        self._moon_dot()
        self._top(self.script.hooks[self.hook], text_alpha)
        sheet = next(sub.text for sub in self.script.subs if sub.id == "sheet")
        self._bottom(0, sheet, text_alpha)

    def _tease(self) -> None:
        self._body(1128, 1.0, lines=self.script.hooks["A"], force_done=42)

    def _body(self, frame: int, text_alpha: float, lines: tuple[str, ...] | None = None, force_done: int | None = None) -> None:
        cfg = self.config
        done = folds_done(frame) if force_done is None else force_done
        space = self._space_mix(frame)
        self.canvas.fill(mix_hex(cfg.background, cfg.space, space))
        if space > 0.35:
            self._stars(space)
        self._accent(frame)
        view = camera_at(frame)
        if view == "tower" and done >= 36:
            self._earth(done)
        if view != "answer":
            self._ground()
        if view == "topdown":
            n, u, phase = fold_motion(frame)
            self._sheet(n or 1, u if n else 0.0, phase if n else "hold", 1.0)
            if frame < 72:
                self._moon_dot()
        elif view == "transition":
            self._transition(frame, done)
        elif view == "reality":
            self._reality()
        elif view == "outro":
            self._sheet(1, 0.0, "hold", 0.62)
        elif view == "answer":
            pass
        else:
            self._tower(frame, done)
            if 672 <= frame < 864:
                self._axis(frame, done)
        shown = lines if lines is not None else top_lines(self.script, frame, self.hook)
        if self.stamp_label and (not shown or shown == ("DOUBLING.",)):
            shown = (self.stamp_label,)
        if view == "answer":
            self._hero(shown, text_alpha, gold=frame < 1536)
            return
        self._top(shown, text_alpha)
        if view == "reality":
            self._center_line(self._credit, 44.0, 1488.0, "word", self.config.text, text_alpha)
        elif view != "outro":
            self._bottom(done, sub_line(self.script, frame, done), text_alpha)

    def _space_mix(self, frame: int) -> float:
        if frame < 576 or frame >= 1248:
            return 0.0
        if frame >= 588:
            return 1.0
        return ease_in_out_cubic((frame - 576) / 12.0)

    def _accent(self, frame: int) -> None:
        color = self.config.bar_colors[(max(0, frame) // 96) % 8]
        self.canvas.rect(0, 0, self.sx(28), self.height, color)

    def _ground(self) -> None:
        y = self.sx(GROUND_Y)
        self.canvas.rect(0, y, self.width, self.height - y, self.config.panel)
        self.canvas.rect(0, y - self.sx(16), self.width, self.sx(16), self.config.seam)

    def _stars(self, alpha: float) -> None:
        for x, y in self.stars:
            self.canvas.circle(self.sx(x), self.sx(y), self.sx(8), self.config.paper, alpha)

    def _moon_dot(self) -> None:
        cfg = self.config
        x, y, r = 980.0, 188.0, 26.0
        self.canvas.circle(self.sx(x), self.sx(y), self.sx(r), cfg.paper)
        label = "384,400 KM"
        self.side_label = label
        size = 28.0 * self.s
        width = text_width("mono", size, label)
        self.canvas.text(label, self.sx(x) - width / 2.0, self.sx(y + 36), "mono", size, cfg.muted)

    def _sheet(self, n: int, u: float, phase: str, scale: float) -> None:
        cfg = self.config
        width = SHEET_W * scale
        height = SHEET_H * scale
        if phase == "squash":
            height *= 0.94
        bottom = SHEET_BOTTOM
        top = bottom - height
        stripes = min(6, max(0, folds_done_for_sheet(n, u)))
        if phase != "flip" or u <= 0.02 or u >= 0.98:
            left = 540.0 - width / 2.0
            self.canvas.rect(self.sx(left), self.sx(top), self.sx(width), self.sx(height), cfg.paper)
            self._stripes(left, bottom, width, stripes)
            return
        span = math.cos(u * math.pi)
        lift = (1.0 - abs(span)) * 110.0
        half = width / 2.0
        moving = abs(span) * half
        if n % 2 == 1:
            foot = half + (moving if span > 0 else 0.0)
            zoom = width / max(foot, 1.0)
            still_w = half * zoom
            flap_w = max(18.0, moving * zoom)
            if span > 0:
                left = 540.0 - width / 2.0
                hinge = left + still_w
                self.canvas.rect(self.sx(left), self.sx(top), self.sx(still_w), self.sx(height), cfg.paper)
                outer = hinge + flap_w
                flap_top = top - lift
                self.canvas.polygon(
                    [
                        (self.sx(hinge), self.sx(top)),
                        (self.sx(outer), self.sx(flap_top)),
                        (self.sx(outer), self.sx(flap_top + height - lift)),
                        (self.sx(hinge), self.sx(top + height)),
                    ],
                    cfg.paper,
                )
                self.canvas.rect(self.sx(hinge - 10), self.sx(top), self.sx(20), self.sx(height), cfg.seam)
            else:
                left = 540.0 - still_w / 2.0
                self.canvas.rect(self.sx(left), self.sx(top), self.sx(still_w), self.sx(height), cfg.seam)
                self.canvas.rect(self.sx(left), self.sx(top), self.sx(still_w), self.sx(22), cfg.paper)
            self._stripes(540.0 - width / 2.0, bottom, width, stripes)
        else:
            flap_h = abs(span) * height / 2.0
            still_h = height / 2.0
            left = 540.0 - width / 2.0
            if span > 0:
                self.canvas.rect(self.sx(left), self.sx(bottom - still_h), self.sx(width), self.sx(still_h), cfg.paper)
                flap_top = bottom - still_h - max(18.0, flap_h) - lift
                self.canvas.rect(self.sx(left), self.sx(flap_top), self.sx(width), self.sx(max(18.0, flap_h)), cfg.paper)
                self.canvas.rect(self.sx(left), self.sx(bottom - still_h - 10), self.sx(width), self.sx(20), cfg.seam)
            else:
                self.canvas.rect(self.sx(left), self.sx(top), self.sx(width), self.sx(height), cfg.seam)
                self.canvas.rect(self.sx(left), self.sx(top), self.sx(22), self.sx(height), cfg.paper)
            self._stripes(left, bottom, width, stripes)

    def _stripes(self, left: float, bottom: float, width: float, stripes: int) -> None:
        cfg = self.config
        for index in range(stripes):
            color = cfg.seam if index % 2 == 0 else cfg.paper
            y = bottom - (index + 1) * 16.0
            self.canvas.rect(self.sx(left), self.sx(y), self.sx(width), self.sx(16), color)

    def _transition(self, frame: int, done: int) -> None:
        t = ease_in_out_cubic((frame - TRANSITION_START) / (TRANSITION_END - TRANSITION_START - 1))
        sheet = (540.0 - SHEET_W / 2.0, SHEET_BOTTOM - SHEET_H, SHEET_W, SHEET_H)
        tower = (540.0 - TOWER_W / 2.0, TOWER_TOP, TOWER_W, TOWER_PX)
        left = sheet[0] + (tower[0] - sheet[0]) * t
        top = sheet[1] + (tower[1] - sheet[1]) * t
        width = sheet[2] + (tower[2] - sheet[2]) * t
        height = sheet[3] + (tower[3] - sheet[3]) * t
        pop = 1.0 + 0.06 * math.sin(t * math.pi)
        height *= pop
        top -= (height - (sheet[3] + (tower[3] - sheet[3]) * t)) * 0.5
        self.canvas.rect(self.sx(left), self.sx(top), self.sx(width), self.sx(height), self.config.paper)
        if frame >= TRANSITION_END - 6:
            self._ghost_band(left, top, width, height, ghost_u(frame))

    def _tower(self, frame: int, done: int) -> None:
        if done <= 0:
            done = 1
        left = 540.0 - TOWER_W / 2.0
        self.canvas.rect(self.sx(left), self.sx(TOWER_TOP), self.sx(TOWER_W), self.sx(TOWER_PX), self.config.paper)
        band = ghost_u(frame)
        if band > 0:
            self._ghost_band(left, TOWER_TOP, TOWER_W, TOWER_PX, band)
        self._milestone(frame, done, left)

    def _ghost_band(self, left: float, top: float, width: float, height: float, u: float) -> None:
        if u <= 0:
            return
        seam_y = top + u * (height / 2.0)
        self.canvas.rect(self.sx(left), self.sx(seam_y), self.sx(width), self.sx(18), self.config.gold)

    def _earth(self, done: int) -> None:
        radius = EARTH_RADIUS_M * (TOWER_PX / height_m(done))
        if radius < 16:
            return
        self.canvas.circle(self.sx(540), self.sx(GROUND_Y), self.sx(radius), self.config.seam)

    def _milestone(self, frame: int, done: int, tower_left: float) -> None:
        tower_m = height_m(done)
        scale = TOWER_PX / tower_m
        upcoming = None
        passed = None
        for name, dist, label in MILESTONES:
            fold = _pass_frame(name)
            if fold <= frame < fold + 24:
                passed = (name, dist, label, fold)
            elif dist > tower_m and upcoming is None and milestone_visible(dist, tower_m):
                upcoming = (name, dist, label)
        if done >= 36:
            self._moon_disc(done, scale)
        target = passed or upcoming
        if target is None or target[0] == "moon":
            if passed is not None and passed[0] == "moon":
                self.stamp_label = "THE MOON"
            return
        name, dist, label = target[0], target[1], target[2]
        if not milestone_visible(dist, tower_m) and passed is None:
            return
        icon_h = _clamp(dist * scale, 120.0, 620.0)
        line_y = max(560.0, GROUND_Y - icon_h)
        icon_h = GROUND_Y - line_y
        if 800.0 - (tower_left + TOWER_W) > 80.0:
            self._dashes(tower_left + TOWER_W + 12.0, 800.0, line_y, self.config.gold if passed else self.config.muted)
        self._icon(name, 840.0, GROUND_Y, icon_h)
        if passed is not None:
            self.stamp_label = label

    def _moon_disc(self, done: int, scale: float) -> None:
        y = GROUND_Y - _clamp(MOON_M * scale, 80.0, 700.0)
        grow = _clamp((done - 36) / 6.0, 0.0, 1.0)
        radius = 56.0 + grow * 64.0
        if y - radius < 540:
            y = 540 + radius
        self.canvas.circle(self.sx(860), self.sx(y), self.sx(radius), self.config.paper)

    def _dashes(self, x0: float, x1: float, y: float, color: str) -> None:
        x = x0
        while x < x1:
            self.canvas.rect(self.sx(x), self.sx(y - 7), self.sx(18), self.sx(14), color)
            x += 32.0

    def _icon(self, name: str, x: float, y: float, size: float) -> None:
        cfg = self.config
        size = max(28.0, size)
        if name == "person":
            self.canvas.rect(self.sx(x - size * 0.16), self.sx(y - size * 0.62), self.sx(size * 0.32), self.sx(size * 0.62), cfg.paper)
            self.canvas.circle(self.sx(x), self.sx(y - size * 0.78), self.sx(size * 0.16), cfg.paper)
        elif name == "burj":
            self.canvas.rect(self.sx(x - size * 0.16), self.sx(y - size * 0.7), self.sx(size * 0.32), self.sx(size * 0.7), cfg.paper)
            self.canvas.polygon(
                [
                    (self.sx(x - size * 0.16), self.sx(y - size * 0.7)),
                    (self.sx(x + size * 0.16), self.sx(y - size * 0.7)),
                    (self.sx(x), self.sx(y - size)),
                ],
                cfg.paper,
            )
        elif name == "everest":
            self.canvas.polygon(
                [
                    (self.sx(x - size * 0.32), self.sx(y)),
                    (self.sx(x), self.sx(y - size)),
                    (self.sx(x + size * 0.32), self.sx(y)),
                ],
                cfg.paper,
            )
            self.canvas.polygon(
                [
                    (self.sx(x - size * 0.16), self.sx(y - size * 0.72)),
                    (self.sx(x), self.sx(y - size)),
                    (self.sx(x + size * 0.16), self.sx(y - size * 0.72)),
                ],
                cfg.gold,
            )
        else:
            self.canvas.circle(self.sx(x), self.sx(y - size * 0.4), self.sx(size * 0.4), cfg.paper)

    def _axis(self, frame: int, done: int) -> None:
        cfg = self.config
        y = 1050.0
        self.canvas.rect(self.sx(130), self.sx(y - 8), self.sx(820), self.sx(16), cfg.muted)
        for meters in (1_000.0, 40_075e3, 149_597_870_700.0, 9.4607e15, 100_000 * 9.4607e15):
            x = log_axis_x(meters)
            self.canvas.rect(self.sx(x - 6), self.sx(y - 22), self.sx(12), self.sx(28), cfg.text)
        marker_n = 12 if frame < 744 else 20 if frame < 792 else 30
        if frame < 696:
            marker_n = 12
        x = log_axis_x(length_m(marker_n))
        self.canvas.rect(self.sx(x - 14), self.sx(y - 36), self.sx(28), self.sx(56), cfg.gold)
        label = axis_label(marker_n)
        self.side_label = label
        size = 40.0 * self.s
        width = text_width("mono", size, label)
        self.canvas.text(label, self.sx(x) - width / 2.0, self.sx(520), "mono", size, cfg.gold)

    def _reality(self) -> None:
        cfg = self.config
        person_h = 560.0
        stack_h = person_h * (height_m(12) / PERSON_M)
        ground = GROUND_Y
        self.canvas.rect(self.sx(300), self.sx(ground - stack_h), self.sx(130), self.sx(stack_h), cfg.paper)
        self._icon("person", 720.0, ground, person_h)

    def _hero(self, lines: tuple[str, ...] | list[str], alpha: float, gold: bool) -> None:
        if not lines or alpha <= 0:
            return
        cursor = 760.0
        for line in lines:
            digits = any(ch.isdigit() for ch in line)
            size = 420.0 if digits else 100.0
            kind = "mono" if digits else "word"
            color = self.config.gold if gold and digits else self.config.text
            self.top_size = max(self.top_size, size if not digits else 96.0)
            width = text_width(kind, size * self.s, line)
            self.canvas.text(line, self.width / 2.0 - width / 2.0, self.sx(cursor), kind, size * self.s, color, alpha)
            cursor += size * 0.85

    def _top(self, lines: tuple[str, ...] | list[str], alpha: float) -> None:
        if not lines or alpha <= 0:
            return
        lines = tuple(lines)
        size = 110.0
        while size > 72 and max(text_width("word", size, line) for line in lines) > (SAFE_R - SAFE_L):
            size -= 2
        self.top_size = size
        cursor = 230.0
        for line in lines:
            kind = "mono" if any(ch.isdigit() for ch in line) else "word"
            width = text_width(kind, size * self.s, line)
            x = self.width / 2.0 - width / 2.0
            self.canvas.text(line, x, self.sx(cursor), kind, size * self.s, self.config.text, alpha)
            cursor += size + 8

    def _center_line(self, text: str, size: float, y: float, kind: str, color: str, alpha: float) -> None:
        if not text or alpha <= 0:
            return
        px = size * self.s
        width = text_width(kind, px, text)
        self.canvas.text(text, self.width / 2.0 - width / 2.0, self.sx(y), kind, px, color, alpha)

    def _bottom(self, done: int, sub: str, alpha: float) -> None:
        cfg = self.config
        if alpha <= 0:
            return
        label = "FOLDS"
        label_size = 36.0 * self.s
        label_w = text_width("word", label_size, label)
        self.canvas.text(label, self.width / 2.0 - label_w / 2.0, self.sx(1360), "word", label_size, cfg.muted, alpha)
        if done > 0 or sub:
            number = str(done)
            num_size = 130.0 * self.s
            num_w = text_width("mono", num_size, number)
            self.canvas.text(number, self.width / 2.0 - num_w / 2.0, self.sx(1405), "mono", num_size, cfg.text, alpha)
        if sub:
            sub_size = 34.0 * self.s
            kind = "mono" if any(ch.isdigit() for ch in sub) else "word"
            sub_w = text_width(kind, sub_size, sub)
            self.canvas.text(sub, self.width / 2.0 - sub_w / 2.0, self.sx(1545), kind, sub_size, cfg.muted, alpha)


def folds_done_for_sheet(n: int, u: float) -> int:
    """Stripes already on the table. The fold in flight has not landed."""
    if n <= 1:
        return 0
    return n - 1


def _pass_frame(name: str) -> int:
    from fc_sat.paperfold_math import milestone_fold

    return fold_frame(milestone_fold(name))


def paper_fraction(image_bgr: np.ndarray) -> float:
    """Share of the frame filled by paper, the pale back of a fold, or a milestone icon."""
    red = image_bgr[:, :, 2].astype(np.int16)
    green = image_bgr[:, :, 1].astype(np.int16)
    blue = image_bgr[:, :, 0].astype(np.int16)
    return float(((red + green + blue) > 500).mean())


def tower_white_span(image_bgr: np.ndarray, scale: float) -> int:
    """Pixel span of the white stack inside the tower's vertical band."""
    x = int(round(540 * scale))
    y0 = int(round(560 * scale))
    y1 = int(round(1340 * scale))
    column = image_bgr[y0:y1, x]
    white = (column[:, 2] > 230) & (column[:, 1] > 230) & (column[:, 0] > 230)
    idx = np.flatnonzero(white)
    if len(idx) == 0:
        return 0
    return int(idx[-1] - idx[0] + 1)
