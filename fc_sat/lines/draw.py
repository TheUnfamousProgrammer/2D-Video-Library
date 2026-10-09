"""Picture for 'Why the other line is always faster'. Line-art lanes, cars and counters; no captions."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import skia

from fc_sat.lines.cues import FPS, RHO, SERVERS, Cues, hops, mm1_wait, mmc_wait, side_serves, ten_finish, trial_time
from fc_sat.rule37.draw import Painter as BasePainter
from fc_sat.rule37.draw import back_out, clamp, ease_in_out, ease_out, glow, lin, mix, paint

ROOT = Path(__file__).resolve().parents[2]
W, H = 1080, 1920

BG = (0x0C, 0x17, 0x1E)
PANEL = (0x13, 0x24, 0x2D)
INK = (0xEE, 0xF6, 0xF2)
DIM = (0x4F, 0x66, 0x6E)
LIME = (0xC6, 0xF2, 0x5B)
CORAL = (0xFF, 0x6B, 0x5B)
AMBER = (0xFF, 0xC2, 0x4B)
SKY = (0x5B, 0xC8, 0xFF)

LANES = (270.0, 540.0, 810.0)
REG_Y = 520.0
SLOT0 = 690.0
SLOT = 124.0
YOU_SLOT = 3


def clock(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 60}:{s % 60:02d}"


class Painter(BasePainter):
    def __init__(self, cues: Cues, captions=None) -> None:  # noqa: super().__init__ needs the 37% cues
        self.cues = cues
        self.captions = captions
        self.surface = skia.Surface(W, H)
        self.mono = skia.Typeface.MakeFromFile(str(ROOT / "assets" / "fonts" / "JetBrainsMono-ExtraBold.ttf"))
        self.bg = self._background()
        self.wait_sep = mm1_wait(RHO)
        self.wait_one = mmc_wait(SERVERS, RHO)
        self.serves = side_serves(cues)
        rng = np.random.default_rng(11)
        self.rain = [(rng.uniform(-90, 90), rng.uniform(0, 1)) for _ in range(14)]

    def _background(self) -> skia.Image:
        s = skia.Surface(W, H)
        c = s.getCanvas()
        c.clear(skia.Color(*BG))
        g = paint(INK, 0.04, 1.5)
        for x in range(0, W + 1, 60):
            c.drawLine(x, 0, x, H, g)
        for y in range(0, H + 1, 60):
            c.drawLine(0, y, W, y, g)
        p = skia.Paint(AntiAlias=True)
        p.setShader(skia.GradientShader.MakeRadial(
            skia.Point(W / 2, H * 0.42), H * 0.75,
            [skia.Color(0x1A, 0x3A, 0x44, 150), skia.Color(0, 0, 0, 0), skia.Color(0, 0, 0, 200)], [0.0, 0.55, 1.0]))
        c.drawRect(skia.Rect(0, 0, W, H), p)
        return s.makeImageSnapshot()

    # --- props ----------------------------------------------------------------------------------

    def register(self, c, x, y, scale=1.0, alpha=1.0, light: float = 0.0, light_col=CORAL, done: float = 0.0) -> None:
        w, h = 170 * scale, 86 * scale
        c.drawRoundRect(skia.Rect(x - w / 2, y - h / 2, x + w / 2, y + h / 2), 14 * scale, 14 * scale, paint(PANEL, alpha))
        c.drawRoundRect(skia.Rect(x - w / 2, y - h / 2, x + w / 2, y + h / 2), 14 * scale, 14 * scale, paint(INK, 0.8 * alpha, 4 * scale))
        c.drawRoundRect(skia.Rect(x - w * 0.36, y - h * 0.26, x - w * 0.02, y + h * 0.18), 6 * scale, 6 * scale,
                        paint(LIME if done > 0 else SKY, (0.25 + 0.5 * done) * alpha))
        c.drawLine(x + w * 0.1, y, x + w * 0.38, y, paint(INK, 0.5 * alpha, 5 * scale))
        c.drawLine(x, y - h / 2, x, y - h / 2 - 46 * scale, paint(INK, 0.6 * alpha, 4 * scale))
        ly = y - h / 2 - 62 * scale
        if light > 0:
            c.drawCircle(x, ly, 46 * scale * light, glow(light_col, 0.55 * alpha * light, 20 * scale))
        c.drawCircle(x, ly, 16 * scale, paint(light_col if light > 0 else DIM, alpha))
        c.drawCircle(x, ly, 16 * scale, paint(INK, 0.6 * alpha, 3 * scale))

    def cloud(self, c, x, y, s, alpha, t, bolt: float = 0.0) -> None:
        if alpha <= 0:
            return
        for dx, dy, r in ((-0.35, 0.05, 0.28), (0.0, -0.12, 0.36), (0.36, 0.04, 0.27), (0.0, 0.12, 0.3)):
            c.drawCircle(x + dx * s, y + dy * s, r * s, paint(DIM, alpha))
        for dx, ph in self.rain:
            k = (t * 2.2 + ph) % 1.0
            yy = y + 0.25 * s + k * 0.9 * s
            c.drawLine(x + dx * s / 180, yy, x + dx * s / 180 - 6, yy + 22, paint(SKY, alpha * (1 - k), 4))
        if bolt > 0:
            p = skia.Path()
            pts = [(0.05, 0.1), (-0.12, 0.55), (0.04, 0.5), (-0.08, 0.95), (0.2, 0.38), (0.03, 0.43), (0.14, 0.1)]
            p.moveTo(x + pts[0][0] * s, y + pts[0][1] * s)
            for px, py in pts[1:]:
                p.lineTo(x + px * s, y + py * s)
            p.close()
            c.drawPath(p, glow(AMBER, 0.6 * bolt * alpha, 18))
            c.drawPath(p, paint(AMBER, bolt * alpha))

    def car(self, c, x, y, col, alpha=1.0, s=1.0) -> None:
        w, h = 108 * s, 190 * s
        r = skia.Rect(x - w / 2, y - h / 2, x + w / 2, y + h / 2)
        c.drawRoundRect(r, 30 * s, 30 * s, paint(BG, alpha))
        c.drawRoundRect(r, 30 * s, 30 * s, paint(col, alpha, 6 * s))
        c.drawRoundRect(skia.Rect(x - w * 0.34, y - h * 0.3, x + w * 0.34, y - h * 0.1), 10 * s, 10 * s, paint(col, alpha, 4 * s))
        c.drawRoundRect(skia.Rect(x - w * 0.34, y + h * 0.18, x + w * 0.34, y + h * 0.32), 8 * s, 8 * s, paint(col, alpha * 0.7, 4 * s))
        for sx in (-1, 1):
            for sy in (-1, 1):
                c.drawRoundRect(skia.Rect(x + sx * w / 2 - 7 * s, y + sy * h * 0.3 - 16 * s, x + sx * w / 2 + 7 * s,
                                          y + sy * h * 0.3 + 16 * s), 4 * s, 4 * s, paint(col, alpha))

    def you_tag(self, c, x, y, alpha=1.0, side: bool = False) -> None:
        if alpha <= 0:
            return
        if side:  # arrow + label to the right of the figure, clear of the queue
            tri = skia.Path()
            tri.moveTo(x, y)
            tri.lineTo(x + 18, y - 12)
            tri.lineTo(x + 18, y + 12)
            tri.close()
            c.drawPath(tri, paint(LIME, alpha))
            self.text(c, "YOU", x + 24, y + 12, 34, LIME, alpha, "left", spacing=2)
            return
        self.text(c, "YOU", x, y, 34, LIME, alpha, "center", spacing=3)
        tri = skia.Path()
        tri.moveTo(x - 12, y + 12)
        tri.lineTo(x + 12, y + 12)
        tri.lineTo(x, y + 26)
        tri.close()
        c.drawPath(tri, paint(LIME, alpha))

    def burst(self, c, x, y, d: float, syms, alpha=1.0) -> None:
        if not 0 <= d < 1.0:
            return
        for k, sym in enumerate(syms):
            ang = -math.pi / 2 + (k - (len(syms) - 1) / 2) * 0.7
            r = 40 + 170 * ease_out(clamp(d / 0.6))
            self.text(c, sym, x + r * math.cos(ang), y + r * math.sin(ang), 58, (LIME, AMBER, SKY)[k % 3],
                      alpha * (1 - clamp((d - 0.5) / 0.5)), "center")

    # --- frame ---------------------------------------------------------------------------------

    def render(self, frame: int) -> np.ndarray:
        t = frame / FPS
        cu = self.cues.t
        c = self.surface.getCanvas()
        c.drawImage(self.bg, 0, 0)
        sx, sy = self._shake(t)
        c.save()
        c.translate(sx, sy)
        zoom = self._zoom(t)
        if zoom != 1.0:
            fx, fy = LANES[1], SLOT0 + SLOT * YOU_SLOT - 120
            c.translate(fx, fy)
            c.scale(zoom, zoom)
            c.translate(-fx, -fy)
        scenes = [
            (0.0, cu["ten"], self.s_hook),
            (cu["ten"] - 0.2, cu["traffic"] + 0.2, self.s_ten),
            (cu["traffic"] - 0.15, cu["fix"] + 0.2, self.s_road),
            (cu["fix"], cu["so3"] + 0.3, self.s_snake),
            (cu["so3"], self.cues.end + 1, self.s_outro),
        ]
        for a, b, fn in scenes:
            if a - 0.001 <= t < b:
                fn(c, t)
        c.restore()
        self.counters(c, t)
        if self.captions is not None:
            self.captions.draw(c, t)
        img = self.surface.makeImageSnapshot().toarray(colorType=skia.kBGRA_8888_ColorType)
        return np.ascontiguousarray(img[:, :, :3])

    def _zoom(self, t: float) -> float:
        """Push in on you at the open; the outro pushes back in so the last frame matches frame 0."""
        z0 = 1.14
        if t < 1.8:
            return z0 + (1.0 - z0) * ease_in_out(t / 1.8)
        end = self.cues.end
        if t > end - 1.0:
            return 1.0 + (z0 - 1.0) * ease_in_out(lin(t, end - 1.0, end - 1.0 / FPS))
        return 1.0

    def _shake(self, t: float) -> tuple[float, float]:
        cu = self.cues.t
        out = [0.0, 0.0]
        for start, amp in ((cu["targeting"], 12.0), (cu["one1"], 14.0), (cu["traffic"] - 0.1, 16.0),
                           (cu["slower"], 12.0), (cu["half"], 12.0)):
            d = t - start
            if 0 <= d < 0.35:
                k = amp * (1 - d / 0.35) ** 2
                out[0] += k * math.sin(d * 97)
                out[1] += k * math.cos(d * 71)
        return out[0], out[1]

    # --- counters -------------------------------------------------------------------------------

    def counters(self, c, t: float) -> None:
        cu, cues = self.cues.t, self.cues
        end = cues.end
        if t < cu["three1"] or t >= end - 0.5:
            hook_t = t if t < cu["three1"] else 0.0
            a = (1 - lin(t, cu["three1"] - 0.2, cu["three1"])) if t < cu["three1"] else lin(t, end - 0.5, end - 0.3)
            self.counter(c, "left", "YOUR WAIT", clock(hook_t * 28), a, CORAL if hook_t > 2 else INK)
            served = sum(1 for s, _ in self.serves if hook_t >= s)
            self.counter(c, "right", "THEM, SERVED", f"{served}", a, INK)
        elif t < cu["ten"]:
            if t >= cu["two"] - 0.1:
                done = [w for k, w in enumerate(cues.trials) if t >= trial_time(cues, k)]
                if done:
                    pct = round(100 * sum(1 for w in done if w != 1) / len(done))
                    a = 1 - lin(t, cu["ten"] - 0.25, cu["ten"])
                    self.counter(c, "right", "OTHER LINE WINS", f"{pct}%", a, CORAL)
        elif t < cu["traffic"]:
            v = round(90 * ease_out(lin(t, cu["lose"], cu["ninety"] + 0.25)))
            a = lin(t, cu["lose"] - 0.2, cu["lose"]) * (1 - lin(t, cu["traffic"] - 0.25, cu["traffic"]))
            self.counter(c, "left", "YOU LOSE", f"{v}%", a, CORAL)
        elif t < cu["heres"]:
            n = sum(1 for k in range(10) if t >= cu["seventy"] + 0.08 * k and k < 7)
            a = lin(t, cu["seventy"], cu["seventy"] + 0.15) * (1 - lin(t, cu["heres"] - 0.2, cu["heres"]))
            self.counter(c, "right", "SAID FASTER", f"{n * 10}%", a, CORAL)
        elif t < cu["fix"]:
            a_out = 1 - lin(t, cu["so2"] - 0.1, cu["so2"] + 0.2)
            passed = 47 * lin(t, cu["cars"] - 0.2, cu["when3"] - 0.1)
            passing = 35 * lin(t, cu["blow"] - 0.2, cu["seconds"] + 0.5)
            self.counter(c, "left", "GETTING PASSED", f"{int(passed)}s", lin(t, cu["slow"], cu["slow"] + 0.2) * a_out, CORAL)
            self.counter(c, "right", "PASSING", f"{int(passing)}s", lin(t, cu["when3"], cu["when3"] + 0.2) * a_out, LIME)
        elif t < cu["so3"]:
            k = ease_in_out(lin(t, cu["cut"], cu["half"] + 0.1))
            wait = (self.wait_sep + (self.wait_one - self.wait_sep) * k) * 60
            a = lin(t, cu["busy"], cu["busy"] + 0.2) * (1 - lin(t, cu["so3"] - 0.2, cu["so3"]))
            pop = lin(t, cu["half"], cu["half"] + 0.3) if cu["half"] <= t < cu["half"] + 0.3 else 0
            self.counter(c, "left", "ONE LINE", clock(wait), a, LIME if k > 0.5 else INK, pop)
            self.counter(c, "right", "SEPARATE LINES", clock(self.wait_sep * 60), a * 0.9, CORAL)

    # --- scenes ---------------------------------------------------------------------------------

    def _lanes(self, c, t: float, alpha: float, you_lane: float = 1.0, you_dy: float = 0.0, hook_t: float | None = None) -> None:
        """Three checkout lanes. Side lanes advance on their serve times; the middle one is stuck."""
        ht = t if hook_t is None else hook_t
        blink = 0.5 + 0.5 * math.cos(ht * 9.0)
        for li, x in enumerate(LANES):
            stuck = li == 1
            c.drawLine(x - 96, REG_Y + 60, x - 96, SLOT0 + SLOT * 4.6, paint(INK, 0.12 * alpha, 3))
            c.drawLine(x + 96, REG_Y + 60, x + 96, SLOT0 + SLOT * 4.6, paint(INK, 0.12 * alpha, 3))
            self.register(c, x, REG_Y, 1.0, alpha, light=blink if stuck else 0.0, light_col=CORAL,
                          done=0.0 if stuck else 1.0)
            if stuck:
                for k in range(5):
                    if k == YOU_SLOT:
                        continue
                    y = SLOT0 + SLOT * k
                    self.person(c, x, y, 104, INK, alpha, accessory=(k * 2 + 1) % 5)
                # front customer's mountain of shopping
                for b in range(4):
                    c.drawRoundRect(skia.Rect(x + 52, SLOT0 - 20 - b * 26, x + 92, SLOT0 + 2 - b * 26), 5, 5, paint(AMBER, 0.85 * alpha, 3))
                continue
            done = [s for s, lane in self.serves if lane == li and ht >= s]
            n = len(done)
            p = ease_in_out(clamp((ht - done[-1]) / 0.35)) if done else 1.0
            for j in range(n - 1, n + 5):
                if j < 0:
                    continue
                slot = j - n + (1 - p)
                if j == n - 1:  # leaving past the register
                    y = SLOT0 - (SLOT0 - REG_Y + 120) * p
                    a = alpha * (1 - p)
                else:
                    y = SLOT0 + SLOT * slot
                    a = alpha * clamp(5.0 - slot)
                if a > 0:
                    self.person(c, x, y, 104, INK, a, accessory=(j * 3 + li) % 5)
        # you
        yx = LANES[0] + (LANES[2] - LANES[0]) * you_lane / 2
        yy = SLOT0 + SLOT * YOU_SLOT + you_dy
        self.person(c, yx, yy, 104, LIME, alpha, accessory=0, fill=None)
        self.you_tag(c, yx + 50, yy - 10, alpha, side=True)

    def s_hook(self, c, t: float) -> None:
        cu = self.cues.t
        a_lanes = 1 - 0.8 * lin(t, cu["three1"], cu["three1"] + 0.3)
        a_lanes *= 1 - lin(t, cu["one1"] - 0.3, cu["one1"])
        if a_lanes > 0:
            self._lanes(c, t, a_lanes, hook_t=min(t, cu["three1"]))
        # storm cloud over you
        yx, yy = LANES[1], SLOT0 + SLOT * YOU_SLOT
        ca = ease_out(lin(t, cu["universe"] - 0.1, cu["universe"] + 0.3)) * (1 - lin(t, cu["math"], cu["math"] + 0.15))
        bolt = 1.0 if cu["targeting"] <= t < cu["targeting"] + 0.12 or cu["targeting"] + 0.2 <= t < cu["targeting"] + 0.32 else 0.0
        bolt = max(bolt, 0.6 * (1 - lin(t, cu["targeting"] + 0.32, cu["targeting"] + 0.9)) if t >= cu["targeting"] + 0.32 else 0)
        self.cloud(c, yx, yy - 250, 170 * (0.6 + 0.4 * ca), ca, t, bolt if t >= cu["targeting"] else 0.0)
        self.burst(c, yx, yy - 250, t - cu["math"], ["1/3", "%", "÷", "="])
        # odds section
        if t >= cu["three1"] - 0.05:
            self._odds(c, t)

    def _odds(self, c, t: float) -> None:
        cu, cues = self.cues.t, self.cues
        out = 1 - lin(t, cu["ten"] - 0.25, cu["ten"])
        # question mark over you
        q = back_out(lin(t, cu["fastest"], cu["fastest"] + 0.25)) * (1 - lin(t, cu["one1"] - 0.2, cu["one1"]))
        if q > 0:
            self.text(c, "?", LANES[1], SLOT0 + SLOT * YOU_SLOT - 150, 150 * q, AMBER, q, "center")
        # 1 in 3 donut
        dk = back_out(lin(t, cu["one1"] - 0.05, cu["one1"] + 0.25), 1.8)
        d_out = ease_in_out(lin(t, cu["two"] - 0.2, cu["two"] + 0.2))
        if dk > 0 and d_out < 1:
            cx, cy = 540, 860 - 380 * d_out
            r = 230 * dk * (1 - 0.6 * d_out)
            rect = skia.Rect(cx - r, cy - r, cx + r, cy + r)
            for k in range(3):
                col = LIME if k == 0 else CORAL
                arc = skia.Path()
                arc.addArc(rect, -90 + k * 120 + 3, 114)
                c.drawPath(arc, paint(col, out * (1 - d_out), 46 * dk * (1 - 0.6 * d_out), skia.Paint.kButt_Cap))
            self.text(c, "1/3", cx, cy + 48 * dk * (1 - 0.6 * d_out), 140 * dk * (1 - 0.6 * d_out), LIME, out * (1 - d_out), "center")
        # 30 race tiles
        if t >= cu["two"] - 0.15:
            for k, w in enumerate(cues.trials):
                tk = trial_time(cues, k)
                pk = back_out(lin(t, tk, tk + 0.18), 2.0)
                if pk <= 0:
                    continue
                col_i, row = k % 5, k // 5
                cx = 160 + col_i * 190
                cy = 560 + row * 112
                tw, th = 160 * pk, 92 * pk
                edge = LIME if w == 1 else CORAL
                c.drawRoundRect(skia.Rect(cx - tw / 2, cy - th / 2, cx + tw / 2, cy + th / 2), 12, 12, paint(PANEL, out))
                c.drawRoundRect(skia.Rect(cx - tw / 2, cy - th / 2, cx + tw / 2, cy + th / 2), 12, 12, paint(edge, out, 4))
                for lane in range(3):
                    bx = cx - 40 * pk + lane * 40 * pk
                    h = (64 if lane == w else 24 + 18 * ((k * 7 + lane * 3) % 3)) * pk
                    col = (LIME if lane == 1 else DIM) if lane != w else edge
                    c.drawRoundRect(skia.Rect(bx - 9 * pk, cy + 34 * pk - h, bx + 9 * pk, cy + 34 * pk), 4, 4, paint(col, out))

    def s_ten(self, c, t: float) -> None:
        cu, cues = self.cues.t, self.cues
        a = lin(t, cu["ten"] - 0.2, cu["ten"] + 0.15) * (1 - lin(t, cu["traffic"] - 0.15, cu["traffic"] + 0.2))
        if a <= 0:
            return
        main = ease_out(lin(t, cu["main"], cu["main"] + 0.3))
        npc = lin(t, cu["npc"], cu["npc"] + 0.6)
        flick = 1.0
        if cu["npc"] <= t < cu["npc"] + 0.5:
            flick = 1.0 if int((t - cu["npc"]) * 14) % 2 == 0 else 0.2
        spot = main * (1 - npc) * flick if t < cu["npc"] + 0.5 else 0.0
        for i in range(10):
            x = 108 + 96 * i
            st = cu["ten"] + 0.04 * i
            pk = ease_out(lin(t, st, st + 0.3))
            if pk <= 0:
                continue
            yoff = 60 * (1 - pk)
            fin = ten_finish(cues, i)
            fk = lin(t, fin, fin + 0.25)
            you = i == 3
            light_col = LIME if you else CORAL
            self.register(c, x, 560 + yoff, 0.5, a * pk, light=fk, light_col=light_col, done=fk)
            n_q = 3
            for k in range(n_q):
                y = 680 + 115 * k + yoff - 115 * (fk if not you else 0) * (1 if k == 0 else 0.5)
                al = a * pk * (1 - fk if (k == 0 and not you) else 1)
                if you and k == 2:
                    col = mix(LIME, DIM, npc)
                    sc = 72 * (1 + 0.25 * main * (1 - npc))
                    if spot > 0:
                        cone = skia.Path()
                        cone.moveTo(x - 20, 380)
                        cone.lineTo(x + 20, 380)
                        cone.lineTo(x + 110, y + 40)
                        cone.lineTo(x - 110, y + 40)
                        cone.close()
                        p = skia.Paint(AntiAlias=True)
                        p.setShader(skia.GradientShader.MakeLinear([skia.Point(0, 380), skia.Point(0, y + 40)],
                                                                   [skia.Color(*AMBER, 0), skia.Color(*AMBER, int(110 * spot * a))]))
                        c.drawPath(cone, p)
                        for s_i in range(4):
                            ang = t * 2 + s_i * 1.57
                            self.sparkle(c, x + 80 * math.cos(ang), y - 40 + 60 * math.sin(ang), 12, AMBER, spot * a)
                        self.crown(c, x, y - 0.6 * sc, 40, AMBER, spot * a)
                    self.person(c, x, y, sc, col, a, accessory=0)
                    self.you_tag(c, x, y - 0.95 * sc, a * (1 - npc * 0.6))
                    if npc > 0:
                        self.cloud(c, x, y - 150, 90, npc * a, t)
                else:
                    self.person(c, x, y, 72, INK, al, accessory=(i * 3 + k) % 5)

    def _road(self, c, t: float, cx: float, top: float, bot: float, width: float, alpha: float, scroll: float) -> None:
        l, r = cx - width / 2, cx + width / 2
        c.drawRect(skia.Rect(l, top, r, bot), paint(PANEL, 0.9 * alpha))
        c.drawLine(l, top, l, bot, paint(INK, 0.8 * alpha, 6))
        c.drawLine(r, top, r, bot, paint(INK, 0.8 * alpha, 6))
        p = paint(INK, 0.6 * alpha, 6, skia.Paint.kButt_Cap)
        p.setPathEffect(skia.DashPathEffect.Make([50, 40], -scroll))
        c.drawLine(cx, top, cx, bot, p)

    def s_road(self, c, t: float) -> None:
        cu = self.cues.t
        a = lin(t, cu["traffic"] - 0.15, cu["traffic"] + 0.25) * (1 - lin(t, cu["fix"] - 0.1, cu["fix"] + 0.2))
        if a <= 0:
            return
        framed = ease_in_out(lin(t, cu["researchers"], cu["video"])) * (1 - ease_in_out(lin(t, cu["heres"] - 0.2, cu["heres"] + 0.3)))
        top = 380 + 60 * framed
        bot = 1250 - 140 * framed
        cx = 540
        width = 440 - 60 * framed
        my_x, next_x = cx + width / 4, cx - width / 4
        # speeds (px/s): positive dash scroll means you move forward
        slow_ph = cu["slow"] - 0.2 <= t < cu["when3"] - 0.1
        fast_ph = cu["when3"] - 0.1 <= t < cu["so2"]
        scroll = t * 420
        if t >= cu["slow"] - 0.2:
            t0 = cu["slow"] - 0.2
            t1 = cu["when3"] - 0.1
            scroll = t0 * 420 + (min(t, t1) - t0) * 120 + max(0.0, min(t, cu["so2"]) - t1) * 900 + max(0.0, t - cu["so2"]) * 300
        c.save()
        if framed > 0:
            c.clipRRect(skia.RRect.MakeRectXY(skia.Rect(250, top, 830, bot), 30, 30), doAntiAlias=True)
        self._road(c, t, cx, top - 20, bot + 20, width, a, scroll)
        you_y = top + (bot - top) * 0.68
        # next-lane cars: relative motion
        if slow_ph:
            rel = -(t - (cu["slow"] - 0.2)) * 520  # they stream past (upward)
            gap = 300
        elif fast_ph:
            rel = (t - (cu["when3"] - 0.1)) * 1150  # you blow past them (they slide down)
            gap = 230
        else:
            rel = -(t * 60)
            gap = 330
        for k in range(-6, 8):
            y = you_y + k * gap + (rel % gap)
            if top - 120 < y < bot + 120:
                self.car(c, next_x, y, CORAL, a * 0.95, 0.85 - 0.1 * framed)
        if slow_ph or not fast_ph:
            for k in (-1, 1):
                self.car(c, my_x, you_y + k * 240, DIM, a * 0.8, 0.85 - 0.1 * framed)
        self.car(c, my_x, you_y, LIME, a, 0.85 - 0.1 * framed)
        if fast_ph:
            for k in range(5):
                yy = you_y + 120 + k * 36
                c.drawLine(my_x - 30 + k * 15, yy, my_x - 30 + k * 15, yy + 40, paint(LIME, a * 0.5, 4))
        c.restore()
        if framed > 0:
            fr = skia.Rect(250, top, 830, bot)
            c.drawRoundRect(fr, 30, 30, paint(INK, a * framed, 6))
            # progress bar and play icon
            c.drawLine(290, bot - 34, 790, bot - 34, paint(INK, 0.3 * a * framed, 6))
            prog = lin(t, cu["video"], cu["heres"])
            c.drawLine(290, bot - 34, 290 + 500 * prog, bot - 34, paint(CORAL, a * framed, 6))
            pk = back_out(lin(t, cu["video"], cu["video"] + 0.25)) * (1 - lin(t, cu["video"] + 0.5, cu["video"] + 0.8))
            if pk > 0:
                tri = skia.Path()
                tri.moveTo(510, 720)
                tri.lineTo(510, 800)
                tri.lineTo(580, 760)
                tri.close()
                c.drawCircle(540, 760, 80 * pk, paint(BG, 0.7 * pk))
                c.drawPath(tri, paint(INK, pk))
            # 10 surveyed drivers
            for k in range(10):
                x = 180 + 80 * k
                y = 1225
                turned = k < 7 and t >= cu["seventy"] + 0.08 * k
                col = CORAL if turned else INK
                self.person(c, x, y, 62, col, a * framed, accessory=k % 5)
                if turned:
                    uk = back_out(lin(t, cu["seventy"] + 0.08 * k, cu["seventy"] + 0.08 * k + 0.2))
                    tri = skia.Path()
                    tri.moveTo(x, y - 70 - 16 * uk)
                    tri.lineTo(x - 12 * uk, y - 52)
                    tri.lineTo(x + 12 * uk, y - 52)
                    tri.close()
                    c.drawPath(tri, paint(CORAL, a * framed))
            sk = back_out(lin(t, cu["slower"] - 0.05, cu["slower"] + 0.15), 2.2)
            if sk > 0:
                c.save()
                c.translate(next_x, top + (bot - top) * 0.35)
                c.rotate(-10)
                c.scale(1.6 - 0.6 * sk, 1.6 - 0.6 * sk)
                lab = "SLOWER"
                f = skia.Font(self.mono, 64)
                lw = f.measureText(lab)
                c.drawRoundRect(skia.Rect(-lw / 2 - 20, -58, lw / 2 + 20, 18), 10, 10, paint(BG, 0.85 * a))
                c.drawRoundRect(skia.Rect(-lw / 2 - 20, -58, lw / 2 + 20, 18), 10, 10, paint(CORAL, a, 6))
                self.text(c, lab, 0, 0, 64, CORAL, a, "center")
                c.restore()
        # receipt of Ls
        rk = lin(t, cu["receipts"] - 0.25, cu["receipts"] + 0.9)
        r_out = lin(t, cu["fix"] - 0.15, cu["fix"] + 0.15)
        if rk > 0:
            c.drawRect(skia.Rect(0, 0, W, H), paint(BG, 0.6 * (1 - r_out)))
            px, py = 540, 520
            c.drawRoundRect(skia.Rect(px - 230, py - 70, px + 230, py + 30), 22, 22, paint(PANEL, 1 - r_out))
            c.drawRoundRect(skia.Rect(px - 230, py - 70, px + 230, py + 30), 22, 22, paint(INK, 1 - r_out, 5))
            length = 640 * ease_out(rk)
            paper = skia.Rect(px - 170, py, px + 170, py + length)
            c.save()
            c.clipRect(paper)
            c.drawRect(paper, paint(INK, 1 - r_out))
            rows = ["RECEIPT", "--------------", "L ......... 1", "L ......... 1", "L ......... 1", "L ......... 1",
                    "L ......... 1", "--------------", "TOTAL     5 L"]
            for k, row in enumerate(rows):
                yy = py + length - 640 + 70 + k * 64
                big = row.startswith("L") or row.startswith("TOTAL")
                self.text(c, row, px, yy, 40 if big else 34, CORAL if row.startswith("TOTAL") else BG, 1 - r_out, "center")
            c.restore()

    def _snake_path(self) -> skia.Path:
        p = skia.Path()
        rows = [1200, 1080, 960, 840]
        p.moveTo(980, rows[0])
        p.lineTo(150, rows[0])
        p.quadTo(100, rows[0], 100, rows[0] - 60)
        p.quadTo(100, rows[1], 150, rows[1])
        p.lineTo(930, rows[1])
        p.quadTo(980, rows[1], 980, rows[1] - 60)
        p.quadTo(980, rows[2], 930, rows[2])
        p.lineTo(150, rows[2])
        p.quadTo(100, rows[2], 100, rows[2] - 60)
        p.quadTo(100, rows[3], 150, rows[3])
        p.lineTo(540, rows[3])
        p.lineTo(540, 700)
        return p

    def s_snake(self, c, t: float) -> None:
        cu = self.cues.t
        a = lin(t, cu["fix"], cu["fix"] + 0.3) * (1 - lin(t, cu["so3"], cu["so3"] + 0.3))
        if a <= 0:
            return
        for li, x in enumerate(LANES):
            dk = ease_out(lin(t, cu["fix"] + 0.05 * li, cu["fix"] + 0.05 * li + 0.35))
            busy = 0.5 + 0.5 * math.sin(t * 5 + li * 2.1)
            self.register(c, x, REG_Y - 80 * (1 - dk), 1.0, a * dk, light=busy if t >= cu["feeding"] else 0, light_col=LIME, done=1.0)
        path = self._snake_path()
        meas = skia.PathMeasure(path, False)
        L = meas.getLength()
        draw_k = ease_in_out(lin(t, cu["single"] - 0.1, cu["register"]))
        if draw_k > 0:
            seg = skia.Path()
            meas.getSegment(0, L * draw_k, seg, True)
            glow_k = lin(t, cu["snake"] - 0.1, cu["snake"] + 0.2) * (1 - lin(t, cu["snake"] + 0.5, cu["snake"] + 1.0))
            if glow_k > 0:
                gp = paint(LIME, 0.7 * glow_k * a, 80)
                gp.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, 18))
                c.drawPath(seg, gp)
            rail = paint(INK, 0.35 * a, 70, skia.Paint.kRound_Cap)
            c.drawPath(seg, rail)
            c.drawPath(seg, paint(BG, a, 58, skia.Paint.kRound_Cap))
            c.drawPath(seg, paint(LIME if glow_k > 0 else DIM, (0.35 + 0.6 * glow_k) * a, 3))
            for k in range(0, int(L / 120)):
                d = k * 120
                if d > L * draw_k:
                    break
                pos, _tan = meas.getPosTan(d)
                if True:
                    c.drawCircle(pos.x(), pos.y() - 34, 6, paint(INK, 0.6 * a))
                    c.drawCircle(pos.x(), pos.y() + 34, 6, paint(INK, 0.6 * a))
        # people flowing along the snake, peeling off to whichever register is free
        if t >= cu["feeding"] - 0.3:
            fa = a * lin(t, cu["feeding"] - 0.3, cu["feeding"] + 0.2)
            speed = 260.0
            gap = 120.0
            head = (t - cu["fix"]) * speed
            for j in range(int(head / gap) + 2):
                d = head - j * gap
                if d < 0:
                    continue
                if d < L:
                    pos, _ = meas.getPosTan(d)
                    if True:
                        self.person(c, pos.x(), pos.y() - 10, 66, INK, fa, accessory=j % 5)
                else:
                    k = clamp((d - L) / 260)
                    if k >= 1:
                        continue
                    lx = LANES[j % 3]
                    x = 540 + (lx - 540) * ease_out(k)
                    y = 700 - (700 - REG_Y - 70) * ease_out(k)
                    self.person(c, x, y, 66, LIME, fa * (1 - k), accessory=j % 5)
        # bank and airport icons
        for key, x, kind in (("banks", 150, "bank"), ("airports", 930, "plane")):
            ik = back_out(lin(t, cu[key], cu[key] + 0.25), 2.0)
            if ik > 0:
                self._icon(c, kind, x, 470, 120 * ik, a)

    def _icon(self, c, kind: str, x: float, y: float, s: float, a: float) -> None:
        p = paint(AMBER, a, max(3.0, s * 0.06))
        if kind == "bank":
            roof = skia.Path()
            roof.moveTo(x - 0.5 * s, y - 0.15 * s)
            roof.lineTo(x, y - 0.45 * s)
            roof.lineTo(x + 0.5 * s, y - 0.15 * s)
            roof.close()
            c.drawPath(roof, p)
            for k in range(4):
                xx = x - 0.33 * s + k * 0.22 * s
                c.drawLine(xx, y - 0.05 * s, xx, y + 0.3 * s, p)
            c.drawLine(x - 0.5 * s, y + 0.4 * s, x + 0.5 * s, y + 0.4 * s, p)
        else:
            body = skia.Path()
            body.moveTo(x - 0.5 * s, y)
            body.lineTo(x + 0.45 * s, y)
            body.moveTo(x + 0.05 * s, y)
            body.lineTo(x - 0.2 * s, y - 0.38 * s)
            body.moveTo(x + 0.05 * s, y)
            body.lineTo(x - 0.2 * s, y + 0.38 * s)
            body.moveTo(x - 0.4 * s, y)
            body.lineTo(x - 0.52 * s, y - 0.18 * s)
            c.drawPath(body, p)

    def s_outro(self, c, t: float) -> None:
        cu, cues = self.cues.t, self.cues
        a = lin(t, cu["so3"] + 0.05, cu["so3"] + 0.4)
        lane = 1.0
        dy = 0.0
        hp = hops(cues)
        prev = 1.0
        for tt, target in hp:
            k = ease_in_out(lin(t, tt, tt + 0.22))
            if k > 0:
                lane = prev + (target - prev) * k
                dy = -110 * math.sin(math.pi * k) if k < 1 else 0.0
            prev = target if t >= tt + 0.22 else prev
        if t >= hp[-1][0] + 0.22:
            lane, dy = 1.0, 0.0
        self._lanes(c, t, a, you_lane=lane, you_dy=dy, hook_t=0.0)
