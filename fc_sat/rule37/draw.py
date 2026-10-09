"""Line-art picture for the 37% rule short. No captions: only drawings and corner counters."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import skia

from fc_sat.rule37.cues import FPS, N, R, Cues, curve_peak_time, grid_pop, label_time, reveal_a, reveal_b, spin_count, spin_time, success_curve

ROOT = Path(__file__).resolve().parents[2]
W, H = 1080, 1920

BG = (0x12, 0x0E, 0x22)
CARD = (0x1D, 0x17, 0x36)
INK = (0xF2, 0xEC, 0xFF)
DIM = (0x5E, 0x56, 0x80)
PINK = (0xFF, 0x4F, 0x8B)
GOLD = (0xFF, 0xC2, 0x4B)
LAV = (0x8E, 0x7C, 0xFF)
GREEN = (0x59, 0xD9, 0x8E)

# chart geometry, kept clear of the lower-middle caption band (y 1330-1650)
X0, X1 = 118.0, 962.0
Y0, YTOP = 1190.0, 640.0
SLOT = (X1 - X0) / N


def clamp(x: float, a: float = 0.0, b: float = 1.0) -> float:
    return a if x < a else b if x > b else x


def lin(t: float, a: float, b: float) -> float:
    return clamp((t - a) / (b - a)) if b > a else float(t >= a)


def ease_out(x: float) -> float:
    return 1 - (1 - x) ** 3


def ease_in_out(x: float) -> float:
    return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2


def back_out(x: float, s: float = 1.9) -> float:
    x -= 1
    return 1 + (s + 1) * x ** 3 + s * x ** 2


def mix(a, b, k: float):
    return tuple(int(round(a[i] + (b[i] - a[i]) * k)) for i in range(3))


def paint(color, alpha: float = 1.0, stroke: float = 0.0, cap=skia.Paint.kRound_Cap) -> skia.Paint:
    p = skia.Paint(AntiAlias=True, Color=skia.Color(*color, int(255 * clamp(alpha))))
    if stroke:
        p.setStyle(skia.Paint.kStroke_Style)
        p.setStrokeWidth(stroke)
        p.setStrokeCap(cap)
        p.setStrokeJoin(skia.Paint.kRound_Join)
    return p


def glow(color, alpha: float, sigma: float) -> skia.Paint:
    p = paint(color, alpha)
    p.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, sigma))
    return p


class Painter:
    def __init__(self, cues: Cues) -> None:
        self.cues = cues
        self.surface = skia.Surface(W, H)
        self.mono = skia.Typeface.MakeFromFile(str(ROOT / "assets" / "fonts" / "JetBrainsMono-ExtraBold.ttf"))
        self.bg = self._background()
        self.curve = success_curve()
        rng = np.random.default_rng(5)
        self.confetti = [(rng.uniform(0, 2 * math.pi), rng.uniform(180, 420), rng.choice(3), rng.uniform(5, 10))
                         for _ in range(40)]
        self.symbols = [(sym, a, r) for sym, a, r in zip(["%", "÷", "1/e", "×", "=", "√", "π", "+"],
                        [3.5, 5.9, 0.5, 2.6, 4.4, 1.2, 0.0, 2.1], [380, 360, 400, 390, 340, 410, 380, 370])]
        self.spin_vals = [int(v) for v in rng.integers(3, 99, size=spin_count(cues))]

    # --- primitives ---------------------------------------------------------------------------

    def _background(self) -> skia.Image:
        s = skia.Surface(W, H)
        c = s.getCanvas()
        c.clear(skia.Color(*BG))
        g = paint(INK, 0.045, 1.5)
        for x in range(0, W + 1, 60):
            c.drawLine(x, 0, x, H, g)
        for y in range(0, H + 1, 60):
            c.drawLine(0, y, W, y, g)
        shader = skia.GradientShader.MakeRadial(
            skia.Point(W / 2, H * 0.42), H * 0.75,
            [skia.Color(0x2A, 0x1C, 0x4A, 150), skia.Color(0, 0, 0, 0), skia.Color(0, 0, 0, 200)],
            [0.0, 0.55, 1.0])
        p = skia.Paint(AntiAlias=True)
        p.setShader(shader)
        c.drawRect(skia.Rect(0, 0, W, H), p)
        return s.makeImageSnapshot()

    def text(self, c, s: str, x: float, y: float, size: float, color, alpha: float = 1.0, align: str = "left",
             spacing: float = 0.0) -> float:
        font = skia.Font(self.mono, size)
        width = font.measureText(s) + spacing * max(len(s) - 1, 0)
        if align == "center":
            x -= width / 2
        elif align == "right":
            x -= width
        if spacing:
            for ch in s:
                c.drawString(ch, x, y, font, paint(color, alpha))
                x += font.measureText(ch) + spacing
        else:
            c.drawString(s, x, y, font, paint(color, alpha))
        return width

    def person(self, c, x, y, s, color, alpha=1.0, accessory: int = 0, fill=None) -> None:
        sw = max(2.0, s * 0.075)
        p = paint(color, alpha, sw)
        if fill is not None:
            f = paint(fill, alpha)
            c.drawCircle(x, y - 0.32 * s, 0.2 * s, f)
        c.drawCircle(x, y - 0.32 * s, 0.2 * s, p)
        body = skia.Path()
        body.moveTo(x - 0.36 * s, y + 0.42 * s)
        body.lineTo(x - 0.36 * s, y + 0.12 * s)
        body.quadTo(x - 0.36 * s, y - 0.06 * s, x - 0.16 * s, y - 0.06 * s)
        body.lineTo(x + 0.16 * s, y - 0.06 * s)
        body.quadTo(x + 0.36 * s, y - 0.06 * s, x + 0.36 * s, y + 0.12 * s)
        body.lineTo(x + 0.36 * s, y + 0.42 * s)
        c.drawPath(body, p)
        hy = y - 0.32 * s
        if accessory == 1:  # bun
            c.drawCircle(x, hy - 0.27 * s, 0.08 * s, p)
        elif accessory == 2:  # glasses
            q = paint(color, alpha, sw * 0.6)
            c.drawCircle(x - 0.08 * s, hy + 0.01 * s, 0.055 * s, q)
            c.drawCircle(x + 0.08 * s, hy + 0.01 * s, 0.055 * s, q)
        elif accessory == 3:  # cap
            cap = skia.Path()
            cap.moveTo(x - 0.2 * s, hy - 0.04 * s)
            cap.quadTo(x - 0.18 * s, hy - 0.24 * s, x, hy - 0.24 * s)
            cap.quadTo(x + 0.18 * s, hy - 0.24 * s, x + 0.2 * s, hy - 0.04 * s)
            cap.lineTo(x + 0.36 * s, hy - 0.04 * s)
            c.drawPath(cap, p)
        elif accessory == 4:  # long hair
            hair = skia.Path()
            hair.moveTo(x - 0.2 * s, hy + 0.02 * s)
            hair.quadTo(x - 0.26 * s, hy + 0.22 * s, x - 0.3 * s, hy + 0.3 * s)
            hair.moveTo(x + 0.2 * s, hy + 0.02 * s)
            hair.quadTo(x + 0.26 * s, hy + 0.22 * s, x + 0.3 * s, hy + 0.3 * s)
            c.drawPath(hair, p)

    def heart(self, c, x, y, s, color, alpha=1.0, stroke: float = 0.0) -> None:
        p = skia.Path()
        p.moveTo(x, y + 0.36 * s)
        p.cubicTo(x - 0.6 * s, y - 0.02 * s, x - 0.36 * s, y - 0.56 * s, x, y - 0.24 * s)
        p.cubicTo(x + 0.36 * s, y - 0.56 * s, x + 0.6 * s, y - 0.02 * s, x, y + 0.36 * s)
        p.close()
        c.drawPath(p, paint(color, alpha, stroke) if stroke else paint(color, alpha))

    def half_heart(self, c, x, y, s, color, alpha, side: int) -> None:
        p = skia.Path()
        p.moveTo(x, y + 0.36 * s)
        p.cubicTo(x + side * 0.6 * s, y - 0.02 * s, x + side * 0.36 * s, y - 0.56 * s, x, y - 0.24 * s)
        p.lineTo(x + side * 0.06 * s, y - 0.08 * s)
        p.lineTo(x - side * 0.05 * s, y + 0.06 * s)
        p.lineTo(x + side * 0.04 * s, y + 0.2 * s)
        p.close()
        c.drawPath(p, paint(color, alpha))

    def crown(self, c, x, y, s, color, alpha=1.0) -> None:
        pts = [(-0.5, 0.3), (-0.5, -0.22), (-0.25, 0.04), (0, -0.38), (0.25, 0.04), (0.5, -0.22), (0.5, 0.3)]
        p = skia.Path()
        p.moveTo(x + pts[0][0] * s, y + pts[0][1] * s)
        for px, py in pts[1:]:
            p.lineTo(x + px * s, y + py * s)
        p.close()
        c.drawPath(p, paint(color, alpha * 0.25))
        c.drawPath(p, paint(color, alpha, s * 0.09))

    def cross(self, c, x, y, s, color, alpha=1.0, w=None) -> None:
        p = paint(color, alpha, w or s * 0.22)
        c.drawLine(x - s / 2, y - s / 2, x + s / 2, y + s / 2, p)
        c.drawLine(x + s / 2, y - s / 2, x - s / 2, y + s / 2, p)

    def sparkle(self, c, x, y, s, color, alpha=1.0) -> None:
        p = skia.Path()
        p.moveTo(x, y - s)
        p.quadTo(x, y, x + s, y)
        p.quadTo(x, y, x, y + s)
        p.quadTo(x, y, x - s, y)
        p.quadTo(x, y, x, y - s)
        p.close()
        c.drawPath(p, paint(color, alpha))

    def lock(self, c, x, y, s, color, alpha=1.0) -> None:
        c.drawRoundRect(skia.Rect(x - 0.4 * s, y - 0.1 * s, x + 0.4 * s, y + 0.5 * s), 0.08 * s, 0.08 * s, paint(color, alpha))
        arc = skia.Path()
        arc.addArc(skia.Rect(x - 0.26 * s, y - 0.5 * s, x + 0.26 * s, y + 0.02 * s), 180, 180)
        c.drawPath(arc, paint(color, alpha, 0.1 * s, skia.Paint.kButt_Cap))
        arc2 = skia.Path()
        arc2.moveTo(x - 0.26 * s, y - 0.24 * s)
        arc2.lineTo(x - 0.26 * s, y - 0.1 * s)
        arc2.moveTo(x + 0.26 * s, y - 0.24 * s)
        arc2.lineTo(x + 0.26 * s, y - 0.1 * s)
        c.drawPath(arc2, paint(color, alpha, 0.1 * s, skia.Paint.kButt_Cap))
        c.drawCircle(x, y + 0.18 * s, 0.07 * s, paint(BG, alpha))

    def card(self, c, x, y, rot, scale, idx: int, alpha=1.0, golden: bool = False, stamp: float = 0.0, t: float = 0.0) -> None:
        w, h = 540 * scale, 740 * scale
        c.save()
        c.translate(x, y)
        c.rotate(rot)
        if alpha < 1:
            c.saveLayerAlpha(None, int(255 * clamp(alpha)))
        edge = GOLD if golden else INK
        if golden:
            c.drawRoundRect(skia.Rect(-w / 2, -h / 2, w / 2, h / 2), 44 * scale, 44 * scale, glow(GOLD, 0.35, 30 * scale))
        c.drawRoundRect(skia.Rect(-w / 2, -h / 2, w / 2, h / 2), 44 * scale, 44 * scale, paint(CARD))
        c.drawRoundRect(skia.Rect(-w / 2, -h / 2, w / 2, h / 2), 44 * scale, 44 * scale, paint(edge, 0.95, 5 * scale))
        tints = [LAV, PINK, GREEN, GOLD]
        tint = GOLD if golden else tints[idx % 4]
        photo = skia.Rect(-w / 2 + 34 * scale, -h / 2 + 34 * scale, w / 2 - 34 * scale, -h / 2 + 34 * scale + 0.58 * h)
        c.drawRoundRect(photo, 26 * scale, 26 * scale, paint(tint, 0.10))
        c.drawRoundRect(photo, 26 * scale, 26 * scale, paint(tint, 0.55, 3 * scale))
        c.save()
        c.clipRRect(skia.RRect.MakeRectXY(photo, 26 * scale, 26 * scale), doAntiAlias=True)
        self.person(c, 0, photo.centerY() + 0.12 * photo.height(), 0.86 * photo.height(), edge, 1.0,
                    accessory=(idx * 3 + 1) % 5)
        c.restore()
        if golden:
            self.crown(c, 0, photo.top() + 0.13 * photo.height(), 70 * scale, GOLD)
            for k in range(4):
                a = t * 3 + k * 1.7
                sx = (-0.32 + 0.21 * k) * w
                sy = photo.top() + (0.2 + 0.18 * (k % 2)) * photo.height()
                self.sparkle(c, sx, sy, (10 + 6 * math.sin(a)) * scale, GOLD, 0.8)
        ly = photo.bottom() + 52 * scale
        for k, frac in enumerate((0.62, 0.42, 0.54)):
            c.drawLine(-w / 2 + 44 * scale, ly + k * 34 * scale, -w / 2 + 44 * scale + frac * (w - 88 * scale),
                       ly + k * 34 * scale, paint(DIM if not golden else GOLD, 0.9 if not golden else 0.6, 12 * scale))
        by = h / 2 - 70 * scale
        c.drawCircle(-90 * scale, by, 42 * scale, paint(DIM, 1, 4 * scale))
        self.cross(c, -90 * scale, by, 30 * scale, DIM, 1, 5 * scale)
        c.drawCircle(90 * scale, by, 42 * scale, paint(PINK, 1, 4 * scale))
        self.heart(c, 90 * scale, by + 2 * scale, 52 * scale, PINK)
        if stamp > 0:
            k = back_out(clamp(stamp), 2.4)
            s = (1.6 - 0.6 * k) * w * 0.55
            c.save()
            c.rotate(-12)
            c.drawCircle(0, -40 * scale, s * 0.62, paint(PINK, clamp(stamp * 3) * 0.9, 16 * scale))
            self.cross(c, 0, -40 * scale, s * 0.62, PINK, clamp(stamp * 3), 26 * scale)
            c.restore()
        if alpha < 1:
            c.restore()
        c.restore()

    def counter(self, c, side: str, label: str, value: str, alpha: float, color=INK, pop: float = 0.0) -> None:
        if alpha <= 0:
            return
        x = 76 if side == "left" else W - 76
        align = "left" if side == "left" else "right"
        self.text(c, label, x, 214, 30, INK, 0.55 * alpha, align, spacing=4)
        size = 104 * (1 + 0.14 * math.sin(math.pi * clamp(pop)) if pop > 0 else 1)
        self.text(c, value, x, 214 + 18 + size * 0.86, size, color, alpha, align)

    # --- frame --------------------------------------------------------------------------------

    def render(self, frame: int) -> np.ndarray:
        t = frame / FPS
        cu = self.cues.t
        c = self.surface.getCanvas()
        c.drawImage(self.bg, 0, 0)
        shake = self._shake(t)
        c.save()
        c.translate(*shake)
        scenes = [
            (0.0, cu["its"], self.s_hook),
            (cu["its"], cu["so"], self.s_queue),
            (cu["so"], cu["first2"] - 0.25, self.s_grid),
            (cu["first2"] - 0.25, cu["no2"] + 0.25, self.s_chart),
            (cu["no2"], cu["but2"], self.s_curve),
            (cu["but2"], cu["so2"] + 0.25, self.s_catch),
            (cu["so2"], self.cues.end + 1, self.s_outro),
        ]
        for a, b, fn in scenes:
            if a - 0.001 <= t < b:
                fn(c, t)
        c.restore()
        self.counters(c, t)
        img = self.surface.makeImageSnapshot().toarray(colorType=skia.kBGRA_8888_ColorType)
        return np.ascontiguousarray(img[:, :, :3])

    def _shake(self, t: float) -> tuple[float, float]:
        cu = self.cues.t
        out = [0.0, 0.0]
        for start, amp in ((cu["on"], 18.0), (cu["theres"] + 0.12, 10.0), (cu["but2"], 14.0), (cu["go"] + 0.45, 9.0)):
            d = t - start
            if 0 <= d < 0.35:
                k = amp * (1 - d / 0.35) ** 2
                out[0] += k * math.sin(d * 97)
                out[1] += k * math.cos(d * 71)
        return out[0], out[1]

    # --- counters -----------------------------------------------------------------------------

    def counters(self, c, t: float) -> None:
        cu, cues = self.cues.t, self.cues
        # left counter
        if t < cu["its"]:
            done = sum(1 for s in cues.swipes if t >= s + 0.06)
            pct = round(37 * done / len(cues.swipes))
            pop = lin(t, cu["on"], cu["on"] + 0.25) if t < cu["on"] + 0.25 else 0
            self.counter(c, "left", "REJECTED", f"{pct}%", 1 - lin(t, cu["its"] - 0.2, cu["its"]), PINK if pct >= 37 else INK, pop)
        elif t < cu["so"]:
            pass
        elif t < cu["first2"]:
            n = min(100, sum(1 for d in range(19) for _ in range(min(d, 18 - d) + 1) if t >= grid_pop(cues, d)))
            self.counter(c, "left", "DATES", f"{n}", lin(t, cu["so"], cu["so"] + 0.2) * (1 - lin(t, cu["first2"] - 0.2, cu["first2"])))
        elif t < cu["that"]:
            n = sum(1 for i in range(R) if t >= reveal_a(cues, i) + 0.05)
            a = lin(t, cu["t37b"], cu["t37b"] + 0.2)
            if t < cu["then"]:
                self.counter(c, "left", "RESEARCH", f"{n}", a, LAV)
            else:
                k = cues.pick_a
                cur = R + sum(1 for i in range(R, k + 1) if t >= reveal_a(cues, i))
                self.counter(c, "left", "DATE NO.", f"{min(cur, k + 1)}", 1, PINK if t >= reveal_a(cues, k) else INK,
                             lin(t, reveal_a(cues, k), reveal_a(cues, k) + 0.3) if t < reveal_a(cues, k) + 0.3 else 0)
        elif t < cu["no2"]:
            self.counter(c, "left", "DATE NO.", f"{cues.pick_a + 1}", 1 - lin(t, cu["no2"] - 0.2, cu["no2"]), PINK)
        elif t < cu["dating"]:
            pass
        elif t < cu["but2"]:
            age = 18 + 8.1 * ease_in_out(lin(t, cu["your"], cu["twentysix"]))
            a = lin(t, cu["dating"], cu["dating"] + 0.25) * (1 - lin(t, cu["but2"] - 0.15, cu["but2"]))
            self.counter(c, "left", "AGE", f"{int(age)}", a, PINK if age >= 26 else INK,
                         lin(t, cu["twentysix"], cu["twentysix"] + 0.3) if cu["twentysix"] <= t < cu["twentysix"] + 0.3 else 0)
        elif t < cu["so2"]:
            if t >= cu["and3"]:
                after = sum(1 for i in range(R, N) if t >= reveal_b(cues, i))
                if after < N - R:
                    self.counter(c, "left", "DATE NO.", f"{R + after}", 1, INK)
                else:
                    self.counter(c, "left", "LOCKED IN", "NOBODY", 1 - lin(t, cu["so2"] - 0.2, cu["so2"]), PINK)
        else:
            a = lin(t, cu["so2"] + 0.1, cu["so2"] + 0.4)
            j = sum(1 for k in range(spin_count(cues)) if t >= spin_time(cues, k))
            if t >= self.cues.end - 0.55:
                val = "0%"
            elif j == 0:
                val = "??"
            else:
                val = f"{self.spin_vals[j - 1]}%"
            self.counter(c, "left", "REJECTED", val, a, INK)

        # right counter
        if cu["that"] <= t < cu["no2"]:
            v = round(37 * ease_out(lin(t, cu["t37c"], cu["shot"])))
            self.counter(c, "right", "BEST ONE", f"{v}%", lin(t, cu["t37c"], cu["t37c"] + 0.15) * (1 - lin(t, cu["no2"] - 0.2, cu["no2"])), GOLD)
        elif cu["also"] <= t < cu["so2"]:
            self.counter(c, "right", "FUMBLED", "37%", lin(t, cu["also"], cu["also"] + 0.2) * (1 - lin(t, cu["so2"] - 0.2, cu["so2"])), PINK,
                         lin(t, cu["t37d"], cu["t37d"] + 0.3) if cu["t37d"] <= t < cu["t37d"] + 0.3 else 0)

    # --- scenes -------------------------------------------------------------------------------

    def _stack(self, c, t: float, top_idx: int, y: float, alpha: float = 1.0) -> None:
        for d in (3, 2, 1):
            self.card(c, 540, y + 26 * d, (d % 2 * 2 - 1) * 2.0 * d, 1 - 0.045 * d, top_idx + d, alpha * (0.85 - 0.18 * d))

    def s_hook(self, c, t: float) -> None:
        cu, sw = self.cues.t, self.cues.swipes
        y = 840
        out = 1 - lin(t, cu["its"] - 0.18, cu["its"])
        done = sum(1 for s in sw if t >= s)
        dim = 1 - 0.55 * lin(t, cu["on"], cu["on"] + 0.12) * (1 - lin(t, cu["even"] - 0.1, cu["even"] + 0.1))
        golden_in = lin(t, cu["even"], cu["even"] + 0.35)
        gold_swipe = cu["perfect"] + 0.35
        if t < cu["even"]:
            self._stack(c, t, done, y, dim * out)
        # incoming top card settles as the previous one leaves
        if t < cu["even"]:
            settle = ease_out(lin(t, sw[done - 1], sw[done - 1] + 0.16)) if done else 1.0
            k = 1 - settle
            lean = 0.0
            if done == 0:
                lean = -3.5 * ease_in_out(lin(t, 0.0, cu["dump"]))
            self.card(c, 540 - 10 * lean, y + 26 * k, 2.0 * k + lean, 1 - 0.045 * k, done, dim * out)
        # cards in flight
        for i, s in enumerate(sw):
            d = t - s
            if 0 <= d < 0.42:
                p = ease_in_out(clamp(d / 0.36))
                x = 540 - 1050 * p
                self.card(c, x, y - 60 * p, -4 - 26 * p, 1.0, i, dim * out, stamp=clamp(d / 0.12))
        # math symbols drift up around the card from "Math" until the first swipe
        ma = lin(t, cu["math"], cu["math"] + 0.2) * (1 - lin(t, cu["dump"] - 0.05, cu["dump"] + 0.2))
        if ma > 0:
            for j, (sym, ang, rad) in enumerate(self.symbols):
                pk = back_out(lin(t, cu["math"] + j * 0.06, cu["math"] + j * 0.06 + 0.25), 2.0)
                rise = 40 * (t - cu["math"])
                x = 540 + rad * math.cos(ang)
                yy = 840 + rad * 1.25 * math.sin(ang) - rise
                self.text(c, sym, x, yy, 72 * pk, (LAV, PINK, GOLD)[j % 3], ma * clamp(pk), "center")
        # 37% slam
        if cu["on"] - 0.02 <= t < cu["even"] + 0.25:
            k = back_out(lin(t, cu["on"] - 0.02, cu["on"] + 0.16), 1.6)
            gone = lin(t, cu["even"], cu["even"] + 0.22)
            size = 300 * (2.2 - 1.2 * k) * (1 - 0.6 * gone)
            a = clamp(k * 2) * (1 - gone)
            c.drawCircle(540, 800, 330 * (1 - gone), glow(PINK, 0.18 * a, 90))
            self.text(c, "37%", 540, 800 + size * 0.36, size, PINK, a, "center")
        # the perfect one
        if t >= cu["even"]:
            gy = y + 900 * (1 - ease_out(golden_in))
            if t < gold_swipe:
                self._stack(c, t, 40, y, out * golden_in)
                self.card(c, 540, gy, 0, 1.0, 7, out, golden=True, t=t)
            else:
                self._stack(c, t, 40, y, out)
                d = t - gold_swipe
                p = ease_in_out(clamp(d / 0.3))
                self.card(c, 540 - 1050 * p, y - 60 * p, -4 - 26 * p, 1.0, 7, out, golden=True, stamp=clamp(d / 0.1), t=t)

    def s_queue(self, c, t: float) -> None:
        cu = self.cues.t
        a = lin(t, cu["its"], cu["its"] + 0.25) * (1 - lin(t, cu["so"] - 0.2, cu["so"]))
        if a <= 0:
            return
        c.saveLayerAlpha(None, int(255 * a))
        floor = 1130
        c.drawLine(60, floor + 66, W - 60, floor + 66, paint(INK, 0.35, 3))
        # queue position: drifts, then steps one person per 0.62 s from "meet"
        q = 0.35 * (t - cu["its"]) - 2.2
        if t >= cu["meet"]:
            q0 = 0.35 * (cu["meet"] - cu["its"]) - 2.2
            n = (t - cu["meet"]) / 0.62
            q = q0 + math.floor(n) + ease_in_out(clamp((n % 1) / 0.55))
        spot = lin(t, cu["meet"] - 0.2, cu["meet"] + 0.15)
        if spot > 0:
            cone = skia.Path()
            cone.moveTo(540 - 40, 380)
            cone.lineTo(540 + 40, 380)
            cone.lineTo(540 + 170, floor + 66)
            cone.lineTo(540 - 170, floor + 66)
            cone.close()
            p = skia.Paint(AntiAlias=True)
            p.setShader(skia.GradientShader.MakeLinear([skia.Point(0, 380), skia.Point(0, floor + 66)],
                                                       [skia.Color(*GOLD, 0), skia.Color(*GOLD, int(70 * spot))]))
            c.drawPath(cone, p)
            c.drawOval(skia.Rect(540 - 170, floor + 50, 540 + 170, floor + 82), paint(GOLD, 0.25 * spot))
        gate_x = 540 - 118
        gate = ease_out(lin(t, cu["theres"], cu["theres"] + 0.14))
        for j in range(-6, 14):
            x = 540 + 190 * (j - q)
            if x < -120 or x > W + 120:
                continue
            passed = x < gate_x - 10
            col = INK
            al = 1.0
            if passed:
                col = DIM if gate > 0 else mix(INK, DIM, 0.4)
                al = 1 - 0.55 * gate
            near = 1 - clamp(abs(x - 540) / 190)
            s = 190 + 30 * near * spot
            bob = -abs(math.sin((t * 6 + j) * 1.0)) * 6 if not passed or gate == 0 else 0
            self.person(c, x, floor - 0.42 * s + 66 + bob, s, col, al, accessory=(j * 3 + 1) % 5)
        if gate > 0:
            gtop = 600.0
            gy = gtop + (floor + 66 - gtop) * gate
            for k in range(5):
                xx = gate_x - 128 + k * 32
                c.drawLine(xx, gtop, xx, gy, paint(PINK, 0.9, 7))
            c.drawLine(gate_x - 142, gy, gate_x + 14, gy, paint(PINK, 1, 9))
            c.drawLine(gate_x - 142, gtop, gate_x + 14, gtop, paint(PINK, 1, 9))
            lk = back_out(lin(t, cu["back"], cu["back"] + 0.25))
            if lk > 0:
                self.lock(c, gate_x - 64, 860, 130 * lk, PINK)
        # name badge
        b_in = back_out(lin(t, cu["secretary"], cu["secretary"] + 0.3))
        b_out = lin(t, cu["meet"] - 0.25, cu["meet"])
        if b_in > 0 and b_out < 1:
            sc = b_in * (1 - b_out)
            c.save()
            c.translate(540, 640)
            c.rotate(-4)
            c.scale(sc, sc)
            r = skia.Rect(-250, -150, 250, 150)
            c.drawRoundRect(r, 30, 30, paint(INK))
            top = skia.Path()
            top.addRRect(skia.RRect.MakeRectXY(r, 30, 30))
            c.save()
            c.clipPath(top, doAntiAlias=True)
            c.drawRect(skia.Rect(-250, -150, 250, -60), paint(PINK))
            c.restore()
            self.text(c, "HELLO", 0, -90, 54, INK, 1, "center", spacing=6)
            # handwriting squiggle
            k = lin(t, cu["secretary"] + 0.1, cu["because"])
            sq = skia.Path()
            pts = 60
            for i in range(int(pts * k) + 1):
                u = i / pts
                xx = -190 + 380 * u
                yy = 30 + 22 * math.sin(u * 31) * math.sin(u * 7 + 1) + 8 * math.sin(u * 63)
                (sq.moveTo if i == 0 else sq.lineTo)(xx, yy)
            c.drawPath(sq, paint(BG, 1, 7))
            ks = lin(t, cu["terrible"], cu["terrible"] + 0.45)
            if ks > 0:
                sc2 = skia.Path()
                for i in range(int(40 * ks) + 1):
                    u = i / 40
                    xx = -200 + 400 * u
                    yy = 30 + 40 * math.sin(u * 47)
                    (sc2.moveTo if i == 0 else sc2.lineTo)(xx, yy)
                c.drawPath(sc2, paint(PINK, 1, 9))
            kq = back_out(lin(t, cu["things"] + 0.1, cu["things"] + 0.35))
            if kq > 0:
                self.text(c, "?", 230, -120 + 0, 170 * kq, GOLD, 1, "center")
            c.restore()
        c.restore()

    def _grid_pos(self, i: int) -> tuple[float, float]:
        r, col = divmod(i, 10)
        return 180 + 80 * col, 560 + 68 * r

    def s_grid(self, c, t: float) -> None:
        cu = self.cues.t
        grass = ease_out(lin(t, cu["touch"] - 0.12, cu["touch"] + 0.25)) * (1 - ease_in_out(lin(t, cu["but1"], cu["but1"] + 0.35)))
        for i in range(N):
            r, col = divmod(i, 10)
            pk = lin(t, grid_pop(self.cues, r + col), grid_pop(self.cues, r + col) + 0.2)
            if pk <= 0:
                continue
            x, y = self._grid_pos(i)
            s = 58 * back_out(pk, 2.2)
            col_c = mix(INK, GREEN, 0.6 * grass)
            self.person(c, x, y, s, col_c, 1, accessory=(i * 7) % 5)
            if grass > 0:
                for b in range(5):
                    bx = x - 22 + b * 11
                    hb = (10 + 7 * ((b * 5 + i) % 3)) * grass
                    lean = (b - 2) * 3 * grass
                    c.drawLine(bx, y + 26, bx + lean, y + 26 - hb, paint(GREEN, 0.95, 3.5))
        if grass > 0:
            base = 1250
            for b in range(90):
                bx = 60 + b * 10.7
                hb = (26 + 18 * math.sin(b * 1.7) + 12 * math.sin(b * 0.6)) * grass
                c.drawLine(bx, base, bx + 6 * math.sin(b + t * 3), base - hb, paint(GREEN, 0.9, 4))
            c.drawLine(50, base, W - 50, base, paint(GREEN, 0.9 * grass, 4))

    def _bar_x(self, i: int) -> float:
        return X0 + SLOT * (i + 0.5)

    def _bar_h(self, score: int) -> float:
        return (Y0 - YTOP) * (0.1 + 0.9 * score / N)

    def _zone(self, c, a: float, label_a: float, gold_label: bool = False) -> None:
        if a <= 0:
            return
        xr = X0 + SLOT * R
        c.drawRoundRect(skia.Rect(X0 - 8, YTOP - 50, xr, Y0 + 4), 14, 14, paint(LAV, 0.13 * a))
        c.drawRoundRect(skia.Rect(X0 - 8, YTOP - 50, xr, Y0 + 4), 14, 14, paint(LAV, 0.5 * a, 3))
        br = paint(LAV, a, 4)
        y = Y0 + 34
        c.drawLine(X0, y, xr - 4, y, br)
        c.drawLine(X0, y, X0, y - 16, br)
        c.drawLine(xr - 4, y, xr - 4, y - 16, br)
        self.text(c, "37", (X0 + xr) / 2, y + 62, 46, LAV, label_a, "center")

    def _bars(self, c, t: float, scores, reveal, *, pick: int | None, gold: int | None, line_from: float, grey_at: float,
              wobble_at: float | None, fade_after: float | None, alpha: float, morph_t: float | None) -> float:
        """Draw a chart and return the threshold height currently shown (px above baseline)."""
        c.drawLine(X0 - 10, Y0, X1 + 10, Y0, paint(INK, 0.45 * alpha, 3))
        run_h = 0.0
        line_h = 0.0
        for i in range(N):
            x = self._bar_x(i)
            st = reveal(i)
            g = ease_out(lin(t, st, st + 0.14))
            h = self._bar_h(scores[i])
            if i < R and t >= st + 0.07:
                if h > run_h:
                    run_h = h
                    line_h = h * 0.0 + run_h
            if g <= 0:
                if morph_t is None or t >= morph_t:
                    c.drawCircle(x, Y0 - 4, 2.6, paint(INK, 0.35 * alpha))
                continue
            col = INK
            al = alpha
            if i < R:
                gk = lin(t, grey_at + i * 0.012, grey_at + i * 0.012 + 0.2)
                col = mix(INK, DIM, gk)
            elif pick is not None and i == pick:
                col = PINK
            elif pick is not None and i < pick:
                col = mix(INK, DIM, lin(t, st + 0.1, st + 0.3))
            elif pick is None:
                col = mix(INK, DIM, lin(t, st + 0.1, st + 0.3))
            else:
                col = DIM
            if gold is not None and i == gold:
                col = GOLD
                if fade_after is not None:
                    al = alpha * (1 - 0.8 * lin(t, fade_after, fade_after + 0.45))
            hh = h * g
            if wobble_at is not None and i < R:
                wk = lin(t, wobble_at, wobble_at + 0.2) * (1 - lin(t, wobble_at + 0.7, wobble_at + 1.0))
                hh *= 1 + 0.07 * wk * math.sin(t * 22 + i * 0.7)
            if col in (PINK, GOLD):
                c.drawRoundRect(skia.Rect(x - 7, Y0 - hh - 4, x + 7, Y0), 4, 4, glow(col, 0.55 * al, 12))
            c.drawRoundRect(skia.Rect(x - SLOT * 0.36, Y0 - hh, x + SLOT * 0.36, Y0), 2.5, 2.5, paint(col, al))
        return run_h

    def _threshold(self, c, t: float, h: float, a: float, prev_key: str) -> None:
        if h <= 0 or a <= 0:
            return
        y = Y0 - h - 2
        p = paint(INK, 0.85 * a, 3.5, skia.Paint.kButt_Cap)
        p.setPathEffect(skia.DashPathEffect.Make([16, 12], -t * 60))
        c.drawLine(X0 - 10, y, X1 + 10, y, p)
        self.text(c, "THE BAR", X1 + 6, y - 14, 26, INK, 0.75 * a, "right", spacing=3)

    def _smooth_line(self, t: float, scores, reveal, upto: int = R) -> float:
        """Threshold that springs up to each new research record."""
        h = 0.0
        for i in range(upto):
            st = reveal(i) + 0.07
            if t < st:
                break
            nh = self._bar_h(scores[i])
            if nh > h:
                k = back_out(lin(t, st, st + 0.16), 1.4)
                h = h + (nh - h) * k
        return h

    def s_chart(self, c, t: float) -> None:
        cu, cues = self.cues.t, self.cues
        m0 = cu["first2"] - 0.25
        a = 1 - lin(t, cu["no2"] - 0.05, cu["no2"] + 0.25)
        # morph: grid figures fly to their slots
        if t < m0 + 0.75:
            for i in range(N):
                st = m0 + i * 0.003
                k = ease_in_out(lin(t, st, st + 0.45))
                gx, gy = self._grid_pos(i)
                x = gx + (self._bar_x(i) - gx) * k
                y = gy + (Y0 - 6 - gy) * k
                s = 58 * (1 - k) + 6 * k
                if k < 0.92:
                    self.person(c, x, y, s, INK, 1 - 0.3 * k, accessory=(i * 7) % 5)
                else:
                    c.drawCircle(x, Y0 - 4, 2.6, paint(INK, 0.35))
        c.saveLayerAlpha(None, int(255 * a))
        self._zone(c, lin(t, cu["t37b"], cu["t37b"] + 0.25), lin(t, cu["t37b"], cu["t37b"] + 0.25))
        rv = lambda i: reveal_a(cues, i)
        self._bars(c, t, cues.scores_a, rv, pick=cues.pick_a, gold=None, line_from=cu["pure"], grey_at=cu["zero"],
                   wobble_at=cu["vibes"], fade_after=None, alpha=1.0, morph_t=m0 + 0.6)
        line = self._smooth_line(t, cues.scores_a, rv)
        self._threshold(c, t, line, 1 - lin(t, cu["that"] - 0.1, cu["that"] + 0.2), "a")
        # scan cursor
        k = cues.pick_a
        if cu["then"] <= t < cu["that"]:
            idx = R + sum(1 for i in range(R, k + 1) if t >= reveal_a(cues, i)) - 1
            idx = max(R, min(idx, k))
            x = self._bar_x(idx)
            tri = skia.Path()
            tri.moveTo(x, Y0 + 18)
            tri.lineTo(x - 14, Y0 + 42)
            tri.lineTo(x + 14, Y0 + 42)
            tri.close()
            c.drawPath(tri, paint(PINK if t >= reveal_a(cues, k) else INK))
        # lock-in heart and crown
        tk = reveal_a(cues, k)
        if t >= tk:
            hk = back_out(lin(t, tk + 0.04, tk + 0.3), 2.2)
            x = self._bar_x(k)
            top = Y0 - self._bar_h(cues.scores_a[k])
            beat = 1 + 0.08 * math.sin(max(0.0, t - tk) * 9) * (1 - lin(t, tk, tk + 2.0))
            c.drawCircle(x, top - 70, 70 * hk, glow(PINK, 0.35, 30))
            self.heart(c, x, top - 70, 92 * hk * beat, PINK)
            ck = back_out(lin(t, cu["best"], cu["best"] + 0.3), 2.2)
            if ck > 0:
                self.crown(c, x, top - 160, 86 * ck, GOLD)
                for j in range(6):
                    ang = j * math.pi / 3 + t
                    rr = 90 + 30 * lin(t, cu["best"], cu["best"] + 0.6)
                    self.sparkle(c, x + rr * math.cos(ang), top - 160 + rr * 0.6 * math.sin(ang), 10 * ck, GOLD,
                                 1 - lin(t, cu["best"] + 0.4, cu["best"] + 1.2))
        c.restore()

    def s_curve(self, c, t: float) -> None:
        cu = self.cues.t
        a_in = lin(t, cu["no2"], cu["no2"] + 0.2)
        a_curve = a_in * (1 - lin(t, cu["dating"], cu["dating"] + 0.3))
        a_axis = a_in * (1 - lin(t, cu["but2"] - 0.15, cu["but2"]))
        ax0, ax1 = 140.0, 940.0
        base = Y0
        c.drawLine(ax0, base, ax1, base, paint(INK, 0.7 * a_axis, 4))
        top = 0.40

        def pt(r: float, p: float) -> tuple[float, float]:
            return ax0 + (ax1 - ax0) * r / N, base - (base - YTOP) * p / top

        if a_curve > 0:
            c.drawLine(ax0, base, ax0, YTOP - 20, paint(INK, 0.35 * a_curve, 3))
            k = ease_in_out(lin(t, cu["no2"] + 0.05, cu["no2"] + 0.8))
            path = skia.Path()
            last = int(k * (N - 1))
            for r in range(last + 1):
                x, y = pt(r, self.curve[r])
                (path.moveTo if r == 0 else path.lineTo)(x, y)
            c.drawPath(path, paint(LAV, a_curve, 7))
            pk = back_out(lin(t, curve_peak_time(self.cues), curve_peak_time(self.cues) + 0.3), 2.2)
            if pk > 0:
                x, y = pt(R, self.curve[R])
                dp = paint(PINK, a_curve, 3, skia.Paint.kButt_Cap)
                dp.setPathEffect(skia.DashPathEffect.Make([10, 8], 0))
                c.drawLine(x, y, x, base, dp)
                c.drawCircle(x, y, 40 * pk, glow(PINK, 0.4 * a_curve, 20))
                c.drawCircle(x, y, 16 * pk, paint(PINK, a_curve))
                self.text(c, "37%", x, base + 70, 50 * pk, PINK, a_curve, "center")
        # age ruler
        if t >= cu["dating"] - 0.1:
            a_r = lin(t, cu["dating"], cu["dating"] + 0.3) * a_axis
            for j in range(12):
                age = 18 + 2 * j
                x = ax0 + (ax1 - ax0) * j / 11
                lk = back_out(lin(t, label_time(self.cues, j), label_time(self.cues, j) + 0.2), 2.0)
                c.drawLine(x, base, x, base - 18, paint(INK, 0.6 * a_r, 3))
                if lk > 0 and (j % 2 == 0 or j == 11):
                    self.text(c, f"{age}", x, base + 64, 40 * lk, INK, 0.8 * a_r, "center")
            prog = ease_in_out(lin(t, cu["your"], cu["twentysix"]))
            xm = ax0 + (ax1 - ax0) * (8.1 * prog) / 22
            if prog > 0:
                c.drawRoundRect(skia.Rect(ax0, base - 260, xm, base), 10, 10, paint(LAV, 0.16 * a_r))
                c.drawLine(ax0, base - 2, xm, base - 2, paint(LAV, a_r, 8))
            walk = t * 7 if 0 < prog < 1 else 0
            self.person(c, xm, base - 125 - abs(math.sin(walk)) * 10, 180, INK, a_r, accessory=4)
            hy = base - 300
            self.heart(c, xm, hy, 60, PINK, a_r)
            c.drawLine(xm, hy + 24, xm, base - 230, paint(PINK, a_r, 4))
            # 26 marker + confetti
            tk = cu["twentysix"]
            if t >= tk:
                mk = back_out(lin(t, tk, tk + 0.25), 2.0)
                self.text(c, "26", xm, base + 140, 84 * mk, PINK, a_r, "center")
                d = t - tk
                if d < 1.4:
                    for ang, sp, ci, sz in self.confetti:
                        r = sp * (1 - math.exp(-d * 4)) / 1.0
                        cx = xm + r * math.cos(ang)
                        cy = hy + r * math.sin(ang) + 260 * d * d
                        col = (PINK, GOLD, LAV)[ci]
                        c.drawRect(skia.Rect(cx - sz / 2, cy - sz / 2, cx + sz / 2, cy + sz / 2), paint(col, a_r * (1 - d / 1.4)))

    def s_catch(self, c, t: float) -> None:
        cu, cues = self.cues.t, self.cues
        a = lin(t, cu["but2"] + 0.05, cu["but2"] + 0.35) * (1 - lin(t, cu["so2"], cu["so2"] + 0.25))
        if a <= 0:
            return
        c.saveLayerAlpha(None, int(255 * a))
        za = lin(t, cu["also"], cu["also"] + 0.25)
        self._zone(c, za, za)
        # warning sign on "But here's the catch"
        wk = back_out(lin(t, cu["but2"] + 0.05, cu["but2"] + 0.3), 2.0) * (1 - ease_in_out(lin(t, cu["also"] - 0.1, cu["also"] + 0.2)))
        if wk > 0:
            pulse = 1 + 0.06 * math.sin((t - cu["but2"]) * 14)
            sz = 210 * wk * pulse
            tri = skia.Path()
            tri.moveTo(540, 820 - sz * 0.62)
            tri.lineTo(540 + sz * 0.62, 820 + sz * 0.45)
            tri.lineTo(540 - sz * 0.62, 820 + sz * 0.45)
            tri.close()
            c.drawPath(tri, glow(PINK, 0.25, 30))
            c.drawPath(tri, paint(PINK, 1, 14))
            c.drawLine(540, 820 - sz * 0.2, 540, 820 + sz * 0.12, paint(PINK, 1, 16))
            c.drawCircle(540, 820 + sz * 0.28, 9 * wk, paint(PINK))
        rv = lambda i: reveal_b(cues, i)
        g = cues.best_b
        self._bars(c, t, cues.scores_b, rv, pick=None, gold=g, line_from=0, grey_at=1e9, wobble_at=None,
                   fade_after=cu["go"], alpha=1.0, morph_t=None)
        line = self._smooth_line(t, cues.scores_b, rv)
        self._threshold(c, t, line, 1.0, "b")
        tg = reveal_b(cues, g)
        x = self._bar_x(g)
        top = Y0 - self._bar_h(cues.scores_b[g])
        if t >= tg:
            ck = back_out(lin(t, tg + 0.05, tg + 0.3), 2.2)
            drift = ease_in_out(lin(t, cu["go"], cu["go"] + 0.7))
            ca = 1 - drift
            self.crown(c, x + 40 * drift, top - 60 - 160 * drift, 70 * ck, GOLD, ca)
        if cu["and3"] <= t:
            last = sum(1 for i in range(R, N) if t >= reveal_b(cues, i))
            idx = min(N - 1, R + max(last - 1, 0))
            if last < N - R:
                xx = self._bar_x(idx)
                tri = skia.Path()
                tri.moveTo(xx, Y0 + 18)
                tri.lineTo(xx - 14, Y0 + 42)
                tri.lineTo(xx + 14, Y0 + 42)
                tri.close()
                c.drawPath(tri, paint(INK))
        # broken heart
        hk = back_out(lin(t, cu["go"] - 0.05, cu["go"] + 0.2), 2.0)
        if hk > 0:
            split = ease_out(lin(t, cu["go"] + 0.45, cu["go"] + 0.8))
            hx, hy = 540, 470
            if split <= 0:
                self.heart(c, hx, hy, 150 * hk, PINK)
            else:
                c.save()
                c.translate(-28 * split, 30 * split * split)
                c.rotate(-12 * split)
                self.half_heart(c, hx, hy, 150, PINK, 1 - 0.3 * split, -1)
                c.restore()
                c.save()
                c.translate(28 * split, 30 * split * split)
                c.rotate(12 * split)
                self.half_heart(c, hx, hy, 150, PINK, 1 - 0.3 * split, 1)
                c.restore()
        c.restore()

    def s_outro(self, c, t: float) -> None:
        cu = self.cues.t
        k = ease_out(lin(t, cu["so2"] + 0.1, cu["so2"] + 0.6))
        y = 840 + 1100 * (1 - k)
        self._stack(c, t, 0, y, k)
        # settle to exactly the frame-0 pose
        self.card(c, 540, y, 3.0 * (1 - k), 1.0, 0, k)
