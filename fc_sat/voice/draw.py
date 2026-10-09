"""Picture for 'Why does your voice sound so weird on recordings?'. Line-art head, phone, EQ, lineup."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import skia

from fc_sat.rule37.draw import Painter as BasePainter
from fc_sat.rule37.draw import back_out, clamp, ease_in_out, ease_out, glow, lin, mix, paint
from fc_sat.voice.cues import FPS, Cues, rerecord_times

ROOT = Path(__file__).resolve().parents[2]
W, H = 1080, 1920

BG = (0x10, 0x10, 0x1A)
PANEL = (0x1A, 0x1A, 0x29)
INK = (0xF4, 0xF1, 0xEA)
DIM = (0x55, 0x56, 0x6E)
GOLD = (0xFF, 0xC8, 0x57)  # the voice in your head: deep, rich
CYAN = (0x5C, 0xE1, 0xE6)  # the voice through the air / on recordings: thin
RED = (0xFF, 0x5A, 0x6E)

# head profile (facing right), unit coordinates; ~1 unit = head height
HEAD_PTS = [(-0.22, 0.78), (-0.27, 0.45), (-0.47, 0.12), (-0.44, -0.3), (-0.2, -0.58), (0.12, -0.6),
            (0.36, -0.38), (0.41, -0.16), (0.53, 0.0), (0.42, 0.07), (0.45, 0.17), (0.41, 0.24),
            (0.39, 0.36), (0.18, 0.46), (0.14, 0.78)]
EAR = (-0.1, 0.03)
MOUTH = (0.44, 0.2)


def _catmull(pts: list[tuple[float, float]]) -> skia.Path:
    p = skia.Path()
    p.moveTo(*pts[0])
    for i in range(len(pts) - 1):
        p0 = pts[i - 1] if i > 0 else pts[i]
        p1, p2 = pts[i], pts[i + 1]
        p3 = pts[i + 2] if i + 2 < len(pts) else p2
        c1 = (p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6)
        c2 = (p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6)
        p.cubicTo(*c1, *c2, *p2)
    return p


class Painter(BasePainter):
    def __init__(self, cues: Cues, captions=None) -> None:
        self.cues = cues
        self.captions = captions
        self.surface = skia.Surface(W, H)
        self.mono = skia.Typeface.MakeFromFile(str(ROOT / "assets" / "fonts" / "JetBrainsMono-ExtraBold.ttf"))
        self.bg = self._background()
        self.head_unit = _catmull(HEAD_PTS)
        rng = np.random.default_rng(4)
        self.lineup_bars = [rng.uniform(0.25, 1.0, 14) for _ in range(10)]
        self.lineup_stars = [3, 4, 2, 3, 4, 3, 2, 4, 3, 3]
        self.you_card = 6

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
            [skia.Color(0x2A, 0x24, 0x48, 150), skia.Color(0, 0, 0, 0), skia.Color(0, 0, 0, 200)], [0.0, 0.55, 1.0]))
        c.drawRect(skia.Rect(0, 0, W, H), p)
        return s.makeImageSnapshot()

    # --- props ----------------------------------------------------------------------------------

    def head_path(self, cx: float, cy: float, s: float) -> skia.Path:
        p = skia.Path(self.head_unit)
        p.transform(skia.Matrix.Translate(cx, cy).preScale(s, s))
        return p

    def head(self, c, cx, cy, s, alpha=1.0, skull_glow: float = 0.0, jitter: float = 0.0) -> skia.Path:
        dx = jitter * math.sin(jitter * 50.0)
        p = self.head_path(cx + dx, cy, s)
        c.drawPath(p, paint(PANEL, 0.9 * alpha))
        if skull_glow > 0:
            g = paint(GOLD, 0.6 * skull_glow * alpha, 22)
            g.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, 14))
            c.drawPath(p, g)
        c.drawPath(p, paint(mix(INK, GOLD, skull_glow), alpha, s * 0.018))
        ex, ey = cx + dx + EAR[0] * s, cy + EAR[1] * s
        ear = skia.Path()
        ear.addArc(skia.Rect(ex - 0.07 * s, ey - 0.1 * s, ex + 0.07 * s, ey + 0.1 * s), 100, 220)
        c.drawPath(ear, paint(INK, alpha, s * 0.014))
        c.drawCircle(cx + dx + 0.22 * s, cy - 0.17 * s, 0.025 * s, paint(INK, alpha))
        return p

    def wave(self, c, x0, x1, y, amp, cycles, phase, col, alpha, width, clip: skia.Path | None = None) -> None:
        if alpha <= 0:
            return
        p = skia.Path()
        n = 90
        for i in range(n + 1):
            u = i / n
            env = math.sin(math.pi * u) ** 0.6
            yy = y + amp * env * math.sin(2 * math.pi * cycles * u + phase)
            (p.moveTo if i == 0 else p.lineTo)(x0 + (x1 - x0) * u, yy)
        c.save()
        if clip is not None:
            c.clipPath(clip, doAntiAlias=True)
        c.drawPath(p, glow(col, 0.0, 1) if False else paint(col, alpha, width))
        gp = paint(col, 0.35 * alpha, width * 2.2)
        gp.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, width))
        c.drawPath(p, gp)
        c.restore()

    def phone(self, c, cx, cy, s, alpha, t, playing: float, progress: float, shake: float = 0.0) -> None:
        if alpha <= 0:
            return
        c.save()
        c.translate(cx + shake * math.sin(t * 90), cy)
        c.scale(s, s)
        w, h = 480, 840
        r = skia.Rect(-w / 2, -h / 2, w / 2, h / 2)
        c.drawRoundRect(r, 60, 60, paint(PANEL, alpha))
        c.drawRoundRect(r, 60, 60, paint(INK, alpha, 7))
        c.drawRoundRect(skia.Rect(-60, -h / 2 + 22, 60, -h / 2 + 46), 12, 12, paint(INK, 0.5 * alpha))
        # chat bubble with a voice note
        b = skia.Rect(-200, -70, 200, 70)
        c.drawRoundRect(b, 40, 40, paint(mix(PANEL, CYAN, 0.18), alpha))
        c.drawRoundRect(b, 40, 40, paint(CYAN, alpha, 4))
        c.drawCircle(-140, 0, 38, paint(CYAN, alpha))
        tri = skia.Path()
        tri.moveTo(-152, -18)
        tri.lineTo(-152, 18)
        tri.lineTo(-122, 0)
        tri.close()
        c.drawPath(tri, paint(BG, alpha))
        n = 16
        for i in range(n):
            x = -80 + i * 16
            hh = (10 + 34 * abs(math.sin(i * 1.7 + 0.4))) * (0.55 + 0.45 * playing * abs(math.sin(t * 9 + i)))
            col = CYAN if i / n <= progress else DIM
            c.drawLine(x, -hh / 2, x, hh / 2, paint(col, alpha, 7))
        self.text(c, "0:07", 175, 58, 26, INK, 0.6 * alpha, "right")
        c.restore()

    def cringe(self, c, x, y, s, alpha) -> None:
        if alpha <= 0:
            return
        c.drawCircle(x, y, s, paint(GOLD, alpha))
        c.drawCircle(x, y, s, paint(BG, alpha, s * 0.06))
        for side in (-1, 1):  # squeezed > < eyes
            ex = x + side * 0.38 * s
            p = skia.Path()
            p.moveTo(ex - side * 0.16 * s, y - 0.32 * s)
            p.lineTo(ex + side * 0.08 * s, y - 0.2 * s)
            p.lineTo(ex - side * 0.16 * s, y - 0.08 * s)
            c.drawPath(p, paint(BG, alpha, s * 0.08))
        teeth = skia.Rect(x - 0.5 * s, y + 0.12 * s, x + 0.5 * s, y + 0.45 * s)
        c.drawRoundRect(teeth, 0.12 * s, 0.12 * s, paint(INK, alpha))
        c.drawRoundRect(teeth, 0.12 * s, 0.12 * s, paint(BG, alpha, s * 0.06))
        c.drawLine(teeth.left(), teeth.centerY(), teeth.right(), teeth.centerY(), paint(BG, alpha, s * 0.05))
        for k in range(1, 5):
            xx = teeth.left() + k * teeth.width() / 5
            c.drawLine(xx, teeth.top(), xx, teeth.bottom(), paint(BG, alpha, s * 0.04))

    def star(self, c, x, y, r, col, alpha, filled: bool) -> None:
        p = skia.Path()
        for k in range(10):
            ang = -math.pi / 2 + k * math.pi / 5
            rr = r if k % 2 == 0 else r * 0.45
            (p.moveTo if k == 0 else p.lineTo)(x + rr * math.cos(ang), y + rr * math.sin(ang))
        p.close()
        if filled:
            c.drawPath(p, paint(col, alpha))
        c.drawPath(p, paint(col, alpha, max(2.0, r * 0.12)))

    def toggle(self, c, x, y, s, on: float, alpha, label: str = "FILTER") -> None:
        if alpha <= 0:
            return
        w, h = 200 * s, 100 * s
        col = mix(DIM, GOLD, on)
        c.drawRoundRect(skia.Rect(x - w / 2, y - h / 2, x + w / 2, y + h / 2), h / 2, h / 2, paint(col, 0.3 * alpha))
        c.drawRoundRect(skia.Rect(x - w / 2, y - h / 2, x + w / 2, y + h / 2), h / 2, h / 2, paint(col, alpha, 6 * s))
        kx = x - w / 2 + h / 2 + (w - h) * on
        c.drawCircle(kx, y, h * 0.38, paint(INK, alpha))
        self.text(c, label, x, y - h / 2 - 22 * s, 40 * s, col, alpha, "center", spacing=4 * s)
        self.text(c, "ON" if on > 0.5 else "OFF", x, y + h / 2 + 52 * s, 38 * s, col, alpha, "center")

    def eq(self, c, x0, y0, w, h, lows: float, alpha, t, col, label: str) -> None:
        """Twelve EQ bars; ``lows`` scales the bass end (1 = boosted, 0 = cut)."""
        if alpha <= 0:
            return
        c.drawRoundRect(skia.Rect(x0, y0, x0 + w, y0 + h), 26, 26, paint(PANEL, 0.85 * alpha))
        c.drawRoundRect(skia.Rect(x0, y0, x0 + w, y0 + h), 26, 26, paint(col, 0.6 * alpha, 4))
        self.text(c, label, x0 + 28, y0 + 52, 32, col, alpha, "left", spacing=3)
        n = 12
        base = y0 + h - 34
        for i in range(n):
            x = x0 + 50 + i * (w - 100) / (n - 1)
            u = i / (n - 1)
            shape = 0.35 + 0.25 * math.sin(i * 1.3 + 0.5) ** 2
            bass = max(0.0, 1 - u * 2.6)
            hh = (h - 110) * clamp(shape * (1 - bass) + bass * (0.1 + 0.9 * lows) + bass * 0.0)
            hh *= 0.9 + 0.1 * math.sin(t * 8 + i * 1.9)
            c.drawRoundRect(skia.Rect(x - 13, base - hh, x + 13, base), 6, 6, paint(col if bass < 0.01 else mix(col, GOLD if col == GOLD else col, 1), alpha))
        self.text(c, "BASS", x0 + 50, base + 28, 22, INK, 0.45 * alpha, "left", spacing=2)
        self.text(c, "TREBLE", x0 + w - 50, base + 28, 22, INK, 0.45 * alpha, "right", spacing=2)

    # --- frame ----------------------------------------------------------------------------------

    def render(self, frame: int) -> np.ndarray:
        t = frame / FPS
        cu = self.cues.t
        c = self.surface.getCanvas()
        c.drawImage(self.bg, 0, 0)
        sx, sy = self._shake(t)
        c.save()
        c.translate(sx, sy)
        z = self._zoom(t)
        if z != 1.0:
            c.translate(540, 820)
            c.scale(z, z)
            c.translate(-540, -820)
        scenes = [
            (0.0, cu["when"] + 0.1, self.s_hook),
            (cu["when"] - 0.2, cu["recording"] + 0.2, self.s_paths),
            (cu["bones"] - 0.2, cu["plot"] + 0.1, self.s_eq),
            (cu["plot"], cu["so3"] + 0.25, self.s_study),
            (cu["so3"], cu["so4"] + 0.25, self.s_unfiltered),
            (cu["so4"], self.cues.end + 1, self.s_outro),
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
        z0 = 1.12
        if t < 1.6:
            return z0 + (1 - z0) * ease_in_out(t / 1.6)
        end = self.cues.end
        if t > end - 0.9:
            return 1 + (z0 - 1) * ease_in_out(lin(t, end - 0.9, end - 1 / FPS))
        return 1.0

    def _shake(self, t: float) -> tuple[float, float]:
        cu = self.cues.t
        out = [0.0, 0.0]
        for start, amp in ((cu["once"], 10.0), (cu["yeah"], 14.0), (cu["skull"], 12.0), (cu["plot"], 14.0),
                           (cu["more"], 8.0)):
            d = t - start
            if 0 <= d < 0.35:
                k = amp * (1 - d / 0.35) ** 2
                out[0] += k * math.sin(d * 97)
                out[1] += k * math.cos(d * 71)
        return out[0], out[1]

    def counters(self, c, t: float) -> None:
        cu, cues, end = self.cues.t, self.cues, self.cues.end
        if t < cu["voice2"] or t >= end - 0.55:
            a = 1 - lin(t, cu["voice2"] - 0.2, cu["voice2"]) if t < cu["voice2"] else lin(t, end - 0.55, end - 0.35)
            pop = lin(t, cu["once"], cu["once"] + 0.3) if cu["once"] <= t < cu["once"] + 0.3 else 0
            self.counter(c, "left", "HEARD IT", "0", a, GOLD, pop)
        elif t < cu["when"]:
            n = sum(1 for tt in rerecord_times(cues, "hook") if t >= tt)
            if n:
                self.counter(c, "left", "RE-RECORDS", f"{n}", 1 - lin(t, cu["when"] - 0.2, cu["when"]), RED)
        elif t < cu["bones"]:
            n = (1 if t >= cu["through1"] else 0) + (1 if t >= cu["through2"] else 0)
            if n:
                self.counter(c, "left", "PATHS", f"{n}", 1 - lin(t, cu["bones"] - 0.2, cu["bones"]), GOLD if n == 2 else CYAN)
        elif t < cu["plot"]:
            low = t >= cu["no"]
            a = lin(t, cu["deep"], cu["deep"] + 0.2) * (1 - lin(t, cu["plot"] - 0.2, cu["plot"]))
            pop = lin(t, cu["no"], cu["no"] + 0.3) if cu["no"] <= t < cu["no"] + 0.3 else 0
            self.counter(c, "left", "BASS", "LOW" if low else "MAX", a, CYAN if low else GOLD, pop)
        elif t < cu["so3"]:
            a = lin(t, cu["study"], cu["study"] + 0.2) * (1 - lin(t, cu["so3"] - 0.2, cu["so3"]))
            self.counter(c, "left", "STUDY", "2013", a, INK)
        elif t >= cu["how"]:
            n = sum(1 for tt in rerecord_times(cues, "outro") if t >= tt)
            if n:
                self.counter(c, "left", "RE-RECORDS", f"{n}", 1 - lin(t, end - 0.75, end - 0.55), RED)

    # --- scenes ---------------------------------------------------------------------------------

    def _locked_head(self, c, t: float, alpha: float, cx: float = 440, cy: float = 720, s: float = 580) -> None:
        hp = self.head(c, cx, cy, s, alpha)
        mx, ex = cx + MOUTH[0] * s, cx + EAR[0] * s
        y = cy + 0.06 * s
        self.wave(c, ex + 0.05 * s, mx - 0.04 * s, y, 0.09 * s, 2.2, t * 5.0, GOLD, alpha, 12, hp)
        # padlock over the voice
        lx, ly = cx - 0.08 * s, cy - 0.3 * s
        wob = 4 * math.sin(t * 3.0)
        c.save()
        c.translate(lx, ly)
        c.rotate(wob)
        c.drawRoundRect(skia.Rect(-46, -6, 46, 64), 12, 12, paint(GOLD, alpha))
        arc = skia.Path()
        arc.addArc(skia.Rect(-30, -50, 30, 10), 180, 180)
        c.drawPath(arc, paint(GOLD, alpha, 11, skia.Paint.kButt_Cap))
        c.drawLine(-30, -20, -30, -6, paint(GOLD, alpha, 11, skia.Paint.kButt_Cap))
        c.drawLine(30, -20, 30, -6, paint(GOLD, alpha, 11, skia.Paint.kButt_Cap))
        c.drawCircle(0, 24, 9, paint(BG, alpha))
        c.drawLine(0, 28, 0, 46, paint(BG, alpha, 7))
        c.restore()

    def s_hook(self, c, t: float) -> None:
        cu = self.cues.t
        a_head = 1 - lin(t, cu["voice2"] - 0.1, cu["voice2"] + 0.2)
        if a_head > 0:
            self._locked_head(c, t, a_head)
            # outside: the thin air version reaching the crowd
            ck = lin(t, cu["nobody"], cu["nobody"] + 0.3)
            if ck > 0:
                mx = 440 + MOUTH[0] * 580
                self.wave(c, mx + 10, 1010, 720 + 0.2 * 580, 22, 9, -t * 14, CYAN, ck * a_head, 4)
            # mum, friends, ex: each gets an X
            for k, (key, acc, x) in enumerate((("mum", 4, 230), ("friends", 2, 540), ("ex", 3, 850))):
                pk = back_out(lin(t, cu[key] - 0.12, cu[key] + 0.12), 2.0)
                if pk <= 0:
                    continue
                y = 1300
                if key == "friends":
                    self.person(c, x - 50, y, 120 * pk, INK, a_head, accessory=2)
                    self.person(c, x + 50, y, 120 * pk, INK, a_head, accessory=1)
                else:
                    self.person(c, x, y, 130 * pk, INK, a_head, accessory=acc)
                sk = lin(t, cu[key] + 0.05, cu[key] + 0.2)
                if sk > 0:
                    big = 1.35 if key == "ex" else 1.0
                    sz = 150 * big * (1.5 - 0.5 * back_out(sk, 2.0))
                    self.cross(c, x, y - 20, sz, RED, a_head * clamp(sk * 2), 16 * big)
                    if key == "ex":
                        self.heart(c, x + 90, y - 120, 60 * clamp(sk * 2), RED, a_head)
        # the voice note
        if t >= cu["voice2"] - 0.1:
            k = ease_out(lin(t, cu["voice2"] - 0.1, cu["voice2"] + 0.3))
            out = 1 - lin(t, cu["when"] - 0.15, cu["when"] + 0.1)
            frozen = t >= cu["yeah"]
            tt = cu["yeah"] if frozen else t
            prog = lin(tt, cu["notes1"] - 0.2, cu["yeah"])
            shake = 10.0 if frozen and t < cu["yeah"] + 0.5 else 0.0
            self.phone(c, 540, 820 + 900 * (1 - k), 1.0, out, tt, 0.0 if frozen else 1.0, prog, shake)
            fk = back_out(lin(t, cu["yeah"], cu["yeah"] + 0.25), 2.2)
            if fk > 0:
                self.cringe(c, 830, 470, 110 * fk, out)
                dk = 0.75 + 0.25 * math.sin((t - cu["yeah"]) * 12)
                r = skia.Rect(540 - 150, 1080, 540 + 150, 1170)
                c.drawRoundRect(r, 45, 45, paint(RED, out * dk))
                self.text(c, "DELETE?", 540, 1142, 46, INK, out, "center", spacing=2)

    def s_paths(self, c, t: float) -> None:
        cu = self.cues.t
        a = lin(t, cu["when"] - 0.2, cu["when"] + 0.15) * (1 - lin(t, cu["bones"] - 0.2, cu["bones"] + 0.15))
        if a <= 0:
            return
        cx, cy, s = 430, 860, 620
        skull = lin(t, cu["skull"], cu["skull"] + 0.15) * (1 - 0.4 * lin(t, cu["skull"] + 0.4, cu["skull"] + 1.0))
        jit = 6.0 * (1 - lin(t, cu["skull"], cu["skull"] + 0.5)) if t >= cu["skull"] else 0.0
        hp = self.head(c, cx, cy, s, a, skull_glow=skull, jitter=jit)
        mx, my = cx + MOUTH[0] * s, cy + MOUTH[1] * s
        ex, ey = cx + EAR[0] * s, cy + EAR[1] * s
        # sound ripples from the mouth while "you talk"
        rk = lin(t, cu["when"], cu["when"] + 0.2) * (1 - lin(t, cu["through1"] + 0.4, cu["through1"] + 0.8))
        if rk > 0:
            for j in range(3):
                ph = ((t - cu["when"]) * 1.6 + j / 3) % 1.0
                rr = 30 + 160 * ph
                arc = skia.Path()
                arc.addArc(skia.Rect(mx - rr, my - rr, mx + rr, my + rr), -40, 80)
                c.drawPath(arc, paint(INK, a * rk * (1 - ph), 6))
        fk = back_out(lin(t, cu["ways"] - 0.1, cu["ways"] + 0.2), 2.0) * (1 - lin(t, cu["through1"] + 0.3, cu["through1"] + 0.6))
        if fk > 0:
            self.text(c, "2", 880, 560, 220 * fk, GOLD, a, "center")
        # air path: out of the mouth, around in front of the face, back to the ear
        air = skia.Path()
        air.moveTo(mx + 10, my)
        air.cubicTo(mx + 260, my + 40, mx + 260, cy - 0.85 * s, cx, cy - 0.75 * s)
        air.cubicTo(cx - 0.55 * s, cy - 0.7 * s, ex - 0.35 * s, ey - 0.1 * s, ex - 0.02 * s, ey)
        meas = skia.PathMeasure(air, False)
        L = meas.getLength()
        ak = ease_in_out(lin(t, cu["through1"] - 0.1, cu["through1"] + 0.6))
        if ak > 0:
            seg = skia.Path()
            meas.getSegment(0, L * ak, seg, True)
            dp = paint(CYAN, a, 7)
            dp.setPathEffect(skia.DashPathEffect.Make([18, 14], -t * 80))
            c.drawPath(seg, dp)
            for k in range(5):
                d = ((t * 380 + k * L / 5) % L)
                if d <= L * ak:
                    pos, _ = meas.getPosTan(d)
                    c.drawCircle(pos.x(), pos.y(), 9, paint(CYAN, a))
            # someone else hears the same air version
            lk = back_out(lin(t, cu["everyone"], cu["everyone"] + 0.25), 2.0)
            if lk > 0:
                self.person(c, 930, 1200, 150 * lk, INK, a, accessory=1)
                self.wave(c, mx + 30, 860, 1130, 16, 7, -t * 14, CYAN, a * lk, 4)
        # bone path: jaw to inner ear, through the skull
        bk = ease_in_out(lin(t, cu["through2"], cu["skull"] + 0.3))
        if bk > 0:
            bone = skia.Path()
            bone.moveTo(cx + 0.3 * s, cy + 0.32 * s)
            bone.quadTo(cx + 0.05 * s, cy + 0.3 * s, ex + 0.03 * s, ey + 0.02 * s)
            bm = skia.PathMeasure(bone, False)
            seg = skia.Path()
            bm.getSegment(0, bm.getLength() * bk, seg, True)
            gp = paint(GOLD, a, 16)
            c.drawPath(seg, glow(GOLD, 0, 1) if False else gp)
            g2 = paint(GOLD, 0.5 * a, 34)
            g2.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, 14))
            c.drawPath(seg, g2)
            if bk >= 1:
                ring = (t - cu["skull"]) * 2.2 % 1.0
                c.drawCircle(ex, ey, 20 + 70 * ring, paint(GOLD, a * (1 - ring), 5))

    def s_eq(self, c, t: float) -> None:
        cu = self.cues.t
        a = lin(t, cu["bones"] - 0.2, cu["bones"] + 0.2) * (1 - lin(t, cu["plot"] - 0.15, cu["plot"] + 0.1))
        if a <= 0:
            return
        split = ease_in_out(lin(t, cu["recording"] - 0.1, cu["recording"] + 0.4))
        boost = ease_out(lin(t, cu["deep"], cu["frequencies"] + 0.3))
        boost = max(boost, 1.0 if t >= cu["bass1"] else 0) * (1 + 0.25 * math.sin(max(0.0, t - cu["bass1"]) * 18) * (1 - lin(t, cu["bass1"], cu["bass1"] + 0.8)) if t >= cu["bass1"] else boost)
        # small head icon with gold inner wave
        hx, hy, hs = 300, 560 - 120 * split, 300
        hp = self.head(c, hx, hy, hs, a * (1 - 0.5 * split))
        self.wave(c, hx + EAR[0] * hs + 10, hx + MOUTH[0] * hs - 10, hy + 0.06 * hs, 0.08 * hs, 2.2, t * 5, GOLD, a * (1 - 0.5 * split), 8, hp)
        if split <= 0:
            self.eq(c, 110, 820, 860, 420, boost, a, t, GOLD, "IN YOUR HEAD")
        else:
            y1 = 820 - 210 * split
            self.eq(c, 110, y1, 860, 380, 1.0, a, t, GOLD, "IN YOUR HEAD")
            cut = ease_in_out(lin(t, cu["no"], cu["bass2"] + 0.4))
            y2 = y1 + 420
            self.eq(c, 110, y2, 860, 380, 1.0 - cut, a * split, t, CYAN, "RECORDING")
            if t >= cu["no"]:
                sk = lin(t, cu["no"], cu["no"] + 0.2)
                c.drawLine(140, y2 + 150, 140 + 340 * sk, y2 + 360, paint(RED, a, 10))
        # thin vs deep waves and the nose gag
        if t >= cu["thinner"] - 0.1:
            k = lin(t, cu["thinner"] - 0.1, cu["thinner"] + 0.2) * (1 - lin(t, cu["filter"] - 0.1, cu["filter"] + 0.2))
            nk = back_out(lin(t, cu["nasal"], cu["nasal"] + 0.3), 2.4)
            if nk > 0:
                nx, ny = 780, 470
                nose = skia.Path()
                nose.moveTo(nx - 10, ny - 70 * nk)
                nose.quadTo(nx + 10, ny - 10, nx + 50 * nk, ny + 30 * nk)
                nose.quadTo(nx + 10, ny + 50 * nk, nx - 20, ny + 30 * nk)
                c.drawPath(nose, paint(CYAN, a * k, 9))
                wig = 6 * math.sin((t - cu["nasal"]) * 30) * (1 - lin(t, cu["nasal"], cu["nasal"] + 0.8))
                self.text(c, "~", nx + 70 + wig, ny - 20, 70, CYAN, a * k, "center")
        fk = back_out(lin(t, cu["filter"] - 0.1, cu["filter"] + 0.2), 2.0)
        if fk > 0:
            self.toggle(c, 780, 470, 1.0 * fk, 1.0, a)
            if t >= cu["life"]:
                d = t - cu["life"]
                for j in range(6):
                    ang = j * math.pi / 3 + d * 2
                    self.sparkle(c, 780 + 150 * math.cos(ang), 470 + 90 * math.sin(ang), 12 * (1 - clamp(d / 0.8)), GOLD, a)

    def s_study(self, c, t: float) -> None:
        cu = self.cues.t
        a = lin(t, cu["plot"], cu["plot"] + 0.25) * (1 - lin(t, cu["so3"] - 0.1, cu["so3"] + 0.25))
        if a <= 0:
            return
        # flash on "plot twist"
        fl = 1 - lin(t, cu["plot"], cu["plot"] + 0.35)
        if fl > 0:
            c.drawRect(skia.Rect(0, 0, W, H), paint(INK, 0.18 * fl))
        ck = back_out(lin(t, cu["plot"] + 0.1, cu["plot"] + 0.4), 1.8) * (1 - ease_in_out(lin(t, cu["secretly"] - 0.2, cu["secretly"] + 0.2)))
        if ck > 0:
            c.save()
            c.translate(540, 820)
            c.rotate(-4)
            c.scale(ck, ck)
            board = skia.Rect(-230, -300, 230, 300)
            c.drawRoundRect(board, 30, 30, paint(PANEL, a))
            c.drawRoundRect(board, 30, 30, paint(INK, a, 7))
            c.drawRoundRect(skia.Rect(-90, -330, 90, -270), 16, 16, paint(GOLD, a))
            self.text(c, "STUDY", 0, -180, 56, INK, a, "center", spacing=6)
            for j in range(5):
                y = -90 + j * 70
                c.drawCircle(-160, y, 12, paint(CYAN, a))
                w = (0.8, 0.55, 0.7, 0.45, 0.65)[j]
                c.drawLine(-120, y, -120 + 280 * w, y, paint(DIM, a, 12))
            qk = back_out(lin(t, cu["researchers"], cu["researchers"] + 0.25), 2.0)
            if qk > 0:
                c.drawCircle(170, 230, 58 * qk, paint(GOLD, a))
                self.text(c, "?", 170, 262, 90 * qk, BG, a, "center")
            c.restore()
        focus = ease_in_out(lin(t, cu["rated"] - 0.1, cu["rated"] + 0.4))
        for k in range(10):
            col_i, row = k % 2, k // 2
            st = cu["secretly"] + 0.1 * k
            pk = back_out(lin(t, st, st + 0.25), 1.8)
            if pk <= 0:
                continue
            x = 300 + col_i * 480
            y = 470 + row * 150
            is_you = k == self.you_card
            al = a * (1 - focus)
            if al <= 0.01:
                continue
            c.save()
            c.translate(x, y)
            c.scale(pk, pk)
            r = skia.Rect(-215, -58, 215, 58)
            c.drawRoundRect(r, 30, 30, paint(PANEL, al))
            c.drawRoundRect(r, 30, 30, paint(CYAN, al, 4))
            c.drawCircle(-170, 0, 28, paint(CYAN, al))
            for i, v in enumerate(self.lineup_bars[k]):
                xx = -120 + i * 13
                hh = 60 * v * (0.7 + 0.3 * abs(math.sin(t * 6 + i + k)))
                c.drawLine(xx, -hh / 2, xx, hh / 2, paint(CYAN, al, 6))
            # stars from other listeners
            sk = lin(t, cu["most"] + 0.05 * k, cu["most"] + 0.05 * k + 0.3)
            if sk > 0 and not is_you:
                for j in range(5):
                    self.star(c, 95 + j * 26, 0, 11, GOLD, al * sk, j < self.lineup_stars[k])
            c.restore()
            if is_you:
                q = back_out(lin(t, cu["lineup"], cu["lineup"] + 0.25), 2.0) * (1 - lin(t, cu["more"] - 0.05, cu["more"] + 0.1))
                if q > 0:
                    c.drawCircle(x + 185, y - 50, 34 * q, paint(GOLD, a))
                    self.text(c, "?", x + 185, y - 30, 52 * q, BG, a, "center")
        # the reveal: you rated yourself higher than others rated you
        if focus > 0:
            k = self.you_card
            x0 = 300 + (k % 2) * 480
            y0 = 470 + (k // 2) * 150
            cx, cy = x0 + (540 - x0) * focus, y0 + (700 - y0) * focus
            c.save()
            c.translate(cx, cy)
            sc = 1 + 0.25 * focus
            c.scale(sc, sc)
            r = skia.Rect(-215, -58, 215, 58)
            c.drawRoundRect(r, 30, 30, glow(GOLD, 0.4 * focus * a, 20))
            c.drawRoundRect(r, 30, 30, paint(PANEL, a))
            c.drawRoundRect(r, 30, 30, paint(GOLD, a, 5))
            c.drawCircle(-170, 0, 28, paint(GOLD, a))
            for i, v in enumerate(self.lineup_bars[k]):
                xx = -120 + i * 13
                hh = 60 * v
                c.drawLine(xx, -hh / 2, xx, hh / 2, paint(GOLD, a, 6))
            rv = back_out(lin(t, cu["more"], cu["more"] + 0.25), 2.0)
            if rv > 0:
                self.text(c, "YOU", 150, 16, 46 * rv, GOLD, a, "center", spacing=3)
            c.restore()
            rows = (("YOU RATED IT", 5, GOLD, cu["rated"] + 0.3), ("OTHERS RATED IT", 3, CYAN, cu["more"] + 0.5))
            for j, (lab, n, col, st) in enumerate(rows):
                rk = lin(t, st, st + 0.5)
                if rk <= 0:
                    continue
                y = 920 + j * 150
                self.text(c, lab, 540, y, 34, col, a * clamp(rk * 3), "center", spacing=3)
                for s_i in range(5):
                    on = s_i < n and rk >= (s_i + 1) / 5
                    self.star(c, 380 + s_i * 80, y + 60, 30, col, a * clamp(rk * 3), on)

    def s_unfiltered(self, c, t: float) -> None:
        cu = self.cues.t
        a = lin(t, cu["so3"], cu["so3"] + 0.25) * (1 - lin(t, cu["so4"] - 0.1, cu["so4"] + 0.25))
        if a <= 0:
            return
        off = ease_in_out(lin(t, cu["unfiltered"], cu["unfiltered"] + 0.25))
        cx, cy, s = 470, 900, 600
        hp = self.head(c, cx, cy, s, a)
        ex, mx = cx + EAR[0] * s, cx + MOUTH[0] * s
        y = cy + 0.06 * s
        self.wave(c, ex + 0.05 * s, mx - 0.04 * s, y, 0.09 * s * (1 - off) + 0.03 * s * off, 2.2 + 6.8 * off, t * 6,
                  mix(GOLD, CYAN, off), a, 12 - 7 * off, hp)
        hk = back_out(lin(t, cu["so3"] + 0.1, cu["so3"] + 0.4), 2.0) * (1 - off)
        if hk > 0:
            self.heart(c, cx + 0.1 * s, cy - 0.85 * s, 110 * hk, GOLD, a)
        self.toggle(c, 820, 470, 0.9, 1.0 - off, a)
        if off > 0:
            self.cringe(c, cx + 0.1 * s, cy - 0.85 * s, 90 * off, a)

    def s_outro(self, c, t: float) -> None:
        cu, cues = self.cues.t, self.cues
        end = cues.end
        a_phone = lin(t, cu["so4"], cu["so4"] + 0.3) * (1 - lin(t, end - 0.8, end - 0.55))
        if a_phone > 0:
            n = sum(1 for tt in rerecord_times(cues, "outro") if t >= tt)
            last = max([tt for tt in rerecord_times(cues, "outro") if t >= tt], default=cu["so4"])
            prog = lin(t, last, last + 0.4)
            self.phone(c, 540, 820, 1.0, a_phone, t, 1.0, prog)
            r = skia.Rect(540 - 170, 1080, 540 + 170, 1170)
            press = 1 - 0.08 * (1 - lin(t, last, last + 0.12)) if n else 1
            c.save()
            c.translate(540, 1125)
            c.scale(press, press)
            c.translate(-540, -1125)
            c.drawRoundRect(r, 45, 45, paint(RED, a_phone))
            self.text(c, "RE-RECORD", 540, 1142, 42, INK, a_phone, "center", spacing=2)
            c.restore()
        a_head = lin(t, end - 0.8, end - 0.55)
        if a_head > 0:
            self._locked_head(c, 0.0 if t >= end - 0.55 else t - (end - 0.55), a_head)
