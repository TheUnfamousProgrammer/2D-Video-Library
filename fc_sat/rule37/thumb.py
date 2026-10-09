"""Cover frame: 'How to find your better half' with one half of the heart missing."""

from __future__ import annotations

from pathlib import Path

import skia

from fc_sat.rule37.draw import BG, DIM, GOLD, INK, PINK, W, H, Painter, glow, paint

ROOT = Path(__file__).resolve().parents[2]


def _line(c, face, text, y, size, color, max_w=980.0, glow_a=0.0):
    font = skia.Font(face, size)
    w = font.measureText(text)
    if w > max_w:
        font = skia.Font(face, size * max_w / w)
        w = font.measureText(text)
    blob = skia.TextBlob.MakeFromString(text, font)
    x = (W - w) / 2
    if glow_a:
        g = skia.Paint(AntiAlias=True, Color=skia.Color(*color, int(255 * glow_a)))
        g.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, 28))
        c.drawTextBlob(blob, x, y, g)
    sh = skia.Paint(AntiAlias=True, Color=skia.Color(0, 0, 0, 150))
    sh.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, 12))
    c.drawTextBlob(blob, x, y + 8, sh)
    c.drawTextBlob(blob, x, y, paint(color))


def render(painter: Painter, out: Path) -> None:
    s = skia.Surface(W, H)
    c = s.getCanvas()
    c.drawImage(painter.bg, 0, 0)
    face = skia.Typeface.MakeFromFile(str(ROOT / "assets" / "fonts" / "Montserrat-ExtraBold.ttf"))

    _line(c, face, "HOW TO FIND YOUR", 400, 104, INK)
    _line(c, face, "BETTER HALF", 560, 168, PINK, glow_a=0.55)

    # the heart: one half found, the other half a dashed '?'
    hx, hy, hs = 540, 900, 560
    c.drawCircle(hx - 90, hy, 300, glow(PINK, 0.28, 90))
    c.save()
    c.translate(-26, 0)
    painter.half_heart(c, hx, hy, hs, PINK, 1.0, -1)
    c.restore()
    right = skia.Path()
    x, y, s_ = hx + 26, hy, hs
    right.moveTo(x, y + 0.36 * s_)
    right.cubicTo(x + 0.6 * s_, y - 0.02 * s_, x + 0.36 * s_, y - 0.56 * s_, x, y - 0.24 * s_)
    right.lineTo(x + 0.06 * s_, y - 0.08 * s_)
    right.lineTo(x - 0.05 * s_, y + 0.06 * s_)
    right.lineTo(x + 0.04 * s_, y + 0.2 * s_)
    right.close()
    dash = paint(GOLD, 1.0, 10, skia.Paint.kRound_Cap)
    dash.setPathEffect(skia.DashPathEffect.Make([26, 20], 0))
    c.drawPath(right, paint(GOLD, 0.08))
    c.drawPath(right, dash)
    qf = skia.Font(face, 190)
    qw = qf.measureText("?")
    c.drawTextBlob(skia.TextBlob.MakeFromString("?", qf), x + 0.24 * s_ - qw / 2, y + 40, glow(GOLD, 0.6, 18))
    c.drawTextBlob(skia.TextBlob.MakeFromString("?", qf), x + 0.24 * s_ - qw / 2, y + 40, paint(GOLD))

    # stamp
    c.save()
    c.translate(540, 1300)
    c.rotate(-5)
    sf = skia.Font(face, 84)
    label = "REJECT 37% FIRST"
    lw = sf.measureText(label)
    box = skia.Rect(-lw / 2 - 36, -82, lw / 2 + 36, 38)
    c.drawRoundRect(box, 18, 18, paint(PINK, 0.14))
    c.drawRoundRect(box, 18, 18, paint(PINK, 1.0, 9))
    c.drawTextBlob(skia.TextBlob.MakeFromString(label, sf), -lw / 2, 0, paint(PINK))
    c.restore()

    # row of mini cards: the first ones get the X, then the gold one
    for k in range(5):
        cx = 170 + k * 185
        golden = k == 3
        painter.card(c, cx, 1560, (-6 + 3 * k) if not golden else 0, 0.27 if not golden else 0.33, k + 2,
                     0.55 if k < 3 else (1.0 if golden else 0.35), golden=golden, stamp=1.0 if k < 3 else 0.0, t=0.4)
    out.parent.mkdir(parents=True, exist_ok=True)
    s.makeImageSnapshot().save(str(out), skia.kPNG)
    print(f"wrote {out}")
