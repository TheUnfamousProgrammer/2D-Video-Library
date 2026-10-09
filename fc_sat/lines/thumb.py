"""Cover frame: your lane is packed and rained on while both neighbours are empty."""

from __future__ import annotations

from pathlib import Path

import skia

from fc_sat.lines.draw import AMBER, BG, CORAL, INK, LANES, LIME, W, H, Painter
from fc_sat.rule37.draw import glow, paint

ROOT = Path(__file__).resolve().parents[2]


def _line(c, face, text, y, size, color, glow_a=0.0, max_w=980.0):
    font = skia.Font(face, size)
    w = font.measureText(text)
    if w > max_w:
        font = skia.Font(face, size * max_w / w)
        w = font.measureText(text)
    blob = skia.TextBlob.MakeFromString(text, font)
    x = (W - w) / 2
    if glow_a:
        g = skia.Paint(AntiAlias=True, Color=skia.Color(*color, int(255 * glow_a)))
        g.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, 26))
        c.drawTextBlob(blob, x, y, g)
    sh = skia.Paint(AntiAlias=True, Color=skia.Color(0, 0, 0, 160))
    sh.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, 12))
    c.drawTextBlob(blob, x, y + 8, sh)
    c.drawTextBlob(blob, x, y, paint(color))


def render(painter: Painter, out: Path) -> None:
    s = skia.Surface(W, H)
    c = s.getCanvas()
    c.drawImage(painter.bg, 0, 0)
    face = skia.Typeface.MakeFromFile(str(ROOT / "assets" / "fonts" / "Montserrat-ExtraBold.ttf"))
    _line(c, face, "WHY YOU ALWAYS PICK", 330, 96, INK)
    _line(c, face, "THE SLOWEST LINE", 480, 150, CORAL, glow_a=0.5)

    reg_y, top, gap = 640, 790, 118
    for li, x in enumerate(LANES):
        stuck = li == 1
        painter.register(c, x, reg_y, 1.0, 1.0, light=1.0 if stuck else 0.6, light_col=CORAL if stuck else LIME, done=0.0 if stuck else 1.0)
        c.drawLine(x - 96, reg_y + 60, x - 96, top + gap * 5, paint(INK, 0.15, 3))
        c.drawLine(x + 96, reg_y + 60, x + 96, top + gap * 5, paint(INK, 0.15, 3))
        if stuck:
            for k in range(5):
                y = top + gap * k
                if k == 4:
                    painter.person(c, x, y, 100, LIME, 1.0, accessory=0)
                else:
                    painter.person(c, x, y, 100, INK, 1.0, accessory=(k * 2 + 1) % 5)
            for b in range(5):
                c.drawRoundRect(skia.Rect(x + 50, top - 18 - b * 26, x + 90, top + 4 - b * 26), 5, 5, paint(AMBER, 0.9, 3))
        else:
            painter.person(c, x, top, 100, INK, 1.0, accessory=li + 1)
            painter.text(c, "0:05", x, top + 170, 54, LIME, 1.0, "center")
    you_y = top + gap * 4
    painter.you_tag(c, LANES[1] + 50, you_y - 10, 1.0, side=True)
    painter.cloud(c, LANES[1] + 150, you_y - 150, 130, 1.0, 0.35, bolt=1.0)
    painter.text(c, "47:00", LANES[1], you_y + 170, 54, CORAL, 1.0, "center")

    # stamp
    c.save()
    c.translate(540, 1560)
    c.rotate(-5)
    sf = skia.Font(face, 92)
    label = "IT'S MATH"
    lw = sf.measureText(label)
    box = skia.Rect(-lw / 2 - 40, -90, lw / 2 + 40, 40)
    c.drawRoundRect(box, 18, 18, glow(LIME, 0.25, 24))
    c.drawRoundRect(box, 18, 18, paint(BG, 0.9))
    c.drawRoundRect(box, 18, 18, paint(LIME, 1.0, 9))
    c.drawTextBlob(skia.TextBlob.MakeFromString(label, sf), -lw / 2, 0, paint(LIME))
    c.restore()
    out.parent.mkdir(parents=True, exist_ok=True)
    s.makeImageSnapshot().save(str(out), skia.kPNG)
    print(f"wrote {out}")
