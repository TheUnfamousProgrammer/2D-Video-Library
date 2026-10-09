"""Cover frame: the locked voice in your head, and the thin one everyone else hears."""

from __future__ import annotations

from pathlib import Path

import skia

from fc_sat.rule37.draw import glow, paint
from fc_sat.voice.draw import BG, CYAN, GOLD, INK, MOUTH, RED, W, H, Painter

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
    _line(c, face, "NOBODY HAS EVER HEARD", 330, 92, INK)
    _line(c, face, "YOUR VOICE", 490, 170, GOLD, glow_a=0.55)

    cx, cy, hs = 400, 960, 600
    painter._locked_head(c, 0.3, 1.0, cx, cy, hs)
    mx = cx + MOUTH[0] * hs
    painter.wave(c, mx + 10, 1020, cy + 0.2 * hs, 26, 10, 0.0, CYAN, 1.0, 5)
    for k, x in enumerate((780, 900, 1010)):
        painter.person(c, x, cy + 0.2 * hs + 170, 110, INK, 0.9, accessory=k + 1)

    c.save()
    c.translate(540, 1640)
    c.rotate(-5)
    sf = skia.Font(face, 86)
    label = "NOT EVEN ONCE"
    lw = sf.measureText(label)
    box = skia.Rect(-lw / 2 - 40, -86, lw / 2 + 40, 36)
    c.drawRoundRect(box, 18, 18, glow(RED, 0.25, 24))
    c.drawRoundRect(box, 18, 18, paint(BG, 0.9))
    c.drawRoundRect(box, 18, 18, paint(RED, 1.0, 9))
    c.drawTextBlob(skia.TextBlob.MakeFromString(label, sf), -lw / 2, 0, paint(RED))
    c.restore()
    out.parent.mkdir(parents=True, exist_ok=True)
    s.makeImageSnapshot().save(str(out), skia.kPNG)
    print(f"wrote {out}")
