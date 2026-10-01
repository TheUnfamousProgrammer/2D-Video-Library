"""HUD rectangles for Odd One Out.

HUD text stays outside the field. The reveal pill is the one label allowed
inside the field; it is clamped there and offset from the odd disc.
The caption is centered in the band left of the seconds digits. The CTA sits
in the right side of the pip row, because a full-width line at y=1450 would
cover the pips at y=1440.
"""

from __future__ import annotations

from PIL import ImageFont

from fc_sat.odd_config import OddConfig
from fc_sat.visual import find_font


def _box(cx: float, cy: float, w: float, h: float) -> tuple[float, float, float, float]:
    return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)


def text_size(text: str, px: int, font_path: str | None = None) -> tuple[float, float]:
    font = ImageFont.truetype(font_path or find_font(), int(px))
    bbox = font.getbbox(text)
    stroke = max(1, int(px) // 28)
    shadow = max(2, int(px) // 18)
    pad = stroke + shadow + 4
    return float(bbox[2] - bbox[0] + pad * 2), float(bbox[3] - bbox[1] + pad * 2)


def _intersects(a: tuple[float, float, float, float], b: tuple[float, float, float, float], gap: float) -> bool:
    return not (a[2] + gap <= b[0] or b[2] + gap <= a[0] or a[3] + gap <= b[1] or b[3] + gap <= a[1])


def field_box(cfg: OddConfig, scale: float = 1.0) -> tuple[float, float, float, float]:
    return (
        cfg.field_x0 * scale,
        cfg.field_y0 * scale,
        cfg.field_x1 * scale,
        cfg.field_y1 * scale,
    )


def pip_centers(cfg: OddConfig, *, outro: bool, scale: float = 1.0) -> list[tuple[float, float]]:
    count = len(cfg.levels)
    gap = 72.0 * scale
    if outro:
        origin = (cfg.cta_x[0] - 40.0) * scale - gap * (count - 1)
        # Keep the row in the left portion of the bottom band.
        origin = max(cfg.safe_x[0] * scale + 30.0, 200.0 * scale)
    else:
        origin = (cfg.safe_x[0] + cfg.safe_x[1]) * 0.5 * scale - gap * (count - 1) / 2.0
    y = cfg.pip_y * scale
    return [(origin + index * gap, y) for index in range(count)]


def hud_boxes(
    cfg: OddConfig,
    *,
    caption: str,
    seconds: str,
    outro: bool,
    scale: float = 1.0,
    font_path: str | None = None,
) -> list[tuple[str, tuple[float, float, float, float]]]:
    path = font_path or find_font()
    boxes: list[tuple[str, tuple[float, float, float, float]]] = []
    if outro:
        question = "How far did you get?"
        qw, qh = text_size(question, int(round(cfg.outro_px * scale)), path)
        qcx = (cfg.safe_x[0] + cfg.safe_x[1]) * 0.5 * scale
        boxes.append(("outro", _box(qcx, cfg.outro_y * scale, qw, qh)))
        cw, ch = text_size(cfg.cta, int(round(cfg.cta_px * scale)), path)
        ccx = (cfg.cta_x[0] + cfg.cta_x[1]) * 0.5 * scale
        boxes.append(("cta", _box(ccx, cfg.cta_y * scale, cw, ch)))
        for index, (px, py) in enumerate(pip_centers(cfg, outro=True, scale=scale)):
            r = cfg.pip_r * scale
            boxes.append((f"pip{index}", (px - r, py - r, px + r, py + r)))
        return boxes
    label = "LEVEL 1  EASY"
    lw, lh = text_size(label, int(round(cfg.label_px * scale)), path)
    lcx = (cfg.safe_x[0] + cfg.safe_x[1]) * 0.5 * scale
    boxes.append(("label", _box(lcx, cfg.label_y * scale, lw, lh)))
    if caption:
        cw, ch = text_size(caption, int(round(cfg.caption_px * scale)), path)
        ccx = (cfg.caption_x[0] + cfg.caption_x[1]) * 0.5 * scale
        boxes.append(("caption", _box(ccx, cfg.caption_y * scale, cw, ch)))
    bar = (
        cfg.timer_x0 * scale,
        cfg.timer_y * scale,
        cfg.timer_x1 * scale,
        (cfg.timer_y + cfg.timer_h) * scale,
    )
    boxes.append(("timer", bar))
    tw, th = text_size(seconds or "5", int(round(cfg.timer_num_px * scale)), path)
    tcx = (cfg.timer_num_x0 + cfg.timer_num_x1) * 0.5 * scale
    tcy = (cfg.timer_y + cfg.timer_h * 0.5) * scale
    boxes.append(("seconds", _box(tcx, tcy, tw, th)))
    for index, (px, py) in enumerate(pip_centers(cfg, outro=False, scale=scale)):
        r = cfg.pip_r * scale
        boxes.append((f"pip{index}", (px - r, py - r, px + r, py + r)))
    return boxes


def layout_failures(
    boxes: list[tuple[str, tuple[float, float, float, float]]],
    field: tuple[float, float, float, float],
    safe: tuple[float, float, float, float],
    *,
    gap: float = 4.0,
    allow_inside_field: frozenset[str] = frozenset(),
) -> list[str]:
    failures: list[str] = []
    for name, box in boxes:
        if box[0] < safe[0] - 0.5 or box[1] < safe[1] - 0.5 or box[2] > safe[2] + 0.5 or box[3] > safe[3] + 0.5:
            failures.append(f"{name} leaves the safe zone {tuple(round(v, 1) for v in box)}")
        if name not in allow_inside_field and _intersects(box, field, gap=0.0):
            failures.append(f"{name} overlaps the field")
    for i, (name_a, box_a) in enumerate(boxes):
        for name_b, box_b in boxes[i + 1 :]:
            if _intersects(box_a, box_b, gap):
                failures.append(f"{name_a} overlaps {name_b}")
    return failures


def math_layout_failures(cfg: OddConfig, scale: float = 1.0) -> list[str]:
    """Check the captions that actually appear, plus the outro."""
    field = field_box(cfg, scale)
    safe = (cfg.safe_x[0] * scale, cfg.safe_y[0] * scale, cfg.safe_x[1] * scale, cfg.safe_y[1] * scale)
    failures: list[str] = []
    samples = [
        ("", "5"),
        (cfg.hook, "5"),
        ("Pausing won't help", "8"),
        ("Last chance", "2"),
    ]
    for caption, seconds in samples:
        boxes = hud_boxes(cfg, caption=caption, seconds=seconds, outro=False, scale=scale)
        failures.extend(layout_failures(boxes, field, safe))
    failures.extend(layout_failures(hud_boxes(cfg, caption="", seconds="", outro=True, scale=scale), field, safe))
    return failures
