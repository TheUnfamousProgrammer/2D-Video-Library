"""Text boxes for Odd One Out. Flat text: the box is the glyph, plus 2 px of pad."""

from __future__ import annotations

from PIL import ImageFont

from fc_sat.odd_config import OddConfig
from fc_sat.visual import find_font


def text_size(text: str, px: int, max_width: int) -> tuple[float, float]:
    font_px = max(12, int(px))
    font = ImageFont.truetype(find_font(), font_px)
    while font_px >= 12:
        font = ImageFont.truetype(find_font(), font_px)
        if font.getlength(text) <= max_width or font_px == 12:
            break
        font_px = int(font_px * 0.9)
    box = font.getbbox(text)
    return float(box[2] - box[0] + 4), float(box[3] - box[1] + 4)


def _box(cx: float, cy: float, width: float, height: float) -> tuple[float, float, float, float]:
    return (cx - width / 2.0, cy - height / 2.0, cx + width / 2.0, cy + height / 2.0)


def field_box(cfg: OddConfig, scale: float = 1.0) -> tuple[float, float, float, float]:
    return (
        cfg.field_x0 * scale,
        cfg.field_y0 * scale,
        cfg.field_x1 * scale,
        cfg.field_y1 * scale,
    )


def _intersects(a: tuple[float, float, float, float], b: tuple[float, float, float, float], gap: float) -> bool:
    return not (a[2] + gap <= b[0] or b[2] + gap <= a[0] or a[3] + gap <= b[1] or b[3] + gap <= a[1])


def layout_failures(
    boxes: list[tuple[str, tuple[float, float, float, float]]],
    field: tuple[float, float, float, float],
    safe: tuple[float, float, float, float],
    *,
    gap: float = 4.0,
) -> list[str]:
    failures: list[str] = []
    for name, box in boxes:
        if box[0] < safe[0] - 0.5 or box[1] < safe[1] - 0.5 or box[2] > safe[2] + 0.5 or box[3] > safe[3] + 0.5:
            failures.append(f"{name} leaves the safe zone {tuple(round(v, 1) for v in box)}")
        if _intersects(box, field, gap=0.0):
            failures.append(f"{name} overlaps the field")
    for i, (name_a, box_a) in enumerate(boxes):
        for name_b, box_b in boxes[i + 1 :]:
            if _intersects(box_a, box_b, gap):
                failures.append(f"{name_a} overlaps {name_b}")
    return failures


def hud_boxes(
    cfg: OddConfig,
    *,
    caption: str,
    seconds: str,
    outro: bool,
    scale: float = 1.0,
) -> list[tuple[str, tuple[float, float, float, float]]]:
    boxes: list[tuple[str, tuple[float, float, float, float]]] = []
    if outro:
        question = "How far did you get?"
        qw, qh = text_size(question, int(cfg.outro_px * scale), int((cfg.safe_x[1] - cfg.safe_x[0]) * scale))
        boxes.append(("question", _box(cfg.width * 0.5 * scale, cfg.outro_y * scale, qw, qh)))
        cw, ch = text_size(cfg.cta, int(cfg.cta_px * scale), int((cfg.safe_x[1] - cfg.safe_x[0]) * scale))
        boxes.append(("cta", _box(cfg.width * 0.5 * scale, cfg.cta_y * scale, cw, ch)))
        return boxes
    lw, lh = text_size("LEVEL 8", int(cfg.label_px * scale), int((cfg.safe_x[1] - cfg.safe_x[0]) * scale))
    boxes.append(("label", _box(cfg.width * 0.5 * scale, cfg.label_y * scale, lw, lh)))
    if caption:
        cap_w = int((cfg.caption_x[1] - cfg.caption_x[0]) * scale)
        cw, ch = text_size(caption, int(cfg.caption_px * scale), cap_w)
        boxes.append(("caption", _box((cfg.caption_x[0] + cfg.caption_x[1]) * 0.5 * scale, cfg.caption_y * scale, cw, ch)))
    num_w = int((cfg.timer_num_x1 - cfg.timer_num_x0) * scale)
    sw, sh = text_size(seconds, int(cfg.timer_num_px * scale), max(8, num_w))
    num_cx = (cfg.timer_num_x0 + cfg.timer_num_x1) * 0.5 * scale
    num_cy = (cfg.timer_y + cfg.timer_h * 0.5) * scale
    boxes.append(("seconds", _box(num_cx, num_cy, sw, sh)))
    return boxes


def math_layout_failures(cfg: OddConfig, scale: float = 1.0) -> list[str]:
    field = field_box(cfg, scale)
    safe = (cfg.safe_x[0] * scale, cfg.safe_y[0] * scale, cfg.safe_x[1] * scale, cfg.safe_y[1] * scale)
    failures: list[str] = []
    samples = [
        (cfg.hook, "5", False),
        ("", "8", False),
        ("It was a different color", "0", False),
        ("It was tilted", "1", False),
        ("It had no dot", "2", False),
        ("", "", True),
    ]
    for caption, seconds, outro in samples:
        boxes = hud_boxes(cfg, caption=caption, seconds=seconds, outro=outro, scale=scale)
        if len(boxes) > 4:
            failures.append(f"more than 4 text elements: {[name for name, _box in boxes]}")
        failures.extend(layout_failures(boxes, field, safe))
    return failures
