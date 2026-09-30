"""Shared drawing helpers used by the bounce renderer and Melody Hop.

Bloom is the bounce recipe: 4x area downscale, Gaussian sigmas 1.0 and 2.2
summed, then ``1 - exp(-x/90)`` with strength taken from the caller.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def find_font() -> str:
    bundled = Path(__file__).resolve().parents[1] / "assets" / "Montserrat-ExtraBold.ttf"
    if bundled.exists():
        return str(bundled)
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
        Path("/Library/Fonts/Arial Bold.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/System/Library/Fonts/Helvetica.ttc"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        Path("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    raise SystemExit(
        "No font found. Place an OFL font at assets/Montserrat-ExtraBold.ttf "
        "or install Arial Bold (Windows/macOS) or DejaVu Sans Bold (Linux)."
    )


def vignette_bgr(width: int, height: int) -> np.ndarray:
    """Near-black #07070B with a soft radial vignette. Deterministic."""
    yy, xx = np.mgrid[0:height, 0:width]
    nx = (xx - (width - 1) / 2.0) / (width / 2.0)
    ny = (yy - (height - 1) / 2.0) / (height / 2.0)
    radius = np.sqrt(nx * nx + ny * ny)
    shade = np.clip(1.0 - 0.42 * np.power(radius, 1.55), 0.35, 1.0)
    base = np.array([11.0, 7.0, 7.0], dtype=np.float32)  # BGR of #07070B
    image = base * shade[..., None]
    return np.clip(image + 0.5, 0, 255).astype(np.uint8)


def _wrap(text: str, font: ImageFont.FreeTypeFont, max_width: float, max_lines: int) -> list[str] | None:
    words = text.split()
    if not words:
        return [""]
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        trial = f"{current} {word}"
        if font.getlength(trial) <= max_width:
            current = trial
        else:
            lines.append(current)
            current = word
    lines.append(current)
    if len(lines) > max_lines:
        return None
    return lines


def raster_text(
    text: str,
    font_path: str,
    font_px: int,
    max_width: int,
    max_lines: int = 2,
) -> np.ndarray:
    """White text, dark stroke, soft shadow. RGBA uint8."""
    px = max(8, int(font_px))
    stroke = max(1, px // 28)
    shadow = max(2, px // 18)
    lines: list[str] | None = None
    font: ImageFont.FreeTypeFont | None = None
    while px >= 12:
        font = ImageFont.truetype(font_path, px)
        lines = _wrap(text, font, max_width, max_lines)
        if lines is not None:
            break
        px = int(px * 0.9)
        stroke = max(1, px // 28)
        shadow = max(2, px // 18)
    if font is None or lines is None:
        font = ImageFont.truetype(font_path, 12)
        lines = [text]
    bboxes = [font.getbbox(line) for line in lines]
    widths = [box[2] - box[0] for box in bboxes]
    heights = [box[3] - box[1] for box in bboxes]
    gap = max(0, int(px * 0.12))
    text_w = max(widths) if widths else 1
    text_h = sum(heights) + gap * (len(lines) - 1)
    pad = stroke + shadow + 6
    size = (text_w + pad * 2, text_h + pad * 2)
    shadow_img = Image.new("RGBA", size, (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow_img)
    y = pad
    for line, box, width in zip(lines, bboxes, widths):
        x = pad + (text_w - width) / 2 - box[0]
        shadow_draw.text((x, y - box[1] + shadow), line, font=font, fill=(0, 0, 0, 170))
        y += (box[3] - box[1]) + gap
    shadow_np = np.array(shadow_img)
    shadow_np[..., 3] = cv2.GaussianBlur(shadow_np[..., 3], (0, 0), 1.4)
    image = Image.fromarray(shadow_np, "RGBA")
    draw = ImageDraw.Draw(image)
    y = pad
    for line, box, width in zip(lines, bboxes, widths):
        x = pad + (text_w - width) / 2 - box[0]
        draw.text(
            (x, y - box[1]),
            line,
            font=font,
            fill=(255, 255, 255, 255),
            stroke_width=stroke,
            stroke_fill=(0, 0, 0, 255),
        )
        y += (box[3] - box[1]) + gap
    return np.array(image)


def composite_rgba(dst: np.ndarray, rgba: np.ndarray, x: int, y: int) -> None:
    """Alpha-composite an RGBA patch onto a BGR uint8 frame, in place."""
    h, w = rgba.shape[:2]
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(dst.shape[1], x + w)
    y1 = min(dst.shape[0], y + h)
    if x0 >= x1 or y0 >= y1:
        return
    src = rgba[y0 - y : y1 - y, x0 - x : x1 - x]
    alpha = src[..., 3:4].astype(np.float32) * (1.0 / 255.0)
    if float(alpha.max()) <= 0:
        return
    bgr = src[..., :3][..., ::-1].astype(np.float32)
    region = dst[y0:y1, x0:x1].astype(np.float32)
    dst[y0:y1, x0:x1] = np.clip(bgr * alpha + region * (1.0 - alpha), 0, 255).astype(np.uint8)


def apply_bloom(signal: np.ndarray, strength: float) -> None:
    """Tone-map ``signal`` in place. ``strength`` scales the summed Gaussian bloom."""
    if signal.size == 0 or strength <= 0:
        mapped = 255.0 * (1.0 - np.exp(-signal / 90.0))
        signal[:] = mapped
        return
    small_w = max(2, signal.shape[1] // 4)
    small_h = max(2, signal.shape[0] // 4)
    small = cv2.resize(signal, (small_w, small_h), interpolation=cv2.INTER_AREA)
    blur = cv2.GaussianBlur(small, (0, 0), 1.0) + cv2.GaussianBlur(small, (0, 0), 2.2)
    bloom = cv2.resize(blur, (signal.shape[1], signal.shape[0]), interpolation=cv2.INTER_LINEAR)
    combined = signal + float(strength) * bloom
    signal[:] = 255.0 * (1.0 - np.exp(-combined / 90.0))
