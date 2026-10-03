"""Pull a cycle of hues to one OKLab lightness so a downbeat does not flash."""

from __future__ import annotations

import numpy as np

from fc_sat.color import hex_to_rgb, oklab_to_rgb_u8, rgb_u8_to_oklab


class PaletteError(SystemExit):
    pass


def _rgb_u8(hex_color: str) -> np.ndarray:
    return np.clip(np.rint(hex_to_rgb(hex_color) * 255.0), 0, 255).astype(np.uint8)


def _hex_of(rgb: np.ndarray) -> str:
    red, green, blue = (int(channel) for channel in rgb)
    return f"#{red:02X}{green:02X}{blue:02X}"


def lightness_spread(hexes: list[str] | tuple[str, ...]) -> float:
    lights = [float(rgb_u8_to_oklab(_rgb_u8(color))[0]) for color in hexes]
    return max(lights) - min(lights)


def equalize_lightness(hexes: list[str]) -> list[str]:
    """Move every color to the median OKLab L. Chroma is reduced only if the sRGB fit misses that L."""
    labs = [rgb_u8_to_oklab(_rgb_u8(color)) for color in hexes]
    target = float(np.median([float(lab[0]) for lab in labs]))
    fitted: list[str] = []
    for lab in labs:
        chosen = None
        for scale in (1.0, 0.85, 0.7, 0.55, 0.4, 0.25):
            trial = np.array([target, float(lab[1]) * scale, float(lab[2]) * scale], dtype=np.float64)
            rgb = oklab_to_rgb_u8(trial)
            back = float(rgb_u8_to_oklab(rgb)[0])
            if abs(back - target) <= 0.015:
                chosen = rgb
                break
        if chosen is None:
            chosen = oklab_to_rgb_u8(np.array([target, 0.0, 0.0], dtype=np.float64))
        fitted.append(_hex_of(chosen))
    spread = lightness_spread(fitted)
    if spread > 0.03:
        raise PaletteError(f"bar colors still differ by {spread:.3f} OKLab L after matching")
    return fitted
