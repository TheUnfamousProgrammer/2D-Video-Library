"""OKLab palette interpolation. Kept separate so render and tests share one path."""

from __future__ import annotations

import numpy as np


def hex_to_rgb(value: str) -> np.ndarray:
    text = value[1:] if value.startswith("#") else value
    number = int(text, 16)
    return np.array([(number >> 16) & 255, (number >> 8) & 255, number & 255], dtype=np.float64) / 255.0


def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(np.clip(c, 0, None), 1 / 2.4) - 0.055)


def _linear_to_oklab(rgb: np.ndarray) -> np.ndarray:
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l_ = np.cbrt(l)
    m_ = np.cbrt(m)
    s_ = np.cbrt(s)
    return np.stack(
        (
            0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
            1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
            0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
        ),
        axis=-1,
    )


def _oklab_to_linear(lab: np.ndarray) -> np.ndarray:
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b
    l = l_ ** 3
    m = m_ ** 3
    s = s_ ** 3
    return np.stack(
        (
            +4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
            -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
            -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s,
        ),
        axis=-1,
    )


def rgb_u8_to_oklab(rgb: np.ndarray) -> np.ndarray:
    """sRGB uint8 ``(..., 3)`` to OKLab float64. Shared by palette and Pixel Morph."""
    unit = np.asarray(rgb, dtype=np.float64) / 255.0
    return _linear_to_oklab(_srgb_to_linear(unit))


def oklab_to_rgb_u8(lab: np.ndarray) -> np.ndarray:
    """OKLab float to sRGB uint8 ``(..., 3)``."""
    linear = np.clip(_oklab_to_linear(np.asarray(lab, dtype=np.float64)), 0.0, 1.0)
    srgb = np.clip(_linear_to_srgb(linear), 0.0, 1.0)
    return (srgb * 255.0 + 0.5).astype(np.uint8)


def dominant_colors(rgb: np.ndarray, k: int = 3, seed: int = 0) -> np.ndarray:
    """Seeded k-means. Returns ``(k, 3)`` uint8 RGB centers."""
    flat = np.asarray(rgb, dtype=np.float64).reshape(-1, 3)
    if flat.shape[0] == 0:
        return np.zeros((k, 3), dtype=np.uint8)
    rng = np.random.Generator(np.random.PCG64(seed))
    if flat.shape[0] > 2500:
        flat = flat[rng.choice(flat.shape[0], 2500, replace=False)]
    centers = flat[rng.choice(flat.shape[0], size=min(k, flat.shape[0]), replace=False)].copy()
    if len(centers) < k:
        extra = np.repeat(centers[:1], k - len(centers), axis=0)
        centers = np.vstack([centers, extra])
    labels = np.zeros(flat.shape[0], dtype=np.int32)
    for _ in range(8):
        dist = ((flat[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = dist.argmin(axis=1)
        for index in range(k):
            group = flat[labels == index]
            if len(group):
                centers[index] = group.mean(axis=0)
    return np.clip(np.rint(centers), 0, 255).astype(np.uint8)


def palette_bgr(stops: tuple[str, ...], count: int) -> np.ndarray:
    """Return (count, 3) uint8 BGR colors, smoothly spaced across the stops."""
    rgb = np.stack([_srgb_to_linear(hex_to_rgb(stop)) for stop in stops], axis=0)
    lab = _linear_to_oklab(rgb)
    if count <= 1:
        samples = lab[:1]
    else:
        pos = np.linspace(0.0, len(stops) - 1, count)
        left = np.floor(pos).astype(np.int32)
        right = np.minimum(left + 1, len(stops) - 1)
        frac = (pos - left)[:, None]
        samples = lab[left] * (1 - frac) + lab[right] * frac
    linear = np.clip(_oklab_to_linear(samples), 0.0, 1.0)
    srgb = np.clip(_linear_to_srgb(linear), 0.0, 1.0)
    bgr = (srgb[:, ::-1] * 255.0 + 0.5).astype(np.uint8)
    return bgr
