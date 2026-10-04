"""Sepia ramp in linear light. 0 is dark ink, 0.5 is paper, 1 is light ink."""

from __future__ import annotations

import cv2
import numpy as np

DARK_HEX = "#1C1712"
PAPER_HEX = "#8F826D"
LIGHT_HEX = "#F3E9D2"


def _hex_rgb(value: str) -> np.ndarray:
    text = value.removeprefix("#")
    return np.array([int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16)], dtype=np.float64) / 255.0


def srgb_to_linear(u: np.ndarray) -> np.ndarray:
    return np.where(u <= 0.04045, u / 12.92, ((u + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(u: np.ndarray) -> np.ndarray:
    return np.where(u <= 0.0031308, 12.92 * u, 1.055 * np.power(np.maximum(u, 0.0), 1.0 / 2.4) - 0.055)


def make_lut(steps: int = 4096) -> np.ndarray:
    t = np.linspace(0.0, 1.0, steps)
    dark = srgb_to_linear(_hex_rgb(DARK_HEX))
    paper = srgb_to_linear(_hex_rgb(PAPER_HEX))
    light = srgb_to_linear(_hex_rgb(LIGHT_HEX))
    low = t < 0.5
    mix = np.where(low, t / 0.5, (t - 0.5) / 0.5)[:, None]
    start = np.where(low[:, None], dark, paper)
    end = np.where(low[:, None], paper, light)
    rgb = linear_to_srgb(start + (end - start) * mix)
    return np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)


LUT = make_lut()


def colorize(brightness: np.ndarray) -> np.ndarray:
    index = np.clip(np.rint(np.asarray(brightness) * (len(LUT) - 1)), 0, len(LUT) - 1).astype(np.int32)
    return LUT[index]


def paper_rgb(canvas: np.ndarray, width: int, height: int) -> np.ndarray:
    """Scale a grid canvas to the drawing panel and map it through the sepia ramp."""
    scaled = cv2.resize(np.asarray(canvas, dtype=np.float32), (int(width), int(height)), interpolation=cv2.INTER_LINEAR)
    return colorize(scaled)
