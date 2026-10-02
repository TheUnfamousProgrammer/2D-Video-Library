"""Load and check the polycircle config.

Bar colors keep the brief's hues. Their OKLab lightness is pulled to the median
so the spread is at most 0.03 and a downbeat does not flash.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from fc_sat.color import hex_to_rgb, oklab_to_rgb_u8, rgb_u8_to_oklab

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "polycircle.yaml"


class ConfigError(SystemExit):
    pass


@dataclass(frozen=True)
class PolyConfig:
    bpm: float
    fps: int
    width: int
    height: int
    frames: int
    sample_rate: int
    radius: float
    center_x: float
    center_y: float
    zoom_max: float
    gap_px: float
    hook: str
    seed: int
    background: str
    text: str
    muted: str
    gold: str
    bar_hues: tuple[str, ...]
    bar_colors: tuple[str, ...]
    path: Path

    @property
    def center(self) -> np.ndarray:
        return np.array([self.center_x, self.center_y], dtype=np.float64)

    @property
    def top(self) -> np.ndarray:
        return np.array([self.center_x, self.center_y - self.radius], dtype=np.float64)


def _rgb_u8(hex_color: str) -> np.ndarray:
    return np.clip(np.rint(hex_to_rgb(hex_color) * 255.0), 0, 255).astype(np.uint8)


def _hex_of(rgb: np.ndarray) -> str:
    red, green, blue = (int(channel) for channel in rgb)
    return f"#{red:02X}{green:02X}{blue:02X}"


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
        raise ConfigError(f"bar colors still differ by {spread:.3f} OKLab L after matching")
    return fitted


def lightness_spread(hexes: list[str] | tuple[str, ...]) -> float:
    lights = [float(rgb_u8_to_oklab(_rgb_u8(color))[0]) for color in hexes]
    return max(lights) - min(lights)


def load_config(path: Path | None = None) -> PolyConfig:
    path = path or CONFIG_PATH
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} is not a mapping")
    hues = [str(item) for item in raw.get("bar_colors") or []]
    if len(hues) != 8:
        raise ConfigError("bar_colors needs 8 hex hues, one per bar in the cycle")
    center = raw.get("center") or [540, 910]
    cfg = PolyConfig(
        bpm=float(raw["bpm"]),
        fps=int(raw["fps"]),
        width=int(raw["width"]),
        height=int(raw["height"]),
        frames=int(raw["frames"]),
        sample_rate=int(raw["sample_rate"]),
        radius=float(raw["radius"]),
        center_x=float(center[0]),
        center_y=float(center[1]),
        zoom_max=float(raw["zoom_max"]),
        gap_px=float(raw["gap_px"]),
        hook=str(raw.get("hook", "A")),
        seed=int(raw.get("seed", 7)),
        background=str(raw["background"]),
        text=str(raw["text"]),
        muted=str(raw["muted"]),
        gold=str(raw["gold"]),
        bar_hues=tuple(hues),
        bar_colors=tuple(equalize_lightness(hues)),
        path=path,
    )
    validate_config(cfg)
    return cfg


def validate_config(cfg: PolyConfig) -> None:
    if cfg.bpm <= 0 or cfg.fps <= 0:
        raise ConfigError("bpm and fps must be positive")
    if cfg.frames != 1824:
        raise ConfigError(f"this short is 1824 frames, got {cfg.frames}")
    if cfg.sample_rate != 48000:
        raise ConfigError("sample rate is 48000")
    if cfg.fps not in (30, 60):
        raise ConfigError("fps must be 30 or 60")
    if cfg.radius <= 0 or cfg.zoom_max <= 1:
        raise ConfigError("radius and zoom_max are out of range")
    if cfg.hook not in {"A", "B", "C"}:
        raise ConfigError("hook must be A, B, or C")
    if lightness_spread(cfg.bar_colors) > 0.03:
        raise ConfigError("equalized bar colors exceed 0.03 OKLab L")
    exact = cfg.fps * 60.0 / cfg.bpm
    if abs(exact - round(exact)) > 0.5:
        raise ConfigError(f"beat snap error {abs(exact - round(exact)):.3f} frames exceeds 0.5")
