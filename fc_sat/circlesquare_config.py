"""Load the circlesquare config. Bar hues share one OKLab lightness."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from fc_sat.beatkit.palette import equalize_lightness, lightness_spread

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "circlesquare.yaml"


class ConfigError(SystemExit):
    pass


@dataclass(frozen=True)
class CircleConfig:
    bpm: float
    fps: int
    width: int
    height: int
    frames: int
    sample_rate: int
    half_side: float
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
    plate: str
    bar_hues: tuple[str, ...]
    bar_colors: tuple[str, ...]
    path: Path

    @property
    def center(self) -> np.ndarray:
        return np.array([self.center_x, self.center_y], dtype=np.float64)


def load_config(path: Path | None = None) -> CircleConfig:
    path = path or CONFIG_PATH
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} is not a mapping")
    hues = [str(item) for item in raw.get("bar_colors") or []]
    if len(hues) != 8:
        raise ConfigError("bar_colors needs 8 hex hues")
    center = raw.get("center") or [540, 905]
    cfg = CircleConfig(
        bpm=float(raw["bpm"]),
        fps=int(raw["fps"]),
        width=int(raw["width"]),
        height=int(raw["height"]),
        frames=int(raw["frames"]),
        sample_rate=int(raw["sample_rate"]),
        half_side=float(raw["half_side"]),
        center_x=float(center[0]),
        center_y=float(center[1]),
        zoom_max=float(raw["zoom_max"]),
        gap_px=float(raw["gap_px"]),
        hook=str(raw.get("hook", "A")),
        seed=int(raw.get("seed", 11)),
        background=str(raw["background"]),
        text=str(raw["text"]),
        muted=str(raw["muted"]),
        gold=str(raw["gold"]),
        plate=str(raw.get("plate", "#1B2030")),
        bar_hues=tuple(hues),
        bar_colors=tuple(equalize_lightness(hues)),
        path=path,
    )
    validate_config(cfg)
    return cfg


def validate_config(cfg: CircleConfig) -> None:
    if cfg.bpm <= 0 or cfg.fps <= 0:
        raise ConfigError("bpm and fps must be positive")
    if cfg.frames != 1824:
        raise ConfigError(f"this short is 1824 frames, got {cfg.frames}")
    if cfg.sample_rate != 48000:
        raise ConfigError("sample rate is 48000")
    if cfg.width != 1080 or cfg.height != 1920:
        raise ConfigError("design size is 1080x1920")
    if cfg.half_side != 340 or cfg.zoom_max != 36:
        raise ConfigError("half-side is 340 px and zoom max is 36")
    if cfg.hook not in {"A", "B", "D"}:
        raise ConfigError("hook must be A, B, or D")
    if lightness_spread(cfg.bar_colors) > 0.03:
        raise ConfigError("equalized bar colors exceed 0.03 OKLab L")
    exact = cfg.fps * 60.0 / cfg.bpm
    if abs(exact - round(exact)) > 0.5:
        raise ConfigError(f"beat snap error {abs(exact - round(exact)):.3f} frames exceeds 0.5")
