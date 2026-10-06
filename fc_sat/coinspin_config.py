"""Load the coinspin config. Trace hues share one OKLab lightness."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from fc_sat.beatkit.palette import equalize_lightness, lightness_spread
from fc_sat.coinspin_math import CX, CY, STAGE

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "coinspin.yaml"

_COLOR_KEYS = (
    "background",
    "text",
    "muted",
    "gold",
    "plate",
    "strike",
    "fixed_coin",
    "fixed_rim",
    "rolling_coin",
    "rolling_rim",
    "sun",
    "earth",
    "night",
)


class ConfigError(SystemExit):
    pass


@dataclass(frozen=True)
class CoinConfig:
    bpm: float
    fps: int
    width: int
    height: int
    frames: int
    sample_rate: int
    stage_radius: float
    center_x: float
    center_y: float
    hook: str
    seed: int
    background: str
    text: str
    muted: str
    gold: str
    plate: str
    strike: str
    fixed_coin: str
    fixed_rim: str
    rolling_coin: str
    rolling_rim: str
    sun: str
    earth: str
    night: str
    trace_hues: tuple[str, ...]
    trace_colors: tuple[str, ...]
    path: Path

    @property
    def center(self) -> np.ndarray:
        return np.array([self.center_x, self.center_y], dtype=np.float64)


def load_config(path: Path | None = None) -> CoinConfig:
    path = path or CONFIG_PATH
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} is not a mapping")
    hues = [str(item) for item in raw.get("trace_colors") or []]
    if len(hues) != 8:
        raise ConfigError("trace_colors needs 8 hex hues")
    missing = [key for key in _COLOR_KEYS if key not in raw]
    if missing:
        raise ConfigError(f"missing colors: {', '.join(missing)}")
    center = raw.get("center") or [CX, CY]
    cfg = CoinConfig(
        bpm=float(raw["bpm"]),
        fps=int(raw["fps"]),
        width=int(raw["width"]),
        height=int(raw["height"]),
        frames=int(raw["frames"]),
        sample_rate=int(raw["sample_rate"]),
        stage_radius=float(raw["stage_radius"]),
        center_x=float(center[0]),
        center_y=float(center[1]),
        hook=str(raw.get("hook", "A")),
        seed=int(raw.get("seed", 13)),
        trace_hues=tuple(hues),
        trace_colors=tuple(equalize_lightness(hues)),
        path=path,
        **{key: str(raw[key]) for key in _COLOR_KEYS},
    )
    validate_config(cfg)
    return cfg


def validate_config(cfg: CoinConfig) -> None:
    if cfg.bpm != 150 or cfg.fps != 60:
        raise ConfigError("the lap schedule is written for 150 bpm at 60 fps")
    if cfg.frames != 1824:
        raise ConfigError(f"this short is 1824 frames, got {cfg.frames}")
    if cfg.sample_rate != 48000:
        raise ConfigError("sample rate is 48000")
    if cfg.width != 1080 or cfg.height != 1920:
        raise ConfigError("design size is 1080x1920")
    if (cfg.stage_radius, cfg.center_x, cfg.center_y) != (STAGE, CX, CY):
        raise ConfigError(f"stage is radius {STAGE:g} px at ({CX:g}, {CY:g}); the rolling math assumes it")
    if cfg.hook not in {"A", "B", "D"}:
        raise ConfigError("hook must be A, B, or D")
    if lightness_spread(cfg.trace_colors) > 0.03:
        raise ConfigError("equalized trace colors exceed 0.03 OKLab L")
