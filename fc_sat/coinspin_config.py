"""Load the coinspin config. The coin sizes are pinned to the rolling math."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from fc_sat.coinspin_math import CX, CY, R1, R_BIG, R_SMALL

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "coinspin.yaml"

_COLOR_KEYS = (
    "background",
    "text",
    "muted",
    "gold",
    "plate",
    "strike",
    "grey_coin",
    "grey_rim",
    "gold_coin",
    "gold_rim",
    "face",
    "rolling",
    "trip",
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
    center_x: float
    center_y: float
    coin_radius: float
    sat_grey_radius: float
    sat_gold_radius: float
    hook: str
    seed: int
    background: str
    text: str
    muted: str
    gold: str
    plate: str
    strike: str
    grey_coin: str
    grey_rim: str
    gold_coin: str
    gold_rim: str
    face: str
    rolling: str
    trip: str
    path: Path

    @property
    def center(self) -> np.ndarray:
        return np.array([self.center_x, self.center_y], dtype=np.float64)


def load_config(path: Path | None = None) -> CoinConfig:
    path = path or CONFIG_PATH
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} is not a mapping")
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
        center_x=float(center[0]),
        center_y=float(center[1]),
        coin_radius=float(raw["coin_radius"]),
        sat_grey_radius=float(raw["sat_grey_radius"]),
        sat_gold_radius=float(raw["sat_gold_radius"]),
        hook=str(raw.get("hook", "A")),
        seed=int(raw.get("seed", 13)),
        path=path,
        **{key: str(raw[key]) for key in _COLOR_KEYS},
    )
    validate_config(cfg)
    return cfg


def validate_config(cfg: CoinConfig) -> None:
    if cfg.bpm != 150 or cfg.fps != 60:
        raise ConfigError("the schedule is written for 150 bpm at 60 fps")
    if cfg.frames != 1824:
        raise ConfigError(f"this short is 1824 frames, got {cfg.frames}")
    if cfg.sample_rate != 48000:
        raise ConfigError("sample rate is 48000")
    if cfg.width != 1080 or cfg.height != 1920:
        raise ConfigError("design size is 1080x1920")
    geometry = (cfg.center_x, cfg.center_y, cfg.coin_radius, cfg.sat_grey_radius, cfg.sat_gold_radius)
    if geometry != (CX, CY, R1, R_BIG, R_SMALL):
        raise ConfigError(
            f"coins are radius {R1:g} at ({CX:g}, {CY:g}), and {R_BIG:g} / {R_SMALL:g} in the SAT act; the rolling math assumes it"
        )
    if cfg.sat_grey_radius != 3.0 * cfg.sat_gold_radius:
        raise ConfigError("the SAT grey coin must be exactly 3x wider than the gold coin")
    if cfg.hook not in {"A", "B", "D"}:
        raise ConfigError("hook must be A, B, or D")
