"""Load and check the paperfold config. Full renders are 1080x1920 at 60 fps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from fc_sat.beatkit.palette import equalize_lightness, lightness_spread

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "paperfold.yaml"


class ConfigError(SystemExit):
    pass


@dataclass(frozen=True)
class PaperConfig:
    bpm: float
    fps: int
    width: int
    height: int
    frames: int
    sample_rate: int
    hook: str
    seed: int
    background: str
    panel: str
    paper: str
    seam: str
    text: str
    muted: str
    gold: str
    coral: str
    teal: str
    space: str
    bar_hues: tuple[str, ...]
    bar_colors: tuple[str, ...]
    path: Path


def load_config(path: Path | None = None) -> PaperConfig:
    path = path or CONFIG_PATH
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} is not a mapping")
    hues = [str(item) for item in raw.get("bar_colors") or []]
    if len(hues) != 8:
        raise ConfigError("bar_colors needs 8 hex hues, one per bar in the cycle")
    cfg = PaperConfig(
        bpm=float(raw["bpm"]),
        fps=int(raw["fps"]),
        width=int(raw["width"]),
        height=int(raw["height"]),
        frames=int(raw["frames"]),
        sample_rate=int(raw["sample_rate"]),
        hook=str(raw.get("hook", "A")),
        seed=int(raw.get("seed", 7)),
        background=str(raw["background"]),
        panel=str(raw["panel"]),
        paper=str(raw["paper"]),
        seam=str(raw["seam"]),
        text=str(raw["text"]),
        muted=str(raw["muted"]),
        gold=str(raw["gold"]),
        coral=str(raw["coral"]),
        teal=str(raw["teal"]),
        space=str(raw["space"]),
        bar_hues=tuple(hues),
        bar_colors=tuple(equalize_lightness(hues)),
        path=path,
    )
    validate_config(cfg)
    return cfg


def validate_config(cfg: PaperConfig) -> None:
    if (cfg.width, cfg.height, cfg.fps) != (1080, 1920, 60):
        raise ConfigError("the master is 1080x1920 at 60 fps; 540x960 at 30 fps is a preview mode")
    if cfg.frames != 1824:
        raise ConfigError(f"this short is 1824 frames, got {cfg.frames}")
    if cfg.bpm != 150:
        raise ConfigError("tempo is 150 bpm so a beat is exactly 24 frames")
    if cfg.sample_rate != 48000:
        raise ConfigError("sample rate is 48000")
    if cfg.hook not in {"A", "B", "C", "D"}:
        raise ConfigError("hook must be A, B, C, or D")
    if lightness_spread(cfg.bar_colors) > 0.03:
        raise ConfigError("equalized bar colors exceed 0.03 OKLab L")
