"""Load the Collatz configs and refuse a palette that fails the contrast floors."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_yaml(name: str) -> dict:
    path = ROOT / "configs" / name
    if not path.exists():
        raise SystemExit(f"missing {path}")
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise SystemExit(f"{path} must be a mapping")
    return data


def _channel(value: int) -> float:
    channel = value / 255.0
    if channel <= 0.04045:
        return channel / 12.92
    return ((channel + 0.055) / 1.055) ** 2.4


def luminance(hex_color: str) -> float:
    text = hex_color.removeprefix("#")
    red, green, blue = int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16)
    return 0.2126 * _channel(red) + 0.7152 * _channel(green) + 0.0722 * _channel(blue)


def contrast_ratio(foreground: str, background: str) -> float:
    hi = max(luminance(foreground), luminance(background))
    lo = min(luminance(foreground), luminance(background))
    return (hi + 0.05) / (lo + 0.05)


def validate_palette(spec: dict) -> None:
    palette = spec["palette"]
    floors = spec["contrast"]
    primary = contrast_ratio(palette["text"], palette["bg"])
    muted = contrast_ratio(palette["muted"], palette["bg"])
    if primary < float(floors["primary"]):
        raise SystemExit(f"primary contrast {primary:.2f} is below {floors['primary']}")
    if muted < float(floors["muted"]):
        raise SystemExit(f"muted contrast {muted:.2f} is below {floors['muted']}")
