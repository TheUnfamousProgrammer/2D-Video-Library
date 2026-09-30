"""Melody Hop config. Validation errors name the field that failed."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any

from fc_sat.config import _fail, _load_hooks, _load_mapping, _load_palettes


@dataclass(frozen=True)
class HopConfig:
    seed: int
    song: str
    octave_shift: int
    bpm: float | None
    repeats: int
    palette: str
    palette_stops: tuple[str, ...]
    hook: str
    hooks: tuple[str, ...]
    show_note_names: bool
    width: int
    height: int
    fps: int
    workers: int
    bloom_strength: float
    audio_offset_ms: float
    span: float
    pad_max_width: float
    pad_top_y: float
    pad_thickness: float
    ball_radius: float
    corner: float
    h_ref: float
    h_min: float
    h_max: float
    exponent: float
    trail_seconds: float
    sparks: int
    ring: bool
    reverb_wet: float
    rt60: float

    def to_public_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["palette_stops"] = list(self.palette_stops)
        payload["hooks"] = list(self.hooks)
        return payload


def _as_bool(field: str, value: Any) -> bool:
    if isinstance(value, bool):
        return value
    _fail(field, f"must be a boolean, got {type(value).__name__}")
    raise AssertionError


def _as_int(field: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(field, f"must be an integer, got {value!r}")
    return int(value)


def _as_float(field: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(field, f"must be a number, got {value!r}")
    return float(value)


def _section(raw: dict, name: str) -> dict:
    value = raw.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        _fail(name, "must be a mapping")
    return value


def validate_hop(raw: dict[str, Any], *, palettes: dict[str, tuple[str, ...]], hooks: tuple[str, ...]) -> HopConfig:
    seed = _as_int("seed", raw.get("seed", 1))
    if seed < 0:
        _fail("seed", "must be >= 0")
    song = raw.get("song", "songs/ode_to_joy.yaml")
    if not isinstance(song, str) or not song.strip():
        _fail("song", "must be a non-empty path")
    octave = _as_int("octave_shift", raw.get("octave_shift", 1))
    if octave < -4 or octave > 4:
        _fail("octave_shift", "must be in [-4, 4]")
    bpm_raw = raw.get("bpm", None)
    bpm: float | None
    if bpm_raw is None:
        bpm = None
    else:
        bpm = _as_float("bpm", bpm_raw)
        if bpm <= 0:
            _fail("bpm", "must be > 0")
    repeats = _as_int("repeats", raw.get("repeats", 1))
    if repeats < 1 or repeats > 3:
        _fail("repeats", "must be in [1, 3]")
    palette = raw.get("palette", "sunset")
    if not isinstance(palette, str) or palette not in palettes:
        _fail("palette", f"unknown palette {palette!r}")
    hook = raw.get("hook", hooks[0] if hooks else "Guess the song")
    if not isinstance(hook, str) or not hook.strip():
        _fail("hook", "must be a non-empty string")
    show_names = _as_bool("show_note_names", raw.get("show_note_names", False))
    width = _as_int("width", raw.get("width", 1080))
    height = _as_int("height", raw.get("height", 1920))
    fps = _as_int("fps", raw.get("fps", 60))
    if width != 1080 or height != 1920:
        _fail("width", "full renders are 1080x1920")
    if fps != 60:
        _fail("fps", "must be 60")
    workers = _as_int("workers", raw.get("workers", 1))
    if workers < 1 or workers > 64:
        _fail("workers", "must be in [1, 64]")
    bloom = _as_float("bloom_strength", raw.get("bloom_strength", 0.6))
    if bloom < 0 or bloom > 2:
        _fail("bloom_strength", "must be in [0, 2]")
    offset = _as_float("audio_offset_ms", raw.get("audio_offset_ms", 0))
    if offset < -500 or offset > 500:
        _fail("audio_offset_ms", "must be in [-500, 500]")

    layout = _section(raw, "layout")
    span = _as_float("layout.span", layout.get("span", 700))
    pad_max = _as_float("layout.pad_max_width", layout.get("pad_max_width", 112))
    pad_top = _as_float("layout.pad_top_y", layout.get("pad_top_y", 1180))
    thick = _as_float("layout.pad_thickness", layout.get("pad_thickness", 36))
    ball = _as_float("layout.ball_radius", layout.get("ball_radius", 34))
    corner = _as_float("layout.corner", layout.get("corner", 14))
    if span <= 0 or pad_max <= 0 or thick <= 0 or ball <= 0 or corner < 0:
        _fail("layout.span", "span, pad size, thickness, and ball radius must be > 0")

    hop = _section(raw, "hop")
    h_ref = _as_float("hop.h_ref", hop.get("h_ref", 220))
    h_min = _as_float("hop.h_min", hop.get("h_min", 60))
    h_max = _as_float("hop.h_max", hop.get("h_max", 600))
    exponent = _as_float("hop.exponent", hop.get("exponent", 1.2))
    if h_min <= 0 or h_max < h_min or h_ref <= 0:
        _fail("hop.h_min", "need 0 < h_min <= h_max and h_ref > 0")

    effects = _section(raw, "effects")
    trail = _as_float("effects.trail_seconds", effects.get("trail_seconds", 0.30))
    sparks = _as_int("effects.sparks", effects.get("sparks", 10))
    ring = _as_bool("effects.ring", effects.get("ring", True))
    if trail <= 0:
        _fail("effects.trail_seconds", "must be > 0")
    if sparks < 0 or sparks > 64:
        _fail("effects.sparks", "must be in [0, 64]")

    audio = _section(raw, "audio")
    wet = _as_float("audio.reverb_wet", audio.get("reverb_wet", 0.22))
    rt60 = _as_float("audio.rt60", audio.get("rt60", 1.4))
    if wet < 0 or wet > 1:
        _fail("audio.reverb_wet", "must be in [0, 1]")
    if rt60 <= 0:
        _fail("audio.rt60", "must be > 0")

    return HopConfig(
        seed=seed,
        song=str(song),
        octave_shift=octave,
        bpm=bpm,
        repeats=repeats,
        palette=str(palette),
        palette_stops=palettes[str(palette)],
        hook=str(hook).strip(),
        hooks=hooks,
        show_note_names=show_names,
        width=width,
        height=height,
        fps=fps,
        workers=workers,
        bloom_strength=bloom,
        audio_offset_ms=offset,
        span=span,
        pad_max_width=pad_max,
        pad_top_y=pad_top,
        pad_thickness=thick,
        ball_radius=ball,
        corner=corner,
        h_ref=h_ref,
        h_min=h_min,
        h_max=h_max,
        exponent=exponent,
        trail_seconds=trail,
        sparks=sparks,
        ring=ring,
        reverb_wet=wet,
        rt60=rt60,
    )


def load_hop_config(path: str | Path) -> HopConfig:
    config_path = Path(path)
    raw = _load_mapping(config_path)
    palette_path = config_path.parent / "palettes.yaml"
    hooks_path = config_path.parent / "hop_hooks.yaml"
    if not palette_path.exists():
        _fail("palettes", f"missing {palette_path}")
    if not hooks_path.exists():
        _fail("hooks", f"missing {hooks_path}")
    return validate_hop(raw, palettes=_load_palettes(palette_path), hooks=_load_hooks(hooks_path))


def with_hop_overrides(cfg: HopConfig, **changes: Any) -> HopConfig:
    known = {item.name for item in fields(HopConfig)}
    for key in changes:
        if key not in known:
            _fail(key, "is not a config field")
    updated = replace(cfg, **changes)
    if updated.repeats < 1 or updated.repeats > 3:
        _fail("repeats", "must be in [1, 3]")
    if updated.workers < 1:
        _fail("workers", "must be in [1, 64]")
    if updated.octave_shift < -4 or updated.octave_shift > 4:
        _fail("octave_shift", "must be in [-4, 4]")
    return updated
