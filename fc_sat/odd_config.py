"""Odd One Out config, tiers, caption checks, and timeline.

Validation errors name the field, same as the other generators.
Frames for a segment are ``round(seconds * fps)``. The normal tier is
5+6+7 plus three 1.2 s reveals, two 0.2 s dissolves, and a 2.0 s outro:
24.0 s and 1440 frames.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fc_sat.config import (
    _as_float,
    _as_int,
    _as_str,
    _fail,
    _load_hooks,
    _load_mapping,
    _sibling,
)

DIFFERENCES = ("hue", "tilt", "detail", "hue_subtle")
TIERS = ("easy", "normal", "hard")
PERCENT_CLAIM = re.compile(r"\d+(?:\.\d+)?\s*%", re.IGNORECASE)
HUE_CVD_FLOOR = 0.12
SUBTLE_CVD_FLOOR = 0.06

_TOP = frozenset(
    {
        "generator",
        "seed",
        "tier",
        "width",
        "height",
        "fps",
        "workers",
        "reveal_seconds",
        "dissolve_seconds",
        "outro_seconds",
        "pop_seconds",
        "pop_floor",
        "hook",
        "cta",
        "background",
        "levels",
        "field",
        "item",
        "layout",
        "constraints",
        "audio",
        "tiers",
    }
)
_TIER_KEYS = frozenset(
    {
        "level_ids",
        "hue_min_distance",
        "hue_min_lightness",
        "cvd_min_distance",
        "tilt_degrees",
        "tilt_range",
        "dot_fraction",
        "detail_mode",
        "hue_subtle_distance",
        "hue_subtle_lightness",
        "hue_subtle_cvd",
        "base_l",
        "base_c",
    }
)
_LEVEL_KEYS = frozenset({"id", "difference", "grid", "size", "timer"})
_FIELD_KEYS = frozenset({"x0", "x1", "y0", "y1", "corner_radius", "outline", "outline_color"})
_ITEM_KEYS = frozenset(
    {"corner_frac", "ring_color", "ring_px", "ring_from", "ring_to", "ring_grow", "fade_seconds", "fade_alpha"}
)
_LAYOUT_KEYS = frozenset(
    {
        "safe_x",
        "safe_y",
        "label_y",
        "label_px",
        "caption_y",
        "caption_px",
        "caption_max_chars",
        "caption_x",
        "timer_y",
        "timer_x0",
        "timer_x1",
        "timer_h",
        "timer_num_x0",
        "timer_num_x1",
        "timer_num_px",
        "timer_color",
        "timer_red",
        "red_seconds",
        "outro_y",
        "outro_px",
        "cta_y",
        "cta_px",
        "hook_seconds",
    }
)
_CONSTRAINT_KEYS = frozenset({"max_attempts"})
_AUDIO_KEYS = frozenset({"tone_seconds", "double_tick_window"})


def validate_caption(text: str, *, field: str, max_chars: int | None = None) -> str:
    """Reject percentage claims. Optionally enforce the caption-row length."""
    if not isinstance(text, str) or not text.strip():
        _fail(field, f"must be a non-empty string, got {text!r}")
    cleaned = text.strip()
    if PERCENT_CLAIM.search(cleaned):
        _fail(field, f"must not claim a percentage, got {cleaned!r}")
    if max_chars is not None and len(cleaned) > max_chars:
        _fail(field, f"must be at most {max_chars} characters, got {len(cleaned)} in {cleaned!r}")
    return cleaned


def frames_for(seconds: float, fps: int) -> int:
    return int(round(float(seconds) * int(fps)))


def rotated_extent(size: float, degrees: float) -> float:
    """Axis-aligned width of a square of side ``size`` rotated by ``degrees``."""
    angle = math.radians(abs(float(degrees)))
    return float(size) * (abs(math.cos(angle)) + abs(math.sin(angle)))


@dataclass(frozen=True)
class LevelSpec:
    id: int
    difference: str
    grid: int
    size: float
    timer: float

    @property
    def count(self) -> int:
        return self.grid * self.grid

    @property
    def label(self) -> str:
        return f"LEVEL {self.id}"


@dataclass(frozen=True)
class TierParams:
    name: str
    hue_min_distance: float
    hue_min_lightness: float
    cvd_min_distance: float
    tilt_degrees: float
    tilt_min: float
    tilt_max: float
    dot_fraction: float
    detail_mode: str
    hue_subtle_distance: float
    hue_subtle_lightness: float
    hue_subtle_cvd: float
    base_l: float
    base_c: float


@dataclass(frozen=True)
class Segment:
    kind: str
    level_id: int
    start_s: float
    duration_s: float
    start_frame: int
    n_frames: int

    @property
    def end_s(self) -> float:
        return self.start_s + self.duration_s

    @property
    def end_frame(self) -> int:
        return self.start_frame + self.n_frames


@dataclass(frozen=True)
class OddConfig:
    generator: str
    seed: int
    tier_name: str
    tier: TierParams
    width: int
    height: int
    fps: int
    workers: int
    reveal_seconds: float
    dissolve_seconds: float
    outro_seconds: float
    pop_seconds: float
    pop_floor: float
    hook: str
    hooks: tuple[str, ...]
    cta: str
    background: str
    levels: tuple[LevelSpec, ...]
    field_x0: float
    field_x1: float
    field_y0: float
    field_y1: float
    field_radius: float
    outline_px: float
    outline_color: str
    corner_frac: float
    ring_color: str
    ring_px: float
    ring_from: float
    ring_to: float
    ring_grow: float
    fade_seconds: float
    fade_alpha: float
    safe_x: tuple[float, float]
    safe_y: tuple[float, float]
    label_y: float
    label_px: int
    caption_y: float
    caption_px: int
    caption_max_chars: int
    caption_x: tuple[float, float]
    timer_y: float
    timer_x0: float
    timer_x1: float
    timer_h: float
    timer_num_x0: float
    timer_num_x1: float
    timer_num_px: int
    timer_color: str
    timer_red: str
    red_seconds: float
    outro_y: float
    outro_px: int
    cta_y: float
    cta_px: int
    hook_seconds: float
    max_attempts: int
    tone_seconds: float
    double_tick_window: float

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "generator": self.generator,
            "seed": self.seed,
            "tier": self.tier_name,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "reveal_seconds": self.reveal_seconds,
            "dissolve_seconds": self.dissolve_seconds,
            "outro_seconds": self.outro_seconds,
            "pop_seconds": self.pop_seconds,
            "pop_floor": self.pop_floor,
            "hook": self.hook,
            "levels": [
                {
                    "id": level.id,
                    "difference": level.difference,
                    "grid": level.grid,
                    "size": level.size,
                    "timer": level.timer,
                }
                for level in self.levels
            ],
            "tier_params": {
                "hue_min_distance": self.tier.hue_min_distance,
                "hue_min_lightness": self.tier.hue_min_lightness,
                "cvd_min_distance": self.tier.cvd_min_distance,
                "tilt_degrees": self.tier.tilt_degrees,
                "dot_fraction": self.tier.dot_fraction,
                "detail_mode": self.tier.detail_mode,
                "hue_subtle_distance": self.tier.hue_subtle_distance,
                "hue_subtle_cvd": self.tier.hue_subtle_cvd,
            },
        }


def build_timeline(cfg: OddConfig) -> tuple[Segment, ...]:
    """Play, reveal, a cross-dissolve between levels, then the outro."""
    segments: list[Segment] = []
    cursor_s = 0.0
    cursor_f = 0
    levels = cfg.levels
    for index, level in enumerate(levels):
        for kind, duration in (("play", level.timer), ("reveal", cfg.reveal_seconds)):
            n = frames_for(duration, cfg.fps)
            segments.append(Segment(kind, level.id, cursor_s, duration, cursor_f, n))
            cursor_s += duration
            cursor_f += n
        if index < len(levels) - 1:
            n = frames_for(cfg.dissolve_seconds, cfg.fps)
            segments.append(Segment("dissolve", level.id, cursor_s, cfg.dissolve_seconds, cursor_f, n))
            cursor_s += cfg.dissolve_seconds
            cursor_f += n
    n = frames_for(cfg.outro_seconds, cfg.fps)
    segments.append(Segment("outro", 0, cursor_s, cfg.outro_seconds, cursor_f, n))
    return tuple(segments)


def timeline_duration(cfg: OddConfig) -> float:
    return float(sum(segment.duration_s for segment in build_timeline(cfg)))


def timeline_frames(cfg: OddConfig) -> int:
    return int(sum(segment.n_frames for segment in build_timeline(cfg)))


def _section(raw: dict[str, Any], field: str, allowed: frozenset[str]) -> dict[str, Any]:
    value = raw.get(field)
    if not isinstance(value, dict):
        _fail(field, "must be a mapping")
    unknown = set(value) - allowed
    if unknown:
        _fail(f"{field}.{sorted(unknown)[0]}", "is not a config field")
    missing = allowed - set(value)
    if missing:
        _fail(f"{field}.{sorted(missing)[0]}", "is required")
    return value


def _pair(field: str, value: Any) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        _fail(field, "must be a pair of numbers")
    left = _as_float(f"{field}[0]", value[0])
    right = _as_float(f"{field}[1]", value[1])
    if right <= left:
        _fail(field, "must be ordered low to high")
    return left, right


def _hex(field: str, value: Any) -> str:
    text = _as_str(field, value)
    if not re.fullmatch(r"#[0-9A-Fa-f]{6}", text):
        _fail(field, f"must be a #RRGGBB color, got {text!r}")
    return text.upper()


def _levels(raw: Any) -> tuple[LevelSpec, ...]:
    if not isinstance(raw, list) or not raw:
        _fail("levels", "must be a non-empty list")
    found: list[LevelSpec] = []
    seen: set[int] = set()
    for index, item in enumerate(raw):
        field = f"levels[{index}]"
        if not isinstance(item, dict):
            _fail(field, "must be a mapping")
        unknown = set(item) - _LEVEL_KEYS
        if unknown:
            _fail(f"{field}.{sorted(unknown)[0]}", "is not a config field")
        missing = _LEVEL_KEYS - set(item)
        if missing:
            _fail(f"{field}.{sorted(missing)[0]}", "is required")
        level_id = _as_int(f"{field}.id", item["id"])
        if level_id in seen:
            _fail(f"{field}.id", f"duplicates level {level_id}")
        seen.add(level_id)
        difference = _as_str(f"{field}.difference", item["difference"])
        if difference not in DIFFERENCES:
            _fail(f"{field}.difference", f"must be one of {DIFFERENCES}")
        grid = _as_int(f"{field}.grid", item["grid"])
        if grid < 2:
            _fail(f"{field}.grid", "must be at least 2")
        size = _as_float(f"{field}.size", item["size"])
        if size < 8:
            _fail(f"{field}.size", "must be at least 8")
        timer = _as_float(f"{field}.timer", item["timer"])
        if timer <= 0:
            _fail(f"{field}.timer", "must be positive")
        found.append(LevelSpec(level_id, difference, grid, size, timer))
    return tuple(found)


def _tier(name: str, block: dict[str, Any]) -> tuple[TierParams, tuple[int, ...]]:
    ids = block.get("level_ids")
    if not isinstance(ids, list) or not ids:
        _fail(f"tiers.{name}.level_ids", "must be a non-empty list")
    level_ids = tuple(_as_int(f"tiers.{name}.level_ids[{i}]", item) for i, item in enumerate(ids))
    tilt_range = _pair(f"tiers.{name}.tilt_range", block["tilt_range"])
    tilt = _as_float(f"tiers.{name}.tilt_degrees", block["tilt_degrees"])
    if not (tilt_range[0] - 1e-9 <= tilt <= tilt_range[1] + 1e-9):
        _fail(f"tiers.{name}.tilt_degrees", f"must sit inside tilt_range {tilt_range}")
    cvd = _as_float(f"tiers.{name}.cvd_min_distance", block["cvd_min_distance"])
    if cvd + 1e-9 < HUE_CVD_FLOOR:
        _fail(f"tiers.{name}.cvd_min_distance", f"must stay at least {HUE_CVD_FLOOR:.2f}")
    subtle_cvd = _as_float(f"tiers.{name}.hue_subtle_cvd", block["hue_subtle_cvd"])
    if subtle_cvd + 1e-9 < SUBTLE_CVD_FLOOR:
        _fail(f"tiers.{name}.hue_subtle_cvd", f"must stay at least {SUBTLE_CVD_FLOOR:.2f}")
    mode = _as_str(f"tiers.{name}.detail_mode", block["detail_mode"])
    if mode not in ("missing", "moved"):
        _fail(f"tiers.{name}.detail_mode", "must be missing or moved")
    hue_distance = _as_float(f"tiers.{name}.hue_min_distance", block["hue_min_distance"])
    if hue_distance < 0.25 - 1e-9 and name != "hard":
        # Hard keeps the normal hue floor. Easy may only go up. Normal is 0.25.
        pass
    if hue_distance + 1e-9 < 0.25 and name in ("normal", "easy"):
        _fail(f"tiers.{name}.hue_min_distance", "must stay at least 0.25")
    if name == "hard" and hue_distance + 1e-9 < 0.25:
        _fail(f"tiers.{name}.hue_min_distance", "must stay at least 0.25")
    lightness = _as_float(f"tiers.{name}.hue_min_lightness", block["hue_min_lightness"])
    if lightness + 1e-9 < 0.15:
        _fail(f"tiers.{name}.hue_min_lightness", "must stay at least 0.15")
    dot = _as_float(f"tiers.{name}.dot_fraction", block["dot_fraction"])
    if not 0.05 <= dot <= 0.4:
        _fail(f"tiers.{name}.dot_fraction", "must be between 0.05 and 0.4")
    subtle_d = _as_float(f"tiers.{name}.hue_subtle_distance", block["hue_subtle_distance"])
    if subtle_d + 1e-9 < 0.10:
        _fail(f"tiers.{name}.hue_subtle_distance", "must stay at least 0.10")
    subtle_l = _as_float(f"tiers.{name}.hue_subtle_lightness", block["hue_subtle_lightness"])
    if subtle_l <= 0:
        _fail(f"tiers.{name}.hue_subtle_lightness", "must be positive")
    params = TierParams(
        name=name,
        hue_min_distance=hue_distance,
        hue_min_lightness=lightness,
        cvd_min_distance=cvd,
        tilt_degrees=tilt,
        tilt_min=tilt_range[0],
        tilt_max=tilt_range[1],
        dot_fraction=dot,
        detail_mode=mode,
        hue_subtle_distance=subtle_d,
        hue_subtle_lightness=subtle_l,
        hue_subtle_cvd=subtle_cvd,
        base_l=_as_float(f"tiers.{name}.base_l", block["base_l"]),
        base_c=_as_float(f"tiers.{name}.base_c", block["base_c"]),
    )
    return params, level_ids


def _fit_items(cfg_field: tuple[float, float, float, float], levels: tuple[LevelSpec, ...], tier: TierParams) -> None:
    span_x = cfg_field[1] - cfg_field[0]
    span_y = cfg_field[3] - cfg_field[2]
    for level in levels:
        cell = min(span_x, span_y) / level.grid
        need = level.size
        if level.difference == "tilt":
            need = rotated_extent(level.size, tier.tilt_degrees)
        if need >= cell - 1.0:
            _fail(
                f"levels id {level.id}",
                f"size {level.size:g} does not fit a {level.grid}x{level.grid} cell of {cell:.1f}px",
            )


def validate_odd(
    raw: dict[str, Any],
    *,
    hooks: list[str],
    config_path: str,
    tier: str | None = None,
    seed: int | None = None,
    level_ids: list[int] | None = None,
) -> OddConfig:
    if not isinstance(raw, dict):
        _fail("config", "must be a mapping")
    unknown = set(raw) - _TOP
    if unknown:
        _fail(sorted(unknown)[0], "is not a config field")
    missing = _TOP - set(raw)
    if missing:
        _fail(sorted(missing)[0], "is required")
    if _as_str("generator", raw["generator"]) != "odd":
        _fail("generator", "must be odd")
    if "bloom" in raw or any("bloom" in str(key) for key in raw):
        _fail("bloom", "is not used")
    width = _as_int("width", raw["width"])
    height = _as_int("height", raw["height"])
    fps = _as_int("fps", raw["fps"])
    if (width, height, fps) != (1080, 1920, 60):
        _fail("width", "the master is 1080x1920 at 60 fps")
    tier_name = tier or _as_str("tier", raw["tier"])
    if tier_name not in TIERS:
        _fail("tier", f"must be one of {TIERS}")
    tiers = raw["tiers"]
    if not isinstance(tiers, dict):
        _fail("tiers", "must be a mapping")
    tier_unknown = set(tiers) - set(TIERS)
    if tier_unknown:
        _fail(f"tiers.{sorted(tier_unknown)[0]}", "is not a tier")
    for name in TIERS:
        block = tiers.get(name)
        if not isinstance(block, dict):
            _fail(f"tiers.{name}", "must be a mapping")
        extra = set(block) - _TIER_KEYS
        if extra:
            _fail(f"tiers.{name}.{sorted(extra)[0]}", "is not a config field")
        absent = _TIER_KEYS - set(block)
        if absent:
            _fail(f"tiers.{name}.{sorted(absent)[0]}", "is required")
    params, tier_levels = _tier(tier_name, tiers[tier_name])
    catalog = {level.id: level for level in _levels(raw["levels"])}
    chosen_ids = list(level_ids) if level_ids is not None else list(tier_levels)
    if not chosen_ids:
        _fail("levels", "must include at least one level")
    chosen: list[LevelSpec] = []
    for level_id in chosen_ids:
        if level_id not in catalog:
            _fail("levels", f"has no level {level_id}")
        chosen.append(catalog[level_id])
    if tier_name != "hard" and any(level.difference == "hue_subtle" for level in chosen):
        _fail("levels", "hue_subtle is only on the hard tier")
    field = _section(raw, "field", _FIELD_KEYS)
    item = _section(raw, "item", _ITEM_KEYS)
    layout = _section(raw, "layout", _LAYOUT_KEYS)
    constraints = _section(raw, "constraints", _CONSTRAINT_KEYS)
    audio = _section(raw, "audio", _AUDIO_KEYS)
    box = (
        _as_float("field.x0", field["x0"]),
        _as_float("field.x1", field["x1"]),
        _as_float("field.y0", field["y0"]),
        _as_float("field.y1", field["y1"]),
    )
    if box[1] - box[0] != box[3] - box[2]:
        _fail("field", "must be a square")
    _fit_items(box, tuple(chosen), params)
    max_chars = _as_int("layout.caption_max_chars", layout["caption_max_chars"])
    hook = validate_caption(_as_str("hook", raw["hook"]), field="hook", max_chars=max_chars)
    checked_hooks = tuple(validate_caption(item_text, field="hooks", max_chars=max_chars) for item_text in hooks)
    if hook not in checked_hooks:
        _fail("hook", "must be one of the lines in odd_hooks.yaml")
    cta = validate_caption(_as_str("cta", raw["cta"]), field="cta", max_chars=max_chars)
    pop_floor = _as_float("pop_floor", raw["pop_floor"])
    if not 0.0 <= pop_floor <= 1.0:
        _fail("pop_floor", "must be between 0 and 1")
    fade_alpha = _as_float("item.fade_alpha", item["fade_alpha"])
    if not 0.0 < fade_alpha < 1.0:
        _fail("item.fade_alpha", "must be between 0 and 1")
    return OddConfig(
        generator="odd",
        seed=_as_int("seed", raw["seed"]) if seed is None else int(seed),
        tier_name=tier_name,
        tier=params,
        width=width,
        height=height,
        fps=fps,
        workers=_as_int("workers", raw["workers"]),
        reveal_seconds=_as_float("reveal_seconds", raw["reveal_seconds"]),
        dissolve_seconds=_as_float("dissolve_seconds", raw["dissolve_seconds"]),
        outro_seconds=_as_float("outro_seconds", raw["outro_seconds"]),
        pop_seconds=_as_float("pop_seconds", raw["pop_seconds"]),
        pop_floor=pop_floor,
        hook=hook,
        hooks=checked_hooks,
        cta=cta,
        background=_hex("background", raw["background"]),
        levels=tuple(chosen),
        field_x0=box[0],
        field_x1=box[1],
        field_y0=box[2],
        field_y1=box[3],
        field_radius=_as_float("field.corner_radius", field["corner_radius"]),
        outline_px=_as_float("field.outline", field["outline"]),
        outline_color=_hex("field.outline_color", field["outline_color"]),
        corner_frac=_as_float("item.corner_frac", item["corner_frac"]),
        ring_color=_hex("item.ring_color", item["ring_color"]),
        ring_px=_as_float("item.ring_px", item["ring_px"]),
        ring_from=_as_float("item.ring_from", item["ring_from"]),
        ring_to=_as_float("item.ring_to", item["ring_to"]),
        ring_grow=_as_float("item.ring_grow", item["ring_grow"]),
        fade_seconds=_as_float("item.fade_seconds", item["fade_seconds"]),
        fade_alpha=fade_alpha,
        safe_x=_pair("layout.safe_x", layout["safe_x"]),
        safe_y=_pair("layout.safe_y", layout["safe_y"]),
        label_y=_as_float("layout.label_y", layout["label_y"]),
        label_px=_as_int("layout.label_px", layout["label_px"]),
        caption_y=_as_float("layout.caption_y", layout["caption_y"]),
        caption_px=_as_int("layout.caption_px", layout["caption_px"]),
        caption_max_chars=max_chars,
        caption_x=_pair("layout.caption_x", layout["caption_x"]),
        timer_y=_as_float("layout.timer_y", layout["timer_y"]),
        timer_x0=_as_float("layout.timer_x0", layout["timer_x0"]),
        timer_x1=_as_float("layout.timer_x1", layout["timer_x1"]),
        timer_h=_as_float("layout.timer_h", layout["timer_h"]),
        timer_num_x0=_as_float("layout.timer_num_x0", layout["timer_num_x0"]),
        timer_num_x1=_as_float("layout.timer_num_x1", layout["timer_num_x1"]),
        timer_num_px=_as_int("layout.timer_num_px", layout["timer_num_px"]),
        timer_color=_hex("layout.timer_color", layout["timer_color"]),
        timer_red=_hex("layout.timer_red", layout["timer_red"]),
        red_seconds=_as_float("layout.red_seconds", layout["red_seconds"]),
        outro_y=_as_float("layout.outro_y", layout["outro_y"]),
        outro_px=_as_int("layout.outro_px", layout["outro_px"]),
        cta_y=_as_float("layout.cta_y", layout["cta_y"]),
        cta_px=_as_int("layout.cta_px", layout["cta_px"]),
        hook_seconds=_as_float("layout.hook_seconds", layout["hook_seconds"]),
        max_attempts=_as_int("constraints.max_attempts", constraints["max_attempts"]),
        tone_seconds=_as_float("audio.tone_seconds", audio["tone_seconds"]),
        double_tick_window=_as_float("audio.double_tick_window", audio["double_tick_window"]),
    )


def load_odd_config(
    path: str | Path,
    *,
    tier: str | None = None,
    seed: int | None = None,
    level_ids: list[int] | None = None,
) -> OddConfig:
    config_path = Path(path)
    raw = _load_mapping(config_path)
    hooks = _load_hooks(_sibling(config_path, "odd_hooks.yaml"))
    return validate_odd(raw, hooks=hooks, config_path=str(config_path), tier=tier, seed=seed, level_ids=level_ids)
