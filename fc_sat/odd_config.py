"""Odd One Out config, tiers, caption checks, and timeline.

Validation errors name the field, same as the other generators.
Frames for a segment are ``round(seconds * fps)``. With the normal tier that
is 5+6+7+8 plus four 1.4 s reveals, three 0.3 s wipes, and a 2.4 s outro:
34.9 s and 2094 frames.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
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

DIFFERENCES = ("hue", "size", "spin", "pulse")
LABELS = ("EASY", "MEDIUM", "HARD", "BRUTAL")
TIERS = ("normal", "brutal", "quick")
# Digits followed by %, including "99%" and "only 1%".
PERCENT_CLAIM = re.compile(r"\d+(?:\.\d+)?\s*%", re.IGNORECASE)

_TOP = frozenset(
    {
        "generator",
        "seed",
        "tier",
        "width",
        "height",
        "fps",
        "workers",
        "bloom_strength",
        "warmup_seconds",
        "reveal_seconds",
        "wipe_seconds",
        "outro_seconds",
        "hook",
        "cta",
        "background",
        "levels",
        "field",
        "item",
        "motion",
        "layout",
        "constraints",
        "audio",
        "tiers",
    }
)
_TIER_KEYS = frozenset(
    {
        "levels",
        "timers",
        "reveal_seconds",
        "wipe_seconds",
        "outro_seconds",
        "hue_min_distance",
        "hue_min_lightness",
        "cvd_min_distance",
        "size_ratio",
        "size_ratio_range",
        "spin_rev_s",
        "pulse_normal_hz",
        "pulse_odd_hz",
        "pulse_amplitude",
        "base_l",
        "base_c",
    }
)
_FIELD_KEYS = frozenset({"x0", "x1", "y0", "y1", "corner_radius", "margin"})
_ITEM_KEYS = frozenset({"radius", "radius_min", "specular"})
_MOTION_KEYS = frozenset(
    {"speed_mean", "speed_spread", "speed_clip", "angular_std", "omega_clip", "repulse", "substeps"}
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
        "pip_y",
        "pip_r",
        "outro_y",
        "outro_px",
        "cta_y",
        "cta_px",
        "cta_x",
        "hook_seconds",
        "pause_caption_seconds",
        "last_chance_seconds",
        "pause_levels",
    }
)
_CONSTRAINT_KEYS = frozenset({"max_attempts", "speed_low_pct", "speed_high_pct", "position_middle"})
_AUDIO_KEYS = frozenset({"silence_at_reveal", "heartbeat_bpm", "heartbeat_window", "buzz_seconds"})


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


@dataclass(frozen=True)
class LevelSpec:
    id: int
    label: str
    difference: str
    count: int
    timer: float


@dataclass(frozen=True)
class TierParams:
    name: str
    hue_min_distance: float
    hue_min_lightness: float
    cvd_min_distance: float
    size_ratio: float
    spin_rev_s: float
    pulse_normal_hz: float
    pulse_odd_hz: float
    pulse_amplitude: float
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
    bloom_strength: float
    warmup_seconds: float
    reveal_seconds: float
    wipe_seconds: float
    outro_seconds: float
    hook: str
    hooks: tuple[str, ...]
    captions: tuple[str, ...]
    cta: str
    background: str
    levels: tuple[LevelSpec, ...]
    field: tuple[float, float, float, float, float, float]
    radius: float
    radius_min: float
    specular: float
    speed_mean: float
    speed_spread: float
    speed_clip: tuple[float, float]
    angular_std: float
    omega_clip: float
    repulse: float
    substeps: int
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
    pip_y: float
    pip_r: float
    outro_y: float
    outro_px: int
    cta_y: float
    cta_px: int
    cta_x: tuple[float, float]
    hook_seconds: float
    pause_caption_seconds: float
    last_chance_seconds: float
    pause_levels: tuple[int, ...]
    max_attempts: int
    speed_low_pct: float
    speed_high_pct: float
    position_middle: float
    silence_at_reveal: float
    heartbeat_bpm: tuple[float, float]
    heartbeat_window: float
    buzz_seconds: float
    config_path: str

    @property
    def field_x0(self) -> float:
        return self.field[0]

    @property
    def field_x1(self) -> float:
        return self.field[1]

    @property
    def field_y0(self) -> float:
        return self.field[2]

    @property
    def field_y1(self) -> float:
        return self.field[3]

    @property
    def corner_radius(self) -> float:
        return self.field[4]

    @property
    def margin(self) -> float:
        return self.field[5]

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "generator": self.generator,
            "seed": self.seed,
            "tier": self.tier_name,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "workers": self.workers,
            "bloom_strength": self.bloom_strength,
            "warmup_seconds": self.warmup_seconds,
            "reveal_seconds": self.reveal_seconds,
            "wipe_seconds": self.wipe_seconds,
            "outro_seconds": self.outro_seconds,
            "hook": self.hook,
            "cta": self.cta,
            "levels": [
                {
                    "id": level.id,
                    "label": level.label,
                    "difference": level.difference,
                    "count": level.count,
                    "timer": level.timer,
                }
                for level in self.levels
            ],
            "tier_params": {
                "hue_min_distance": self.tier.hue_min_distance,
                "hue_min_lightness": self.tier.hue_min_lightness,
                "cvd_min_distance": self.tier.cvd_min_distance,
                "size_ratio": self.tier.size_ratio,
                "spin_rev_s": self.tier.spin_rev_s,
                "pulse_normal_hz": self.tier.pulse_normal_hz,
                "pulse_odd_hz": self.tier.pulse_odd_hz,
                "pulse_amplitude": self.tier.pulse_amplitude,
            },
        }


def build_timeline(cfg: OddConfig) -> tuple[Segment, ...]:
    """Play, reveal, wipe between levels, then the outro. Frame counts sum the rounded parts."""
    segments: list[Segment] = []
    cursor_s = 0.0
    cursor_f = 0
    levels = cfg.levels
    for index, level in enumerate(levels):
        for kind, duration in (
            ("play", level.timer),
            ("reveal", cfg.reveal_seconds),
        ):
            n = frames_for(duration, cfg.fps)
            segments.append(Segment(kind, level.id, cursor_s, duration, cursor_f, n))
            cursor_s += duration
            cursor_f += n
        if index < len(levels) - 1:
            n = frames_for(cfg.wipe_seconds, cfg.fps)
            segments.append(Segment("wipe", level.id, cursor_s, cfg.wipe_seconds, cursor_f, n))
            cursor_s += cfg.wipe_seconds
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
    left = _as_float(field, value[0])
    right = _as_float(field, value[1])
    if right <= left:
        _fail(field, "must be ordered low to high")
    return left, right


def _validate_tier(name: str, raw: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        _fail(f"tiers.{name}", "must be a mapping")
    unknown = set(raw) - _TIER_KEYS
    if unknown:
        _fail(f"tiers.{name}.{sorted(unknown)[0]}", "is not a config field")
    required = _TIER_KEYS - {"levels", "timers", "reveal_seconds", "wipe_seconds", "outro_seconds"}
    missing = required - set(raw)
    if missing:
        _fail(f"tiers.{name}.{sorted(missing)[0]}", "is required")
    return raw


def _tier_params(name: str, raw: dict[str, Any]) -> TierParams:
    hue = _as_float(f"tiers.{name}.hue_min_distance", raw["hue_min_distance"])
    light = _as_float(f"tiers.{name}.hue_min_lightness", raw["hue_min_lightness"])
    cvd = _as_float(f"tiers.{name}.cvd_min_distance", raw["cvd_min_distance"])
    if cvd < 0.10 - 1e-9:
        _fail(f"tiers.{name}.cvd_min_distance", "must stay at least 0.10")
    if hue <= 0 or light <= 0:
        _fail(f"tiers.{name}.hue_min_distance", "hue distance and lightness must be positive")
    ratio = _as_float(f"tiers.{name}.size_ratio", raw["size_ratio"])
    bounds = _pair(f"tiers.{name}.size_ratio_range", raw["size_ratio_range"])
    if not (bounds[0] - 1e-9 <= ratio <= bounds[1] + 1e-9):
        _fail(f"tiers.{name}.size_ratio", f"must sit inside {bounds}, got {ratio}")
    spin = _as_float(f"tiers.{name}.spin_rev_s", raw["spin_rev_s"])
    normal_hz = _as_float(f"tiers.{name}.pulse_normal_hz", raw["pulse_normal_hz"])
    odd_hz = _as_float(f"tiers.{name}.pulse_odd_hz", raw["pulse_odd_hz"])
    amp = _as_float(f"tiers.{name}.pulse_amplitude", raw["pulse_amplitude"])
    if spin <= 0:
        _fail(f"tiers.{name}.spin_rev_s", "must be positive")
    if normal_hz <= 0 or odd_hz <= 0 or normal_hz > 3.0 or odd_hz > 3.0:
        _fail(f"tiers.{name}.pulse_odd_hz", "pulse rates must be in (0, 3] Hz")
    if abs(odd_hz - normal_hz) < 1e-6:
        _fail(f"tiers.{name}.pulse_odd_hz", "must differ from the normal rate")
    if not 0 < amp <= 0.5:
        _fail(f"tiers.{name}.pulse_amplitude", "must be in (0, 0.5]")
    base_l = _as_float(f"tiers.{name}.base_l", raw["base_l"])
    base_c = _as_float(f"tiers.{name}.base_c", raw["base_c"])
    if not 0 < base_l < 1 or base_c < 0:
        _fail(f"tiers.{name}.base_l", "lightness must be in (0, 1) and chroma must be >= 0")
    return TierParams(name, hue, light, cvd, ratio, spin, normal_hz, odd_hz, amp, base_l, base_c)


def _levels(raw_levels: Any, tier_raw: dict[str, Any]) -> tuple[LevelSpec, ...]:
    if not isinstance(raw_levels, list) or not raw_levels:
        _fail("levels", "must be a non-empty list")
    found: dict[int, LevelSpec] = {}
    for index, item in enumerate(raw_levels):
        field = f"levels[{index}]"
        if not isinstance(item, dict):
            _fail(field, "must be a mapping")
        unknown = set(item) - {"id", "label", "difference", "count", "timer"}
        if unknown:
            _fail(f"{field}.{sorted(unknown)[0]}", "is not a config field")
        ident = _as_int(f"{field}.id", item.get("id"))
        label = _as_str(f"{field}.label", item.get("label"))
        if label not in LABELS:
            _fail(f"{field}.label", f"must be one of {', '.join(LABELS)}")
        diff = _as_str(f"{field}.difference", item.get("difference"))
        if diff not in DIFFERENCES:
            _fail(f"{field}.difference", f"must be one of {', '.join(DIFFERENCES)}; a new type needs a measurer")
        count = _as_int(f"{field}.count", item.get("count"))
        timer = _as_float(f"{field}.timer", item.get("timer"))
        if count < 2:
            _fail(f"{field}.count", "must be at least 2 so one item can be odd")
        if timer <= 0:
            _fail(f"{field}.timer", "must be positive")
        if ident in found:
            _fail(f"{field}.id", f"duplicates level {ident}")
        found[ident] = LevelSpec(ident, label, diff, count, timer)
    selected = tier_raw.get("levels")
    if selected is None:
        order = [level.id for level in found.values()]
    else:
        if not isinstance(selected, list) or not selected:
            _fail("tiers.levels", "must be a non-empty list of level ids")
        order = []
        for ident in selected:
            ident_i = _as_int("tiers.levels", ident)
            if ident_i not in found:
                _fail("tiers.levels", f"unknown level {ident_i}")
            order.append(ident_i)
    timers = tier_raw.get("timers") or {}
    if not isinstance(timers, dict):
        _fail("tiers.timers", "must be a mapping of level id to seconds")
    levels: list[LevelSpec] = []
    for ident in order:
        level = found[ident]
        if ident in timers or str(ident) in timers:
            raw_timer = timers[ident] if ident in timers else timers[str(ident)]
            timer = _as_float(f"tiers.timers.{ident}", raw_timer)
            if timer <= 0:
                _fail(f"tiers.timers.{ident}", "must be positive")
            level = replace(level, timer=timer)
        levels.append(level)
    return tuple(levels)


def validate_odd(
    raw: dict[str, Any],
    *,
    hooks: tuple[str, ...],
    captions: tuple[str, ...],
    config_path: str,
    tier_name: str | None = None,
    seed: int | None = None,
    level_ids: list[int] | None = None,
) -> OddConfig:
    if not isinstance(raw, dict):
        _fail("config", "root must be a mapping")
    unknown = set(raw) - _TOP
    if unknown:
        _fail(sorted(unknown)[0], "is not a config field")
    missing = _TOP - set(raw)
    if missing:
        _fail(sorted(missing)[0], "is required")
    generator = _as_str("generator", raw["generator"])
    if generator != "odd":
        _fail("generator", "must be 'odd'")
    chosen_tier = tier_name if tier_name is not None else _as_str("tier", raw["tier"])
    if chosen_tier not in TIERS:
        _fail("tier", f"must be one of {', '.join(TIERS)}")
    tiers_raw = raw["tiers"]
    if not isinstance(tiers_raw, dict):
        _fail("tiers", "must be a mapping")
    tier_unknown = set(tiers_raw) - set(TIERS)
    if tier_unknown:
        _fail(f"tiers.{sorted(tier_unknown)[0]}", "is not a config field")
    for name in TIERS:
        if name not in tiers_raw:
            _fail(f"tiers.{name}", "is required")
        _validate_tier(name, tiers_raw[name])
    tier_raw = tiers_raw[chosen_tier]
    tier = _tier_params(chosen_tier, tier_raw)
    levels = _levels(raw["levels"], tier_raw)
    if level_ids is not None:
        by_id = {level.id: level for level in _levels(raw["levels"], {})}
        picked: list[LevelSpec] = []
        for ident in level_ids:
            if ident not in by_id:
                _fail("levels", f"unknown level {ident}")
            level = by_id[ident]
            timers = tier_raw.get("timers") or {}
            if ident in timers or str(ident) in timers:
                raw_timer = timers[ident] if ident in timers else timers[str(ident)]
                level = replace(level, timer=_as_float(f"tiers.timers.{ident}", raw_timer))
            picked.append(level)
        if not picked:
            _fail("levels", "must keep at least one level")
        levels = tuple(picked)

    layout = _section(raw, "layout", _LAYOUT_KEYS)
    max_chars = _as_int("layout.caption_max_chars", layout["caption_max_chars"])
    if max_chars < 8:
        _fail("layout.caption_max_chars", "must be at least 8")
    hook = validate_caption(_as_str("hook", raw["hook"]), field="hook", max_chars=max_chars)
    checked_hooks = tuple(validate_caption(item, field="hooks", max_chars=max_chars) for item in hooks)
    if hook not in checked_hooks:
        _fail("hook", "must be one of the lines in odd_hooks.yaml")
    checked_captions = tuple(validate_caption(item, field="captions", max_chars=max_chars) for item in captions)
    cta = validate_caption(_as_str("cta", raw["cta"]), field="cta", max_chars=max_chars)

    width = _as_int("width", raw["width"])
    height = _as_int("height", raw["height"])
    fps = _as_int("fps", raw["fps"])
    if width != 1080 or height != 1920:
        _fail("width", "full canvas must be 1080x1920")
    if fps != 60:
        _fail("fps", "must be 60")
    workers = _as_int("workers", raw["workers"])
    if workers < 0:
        _fail("workers", "must be >= 0")
    bloom = _as_float("bloom_strength", raw["bloom_strength"])
    if bloom < 0:
        _fail("bloom_strength", "must be >= 0")
    warmup = _as_float("warmup_seconds", raw["warmup_seconds"])
    reveal = _as_float("reveal_seconds", raw["reveal_seconds"])
    wipe = _as_float("wipe_seconds", raw["wipe_seconds"])
    outro = _as_float("outro_seconds", raw["outro_seconds"])
    if "reveal_seconds" in tier_raw:
        reveal = _as_float(f"tiers.{chosen_tier}.reveal_seconds", tier_raw["reveal_seconds"])
    if "wipe_seconds" in tier_raw:
        wipe = _as_float(f"tiers.{chosen_tier}.wipe_seconds", tier_raw["wipe_seconds"])
    if "outro_seconds" in tier_raw:
        outro = _as_float(f"tiers.{chosen_tier}.outro_seconds", tier_raw["outro_seconds"])
    for field, value in (
        ("warmup_seconds", warmup),
        ("reveal_seconds", reveal),
        ("wipe_seconds", wipe),
        ("outro_seconds", outro),
    ):
        if value <= 0:
            _fail(field, "must be positive")

    field = _section(raw, "field", _FIELD_KEYS)
    fx0 = _as_float("field.x0", field["x0"])
    fx1 = _as_float("field.x1", field["x1"])
    fy0 = _as_float("field.y0", field["y0"])
    fy1 = _as_float("field.y1", field["y1"])
    corner = _as_float("field.corner_radius", field["corner_radius"])
    margin = _as_float("field.margin", field["margin"])
    if fx1 <= fx0 or fy1 <= fy0:
        _fail("field", "box must have positive area")
    if corner < 0 or margin < 0:
        _fail("field.margin", "corner radius and margin must be >= 0")

    item = _section(raw, "item", _ITEM_KEYS)
    radius = _as_float("item.radius", item["radius"])
    radius_min = _as_float("item.radius_min", item["radius_min"])
    specular = _as_float("item.specular", item["specular"])
    if radius < radius_min or radius_min <= 0:
        _fail("item.radius", "radius must be >= radius_min > 0")
    if not 0 < specular < 1:
        _fail("item.specular", "must be in (0, 1)")

    motion = _section(raw, "motion", _MOTION_KEYS)
    speed_mean = _as_float("motion.speed_mean", motion["speed_mean"])
    speed_spread = _as_float("motion.speed_spread", motion["speed_spread"])
    speed_clip = _pair("motion.speed_clip", motion["speed_clip"])
    angular_std = _as_float("motion.angular_std", motion["angular_std"])
    omega_clip = _as_float("motion.omega_clip", motion["omega_clip"])
    repulse = _as_float("motion.repulse", motion["repulse"])
    substeps = _as_int("motion.substeps", motion["substeps"])
    if speed_mean <= 0 or speed_spread < 0 or angular_std < 0 or omega_clip <= 0:
        _fail("motion.speed_mean", "speed, spread, and angular settings must be non-negative, clips positive")
    if repulse < 2.0:
        _fail("motion.repulse", "must be >= 2 so repulsion starts before discs touch")
    if substeps < 1:
        _fail("motion.substeps", "must be >= 1")

    safe_x = _pair("layout.safe_x", layout["safe_x"])
    safe_y = _pair("layout.safe_y", layout["safe_y"])
    caption_x = _pair("layout.caption_x", layout["caption_x"])
    cta_x = _pair("layout.cta_x", layout["cta_x"])
    pause_raw = layout["pause_levels"]
    if not isinstance(pause_raw, list) or not pause_raw:
        _fail("layout.pause_levels", "must be a list of level ids")
    pause_levels = tuple(_as_int("layout.pause_levels", item) for item in pause_raw)

    constraints = _section(raw, "constraints", _CONSTRAINT_KEYS)
    max_attempts = _as_int("constraints.max_attempts", constraints["max_attempts"])
    if max_attempts < 1:
        _fail("constraints.max_attempts", "must be >= 1")
    low_pct = _as_float("constraints.speed_low_pct", constraints["speed_low_pct"])
    high_pct = _as_float("constraints.speed_high_pct", constraints["speed_high_pct"])
    middle = _as_float("constraints.position_middle", constraints["position_middle"])
    if not 0 <= low_pct < high_pct <= 100:
        _fail("constraints.speed_low_pct", "percentiles must be ordered inside 0..100")
    if not 0 < middle <= 1:
        _fail("constraints.position_middle", "must be in (0, 1]")

    audio = _section(raw, "audio", _AUDIO_KEYS)
    silence = _as_float("audio.silence_at_reveal", audio["silence_at_reveal"])
    bpm = _pair("audio.heartbeat_bpm", audio["heartbeat_bpm"])
    heart_window = _as_float("audio.heartbeat_window", audio["heartbeat_window"])
    buzz = _as_float("audio.buzz_seconds", audio["buzz_seconds"])
    if silence < 0 or heart_window <= 0 or buzz <= 0:
        _fail("audio.silence_at_reveal", "windows must be positive (silence may be 0)")

    background = _as_str("background", raw["background"])
    if not re.fullmatch(r"#[0-9A-Fa-f]{6}", background):
        _fail("background", "must be a #RRGGBB color")

    return OddConfig(
        generator=generator,
        seed=_as_int("seed", raw["seed"]) if seed is None else int(seed),
        tier_name=chosen_tier,
        tier=tier,
        width=width,
        height=height,
        fps=fps,
        workers=workers,
        bloom_strength=bloom,
        warmup_seconds=warmup,
        reveal_seconds=reveal,
        wipe_seconds=wipe,
        outro_seconds=outro,
        hook=hook,
        hooks=checked_hooks,
        captions=checked_captions,
        cta=cta,
        background=background,
        levels=levels,
        field=(fx0, fx1, fy0, fy1, corner, margin),
        radius=radius,
        radius_min=radius_min,
        specular=specular,
        speed_mean=speed_mean,
        speed_spread=speed_spread,
        speed_clip=speed_clip,
        angular_std=angular_std,
        omega_clip=omega_clip,
        repulse=repulse,
        substeps=substeps,
        safe_x=safe_x,
        safe_y=safe_y,
        label_y=_as_float("layout.label_y", layout["label_y"]),
        label_px=_as_int("layout.label_px", layout["label_px"]),
        caption_y=_as_float("layout.caption_y", layout["caption_y"]),
        caption_px=_as_int("layout.caption_px", layout["caption_px"]),
        caption_max_chars=max_chars,
        caption_x=caption_x,
        timer_y=_as_float("layout.timer_y", layout["timer_y"]),
        timer_x0=_as_float("layout.timer_x0", layout["timer_x0"]),
        timer_x1=_as_float("layout.timer_x1", layout["timer_x1"]),
        timer_h=_as_float("layout.timer_h", layout["timer_h"]),
        timer_num_x0=_as_float("layout.timer_num_x0", layout["timer_num_x0"]),
        timer_num_x1=_as_float("layout.timer_num_x1", layout["timer_num_x1"]),
        timer_num_px=_as_int("layout.timer_num_px", layout["timer_num_px"]),
        pip_y=_as_float("layout.pip_y", layout["pip_y"]),
        pip_r=_as_float("layout.pip_r", layout["pip_r"]),
        outro_y=_as_float("layout.outro_y", layout["outro_y"]),
        outro_px=_as_int("layout.outro_px", layout["outro_px"]),
        cta_y=_as_float("layout.cta_y", layout["cta_y"]),
        cta_px=_as_int("layout.cta_px", layout["cta_px"]),
        cta_x=cta_x,
        hook_seconds=_as_float("layout.hook_seconds", layout["hook_seconds"]),
        pause_caption_seconds=_as_float("layout.pause_caption_seconds", layout["pause_caption_seconds"]),
        last_chance_seconds=_as_float("layout.last_chance_seconds", layout["last_chance_seconds"]),
        pause_levels=pause_levels,
        max_attempts=max_attempts,
        speed_low_pct=low_pct,
        speed_high_pct=high_pct,
        position_middle=middle,
        silence_at_reveal=silence,
        heartbeat_bpm=bpm,
        heartbeat_window=heart_window,
        buzz_seconds=buzz,
        config_path=config_path,
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
    captions = _load_hooks(_sibling(config_path, "odd_captions.yaml"))
    return validate_odd(
        raw,
        hooks=hooks,
        captions=captions,
        config_path=str(config_path),
        tier_name=tier,
        seed=seed,
        level_ids=level_ids,
    )
