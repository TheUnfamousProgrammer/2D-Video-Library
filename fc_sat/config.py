"""Dataclass config, validation, and YAML/JSON loading."""

from __future__ import annotations

import json
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """A config field failed type or range validation."""


def _fail(field: str, message: str) -> None:
    raise ConfigError(f"config field '{field}' {message}")


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


def _as_str(field: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(field, f"must be a non-empty string, got {value!r}")
    return value


@dataclass(frozen=True)
class Config:
    seed: int
    width: int
    height: int
    fps: int
    growth_seconds: float
    hold_seconds: float
    implode_seconds: float
    beat_seconds: float
    reset_seconds: float
    ring_cx: float
    ring_cy: float
    ring_radius: float
    ring_thickness: float
    gravity: float
    restitution: float
    radius_base: float
    radius_exp: float
    radius_min: float
    radius_max: float
    cap: int
    spawn_cooldown: float
    spawn_jitter_deg: float
    throttle: tuple[tuple[float, float], ...]
    min_bounce_speed: float
    max_speed: float
    palette: str
    palette_stops: tuple[str, ...]
    hook: str
    show_wait_text: bool
    wait_text: str
    show_cta: bool
    cta_text: str
    collisions: bool
    sound_preset: str
    audio_offset_ms: float
    bloom_strength: float
    workers: int
    tune_max_attempts: int
    tune_speed_range: tuple[float, float]
    tune_gravity_range: tuple[float, float]
    tune_seed_offset_range: tuple[int, int]
    hooks: tuple[str, ...] = ()

    @property
    def total_seconds(self) -> float:
        return (
            self.growth_seconds
            + self.hold_seconds
            + self.implode_seconds
            + self.beat_seconds
            + self.reset_seconds
        )

    def phase_durations(self) -> tuple[tuple[str, float], ...]:
        return (
            ("growth", self.growth_seconds),
            ("hold", self.hold_seconds),
            ("implode", self.implode_seconds),
            ("beat", self.beat_seconds),
            ("reset", self.reset_seconds),
        )

    def frame_plan(self, preview: bool = False) -> tuple[int, tuple[tuple[str, int], ...]]:
        """Return (fps, ((phase_name, n_frames), ...)).

        Frame counts are round(seconds * fps). At the default 60 fps timeline
        that is 1140 + 30 + 72 + 18 + 30 = 1290 frames (21.5 s).
        """
        fps = 30 if preview else self.fps
        plan = []
        for name, seconds in self.phase_durations():
            plan.append((name, max(1, int(round(seconds * fps)))))
        return fps, tuple(plan)

    def n_frames(self, preview: bool = False) -> int:
        _fps, plan = self.frame_plan(preview)
        return sum(n for _name, n in plan)

    def to_public_dict(self) -> dict[str, Any]:
        """JSON-ready view of the resolved config, including derived timeline."""
        return {
            "seed": self.seed,
            "canvas": {"width": self.width, "height": self.height},
            "fps": self.fps,
            "growth_seconds": self.growth_seconds,
            "hold_seconds": self.hold_seconds,
            "implode_seconds": self.implode_seconds,
            "beat_seconds": self.beat_seconds,
            "reset_seconds": self.reset_seconds,
            "total_seconds": self.total_seconds,
            "ring": {
                "center": [self.ring_cx, self.ring_cy],
                "radius": self.ring_radius,
                "thickness": self.ring_thickness,
            },
            "gravity": self.gravity,
            "restitution": self.restitution,
            "radius_base": self.radius_base,
            "radius_exp": self.radius_exp,
            "radius_min": self.radius_min,
            "radius_max": self.radius_max,
            "cap": self.cap,
            "spawn_cooldown": self.spawn_cooldown,
            "spawn_jitter_deg": self.spawn_jitter_deg,
            "throttle": [list(pair) for pair in self.throttle],
            "min_bounce_speed": self.min_bounce_speed,
            "max_speed": self.max_speed,
            "palette": self.palette,
            "palette_stops": list(self.palette_stops),
            "hook": self.hook,
            "show_wait_text": self.show_wait_text,
            "wait_text": self.wait_text,
            "show_cta": self.show_cta,
            "cta_text": self.cta_text,
            "collisions": self.collisions,
            "sound_preset": self.sound_preset,
            "audio_offset_ms": self.audio_offset_ms,
            "bloom_strength": self.bloom_strength,
            "workers": self.workers,
            "tune": {
                "max_attempts": self.tune_max_attempts,
                "speed_range": list(self.tune_speed_range),
                "gravity_range": list(self.tune_gravity_range),
                "seed_offset_range": list(self.tune_seed_offset_range),
            },
            "hooks": list(self.hooks),
        }


def sim_fingerprint_payload(cfg: Config) -> dict[str, Any]:
    """Fields that change the simulation. Palette, hook, and bloom are excluded."""
    return {
        "seed": cfg.seed,
        "fps": cfg.fps,
        "growth_seconds": cfg.growth_seconds,
        "ring_cx": cfg.ring_cx,
        "ring_cy": cfg.ring_cy,
        "ring_radius": cfg.ring_radius,
        "gravity": cfg.gravity,
        "restitution": cfg.restitution,
        "radius_base": cfg.radius_base,
        "radius_exp": cfg.radius_exp,
        "radius_min": cfg.radius_min,
        "radius_max": cfg.radius_max,
        "cap": cfg.cap,
        "spawn_cooldown": cfg.spawn_cooldown,
        "spawn_jitter_deg": cfg.spawn_jitter_deg,
        "throttle": [list(pair) for pair in cfg.throttle],
        "min_bounce_speed": cfg.min_bounce_speed,
        "max_speed": cfg.max_speed,
        "collisions": cfg.collisions,
        "tune_max_attempts": cfg.tune_max_attempts,
        "tune_speed_range": list(cfg.tune_speed_range),
        "tune_gravity_range": list(cfg.tune_gravity_range),
        "tune_seed_offset_range": list(cfg.tune_seed_offset_range),
    }


def _range_pair(field: str, value: Any, *, integer: bool = False) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        _fail(field, f"must be a pair of numbers, got {value!r}")
    lo = _as_int(field, value[0]) if integer else _as_float(field, value[0])
    hi = _as_int(field, value[1]) if integer else _as_float(field, value[1])
    if lo > hi:
        _fail(field, f"must be ordered low to high, got {value!r}")
    return (lo, hi)  # type: ignore[return-value]


def _parse_throttle(raw: Any, cap: int) -> tuple[tuple[float, float], ...]:
    field = "throttle"
    if not isinstance(raw, list) or len(raw) < 2:
        _fail(field, "must be a list of at least two [fraction, count] keyframes")
    frames: list[tuple[float, float]] = []
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            _fail(field, f"keyframe must be [fraction, count], got {item!r}")
        frac = _as_float(field, item[0])
        if isinstance(item[1], str):
            if item[1].strip().lower() != "cap":
                _fail(field, f"count must be a number or 'cap', got {item[1]!r}")
            count = float(cap)
        else:
            count = _as_float(field, item[1])
        if not 0.0 <= frac <= 1.0:
            _fail(field, f"fraction must be in [0, 1], got {frac}")
        if count < 1:
            _fail(field, f"count must be >= 1, got {count}")
        frames.append((frac, count))
    if frames[0][0] != 0.0:
        _fail(field, "must start at fraction 0.00")
    for prev, nxt in zip(frames, frames[1:]):
        if nxt[0] <= prev[0]:
            _fail(field, "fractions must be strictly increasing")
    if frames[-1][0] > 1.0:
        _fail(field, "last fraction must be <= 1")
    return tuple(frames)


def _load_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        data = yaml.safe_load(text)
    if not isinstance(data, dict):
        _fail(str(path), "root must be a mapping")
    return data


def _load_palettes(path: Path) -> dict[str, tuple[str, ...]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data:
        _fail("palettes", f"{path} must be a mapping of name -> hex stops")
    out: dict[str, tuple[str, ...]] = {}
    for name, stops in data.items():
        if not isinstance(stops, list) or not 3 <= len(stops) <= 5:
            _fail("palettes", f"{name} must have 3 to 5 hex stops")
        cleaned = []
        for stop in stops:
            if not isinstance(stop, str) or not _hex_ok(stop):
                _fail("palettes", f"{name} has an invalid hex stop {stop!r}")
            cleaned.append(stop.upper() if stop.startswith("#") else "#" + stop.upper())
        out[str(name)] = tuple(cleaned)
    return out


def _hex_ok(value: str) -> bool:
    text = value[1:] if value.startswith("#") else value
    if len(text) != 6:
        return False
    try:
        int(text, 16)
    except ValueError:
        return False
    return True


def _load_hooks(path: Path) -> tuple[str, ...]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        _fail("hooks", f"{path} must be a list of strings")
    hooks = []
    for item in data:
        if not isinstance(item, str) or not item.strip():
            _fail("hooks", f"each hook must be a non-empty string, got {item!r}")
        hooks.append(item.strip())
    return tuple(hooks)


def _sibling(config_path: Path, name: str) -> Path:
    return config_path.parent / name


def validate_and_build(
    raw: dict[str, Any],
    *,
    palettes: dict[str, tuple[str, ...]],
    hooks: tuple[str, ...],
) -> Config:
    canvas = raw.get("canvas", {})
    if not isinstance(canvas, dict):
        _fail("canvas", "must be a mapping with width and height")
    ring = raw.get("ring", {})
    if not isinstance(ring, dict):
        _fail("ring", "must be a mapping")
    center = ring.get("center", [540, 880])
    if not isinstance(center, (list, tuple)) or len(center) != 2:
        _fail("ring.center", f"must be [x, y], got {center!r}")
    tune = raw.get("tune", {})
    if tune is None:
        tune = {}
    if not isinstance(tune, dict):
        _fail("tune", "must be a mapping")

    width = _as_int("canvas.width", canvas.get("width", 1080))
    height = _as_int("canvas.height", canvas.get("height", 1920))
    if width < 16 or height < 16:
        _fail("canvas", "width and height must be >= 16")
    fps = _as_int("fps", raw.get("fps", 60))
    if fps < 1 or fps > 120:
        _fail("fps", "must be in [1, 120]")

    growth = _as_float("growth_seconds", raw.get("growth_seconds", 19.0))
    hold = _as_float("hold_seconds", raw.get("hold_seconds", 0.5))
    implode = _as_float("implode_seconds", raw.get("implode_seconds", 1.2))
    beat = _as_float("beat_seconds", raw.get("beat_seconds", 0.3))
    reset = _as_float("reset_seconds", raw.get("reset_seconds", 0.5))
    for name, value in (
        ("growth_seconds", growth),
        ("hold_seconds", hold),
        ("implode_seconds", implode),
        ("beat_seconds", beat),
        ("reset_seconds", reset),
    ):
        if value <= 0:
            _fail(name, "must be > 0")
    total = growth + hold + implode + beat + reset
    if total > 30.0:
        _fail("growth_seconds", f"total duration {total:.3f}s exceeds 30s")

    ring_radius = _as_float("ring.radius", ring.get("radius", 380))
    ring_thickness = _as_float("ring.thickness", ring.get("thickness", 6))
    if ring_radius < 20:
        _fail("ring.radius", "must be >= 20")
    if ring_thickness <= 0 or ring_thickness > ring_radius / 2:
        _fail("ring.thickness", "must be in (0, ring.radius/2]")

    gravity = _as_float("gravity", raw.get("gravity", 900))
    if gravity < 0 or gravity > 5000:
        _fail("gravity", "must be in [0, 5000]")
    restitution = _as_float("restitution", raw.get("restitution", 1.0))
    if not 0 < restitution <= 1:
        _fail("restitution", "must be in (0, 1]")

    radius_base = _as_float("radius_base", raw.get("radius_base", 28))
    radius_exp = _as_float("radius_exp", raw.get("radius_exp", -0.28))
    radius_min = _as_float("radius_min", raw.get("radius_min", 8))
    radius_max = _as_float("radius_max", raw.get("radius_max", 28))
    if radius_base <= 0:
        _fail("radius_base", "must be > 0")
    if radius_min <= 0:
        _fail("radius_min", "must be > 0")
    if radius_max < radius_min:
        _fail("radius_min", f"must be <= radius_max ({radius_max})")
    if radius_min > 80 or radius_max > 80:
        _fail("radius_min", "radius_min and radius_max must be <= 80")

    cap = _as_int("cap", raw.get("cap", 1000))
    if not 1 <= cap <= 5000:
        _fail("cap", "must be in [1, 5000]")
    cooldown = _as_float("spawn_cooldown", raw.get("spawn_cooldown", 0.15))
    if cooldown < 0 or cooldown > 5:
        _fail("spawn_cooldown", "must be in [0, 5]")
    jitter = _as_float("spawn_jitter_deg", raw.get("spawn_jitter_deg", 25))
    if jitter < 0 or jitter > 180:
        _fail("spawn_jitter_deg", "must be in [0, 180]")

    throttle = _parse_throttle(raw.get("throttle"), cap)
    min_bounce = _as_float("min_bounce_speed", raw.get("min_bounce_speed", 450))
    max_speed = _as_float("max_speed", raw.get("max_speed", 1600))
    if min_bounce <= 0:
        _fail("min_bounce_speed", "must be > 0")
    if max_speed < min_bounce:
        _fail("max_speed", "must be >= min_bounce_speed")
    if max_speed > 8000:
        _fail("max_speed", "must be <= 8000")

    palette_name = _as_str("palette", raw.get("palette", "sunset"))
    if palette_name not in palettes:
        known = ", ".join(sorted(palettes))
        _fail("palette", f"unknown name {palette_name!r}. Known: {known}")

    hook = _as_str("hook", raw.get("hook", hooks[0] if hooks else "Every bounce adds a ball"))
    show_wait = _as_bool("show_wait_text", raw.get("show_wait_text", False))
    wait_text = raw.get("wait_text", "Wait for it...")
    if not isinstance(wait_text, str):
        _fail("wait_text", "must be a string")
    show_cta = _as_bool("show_cta", raw.get("show_cta", False))
    cta_text = raw.get("cta_text", "Subscribe")
    if not isinstance(cta_text, str):
        _fail("cta_text", "must be a string")

    collisions = _as_bool("collisions", raw.get("collisions", False))
    sound = _as_str("sound_preset", raw.get("sound_preset", "mallet"))
    if sound != "mallet":
        _fail("sound_preset", "must be 'mallet'")
    offset = _as_float("audio_offset_ms", raw.get("audio_offset_ms", 0))
    if offset < -500 or offset > 500:
        _fail("audio_offset_ms", "must be in [-500, 500]")
    bloom = _as_float("bloom_strength", raw.get("bloom_strength", 0.6))
    if bloom < 0 or bloom > 2:
        _fail("bloom_strength", "must be in [0, 2]")
    workers = _as_int("workers", raw.get("workers", 0))
    if workers < 0 or workers > 64:
        _fail("workers", "must be in [0, 64] (0 means cpu_count - 1)")

    attempts = _as_int("tune.max_attempts", tune.get("max_attempts", 60))
    if not 1 <= attempts <= 200:
        _fail("tune.max_attempts", "must be in [1, 200]")
    speed_range = _range_pair("tune.speed_range", tune.get("speed_range", [700, 1500]))
    gravity_range = _range_pair("tune.gravity_range", tune.get("gravity_range", [200, 1400]))
    offset_range = _range_pair(
        "tune.seed_offset_range",
        tune.get("seed_offset_range", [0, 59]),
        integer=True,
    )
    if speed_range[0] <= 0:
        _fail("tune.speed_range", "speeds must be > 0")
    if gravity_range[0] < 0:
        _fail("tune.gravity_range", "gravity must be >= 0")
    if offset_range[0] < 0:
        _fail("tune.seed_offset_range", "offsets must be >= 0")

    seed = _as_int("seed", raw.get("seed", 1))
    if seed < 0:
        _fail("seed", "must be >= 0")

    return Config(
        seed=seed,
        width=width,
        height=height,
        fps=fps,
        growth_seconds=growth,
        hold_seconds=hold,
        implode_seconds=implode,
        beat_seconds=beat,
        reset_seconds=reset,
        ring_cx=_as_float("ring.center", center[0]),
        ring_cy=_as_float("ring.center", center[1]),
        ring_radius=ring_radius,
        ring_thickness=ring_thickness,
        gravity=gravity,
        restitution=restitution,
        radius_base=radius_base,
        radius_exp=radius_exp,
        radius_min=radius_min,
        radius_max=radius_max,
        cap=cap,
        spawn_cooldown=cooldown,
        spawn_jitter_deg=jitter,
        throttle=throttle,
        min_bounce_speed=min_bounce,
        max_speed=max_speed,
        palette=palette_name,
        palette_stops=palettes[palette_name],
        hook=hook,
        show_wait_text=show_wait,
        wait_text=wait_text,
        show_cta=show_cta,
        cta_text=cta_text,
        collisions=collisions,
        sound_preset=sound,
        audio_offset_ms=offset,
        bloom_strength=bloom,
        workers=workers,
        tune_max_attempts=attempts,
        tune_speed_range=(float(speed_range[0]), float(speed_range[1])),
        tune_gravity_range=(float(gravity_range[0]), float(gravity_range[1])),
        tune_seed_offset_range=(int(offset_range[0]), int(offset_range[1])),
        hooks=hooks,
    )


def load_config(path: str | Path) -> Config:
    config_path = Path(path)
    raw = _load_mapping(config_path)
    palette_path = _sibling(config_path, "palettes.yaml")
    hooks_path = _sibling(config_path, "hooks.yaml")
    if not palette_path.exists():
        _fail("palettes", f"missing {palette_path}")
    if not hooks_path.exists():
        _fail("hooks", f"missing {hooks_path}")
    return validate_and_build(raw, palettes=_load_palettes(palette_path), hooks=_load_hooks(hooks_path))


def with_overrides(cfg: Config, **changes: Any) -> Config:
    known = {item.name for item in fields(Config)}
    for key in changes:
        if key not in known:
            _fail(key, "is not a config field")
    updated = replace(cfg, **changes)
    # Re-run numeric checks that overrides can break.
    if updated.radius_min <= 0 or updated.radius_min > updated.radius_max:
        _fail("radius_min", f"must be in (0, radius_max={updated.radius_max}]")
    if updated.total_seconds > 30:
        _fail("growth_seconds", f"total duration {updated.total_seconds:.3f}s exceeds 30s")
    if updated.cap < 1:
        _fail("cap", "must be >= 1")
    return updated
