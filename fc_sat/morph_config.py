"""Pixel Morph config. Validation errors name the field that failed."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from fc_sat.config import _as_float, _as_int, _as_str, _fail, _load_hooks, _load_mapping, _sibling

_TIMELINE = ("hold_a_start", "morph_ab", "hold_b", "morph_ba", "hold_a_end")
_SAFE = (130, 950, 200, 1536)
_UNCHANGED = ("unchanged", "same pixels", "pixels stay", "pixels don't change", "pixels do not change")


@dataclass(frozen=True)
class MorphConfig:
    seed: int
    cols: int
    rows: int
    cell: int
    origin_x: int
    origin_y: int
    hold_a_start: float
    morph_ab: float
    hold_b: float
    morph_ba: float
    hold_a_end: float
    delay_frac: float
    arc_amp: float
    lift: float
    spatial_weight: float
    recolor_strength: float
    hook: str
    hooks: tuple[str, ...]
    max_particles: int
    whoosh_db: float
    tick_density: int
    audio_offset_ms: float
    workers: int
    max_delta_e: float
    focus_a: tuple[float, float]
    focus_b: tuple[float, float]
    width: int
    height: int
    fps: int

    def timeline(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in _TIMELINE}

    def duration(self) -> float:
        return float(sum(self.timeline().values()))

    def to_public_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["hooks"] = list(self.hooks)
        payload["focus_a"] = list(self.focus_a)
        payload["focus_b"] = list(self.focus_b)
        return payload


def _section(raw: dict, name: str) -> dict:
    value = raw.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        _fail(name, "must be a mapping")
    return value


def _focus(field: str, value: Any) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        _fail(field, f"must be [fx, fy] in 0..1, got {value!r}")
    fx = _as_float(f"{field}[0]", value[0])
    fy = _as_float(f"{field}[1]", value[1])
    if not (0.0 <= fx <= 1.0 and 0.0 <= fy <= 1.0):
        _fail(field, f"must be inside 0..1, got {(fx, fy)}")
    return fx, fy


def _reject_unchanged(field: str, text: str, recolor: float) -> None:
    if recolor <= 0:
        return
    lowered = text.lower()
    for phrase in _UNCHANGED:
        if phrase in lowered:
            _fail(field, f"must not claim the pixels are unchanged while recolor_strength is {recolor}")


def validate_morph(raw: dict[str, Any], hooks: tuple[str, ...]) -> MorphConfig:
    if not isinstance(raw, dict):
        _fail("config", "must be a mapping")
    timeline = _section(raw, "timeline")
    audio = _section(raw, "audio")
    cols = _as_int("cols", raw.get("cols", 68))
    rows = _as_int("rows", raw.get("rows", 90))
    cell = _as_int("cell", raw.get("cell", 12))
    origin_x = _as_int("origin_x", raw.get("origin_x", 132))
    origin_y = _as_int("origin_y", raw.get("origin_y", 400))
    if cols < 2:
        _fail("cols", "must be at least 2")
    if rows < 2:
        _fail("rows", "must be at least 2")
    if cell < 2:
        _fail("cell", "must be at least 2")
    max_particles = _as_int("max_particles", raw.get("max_particles", 9000))
    if max_particles < 1:
        _fail("max_particles", "must be at least 1")
    count = cols * rows
    if count > max_particles:
        _fail("cols", f"particle count {count} exceeds max_particles {max_particles}. Use a smaller grid.")
    box_w = cols * cell
    box_h = rows * cell
    sx0, sx1, sy0, sy1 = _SAFE
    if origin_x < sx0 or origin_y < sy0 or origin_x + box_w > sx1 or origin_y + box_h > sy1:
        _fail(
            "origin_x",
            f"image box ({origin_x},{origin_y})-({origin_x + box_w},{origin_y + box_h}) "
            f"is outside the safe zone x[{sx0},{sx1}] y[{sy0},{sy1}]",
        )
    if origin_x + box_w > 1080 or origin_y + box_h > 1920 or origin_x < 0 or origin_y < 0:
        _fail("origin_x", "image box must sit inside the 1080x1920 frame")
    seconds: dict[str, float] = {}
    for name in _TIMELINE:
        seconds[name] = _as_float(f"timeline.{name}", timeline.get(name))
        if seconds[name] <= 0:
            _fail(f"timeline.{name}", "must be greater than 0")
    delay_frac = _as_float("delay_frac", raw.get("delay_frac", 0.30))
    if not 0.0 <= delay_frac < 1.0:
        _fail("delay_frac", "must be in [0, 1)")
    arc_amp = _as_float("arc_amp", raw.get("arc_amp", 0.15))
    if arc_amp < 0:
        _fail("arc_amp", "must be >= 0")
    lift = _as_float("lift", raw.get("lift", 0.5))
    if lift < 0:
        _fail("lift", "must be >= 0")
    spatial_weight = _as_float("spatial_weight", raw.get("spatial_weight", 0.3))
    if spatial_weight < 0:
        _fail("spatial_weight", "must be >= 0")
    recolor = _as_float("recolor_strength", raw.get("recolor_strength", 1.0))
    if not 0.0 <= recolor <= 1.0:
        _fail("recolor_strength", "must be in 0..1")
    hook = _as_str("hook", raw.get("hook", hooks[0] if hooks else ""))
    if not hook.strip():
        _fail("hook", "must be a non-empty string")
    hook = hook.strip()
    _reject_unchanged("hook", hook, recolor)
    for index, item in enumerate(hooks):
        _reject_unchanged(f"hooks[{index}]", item, recolor)
    whoosh_db = _as_float("audio.whoosh_db", audio.get("whoosh_db", -18.0))
    if not -80.0 <= whoosh_db <= 0.0:
        _fail("audio.whoosh_db", "must be from -80 to 0")
    tick_density = _as_int("audio.tick_density", audio.get("tick_density", 12))
    if tick_density < 1:
        _fail("audio.tick_density", "must be at least 1")
    offset = _as_float("audio_offset_ms", raw.get("audio_offset_ms", 0.0))
    if not -500.0 <= offset <= 500.0:
        _fail("audio_offset_ms", "must be from -500 to 500")
    workers = _as_int("workers", raw.get("workers", 0))
    if workers < 0:
        _fail("workers", "must be >= 0")
    max_delta_e = _as_float("max_delta_e", raw.get("max_delta_e", 0.06))
    if max_delta_e <= 0:
        _fail("max_delta_e", "must be greater than 0")
    focus_a = _focus("focus_a", raw.get("focus_a", [0.5, 0.5]))
    focus_b = _focus("focus_b", raw.get("focus_b", [0.5, 0.5]))
    width = _as_int("width", raw.get("width", 1080))
    height = _as_int("height", raw.get("height", 1920))
    fps = _as_int("fps", raw.get("fps", 60))
    if (width, height, fps) != (1080, 1920, 60):
        _fail("width", "Pixel Morph renders 1080x1920 at 60 fps; preview scaling is a CLI flag")
    return MorphConfig(
        seed=_as_int("seed", raw.get("seed", 7)),
        cols=cols,
        rows=rows,
        cell=cell,
        origin_x=origin_x,
        origin_y=origin_y,
        delay_frac=delay_frac,
        arc_amp=arc_amp,
        lift=lift,
        spatial_weight=spatial_weight,
        recolor_strength=recolor,
        hook=hook,
        hooks=hooks,
        max_particles=max_particles,
        whoosh_db=whoosh_db,
        tick_density=tick_density,
        audio_offset_ms=offset,
        workers=workers,
        max_delta_e=max_delta_e,
        focus_a=focus_a,
        focus_b=focus_b,
        width=width,
        height=height,
        fps=fps,
        **seconds,
    )


def morph_from_public(payload: dict[str, Any], hooks: tuple[str, ...]) -> MorphConfig:
    """Rebuild a config from the sidecar's public dict."""
    raw = dict(payload)
    raw.pop("hooks", None)
    raw["timeline"] = {name: raw.pop(name) for name in _TIMELINE}
    raw["audio"] = {"whoosh_db": raw.pop("whoosh_db"), "tick_density": raw.pop("tick_density")}
    return validate_morph(raw, hooks)


def load_morph_config(path: str | Path) -> MorphConfig:
    path = Path(path)
    raw = _load_mapping(path)
    hooks = _load_hooks(_sibling(path, "morph_hooks.yaml"))
    return validate_morph(raw, hooks)


def with_morph_overrides(cfg: MorphConfig, **changes: Any) -> MorphConfig:
    raw = cfg.to_public_dict()
    raw["timeline"] = {name: raw.pop(name) for name in _TIMELINE}
    raw["audio"] = {"whoosh_db": raw.pop("whoosh_db"), "tick_density": raw.pop("tick_density")}
    for key, value in changes.items():
        if key in _TIMELINE:
            raw["timeline"][key] = value
        elif key in ("whoosh_db", "tick_density"):
            raw["audio"][key] = value
        else:
            raw[key] = value
    return validate_morph(raw, cfg.hooks)
