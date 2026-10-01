"""Flag Arena config, casts, guards, and meme validation.

Validation errors name the field. v1 accepts a cast of exactly 32 countries.
Gate times are the yaml values; they are not scaled with cast size.
"""

from __future__ import annotations

import datetime as dt
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from fc_sat.config import _as_float, _as_int, _as_str, _fail, _load_hooks, _load_mapping

CAST_SIZE = 32
PLACEHOLDERS = frozenset({"code", "name", "cameo"})
MEME_EVENTS = ("eliminated", "meteor", "self_fall", "final3", "cameo", "winner")
_PLACEHOLDER_RE = re.compile(r"\{([a-z0-9_]+)\}")
_WORD_RE = re.compile(r"[A-Za-z']+")

# Whole words only. Phrases may talk about events, not people or groups.
DENYLIST = frozenset(
    {
        "nigger",
        "nigga",
        "faggot",
        "fag",
        "retard",
        "retarded",
        "kike",
        "spic",
        "chink",
        "tranny",
        "nationality",
        "nation",
        "race",
        "racial",
        "ethnic",
        "ethnicity",
        "religion",
        "religious",
        "immigrant",
        "refugee",
        "muslim",
        "christian",
        "jewish",
        "hindu",
        "buddhist",
    }
)

_TOP_LEVEL = frozenset(
    {
        "generator",
        "seed",
        "width",
        "height",
        "fps",
        "workers",
        "bloom_strength",
        "cast_size",
        "cast",
        "hook",
        "hook_index",
        "world",
        "spawn",
        "zone",
        "sweeper",
        "meteors",
        "cameo",
        "pacing",
        "drama",
        "camera",
        "layout",
        "timeline",
        "juice",
        "audio",
        "voice",
    }
)


@dataclass(frozen=True)
class Country:
    name: str
    iso2: str
    iso3: str


@dataclass(frozen=True)
class Guards:
    religious_inscriptions: tuple[str, ...]
    contested: tuple[str, ...]
    active_conflict: tuple[str, ...]

    @property
    def blocked(self) -> frozenset[str]:
        return frozenset(self.religious_inscriptions + self.contested + self.active_conflict)


@dataclass(frozen=True)
class Memes:
    last_reviewed: str
    phrases: tuple[tuple[str, tuple[str, ...]], ...]
    warnings: tuple[str, ...]

    def lines(self, event: str) -> tuple[str, ...]:
        for name, phrases in self.phrases:
            if name == event:
                return phrases
        _fail("memes", f"missing event {event!r}")
        raise AssertionError


@dataclass(frozen=True)
class ArenaConfig:
    generator: str
    seed: int
    width: int
    height: int
    fps: int
    workers: int
    bloom_strength: float
    cast_size: int
    cast_path: str
    countries: tuple[Country, ...]
    hook: str
    hooks: tuple[str, ...]
    hook_index: int | None
    center: tuple[float, float]
    floor_radius: float
    ball_radius: float
    ball_mass: float
    restitution: float
    damping: float
    substeps_hz: int
    disk_radius: float
    speed_min: float
    speed_max: float
    inward_deg: float
    wander_accel: float
    wander_period: tuple[float, float]
    edge_accel: float
    edge_margin: float
    edge_factor: tuple[float, float]
    no_elim_before: float
    fall_seconds: float
    out_margin_radii: float
    zone_keyframes: tuple[tuple[float, float], ...]
    warning_lead: float
    shrink_seconds: float
    sweeper_start: float
    sweeper_thickness: float
    sweeper_inset: float
    sweeper_omega_start: float
    sweeper_omega_end: float
    sweeper_omega_end_time: float
    meteor_start: float
    meteor_gap: tuple[float, float]
    meteor_reticle: float
    meteor_radius: float
    meteor_impulse: float
    cameo_name: str
    cameo_enter: float
    cameo_exit: float
    cameo_radius: float
    cameo_mass: float
    gates: tuple[tuple[float, int, int], ...]
    win_window: tuple[float, float]
    final_duel_min: float
    min_pass_rate: float
    search: int
    near_miss_window: float
    near_miss_margin_radii: float
    near_miss_cap: int
    near_miss_weight: float
    duel_cap: float
    duel_weight: float
    late_elim_window: float
    late_elim_min: int
    late_elim_weight: float
    double_window: float
    double_cap: int
    double_weight: float
    meteor_elim_range: tuple[int, int]
    meteor_elim_weight: float
    repeat_penalty: float
    screen_radius: float
    camera_smooth: float
    punch_seconds: float
    punch_from: float
    punch_to: float
    shake_px: float
    shake_hz: float
    shake_decay: float
    zone_x: tuple[float, float]
    zone_y: tuple[float, float]
    safe_x: tuple[float, float]
    safe_y: tuple[float, float]
    counter_y: float
    counter_px: float
    hook_y: float
    hook_px: float
    hook_lines: int
    kill_x: float
    kill_y: tuple[float, float, float]
    kill_px: float
    kill_lines: int
    kill_fade: float
    cta_y: float
    name_px: float
    name_seconds: float
    name_fade: float
    last_named: int
    slowmo_rate: float
    slowmo_pre: float
    slowmo_post: float
    replay_sim: float
    replay_zoom: float
    celebration: float
    winner_scale: float
    confetti: int
    counter_pop: float
    counter_pop_scale: float
    hook_until: float
    hook_fade: float
    fact_text: str
    fact_window: tuple[float, float]
    wait_text: str
    wait_start: float
    squash: float
    squash_seconds: float
    trail_speed: float
    spark_speed: float
    sparks: int
    eye_frac: float
    blink: tuple[float, float]
    scared_margin: float
    hit_seconds: float
    shadow_offset_radii: float
    highlight_alpha: float
    bpm: tuple[float, float]
    sidechain_db: float
    voice_duck_db: float
    slowmo_duck_db: float
    slowmo_lpf: float
    stinger_gap: float
    pre_drop: float
    max_voices: int
    max_onsets_per_100ms: int
    bonk_noise_ms: float
    bonk_drop_ms: float
    bonk_hz: tuple[float, float]
    cowbell_hz: tuple[float, float]
    airhorn_hz: tuple[float, float]
    ding_hz: tuple[float, float]
    ding_ms: float
    whistle_hz: tuple[float, float]
    whistle_seconds: float
    meteor_beep_hz: float
    meteor_beep_gap: float
    meteor_beeps: int
    sub_hz: tuple[float, float]
    target_lufs: float
    true_peak_db: float
    fade_ms: float
    loudness_iters: int
    voice_model_id: str
    voice_id: str
    memes: Memes
    guards: Guards
    sensitive_warnings: tuple[str, ...]

    @property
    def dt(self) -> float:
        return 1.0 / float(self.substeps_hz)

    def to_public_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["countries"] = [asdict(item) for item in self.countries]
        payload["hooks"] = list(self.hooks)
        payload["memes"] = {
            "last_reviewed": self.memes.last_reviewed,
            "phrases": {name: list(lines) for name, lines in self.memes.phrases},
            "warnings": list(self.memes.warnings),
        }
        payload["guards"] = {
            "religious_inscriptions": list(self.guards.religious_inscriptions),
            "contested": list(self.guards.contested),
            "active_conflict": list(self.guards.active_conflict),
        }
        return payload


def _section(raw: dict, name: str) -> dict:
    value = raw.get(name, {})
    if not isinstance(value, dict):
        _fail(name, "must be a mapping")
    return value


def _require_keys(field: str, section: dict, keys: set[str]) -> None:
    unknown = set(section) - keys
    if unknown:
        _fail(f"{field}.{sorted(unknown)[0]}", "is not a config field")
    missing = keys - set(section)
    if missing:
        _fail(f"{field}.{sorted(missing)[0]}", "is required")


def _hz_pair(field: str, value: Any) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        _fail(field, "must be a pair of frequencies")
    a = _as_float(field, value[0])
    b = _as_float(field, value[1])
    if a <= 0 or b <= 0:
        _fail(field, "frequencies must be > 0")
    return a, b


def _pair(field: str, value: Any, *, lo_positive: bool = False) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        _fail(field, "must be a pair of numbers")
    a = _as_float(field, value[0])
    b = _as_float(field, value[1])
    if b < a:
        _fail(field, "upper bound must be >= lower bound")
    if lo_positive and a <= 0:
        _fail(field, "must be > 0")
    return a, b


def _code_list(field: str, value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        _fail(field, "must be a non-empty list of ISO2 codes")
    out = []
    for item in value:
        if not isinstance(item, str) or not re.fullmatch(r"[A-Z]{2}", item):
            _fail(field, f"must be uppercase ISO2 codes, got {item!r}")
        out.append(item)
    return tuple(out)


def load_guards(path: str | Path) -> Guards:
    raw = _load_mapping(Path(path))
    allowed = {"religious_inscriptions", "contested", "active_conflict"}
    unknown = set(raw) - allowed
    if unknown:
        _fail(f"guards.{sorted(unknown)[0]}", "is not a config field")
    return Guards(
        religious_inscriptions=_code_list("guards.religious_inscriptions", raw.get("religious_inscriptions")),
        contested=_code_list("guards.contested", raw.get("contested")),
        active_conflict=_code_list("guards.active_conflict", raw.get("active_conflict")),
    )


def _resolve_path(config_path: Path, value: str, field: str) -> Path:
    path = Path(value)
    candidates = [path]
    if not path.is_absolute():
        candidates.append(config_path.parent / path)
        candidates.append(config_path.parent.parent / path)
        if value.startswith("configs/"):
            candidates.append(config_path.parent / value[len("configs/") :])
    for item in candidates:
        if item.is_file():
            return item
    _fail(field, f"file not found: {value}")
    raise AssertionError


def load_cast(
    path: str | Path,
    guards: Guards,
    *,
    allow_sensitive: bool = False,
) -> tuple[tuple[Country, ...], tuple[str, ...]]:
    data = __import__("yaml").safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        _fail("cast", "must be a list of countries")
    countries: list[Country] = []
    seen: set[str] = set()
    warnings: list[str] = []
    for index, item in enumerate(data):
        field = f"cast[{index}]"
        if not isinstance(item, dict):
            _fail(field, "must be a mapping")
        unknown = set(item) - {"name", "iso2", "iso3"}
        if unknown:
            _fail(f"{field}.{sorted(unknown)[0]}", "is not a config field")
        name = _as_str(f"{field}.name", item.get("name"))
        iso2 = _as_str(f"{field}.iso2", item.get("iso2"))
        iso3 = _as_str(f"{field}.iso3", item.get("iso3"))
        if not re.fullmatch(r"[A-Z]{2}", iso2):
            _fail(f"{field}.iso2", f"must be an uppercase ISO2 code, got {iso2!r}")
        if not re.fullmatch(r"[A-Z]{3}", iso3):
            _fail(f"{field}.iso3", f"must be an uppercase ISO3 code, got {iso3!r}")
        if iso2 in seen:
            _fail(f"{field}.iso2", f"duplicate code {iso2}")
        seen.add(iso2)
        if iso2 in guards.blocked:
            message = f"cast includes blocked code {iso2} ({name})"
            if not allow_sensitive:
                _fail("cast", message + "; edit configs/guards.yaml or pass --allow-sensitive")
            warnings.append(message)
            print(f"warning: {message}", file=sys.stderr)
        countries.append(Country(name=name, iso2=iso2, iso3=iso3))
    if len(countries) != CAST_SIZE:
        _fail("cast", f"must contain exactly {CAST_SIZE} countries, got {len(countries)}")
    return tuple(countries), tuple(warnings)


def validate_memes(raw: dict[str, Any]) -> Memes:
    if not isinstance(raw, dict):
        _fail("memes", "must be a mapping")
    unknown = set(raw) - set(MEME_EVENTS) - {"last_reviewed"}
    if unknown:
        _fail(f"memes.{sorted(unknown)[0]}", "is not a config field")
    reviewed_raw = raw.get("last_reviewed")
    if isinstance(reviewed_raw, dt.datetime):
        reviewed = reviewed_raw.date().isoformat()
    elif isinstance(reviewed_raw, dt.date):
        reviewed = reviewed_raw.isoformat()
    else:
        reviewed = _as_str("memes.last_reviewed", reviewed_raw)
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", reviewed):
        _fail("memes.last_reviewed", "must be a YYYY-MM-DD date")
    phrases: list[tuple[str, tuple[str, ...]]] = []
    warnings: list[str] = []
    for event in MEME_EVENTS:
        lines = raw.get(event)
        if not isinstance(lines, list) or not lines:
            _fail(f"memes.{event}", "must be a non-empty list of phrases")
        cleaned: list[str] = []
        for index, line in enumerate(lines):
            field = f"memes.{event}[{index}]"
            text = _as_str(field, line)
            for token in _PLACEHOLDER_RE.findall(text):
                if token not in PLACEHOLDERS:
                    _fail(field, f"unknown placeholder {{{token}}}")
            stripped = _PLACEHOLDER_RE.sub(" ", text)
            for word in _WORD_RE.findall(stripped):
                if word.lower() in DENYLIST:
                    _fail(field, f"contains a blocked word {word!r}")
            if len(text) > 28:
                warnings.append(f"{field} is {len(text)} characters; keep phrases at or under 28")
            cleaned.append(text)
        phrases.append((event, tuple(cleaned)))
    return Memes(last_reviewed=reviewed, phrases=tuple(phrases), warnings=tuple(warnings))


def load_memes(path: str | Path) -> Memes:
    return validate_memes(_load_mapping(Path(path)))


def _keyframes(field: str, value: Any) -> tuple[tuple[float, float], ...]:
    if not isinstance(value, list) or len(value) < 2:
        _fail(field, "must be a list of [time, radius] pairs")
    frames = []
    previous = -1.0
    for index, item in enumerate(value):
        name = f"{field}[{index}]"
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            _fail(name, "must be [time, radius]")
        time = _as_float(name, item[0])
        radius = _as_float(name, item[1])
        if time < previous:
            _fail(name, "times must be non-decreasing")
        if radius <= 0:
            _fail(name, "radius must be > 0")
        previous = time
        frames.append((time, radius))
    return tuple(frames)


def _gates(value: Any) -> tuple[tuple[float, int, int], ...]:
    if not isinstance(value, list) or not value:
        _fail("pacing.gates", "must be a list of [time, low, high] rows")
    rows = []
    for index, item in enumerate(value):
        field = f"pacing.gates[{index}]"
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            _fail(field, "must be [time, low, high]")
        time = _as_float(field, item[0])
        low = _as_int(field, item[1])
        high = _as_int(field, item[2])
        if low > high:
            _fail(field, "low must be <= high")
        rows.append((time, low, high))
    return tuple(rows)


def validate_arena(
    raw: dict[str, Any],
    *,
    hooks: tuple[str, ...],
    countries: tuple[Country, ...],
    guards: Guards,
    memes: Memes,
    cast_path: str,
    sensitive_warnings: tuple[str, ...] = (),
) -> ArenaConfig:
    if not isinstance(raw, dict):
        _fail("arena", "root must be a mapping")
    unknown = set(raw) - _TOP_LEVEL
    if unknown:
        _fail(sorted(unknown)[0], "is not a config field")
    generator = _as_str("generator", raw.get("generator"))
    if generator != "arena":
        _fail("generator", "must be 'arena'")
    seed = _as_int("seed", raw.get("seed"))
    if seed < 0:
        _fail("seed", "must be >= 0")
    width = _as_int("width", raw.get("width"))
    height = _as_int("height", raw.get("height"))
    fps = _as_int("fps", raw.get("fps"))
    if (width, height, fps) != (1080, 1920, 60):
        _fail("width", "full renders are 1080x1920 at 60 fps")
    workers = _as_int("workers", raw.get("workers"))
    if workers < 0 or workers > 64:
        _fail("workers", "must be in [0, 64]")
    bloom = _as_float("bloom_strength", raw.get("bloom_strength"))
    if bloom < 0 or bloom > 2:
        _fail("bloom_strength", "must be in [0, 2]")
    cast_size = _as_int("cast_size", raw.get("cast_size"))
    if cast_size != CAST_SIZE:
        _fail("cast_size", f"must be {CAST_SIZE}")
    if len(countries) != CAST_SIZE:
        _fail("cast", f"must contain exactly {CAST_SIZE} countries, got {len(countries)}")
    hook = _as_str("hook", raw.get("hook"))
    hook_index_raw = raw.get("hook_index", None)
    hook_index: int | None
    if hook_index_raw is None:
        hook_index = None
    else:
        hook_index = _as_int("hook_index", hook_index_raw)
        if hook_index < 0 or hook_index >= len(hooks):
            _fail("hook_index", f"must be in [0, {len(hooks) - 1}]")
        hook = hooks[hook_index]

    world = _section(raw, "world")
    _require_keys(
        "world",
        world,
        {"center", "floor_radius", "ball_radius", "ball_mass", "restitution", "damping", "substeps_hz"},
    )
    center_raw = world["center"]
    if not isinstance(center_raw, (list, tuple)) or len(center_raw) != 2:
        _fail("world.center", "must be [x, y]")
    center = (_as_float("world.center", center_raw[0]), _as_float("world.center", center_raw[1]))
    floor_radius = _as_float("world.floor_radius", world["floor_radius"])
    ball_radius = _as_float("world.ball_radius", world["ball_radius"])
    ball_mass = _as_float("world.ball_mass", world["ball_mass"])
    restitution = _as_float("world.restitution", world["restitution"])
    damping = _as_float("world.damping", world["damping"])
    substeps = _as_int("world.substeps_hz", world["substeps_hz"])
    if floor_radius <= 0 or ball_radius <= 0 or ball_mass <= 0:
        _fail("world.floor_radius", "radius and mass must be > 0")
    if restitution < 0 or restitution > 1:
        _fail("world.restitution", "must be in [0, 1]")
    if damping < 0:
        _fail("world.damping", "must be >= 0")
    if substeps != 240:
        _fail("world.substeps_hz", "must be 240")

    spawn = _section(raw, "spawn")
    _require_keys(
        "spawn",
        spawn,
        {
            "disk_radius",
            "speed_min",
            "speed_max",
            "inward_deg",
            "wander_accel",
            "wander_period",
            "edge_accel",
            "edge_margin",
            "edge_factor",
            "no_elim_before",
            "fall_seconds",
            "out_margin_radii",
        },
    )
    disk_radius = _as_float("spawn.disk_radius", spawn["disk_radius"])
    speed_min = _as_float("spawn.speed_min", spawn["speed_min"])
    speed_max = _as_float("spawn.speed_max", spawn["speed_max"])
    if disk_radius <= 0 or speed_min <= 0 or speed_max < speed_min:
        _fail("spawn.speed_min", "need disk_radius > 0 and 0 < speed_min <= speed_max")
    inward = _as_float("spawn.inward_deg", spawn["inward_deg"])
    if inward < 0 or inward > 180:
        _fail("spawn.inward_deg", "must be in [0, 180]")

    zone = _section(raw, "zone")
    _require_keys("zone", zone, {"keyframes", "warning_lead", "shrink_seconds"})
    keyframes = _keyframes("zone.keyframes", zone["keyframes"])
    warning_lead = _as_float("zone.warning_lead", zone["warning_lead"])
    shrink_seconds = _as_float("zone.shrink_seconds", zone["shrink_seconds"])
    if warning_lead < 0 or shrink_seconds <= 0:
        _fail("zone.warning_lead", "warning_lead must be >= 0 and shrink_seconds > 0")

    sweeper = _section(raw, "sweeper")
    _require_keys(
        "sweeper",
        sweeper,
        {"start", "thickness", "inset", "omega_start", "omega_end", "omega_end_time"},
    )
    meteors = _section(raw, "meteors")
    _require_keys("meteors", meteors, {"start", "gap", "reticle", "radius", "impulse"})
    cameo = _section(raw, "cameo")
    _require_keys("cameo", cameo, {"name", "enter", "exit", "radius", "mass"})
    cameo_enter = _as_float("cameo.enter", cameo["enter"])
    cameo_exit = _as_float("cameo.exit", cameo["exit"])
    if cameo_exit < cameo_enter:
        _fail("cameo.exit", "must be >= cameo.enter")

    pacing = _section(raw, "pacing")
    _require_keys("pacing", pacing, {"gates", "win", "final_duel_min", "min_pass_rate", "search"})
    win = _pair("pacing.win", pacing["win"])
    pass_rate = _as_float("pacing.min_pass_rate", pacing["min_pass_rate"])
    if pass_rate <= 0 or pass_rate > 1:
        _fail("pacing.min_pass_rate", "must be in (0, 1]")
    search = _as_int("pacing.search", pacing["search"])
    if search < 1:
        _fail("pacing.search", "must be >= 1")

    drama = _section(raw, "drama")
    _require_keys(
        "drama",
        drama,
        {
            "near_miss_window",
            "near_miss_margin_radii",
            "near_miss_cap",
            "near_miss_weight",
            "duel_cap",
            "duel_weight",
            "late_elim_window",
            "late_elim_min",
            "late_elim_weight",
            "double_window",
            "double_cap",
            "double_weight",
            "meteor_elim_range",
            "meteor_elim_weight",
            "repeat_penalty",
        },
    )
    meteor_range_raw = drama["meteor_elim_range"]
    if not isinstance(meteor_range_raw, (list, tuple)) or len(meteor_range_raw) != 2:
        _fail("drama.meteor_elim_range", "must be [low, high]")
    meteor_lo = _as_int("drama.meteor_elim_range", meteor_range_raw[0])
    meteor_hi = _as_int("drama.meteor_elim_range", meteor_range_raw[1])
    if meteor_lo > meteor_hi:
        _fail("drama.meteor_elim_range", "low must be <= high")

    camera = _section(raw, "camera")
    _require_keys(
        "camera",
        camera,
        {
            "screen_radius",
            "smooth",
            "punch_seconds",
            "punch_from",
            "punch_to",
            "shake_px",
            "shake_hz",
            "shake_decay",
            "zone_x",
            "zone_y",
        },
    )
    layout = _section(raw, "layout")
    _require_keys(
        "layout",
        layout,
        {
            "safe_x",
            "safe_y",
            "counter_y",
            "counter_px",
            "hook_y",
            "hook_px",
            "hook_lines",
            "kill_x",
            "kill_y",
            "kill_px",
            "kill_lines",
            "kill_fade",
            "cta_y",
            "name_px",
            "name_seconds",
            "name_fade",
            "last_named",
        },
    )
    kill_y_raw = layout["kill_y"]
    if not isinstance(kill_y_raw, (list, tuple)) or len(kill_y_raw) != 3:
        _fail("layout.kill_y", "must be three y positions")
    kill_y = tuple(_as_float("layout.kill_y", item) for item in kill_y_raw)

    timeline = _section(raw, "timeline")
    _require_keys(
        "timeline",
        timeline,
        {
            "slowmo_rate",
            "slowmo_pre",
            "slowmo_post",
            "replay_sim",
            "replay_zoom",
            "celebration",
            "winner_scale",
            "confetti",
            "counter_pop",
            "counter_pop_scale",
            "hook_until",
            "hook_fade",
            "fact_text",
            "fact_window",
            "wait_text",
            "wait_start",
        },
    )
    slowmo_rate = _as_float("timeline.slowmo_rate", timeline["slowmo_rate"])
    if slowmo_rate <= 0 or slowmo_rate > 1:
        _fail("timeline.slowmo_rate", "must be in (0, 1]")

    juice = _section(raw, "juice")
    _require_keys(
        "juice",
        juice,
        {
            "squash",
            "squash_seconds",
            "trail_speed",
            "spark_speed",
            "sparks",
            "eye_frac",
            "blink",
            "scared_margin",
            "hit_seconds",
            "shadow_offset_radii",
            "highlight_alpha",
        },
    )
    audio = _section(raw, "audio")
    _require_keys(
        "audio",
        audio,
        {
            "bpm",
            "sidechain_db",
            "voice_duck_db",
            "slowmo_duck_db",
            "slowmo_lpf",
            "stinger_gap",
            "pre_drop",
            "max_voices",
            "max_onsets_per_100ms",
            "bonk_noise_ms",
            "bonk_drop_ms",
            "bonk_hz",
            "cowbell_hz",
            "airhorn_hz",
            "ding_hz",
            "ding_ms",
            "whistle_hz",
            "whistle_seconds",
            "meteor_beep_hz",
            "meteor_beep_gap",
            "meteor_beeps",
            "sub_hz",
            "target_lufs",
            "true_peak_db",
            "fade_ms",
            "loudness_iters",
        },
    )
    loudness_iters = _as_int("audio.loudness_iters", audio["loudness_iters"])
    if loudness_iters < 1 or loudness_iters > 5:
        _fail("audio.loudness_iters", "must be in [1, 5]")
    voice = _section(raw, "voice")
    _require_keys("voice", voice, {"model_id", "voice_id"})
    model_id = voice.get("model_id")
    if not isinstance(model_id, str) or not model_id.strip():
        _fail("voice.model_id", "must be a non-empty string")
    voice_id = voice.get("voice_id", "")
    if not isinstance(voice_id, str):
        _fail("voice.voice_id", "must be a string")

    return ArenaConfig(
        generator=generator,
        seed=seed,
        width=width,
        height=height,
        fps=fps,
        workers=workers,
        bloom_strength=bloom,
        cast_size=cast_size,
        cast_path=cast_path,
        countries=countries,
        hook=hook,
        hooks=hooks,
        hook_index=hook_index,
        center=center,
        floor_radius=floor_radius,
        ball_radius=ball_radius,
        ball_mass=ball_mass,
        restitution=restitution,
        damping=damping,
        substeps_hz=substeps,
        disk_radius=disk_radius,
        speed_min=speed_min,
        speed_max=speed_max,
        inward_deg=inward,
        wander_accel=_as_float("spawn.wander_accel", spawn["wander_accel"]),
        wander_period=_pair("spawn.wander_period", spawn["wander_period"], lo_positive=True),
        edge_accel=_as_float("spawn.edge_accel", spawn["edge_accel"]),
        edge_margin=_as_float("spawn.edge_margin", spawn["edge_margin"]),
        edge_factor=_pair("spawn.edge_factor", spawn["edge_factor"]),
        no_elim_before=_as_float("spawn.no_elim_before", spawn["no_elim_before"]),
        fall_seconds=_as_float("spawn.fall_seconds", spawn["fall_seconds"]),
        out_margin_radii=_as_float("spawn.out_margin_radii", spawn["out_margin_radii"]),
        zone_keyframes=keyframes,
        warning_lead=warning_lead,
        shrink_seconds=shrink_seconds,
        sweeper_start=_as_float("sweeper.start", sweeper["start"]),
        sweeper_thickness=_as_float("sweeper.thickness", sweeper["thickness"]),
        sweeper_inset=_as_float("sweeper.inset", sweeper["inset"]),
        sweeper_omega_start=_as_float("sweeper.omega_start", sweeper["omega_start"]),
        sweeper_omega_end=_as_float("sweeper.omega_end", sweeper["omega_end"]),
        sweeper_omega_end_time=_as_float("sweeper.omega_end_time", sweeper["omega_end_time"]),
        meteor_start=_as_float("meteors.start", meteors["start"]),
        meteor_gap=_pair("meteors.gap", meteors["gap"], lo_positive=True),
        meteor_reticle=_as_float("meteors.reticle", meteors["reticle"]),
        meteor_radius=_as_float("meteors.radius", meteors["radius"]),
        meteor_impulse=_as_float("meteors.impulse", meteors["impulse"]),
        cameo_name=_as_str("cameo.name", cameo["name"]),
        cameo_enter=cameo_enter,
        cameo_exit=cameo_exit,
        cameo_radius=_as_float("cameo.radius", cameo["radius"]),
        cameo_mass=_as_float("cameo.mass", cameo["mass"]),
        gates=_gates(pacing["gates"]),
        win_window=win,
        final_duel_min=_as_float("pacing.final_duel_min", pacing["final_duel_min"]),
        min_pass_rate=pass_rate,
        search=search,
        near_miss_window=_as_float("drama.near_miss_window", drama["near_miss_window"]),
        near_miss_margin_radii=_as_float("drama.near_miss_margin_radii", drama["near_miss_margin_radii"]),
        near_miss_cap=_as_int("drama.near_miss_cap", drama["near_miss_cap"]),
        near_miss_weight=_as_float("drama.near_miss_weight", drama["near_miss_weight"]),
        duel_cap=_as_float("drama.duel_cap", drama["duel_cap"]),
        duel_weight=_as_float("drama.duel_weight", drama["duel_weight"]),
        late_elim_window=_as_float("drama.late_elim_window", drama["late_elim_window"]),
        late_elim_min=_as_int("drama.late_elim_min", drama["late_elim_min"]),
        late_elim_weight=_as_float("drama.late_elim_weight", drama["late_elim_weight"]),
        double_window=_as_float("drama.double_window", drama["double_window"]),
        double_cap=_as_int("drama.double_cap", drama["double_cap"]),
        double_weight=_as_float("drama.double_weight", drama["double_weight"]),
        meteor_elim_range=(meteor_lo, meteor_hi),
        meteor_elim_weight=_as_float("drama.meteor_elim_weight", drama["meteor_elim_weight"]),
        repeat_penalty=_as_float("drama.repeat_penalty", drama["repeat_penalty"]),
        screen_radius=_as_float("camera.screen_radius", camera["screen_radius"]),
        camera_smooth=_as_float("camera.smooth", camera["smooth"]),
        punch_seconds=_as_float("camera.punch_seconds", camera["punch_seconds"]),
        punch_from=_as_float("camera.punch_from", camera["punch_from"]),
        punch_to=_as_float("camera.punch_to", camera["punch_to"]),
        shake_px=_as_float("camera.shake_px", camera["shake_px"]),
        shake_hz=_as_float("camera.shake_hz", camera["shake_hz"]),
        shake_decay=_as_float("camera.shake_decay", camera["shake_decay"]),
        zone_x=_pair("camera.zone_x", camera["zone_x"]),
        zone_y=_pair("camera.zone_y", camera["zone_y"]),
        safe_x=_pair("layout.safe_x", layout["safe_x"]),
        safe_y=_pair("layout.safe_y", layout["safe_y"]),
        counter_y=_as_float("layout.counter_y", layout["counter_y"]),
        counter_px=_as_float("layout.counter_px", layout["counter_px"]),
        hook_y=_as_float("layout.hook_y", layout["hook_y"]),
        hook_px=_as_float("layout.hook_px", layout["hook_px"]),
        hook_lines=_as_int("layout.hook_lines", layout["hook_lines"]),
        kill_x=_as_float("layout.kill_x", layout["kill_x"]),
        kill_y=kill_y,  # type: ignore[arg-type]
        kill_px=_as_float("layout.kill_px", layout["kill_px"]),
        kill_lines=_as_int("layout.kill_lines", layout["kill_lines"]),
        kill_fade=_as_float("layout.kill_fade", layout["kill_fade"]),
        cta_y=_as_float("layout.cta_y", layout["cta_y"]),
        name_px=_as_float("layout.name_px", layout["name_px"]),
        name_seconds=_as_float("layout.name_seconds", layout["name_seconds"]),
        name_fade=_as_float("layout.name_fade", layout["name_fade"]),
        last_named=_as_int("layout.last_named", layout["last_named"]),
        slowmo_rate=slowmo_rate,
        slowmo_pre=_as_float("timeline.slowmo_pre", timeline["slowmo_pre"]),
        slowmo_post=_as_float("timeline.slowmo_post", timeline["slowmo_post"]),
        replay_sim=_as_float("timeline.replay_sim", timeline["replay_sim"]),
        replay_zoom=_as_float("timeline.replay_zoom", timeline["replay_zoom"]),
        celebration=_as_float("timeline.celebration", timeline["celebration"]),
        winner_scale=_as_float("timeline.winner_scale", timeline["winner_scale"]),
        confetti=_as_int("timeline.confetti", timeline["confetti"]),
        counter_pop=_as_float("timeline.counter_pop", timeline["counter_pop"]),
        counter_pop_scale=_as_float("timeline.counter_pop_scale", timeline["counter_pop_scale"]),
        hook_until=_as_float("timeline.hook_until", timeline["hook_until"]),
        hook_fade=_as_float("timeline.hook_fade", timeline["hook_fade"]),
        fact_text=_as_str("timeline.fact_text", timeline["fact_text"]),
        fact_window=_pair("timeline.fact_window", timeline["fact_window"]),
        wait_text=_as_str("timeline.wait_text", timeline["wait_text"]),
        wait_start=_as_float("timeline.wait_start", timeline["wait_start"]),
        squash=_as_float("juice.squash", juice["squash"]),
        squash_seconds=_as_float("juice.squash_seconds", juice["squash_seconds"]),
        trail_speed=_as_float("juice.trail_speed", juice["trail_speed"]),
        spark_speed=_as_float("juice.spark_speed", juice["spark_speed"]),
        sparks=_as_int("juice.sparks", juice["sparks"]),
        eye_frac=_as_float("juice.eye_frac", juice["eye_frac"]),
        blink=_pair("juice.blink", juice["blink"], lo_positive=True),
        scared_margin=_as_float("juice.scared_margin", juice["scared_margin"]),
        hit_seconds=_as_float("juice.hit_seconds", juice["hit_seconds"]),
        shadow_offset_radii=_as_float("juice.shadow_offset_radii", juice["shadow_offset_radii"]),
        highlight_alpha=_as_float("juice.highlight_alpha", juice["highlight_alpha"]),
        bpm=_pair("audio.bpm", audio["bpm"], lo_positive=True),
        sidechain_db=_as_float("audio.sidechain_db", audio["sidechain_db"]),
        voice_duck_db=_as_float("audio.voice_duck_db", audio["voice_duck_db"]),
        slowmo_duck_db=_as_float("audio.slowmo_duck_db", audio["slowmo_duck_db"]),
        slowmo_lpf=_as_float("audio.slowmo_lpf", audio["slowmo_lpf"]),
        stinger_gap=_as_float("audio.stinger_gap", audio["stinger_gap"]),
        pre_drop=_as_float("audio.pre_drop", audio["pre_drop"]),
        max_voices=_as_int("audio.max_voices", audio["max_voices"]),
        max_onsets_per_100ms=_as_int("audio.max_onsets_per_100ms", audio["max_onsets_per_100ms"]),
        bonk_noise_ms=_as_float("audio.bonk_noise_ms", audio["bonk_noise_ms"]),
        bonk_drop_ms=_as_float("audio.bonk_drop_ms", audio["bonk_drop_ms"]),
        bonk_hz=_hz_pair("audio.bonk_hz", audio["bonk_hz"]),
        cowbell_hz=_pair("audio.cowbell_hz", audio["cowbell_hz"], lo_positive=True),
        airhorn_hz=_pair("audio.airhorn_hz", audio["airhorn_hz"], lo_positive=True),
        ding_hz=_pair("audio.ding_hz", audio["ding_hz"], lo_positive=True),
        ding_ms=_as_float("audio.ding_ms", audio["ding_ms"]),
        whistle_hz=_hz_pair("audio.whistle_hz", audio["whistle_hz"]),
        whistle_seconds=_as_float("audio.whistle_seconds", audio["whistle_seconds"]),
        meteor_beep_hz=_as_float("audio.meteor_beep_hz", audio["meteor_beep_hz"]),
        meteor_beep_gap=_as_float("audio.meteor_beep_gap", audio["meteor_beep_gap"]),
        meteor_beeps=_as_int("audio.meteor_beeps", audio["meteor_beeps"]),
        sub_hz=_hz_pair("audio.sub_hz", audio["sub_hz"]),
        target_lufs=_as_float("audio.target_lufs", audio["target_lufs"]),
        true_peak_db=_as_float("audio.true_peak_db", audio["true_peak_db"]),
        fade_ms=_as_float("audio.fade_ms", audio["fade_ms"]),
        loudness_iters=loudness_iters,
        voice_model_id=model_id.strip(),
        voice_id=voice_id.strip(),
        memes=memes,
        guards=guards,
        sensitive_warnings=sensitive_warnings,
    )


def load_arena_config(path: str | Path, *, allow_sensitive: bool = False) -> ArenaConfig:
    config_path = Path(path)
    raw = _load_mapping(config_path)
    hooks = _load_hooks(config_path.parent / "arena_hooks.yaml")
    guards = load_guards(config_path.parent / "guards.yaml")
    memes = load_memes(config_path.parent / "memes.yaml")
    cast_value = raw.get("cast", "configs/casts/world.yaml")
    if not isinstance(cast_value, str) or not cast_value.strip():
        _fail("cast", "must be a path")
    cast_path = _resolve_path(config_path, cast_value, "cast")
    countries, warnings = load_cast(cast_path, guards, allow_sensitive=allow_sensitive)
    for line in memes.warnings:
        print(f"warning: {line}", file=sys.stderr)
    return validate_arena(
        raw,
        hooks=hooks,
        countries=countries,
        guards=guards,
        memes=memes,
        cast_path=str(cast_path),
        sensitive_warnings=warnings,
    )


def render_phrase(template: str, **values: str) -> str:
    missing = [token for token in _PLACEHOLDER_RE.findall(template) if token not in values]
    if missing:
        _fail("memes", f"missing placeholder value {{{missing[0]}}}")
    return _PLACEHOLDER_RE.sub(lambda match: values[match.group(1)], template)
