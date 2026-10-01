"""Flag Arena config, casts, guards, and meme validation.

v2 is top-down. A gravity key is rejected. Cast size stays locked at 32.
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
PLACEHOLDERS = frozenset({"killer", "victim", "cameo", "name"})
MEME_EVENTS = ("ko", "combo", "revenge", "storm", "self", "final", "cameo", "winner")
MEME_WARN_CHARS = 34
_PLACEHOLDER_RE = re.compile(r"\{([a-z0-9_]+)\}")
_WORD_RE = re.compile(r"[A-Za-z']+")
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
        "physics",
        "platform",
        "spawn",
        "ai",
        "attribution",
        "boss",
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
    guards: Guards
    memes: Memes
    hook: str
    hooks: tuple[str, ...]
    hook_index: int | None
    sensitive_warnings: tuple[str, ...]
    dt: float
    ball_radius: float
    ball_mass: float
    restitution: float
    damping: float
    speed_clamp: float
    solver_iters: int
    max_step_px: float
    center: tuple[float, float]
    hw0: float
    hh0: float
    corner_frac: float
    platform_keyframes: tuple[tuple[float, float, float], ...]
    shrink_gain: float
    hw_speed_max: float
    n_target: tuple[tuple[float, float], ...]
    disk_frac: float
    speed_min: float
    speed_max: float
    clash_gap: float
    clash_speed: float
    cooldown: tuple[float, float]
    cooldown_min: float
    initial_cooldown: tuple[float, float]
    clash_cooldown: tuple[float, float]
    aggression_time: float
    aggression_scale: float
    windup: float
    windup_speed: float
    fizzle_cooldown: float
    dash_speed: float
    dash_predict: float
    dash_time: float
    dash_margin_radii: float
    edge_margin_radii: float
    edge_outward_speed: float
    edge_accel: float
    score_edge: float
    score_near: float
    score_revenge: float
    score_leader: float
    noise: float
    edge_range: float
    near_range: float
    revenge_window: float
    leader_kos: int
    controller_gain: float
    controller_clamp: tuple[float, float]
    hit_window: float
    storm_window: float
    combo_window: float
    simultaneous: float
    fall_seconds: float
    cameo_name: str
    cameo_enter: float
    cameo_warning: float
    cameo_speed: float
    cameo_radius: float
    cameo_mass: float
    cameo_restitution: float
    cameo_charge_speed: float
    cameo_windup: float
    cameo_charges: int
    cameo_ko_exit: int
    cameo_min_alive: int
    gates: tuple[tuple[float, int, int], ...]
    win_window: tuple[float, float]
    final_duel_min: float
    first_ko: tuple[float, float]
    first_impact: tuple[float, float]
    first_impact_speed: float
    cause_ball_min: float
    cause_storm_max: float
    cause_self_max: float
    dead_time: float
    dead_impact: float
    kinetic_window: float
    kinetic_min: float
    kinetic_tail: float
    heap_bottom: float
    heap_neighbors: float
    min_pass_rate: float
    search: int
    horizon: float
    escape_window: float
    escape_margin_radii: float
    escape_cap: int
    escape_weight: float
    comeback_margin_radii: float
    comeback_cap: int
    comeback_weight: float
    combo_cap: int
    combo_weight: float
    revenge_cap: int
    revenge_weight: float
    duel_cap: float
    duel_weight: float
    mutual_bonus: float
    winner_kos_bonus: float
    repeat_penalty: float
    screen_hw: float
    screen_hh: float
    camera_smooth: float
    punch_seconds: float
    punch_from: float
    punch_to: float
    impact_shake: float
    platform_x: tuple[float, float]
    platform_y: tuple[float, float]
    safe_x: tuple[float, float]
    safe_y: tuple[float, float]
    counter_y: float
    counter_px: float
    hook_y: float
    hook_px: float
    hook_max_chars: int
    hook_until: float
    sub_y: float
    sub_px: float
    sub_from: float
    sub_until: float
    sub_text: str
    kill_y: tuple[float, float]
    kill_px: float
    kill_x: float
    kill_chars: int
    kill_lines: int
    kill_fade: float
    name_px: float
    name_seconds: float
    last_named: int
    hitstop_speed: float
    hitstop_impact_frames: int
    hitstop_ko_frames: int
    hitstop_cap: float
    slowmo_rate: float
    slowmo_pre: float
    slowmo_post: float
    replay_sim: float
    replay_zoom: float
    celebration: float
    winner_scale: float
    confetti: int
    duration_window: tuple[float, float]
    hard_cap: float
    squash: float
    dash_stretch: float
    sparks: int
    eye_frac: float
    bpm: tuple[float, float]
    sidechain_db: float
    voice_duck_db: float
    slowmo_duck_db: float
    slowmo_lpf: float
    stinger_gap: float
    pre_drop: float
    hitstop_duck_db: float
    max_voices: int
    max_onsets_per_100ms: int
    cowbell_hz: tuple[float, float]
    airhorn_hz: tuple[float, float]
    ding_hz: tuple[float, float]
    ding_ms: float
    whistle_hz: tuple[float, float]
    whistle_seconds: float
    sub_hz: tuple[float, float]
    windup_hz: tuple[float, float]
    target_lufs: float
    true_peak_db: float
    fade_ms: float
    loudness_iters: int
    voice_model_id: str
    voice_id: str

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


def _hz_pair(field: str, value: Any) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        _fail(field, "must be a pair of frequencies")
    a = _as_float(field, value[0])
    b = _as_float(field, value[1])
    if a <= 0 or b <= 0:
        _fail(field, "frequencies must be > 0")
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
            if len(text) > MEME_WARN_CHARS:
                warnings.append(f"{field} is {len(text)} characters; keep phrases at or under {MEME_WARN_CHARS}")
            cleaned.append(text)
        phrases.append((event, tuple(cleaned)))
    return Memes(last_reviewed=reviewed, phrases=tuple(phrases), warnings=tuple(warnings))


def load_memes(path: str | Path) -> Memes:
    return validate_memes(_load_mapping(Path(path)))


def render_phrase(template: str, **values: str) -> str:
    missing = [token for token in _PLACEHOLDER_RE.findall(template) if token not in values]
    if missing:
        _fail("memes", f"missing placeholder value {{{missing[0]}}}")
    return _PLACEHOLDER_RE.sub(lambda match: values[match.group(1)], template)


def _reject_gravity(value: Any, path: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "gravity":
                _fail(f"{path}{key}", "gravity is not allowed in a top-down arena")
            _reject_gravity(item, f"{path}{key}.")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_gravity(item, f"{path}[{index}].")


def _platform_frames(value: Any) -> tuple[tuple[float, float, float], ...]:
    if not isinstance(value, list) or len(value) < 2:
        _fail("platform.keyframes", "must be a list of [time, hw, hh] rows")
    frames = []
    previous = -1.0
    for index, item in enumerate(value):
        field = f"platform.keyframes[{index}]"
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            _fail(field, "must be [time, hw, hh]")
        time = _as_float(field, item[0])
        hw = _as_float(field, item[1])
        hh = _as_float(field, item[2])
        if time < previous:
            _fail(field, "times must be non-decreasing")
        if hw <= 0 or hh <= 0:
            _fail(field, "half-extents must be > 0")
        previous = time
        frames.append((time, hw, hh))
    return tuple(frames)


def _count_frames(value: Any) -> tuple[tuple[float, float], ...]:
    if not isinstance(value, list) or len(value) < 2:
        _fail("platform.n_target", "must be a list of [time, alive] rows")
    frames = []
    previous = -1.0
    for index, item in enumerate(value):
        field = f"platform.n_target[{index}]"
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            _fail(field, "must be [time, alive]")
        time = _as_float(field, item[0])
        alive = _as_float(field, item[1])
        if time < previous:
            _fail(field, "times must be non-decreasing")
        previous = time
        frames.append((time, alive))
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


def _xy(field: str, value: Any) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        _fail(field, "must be [x, y]")
    return _as_float(field, value[0]), _as_float(field, value[1])


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
    _reject_gravity(raw, "")
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

    physics = _section(raw, "physics")
    _require_keys(physics and "physics", physics, {"radius", "mass", "restitution", "damping", "speed_clamp", "solver_iters", "max_step_px"})
    platform = _section(raw, "platform")
    _require_keys(
        "platform",
        platform,
        {"center", "hw0", "hh0", "corner_frac", "keyframes", "shrink_gain", "hw_speed_max", "n_target"},
    )
    spawn = _section(raw, "spawn")
    _require_keys("spawn", spawn, {"disk_frac", "speed_min", "speed_max", "clash_gap", "clash_speed"})
    ai = _section(raw, "ai")
    _require_keys(
        "ai",
        ai,
        {
            "cooldown",
            "cooldown_min",
            "initial_cooldown",
            "clash_cooldown",
            "aggression_time",
            "aggression_scale",
            "windup",
            "windup_speed",
            "fizzle_cooldown",
            "dash_speed",
            "dash_predict",
            "dash_time",
            "dash_margin_radii",
            "edge_margin_radii",
            "edge_outward_speed",
            "edge_accel",
            "score_edge",
            "score_near",
            "score_revenge",
            "score_leader",
            "noise",
            "edge_range",
            "near_range",
            "revenge_window",
            "leader_kos",
            "controller_gain",
            "controller_clamp",
        },
    )
    attribution = _section(raw, "attribution")
    _require_keys(
        "attribution",
        attribution,
        {"hit_window", "storm_window", "combo_window", "simultaneous", "fall_seconds"},
    )
    boss = _section(raw, "boss")
    _require_keys(
        "boss",
        boss,
        {
            "name",
            "enter",
            "warning",
            "speed",
            "radius",
            "mass",
            "restitution",
            "charge_speed",
            "windup",
            "charges",
            "ko_exit",
            "min_alive",
        },
    )
    pacing = _section(raw, "pacing")
    _require_keys(
        "pacing",
        pacing,
        {
            "gates",
            "win",
            "final_duel_min",
            "first_ko",
            "first_impact",
            "first_impact_speed",
            "cause_ball_min",
            "cause_storm_max",
            "cause_self_max",
            "dead_time",
            "dead_impact",
            "kinetic_window",
            "kinetic_min",
            "kinetic_tail",
            "heap_bottom",
            "heap_neighbors",
            "min_pass_rate",
            "search",
            "horizon",
        },
    )
    drama = _section(raw, "drama")
    _require_keys(
        "drama",
        drama,
        {
            "escape_window",
            "escape_margin_radii",
            "escape_cap",
            "escape_weight",
            "comeback_margin_radii",
            "comeback_cap",
            "comeback_weight",
            "combo_cap",
            "combo_weight",
            "revenge_cap",
            "revenge_weight",
            "duel_cap",
            "duel_weight",
            "mutual_bonus",
            "winner_kos_bonus",
            "repeat_penalty",
        },
    )
    camera = _section(raw, "camera")
    _require_keys(
        "camera",
        camera,
        {"screen_hw", "screen_hh", "smooth", "punch_seconds", "punch_from", "punch_to", "impact_shake", "platform_x", "platform_y"},
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
            "hook_max_chars",
            "hook_until",
            "sub_y",
            "sub_px",
            "sub_from",
            "sub_until",
            "sub_text",
            "kill_y",
            "kill_px",
            "kill_x",
            "kill_chars",
            "kill_lines",
            "kill_fade",
            "name_px",
            "name_seconds",
            "last_named",
        },
    )
    timeline = _section(raw, "timeline")
    _require_keys(
        "timeline",
        timeline,
        {
            "hitstop_speed",
            "hitstop_impact_frames",
            "hitstop_ko_frames",
            "hitstop_cap",
            "slowmo_rate",
            "slowmo_pre",
            "slowmo_post",
            "replay_sim",
            "replay_zoom",
            "celebration",
            "winner_scale",
            "confetti",
            "duration",
            "hard_cap",
        },
    )
    juice = _section(raw, "juice")
    _require_keys("juice", juice, {"squash", "dash_stretch", "sparks", "eye_frac"})
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
            "hitstop_duck_db",
            "max_voices",
            "max_onsets_per_100ms",
            "cowbell_hz",
            "airhorn_hz",
            "ding_hz",
            "ding_ms",
            "whistle_hz",
            "whistle_seconds",
            "sub_hz",
            "windup_hz",
            "target_lufs",
            "true_peak_db",
            "fade_ms",
            "loudness_iters",
        },
    )
    voice = _section(raw, "voice")
    _require_keys("voice", voice, {"model_id", "voice_id"})

    radius = _as_float("physics.radius", physics["radius"])
    mass = _as_float("physics.mass", physics["mass"])
    restitution = _as_float("physics.restitution", physics["restitution"])
    damping = _as_float("physics.damping", physics["damping"])
    speed_clamp = _as_float("physics.speed_clamp", physics["speed_clamp"])
    solver_iters = _as_int("physics.solver_iters", physics["solver_iters"])
    max_step = _as_float("physics.max_step_px", physics["max_step_px"])
    if radius <= 0 or mass <= 0 or damping < 0 or speed_clamp <= 0:
        _fail("physics.radius", "radius, mass, and speed clamp must be > 0")
    if restitution < 0 or restitution > 1:
        _fail("physics.restitution", "must be in [0, 1]")
    if solver_iters < 1 or solver_iters > 8:
        _fail("physics.solver_iters", "must be in [1, 8]")
    dt = 1.0 / 240.0
    if speed_clamp * dt > max_step + 1e-6:
        _fail("physics.max_step_px", "must cover speed_clamp / 240 so a step cannot tunnel")

    pass_rate = _as_float("pacing.min_pass_rate", pacing["min_pass_rate"])
    if pass_rate <= 0 or pass_rate > 1:
        _fail("pacing.min_pass_rate", "must be in (0, 1]")
    search = _as_int("pacing.search", pacing["search"])
    if search < 1:
        _fail("pacing.search", "must be >= 1")
    loudness_iters = _as_int("audio.loudness_iters", audio["loudness_iters"])
    if loudness_iters < 1 or loudness_iters > 5:
        _fail("audio.loudness_iters", "must be in [1, 5]")

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
        guards=guards,
        memes=memes,
        hook=hook,
        hooks=hooks,
        hook_index=hook_index,
        sensitive_warnings=sensitive_warnings,
        dt=dt,
        ball_radius=radius,
        ball_mass=mass,
        restitution=restitution,
        damping=damping,
        speed_clamp=speed_clamp,
        solver_iters=solver_iters,
        max_step_px=max_step,
        center=_xy("platform.center", platform["center"]),
        hw0=_as_float("platform.hw0", platform["hw0"]),
        hh0=_as_float("platform.hh0", platform["hh0"]),
        corner_frac=_as_float("platform.corner_frac", platform["corner_frac"]),
        platform_keyframes=_platform_frames(platform["keyframes"]),
        shrink_gain=_as_float("platform.shrink_gain", platform["shrink_gain"]),
        hw_speed_max=_as_float("platform.hw_speed_max", platform["hw_speed_max"]),
        n_target=_count_frames(platform["n_target"]),
        disk_frac=_as_float("spawn.disk_frac", spawn["disk_frac"]),
        speed_min=_as_float("spawn.speed_min", spawn["speed_min"]),
        speed_max=_as_float("spawn.speed_max", spawn["speed_max"]),
        clash_gap=_as_float("spawn.clash_gap", spawn["clash_gap"]),
        clash_speed=_as_float("spawn.clash_speed", spawn["clash_speed"]),
        cooldown=_pair("ai.cooldown", ai["cooldown"], lo_positive=True),
        cooldown_min=_as_float("ai.cooldown_min", ai["cooldown_min"]),
        initial_cooldown=_pair("ai.initial_cooldown", ai["initial_cooldown"], lo_positive=True),
        clash_cooldown=_pair("ai.clash_cooldown", ai["clash_cooldown"], lo_positive=True),
        aggression_time=_as_float("ai.aggression_time", ai["aggression_time"]),
        aggression_scale=_as_float("ai.aggression_scale", ai["aggression_scale"]),
        windup=_as_float("ai.windup", ai["windup"]),
        windup_speed=_as_float("ai.windup_speed", ai["windup_speed"]),
        fizzle_cooldown=_as_float("ai.fizzle_cooldown", ai["fizzle_cooldown"]),
        dash_speed=_as_float("ai.dash_speed", ai["dash_speed"]),
        dash_predict=_as_float("ai.dash_predict", ai["dash_predict"]),
        dash_time=_as_float("ai.dash_time", ai["dash_time"]),
        dash_margin_radii=_as_float("ai.dash_margin_radii", ai["dash_margin_radii"]),
        edge_margin_radii=_as_float("ai.edge_margin_radii", ai["edge_margin_radii"]),
        edge_outward_speed=_as_float("ai.edge_outward_speed", ai["edge_outward_speed"]),
        edge_accel=_as_float("ai.edge_accel", ai["edge_accel"]),
        score_edge=_as_float("ai.score_edge", ai["score_edge"]),
        score_near=_as_float("ai.score_near", ai["score_near"]),
        score_revenge=_as_float("ai.score_revenge", ai["score_revenge"]),
        score_leader=_as_float("ai.score_leader", ai["score_leader"]),
        noise=_as_float("ai.noise", ai["noise"]),
        edge_range=_as_float("ai.edge_range", ai["edge_range"]),
        near_range=_as_float("ai.near_range", ai["near_range"]),
        revenge_window=_as_float("ai.revenge_window", ai["revenge_window"]),
        leader_kos=_as_int("ai.leader_kos", ai["leader_kos"]),
        controller_gain=_as_float("ai.controller_gain", ai["controller_gain"]),
        controller_clamp=_pair("ai.controller_clamp", ai["controller_clamp"], lo_positive=True),
        hit_window=_as_float("attribution.hit_window", attribution["hit_window"]),
        storm_window=_as_float("attribution.storm_window", attribution["storm_window"]),
        combo_window=_as_float("attribution.combo_window", attribution["combo_window"]),
        simultaneous=_as_float("attribution.simultaneous", attribution["simultaneous"]),
        fall_seconds=_as_float("attribution.fall_seconds", attribution["fall_seconds"]),
        cameo_name=_as_str("boss.name", boss["name"]),
        cameo_enter=_as_float("boss.enter", boss["enter"]),
        cameo_warning=_as_float("boss.warning", boss["warning"]),
        cameo_speed=_as_float("boss.speed", boss["speed"]),
        cameo_radius=_as_float("boss.radius", boss["radius"]),
        cameo_mass=_as_float("boss.mass", boss["mass"]),
        cameo_restitution=_as_float("boss.restitution", boss["restitution"]),
        cameo_charge_speed=_as_float("boss.charge_speed", boss["charge_speed"]),
        cameo_windup=_as_float("boss.windup", boss["windup"]),
        cameo_charges=_as_int("boss.charges", boss["charges"]),
        cameo_ko_exit=_as_int("boss.ko_exit", boss["ko_exit"]),
        cameo_min_alive=_as_int("boss.min_alive", boss["min_alive"]),
        gates=_gates(pacing["gates"]),
        win_window=_pair("pacing.win", pacing["win"]),
        final_duel_min=_as_float("pacing.final_duel_min", pacing["final_duel_min"]),
        first_ko=_pair("pacing.first_ko", pacing["first_ko"]),
        first_impact=_pair("pacing.first_impact", pacing["first_impact"]),
        first_impact_speed=_as_float("pacing.first_impact_speed", pacing["first_impact_speed"]),
        cause_ball_min=_as_float("pacing.cause_ball_min", pacing["cause_ball_min"]),
        cause_storm_max=_as_float("pacing.cause_storm_max", pacing["cause_storm_max"]),
        cause_self_max=_as_float("pacing.cause_self_max", pacing["cause_self_max"]),
        dead_time=_as_float("pacing.dead_time", pacing["dead_time"]),
        dead_impact=_as_float("pacing.dead_impact", pacing["dead_impact"]),
        kinetic_window=_as_float("pacing.kinetic_window", pacing["kinetic_window"]),
        kinetic_min=_as_float("pacing.kinetic_min", pacing["kinetic_min"]),
        kinetic_tail=_as_float("pacing.kinetic_tail", pacing["kinetic_tail"]),
        heap_bottom=_as_float("pacing.heap_bottom", pacing["heap_bottom"]),
        heap_neighbors=_as_float("pacing.heap_neighbors", pacing["heap_neighbors"]),
        min_pass_rate=pass_rate,
        search=search,
        horizon=_as_float("pacing.horizon", pacing["horizon"]),
        escape_window=_as_float("drama.escape_window", drama["escape_window"]),
        escape_margin_radii=_as_float("drama.escape_margin_radii", drama["escape_margin_radii"]),
        escape_cap=_as_int("drama.escape_cap", drama["escape_cap"]),
        escape_weight=_as_float("drama.escape_weight", drama["escape_weight"]),
        comeback_margin_radii=_as_float("drama.comeback_margin_radii", drama["comeback_margin_radii"]),
        comeback_cap=_as_int("drama.comeback_cap", drama["comeback_cap"]),
        comeback_weight=_as_float("drama.comeback_weight", drama["comeback_weight"]),
        combo_cap=_as_int("drama.combo_cap", drama["combo_cap"]),
        combo_weight=_as_float("drama.combo_weight", drama["combo_weight"]),
        revenge_cap=_as_int("drama.revenge_cap", drama["revenge_cap"]),
        revenge_weight=_as_float("drama.revenge_weight", drama["revenge_weight"]),
        duel_cap=_as_float("drama.duel_cap", drama["duel_cap"]),
        duel_weight=_as_float("drama.duel_weight", drama["duel_weight"]),
        mutual_bonus=_as_float("drama.mutual_bonus", drama["mutual_bonus"]),
        winner_kos_bonus=_as_float("drama.winner_kos_bonus", drama["winner_kos_bonus"]),
        repeat_penalty=_as_float("drama.repeat_penalty", drama["repeat_penalty"]),
        screen_hw=_as_float("camera.screen_hw", camera["screen_hw"]),
        screen_hh=_as_float("camera.screen_hh", camera["screen_hh"]),
        camera_smooth=_as_float("camera.smooth", camera["smooth"]),
        punch_seconds=_as_float("camera.punch_seconds", camera["punch_seconds"]),
        punch_from=_as_float("camera.punch_from", camera["punch_from"]),
        punch_to=_as_float("camera.punch_to", camera["punch_to"]),
        impact_shake=_as_float("camera.impact_shake", camera["impact_shake"]),
        platform_x=_pair("camera.platform_x", camera["platform_x"]),
        platform_y=_pair("camera.platform_y", camera["platform_y"]),
        safe_x=_pair("layout.safe_x", layout["safe_x"]),
        safe_y=_pair("layout.safe_y", layout["safe_y"]),
        counter_y=_as_float("layout.counter_y", layout["counter_y"]),
        counter_px=_as_float("layout.counter_px", layout["counter_px"]),
        hook_y=_as_float("layout.hook_y", layout["hook_y"]),
        hook_px=_as_float("layout.hook_px", layout["hook_px"]),
        hook_max_chars=_as_int("layout.hook_max_chars", layout["hook_max_chars"]),
        hook_until=_as_float("layout.hook_until", layout["hook_until"]),
        sub_y=_as_float("layout.sub_y", layout["sub_y"]),
        sub_px=_as_float("layout.sub_px", layout["sub_px"]),
        sub_from=_as_float("layout.sub_from", layout["sub_from"]),
        sub_until=_as_float("layout.sub_until", layout["sub_until"]),
        sub_text=_as_str("layout.sub_text", layout["sub_text"]),
        kill_y=_pair("layout.kill_y", layout["kill_y"]),
        kill_px=_as_float("layout.kill_px", layout["kill_px"]),
        kill_x=_as_float("layout.kill_x", layout["kill_x"]),
        kill_chars=_as_int("layout.kill_chars", layout["kill_chars"]),
        kill_lines=_as_int("layout.kill_lines", layout["kill_lines"]),
        kill_fade=_as_float("layout.kill_fade", layout["kill_fade"]),
        name_px=_as_float("layout.name_px", layout["name_px"]),
        name_seconds=_as_float("layout.name_seconds", layout["name_seconds"]),
        last_named=_as_int("layout.last_named", layout["last_named"]),
        hitstop_speed=_as_float("timeline.hitstop_speed", timeline["hitstop_speed"]),
        hitstop_impact_frames=_as_int("timeline.hitstop_impact_frames", timeline["hitstop_impact_frames"]),
        hitstop_ko_frames=_as_int("timeline.hitstop_ko_frames", timeline["hitstop_ko_frames"]),
        hitstop_cap=_as_float("timeline.hitstop_cap", timeline["hitstop_cap"]),
        slowmo_rate=_as_float("timeline.slowmo_rate", timeline["slowmo_rate"]),
        slowmo_pre=_as_float("timeline.slowmo_pre", timeline["slowmo_pre"]),
        slowmo_post=_as_float("timeline.slowmo_post", timeline["slowmo_post"]),
        replay_sim=_as_float("timeline.replay_sim", timeline["replay_sim"]),
        replay_zoom=_as_float("timeline.replay_zoom", timeline["replay_zoom"]),
        celebration=_as_float("timeline.celebration", timeline["celebration"]),
        winner_scale=_as_float("timeline.winner_scale", timeline["winner_scale"]),
        confetti=_as_int("timeline.confetti", timeline["confetti"]),
        duration_window=_pair("timeline.duration", timeline["duration"]),
        hard_cap=_as_float("timeline.hard_cap", timeline["hard_cap"]),
        squash=_as_float("juice.squash", juice["squash"]),
        dash_stretch=_as_float("juice.dash_stretch", juice["dash_stretch"]),
        sparks=_as_int("juice.sparks", juice["sparks"]),
        eye_frac=_as_float("juice.eye_frac", juice["eye_frac"]),
        bpm=_pair("audio.bpm", audio["bpm"], lo_positive=True),
        sidechain_db=_as_float("audio.sidechain_db", audio["sidechain_db"]),
        voice_duck_db=_as_float("audio.voice_duck_db", audio["voice_duck_db"]),
        slowmo_duck_db=_as_float("audio.slowmo_duck_db", audio["slowmo_duck_db"]),
        slowmo_lpf=_as_float("audio.slowmo_lpf", audio["slowmo_lpf"]),
        stinger_gap=_as_float("audio.stinger_gap", audio["stinger_gap"]),
        pre_drop=_as_float("audio.pre_drop", audio["pre_drop"]),
        hitstop_duck_db=_as_float("audio.hitstop_duck_db", audio["hitstop_duck_db"]),
        max_voices=_as_int("audio.max_voices", audio["max_voices"]),
        max_onsets_per_100ms=_as_int("audio.max_onsets_per_100ms", audio["max_onsets_per_100ms"]),
        cowbell_hz=_hz_pair("audio.cowbell_hz", audio["cowbell_hz"]),
        airhorn_hz=_hz_pair("audio.airhorn_hz", audio["airhorn_hz"]),
        ding_hz=_hz_pair("audio.ding_hz", audio["ding_hz"]),
        ding_ms=_as_float("audio.ding_ms", audio["ding_ms"]),
        whistle_hz=_hz_pair("audio.whistle_hz", audio["whistle_hz"]),
        whistle_seconds=_as_float("audio.whistle_seconds", audio["whistle_seconds"]),
        sub_hz=_hz_pair("audio.sub_hz", audio["sub_hz"]),
        windup_hz=_hz_pair("audio.windup_hz", audio["windup_hz"]),
        target_lufs=_as_float("audio.target_lufs", audio["target_lufs"]),
        true_peak_db=_as_float("audio.true_peak_db", audio["true_peak_db"]),
        fade_ms=_as_float("audio.fade_ms", audio["fade_ms"]),
        loudness_iters=loudness_iters,
        voice_model_id=_as_str("voice.model_id", voice["model_id"]),
        voice_id="" if voice.get("voice_id") in (None, "") else _as_str("voice.voice_id", voice.get("voice_id")),
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
