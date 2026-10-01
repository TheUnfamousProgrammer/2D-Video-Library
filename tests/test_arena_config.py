import copy
from pathlib import Path

import pytest
import yaml

from fc_sat.arena_config import (
    CAST_SIZE,
    DENYLIST,
    load_arena_config,
    load_cast,
    load_guards,
    load_memes,
    validate_memes,
)
from fc_sat.config import ConfigError, _load_mapping


def test_default_arena_config_loads():
    cfg = load_arena_config("configs/arena_default.yaml")
    assert cfg.generator == "arena"
    assert cfg.cast_size == CAST_SIZE
    assert len(cfg.countries) == 32
    assert cfg.ball_radius == 34
    assert cfg.zone_keyframes[0] == (0.0, 400.0)
    assert cfg.zone_keyframes[-1] == (29.0, 130.0)
    assert cfg.voice_model_id == "eleven_v4"
    assert cfg.cameo_name == "OHIO"
    assert cfg.gates[0] == (4.0, 27, 30)
    assert cfg.win_window == (26.0, 32.0)


def test_cast_size_is_locked():
    raw = _load_mapping(Path("configs/arena_default.yaml"))
    raw["cast_size"] = 16
    guards = load_guards("configs/guards.yaml")
    countries, warnings = load_cast("configs/casts/world.yaml", guards)
    memes = load_memes("configs/memes.yaml")
    from fc_sat.arena_config import validate_arena
    from fc_sat.config import _load_hooks

    with pytest.raises(ConfigError, match="cast_size"):
        validate_arena(
            raw,
            hooks=_load_hooks(Path("configs/arena_hooks.yaml")),
            countries=countries,
            guards=guards,
            memes=memes,
            cast_path="configs/casts/world.yaml",
            sensitive_warnings=warnings,
        )


def test_unknown_field_names_itself():
    raw = _load_mapping(Path("configs/arena_default.yaml"))
    raw["not_a_field"] = 1
    guards = load_guards("configs/guards.yaml")
    countries, _warnings = load_cast("configs/casts/world.yaml", guards)
    from fc_sat.arena_config import validate_arena
    from fc_sat.config import _load_hooks

    with pytest.raises(ConfigError, match="not_a_field"):
        validate_arena(
            raw,
            hooks=_load_hooks(Path("configs/arena_hooks.yaml")),
            countries=countries,
            guards=guards,
            memes=load_memes("configs/memes.yaml"),
            cast_path="configs/casts/world.yaml",
        )


def test_short_cast_is_rejected(tmp_path: Path):
    guards = load_guards("configs/guards.yaml")
    payload = yaml.safe_load(Path("configs/casts/world.yaml").read_text(encoding="utf-8"))
    path = tmp_path / "short.yaml"
    path.write_text(yaml.safe_dump(payload[:8]), encoding="utf-8")
    with pytest.raises(ConfigError, match="cast"):
        load_cast(path, guards)


def test_blocked_code_rejected_unless_allowed(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    guards = load_guards("configs/guards.yaml")
    payload = yaml.safe_load(Path("configs/casts/world.yaml").read_text(encoding="utf-8"))
    payload = copy.deepcopy(payload)
    payload[0] = {"name": "Saudi Arabia", "iso2": "SA", "iso3": "SAU"}
    path = tmp_path / "sensitive.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ConfigError, match="SA"):
        load_cast(path, guards)
    countries, warnings = load_cast(path, guards, allow_sensitive=True)
    captured = capsys.readouterr()
    assert countries[0].iso2 == "SA"
    assert warnings and "SA" in warnings[0]
    assert "warning:" in captured.err
    assert "SA" in captured.err


def test_no_cast_contains_a_blocked_code():
    guards = load_guards("configs/guards.yaml")
    assert {"SD", "SS", "MM", "SA", "RU", "UA", "IL", "TW"} <= guards.blocked
    paths = sorted(Path("configs/casts").glob("*.yaml"))
    assert {path.name for path in paths} == {
        "africa.yaml",
        "americas.yaml",
        "asia.yaml",
        "europe.yaml",
        "world.yaml",
    }
    for path in paths:
        countries, warnings = load_cast(path, guards)
        assert len(countries) == 32
        assert warnings == ()
        codes = {country.iso2 for country in countries}
        assert codes.isdisjoint(guards.blocked)
        assert len(codes) == 32


def test_meme_denylist_and_placeholders():
    memes = load_memes("configs/memes.yaml")
    assert memes.last_reviewed == "2026-10-01"
    assert any("walked off" in line for line in memes.lines("self_fall"))
    assert any("28" in line or "characters" in line for line in memes.warnings)
    raw = {
        "last_reviewed": "2026-10-01",
        "eliminated": ["{code} is fine"],
        "meteor": ["{code} met the meteor"],
        "self_fall": ["{code} tripped"],
        "final3": ["FINAL 3"],
        "cameo": ["{cameo} has entered the chat"],
        "winner": ["{name} WINS"],
    }
    blocked = copy.deepcopy(raw)
    blocked["eliminated"] = ["{code} nationality"]
    with pytest.raises(ConfigError, match="nationality"):
        validate_memes(blocked)
    slur = next(iter(DENYLIST))
    raw_slur = copy.deepcopy(raw)
    raw_slur["winner"] = [f"{{name}} {slur}"]
    with pytest.raises(ConfigError, match="blocked word"):
        validate_memes(raw_slur)
    unknown = copy.deepcopy(raw)
    unknown["cameo"] = ["{country} arrived"]
    with pytest.raises(ConfigError, match="placeholder"):
        validate_memes(unknown)


def test_world_cast_matches_the_spec_order():
    guards = load_guards("configs/guards.yaml")
    countries, _warnings = load_cast("configs/casts/world.yaml", guards)
    assert [country.iso2 for country in countries[:4]] == ["BR", "AR", "FR", "DE"]
    assert countries[-1].iso3 == "VNM"
    assert "SD" not in {country.iso2 for country in countries}
