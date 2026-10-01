from pathlib import Path

import pytest

from fc_sat.config import ConfigError, _load_hooks, _load_mapping
from fc_sat.odd_config import (
    DETAIL_OFFSET,
    HUE_DISTANCE,
    HUE_LIGHTNESS,
    TILT_DEGREES,
    active_rung,
    build_timeline,
    cvd_floor,
    load_odd_config,
    timeline_frames,
    validate_caption,
    validate_odd,
)


def test_rung_tables():
    assert HUE_DISTANCE == {1: 0.20, 2: 0.15, 3: 0.12, 4: 0.09, 5: 0.06}
    assert TILT_DEGREES == {1: 18.0, 2: 13.0, 3: 9.0, 4: 6.0, 5: 4.0}
    assert DETAIL_OFFSET == {1: 0.55, 2: 0.42, 3: 0.30, 4: 0.22, 5: 0.15}
    for rung, distance in HUE_DISTANCE.items():
        assert cvd_floor(distance) == pytest.approx(0.5 * distance)
    cfg = load_odd_config("configs/odd_default.yaml")
    assert (cfg.tier.hue_rung, cfg.tier.tilt_rung, cfg.tier.detail_rung) == (2, 3, 4)
    assert HUE_DISTANCE[cfg.tier.hue_rung] == pytest.approx(0.15)
    assert cvd_floor(HUE_DISTANCE[2]) == pytest.approx(0.075)
    assert cvd_floor(HUE_DISTANCE[2]) >= 0.07
    assert HUE_LIGHTNESS == pytest.approx(0.05)


def test_default_config_and_timeline():
    cfg = load_odd_config("configs/odd_default.yaml")
    assert cfg.generator == "odd"
    assert cfg.tier_name == "normal"
    assert [level.id for level in cfg.levels] == [1, 2, 3]
    assert [level.difference for level in cfg.levels] == ["hue", "tilt", "detail"]
    assert [level.grid for level in cfg.levels] == [5, 6, 7]
    assert [level.count for level in cfg.levels] == [25, 36, 49]
    assert [level.size for level in cfg.levels] == [90, 84, 76]
    assert [level.timer for level in cfg.levels] == [6.0, 7.0, 8.0]
    assert cfg.detail_mode == "moved"
    assert cfg.dot_fraction == pytest.approx(0.14)
    assert "bloom" not in cfg.to_public_dict()
    segments = build_timeline(cfg)
    assert sum(segment.duration_s for segment in segments) == pytest.approx(27.0)
    assert timeline_frames(cfg) == 1620
    assert segments[0].kind == "play" and segments[0].n_frames == 360
    play = next(segment for segment in segments if segment.kind == "play" and segment.level_id == 1)
    reveal = next(segment for segment in segments if segment.kind == "reveal" and segment.level_id == 1)
    assert play.end_frame == reveal.start_frame
    assert segments[-1].kind == "outro" and segments[-1].n_frames == 120


def test_easy_is_rung_1_and_hard_adds_level_4():
    easy = load_odd_config("configs/odd_default.yaml", tier="easy")
    assert (easy.tier.hue_rung, easy.tier.tilt_rung, easy.tier.detail_rung) == (1, 1, 1)
    assert HUE_DISTANCE[easy.tier.hue_rung] == pytest.approx(0.20)
    assert TILT_DEGREES[easy.tier.tilt_rung] == pytest.approx(18)
    assert DETAIL_OFFSET[easy.tier.detail_rung] == pytest.approx(0.55)
    assert cvd_floor(HUE_DISTANCE[easy.tier.hue_rung]) == pytest.approx(0.10)
    hard = load_odd_config("configs/odd_default.yaml", tier="hard")
    assert [level.id for level in hard.levels] == [1, 2, 3, 4]
    assert hard.levels[-1].difference == "hue"
    assert hard.levels[-1].grid == 7
    assert hard.levels[-1].count == 49
    assert hard.levels[-1].size == 76
    assert hard.levels[-1].timer == 9.0
    assert active_rung(hard.tier, hard.levels[-1]) == 5
    assert HUE_DISTANCE[5] == pytest.approx(0.06)
    assert cvd_floor(HUE_DISTANCE[5]) == pytest.approx(0.03)
    assert sum(segment.duration_s for segment in build_timeline(hard)) == pytest.approx(37.4)
    assert timeline_frames(hard) == 2244


def test_unknown_field_names_itself():
    raw = _load_mapping(Path("configs/odd_default.yaml"))
    raw["not_a_field"] = 1
    hooks = _load_hooks(Path("configs/odd_hooks.yaml"))
    with pytest.raises(ConfigError, match="not_a_field"):
        validate_odd(raw, hooks=hooks, config_path="configs/odd_default.yaml")


def test_caption_validator_rejects_percentages():
    with pytest.raises(ConfigError, match="percentage"):
        validate_caption("only 1% can see this", field="hook")
    with pytest.raises(ConfigError, match="percentage"):
        validate_caption("99%", field="captions")
    with pytest.raises(ConfigError, match="percentage"):
        validate_caption("seen by 12 % of people", field="captions")
    assert validate_caption("Find the odd one", field="hook", max_chars=26) == "Find the odd one"
    with pytest.raises(ConfigError, match="26"):
        validate_caption("this caption is definitely too long", field="hook", max_chars=26)


def test_level_subset_keeps_order():
    cfg = load_odd_config("configs/odd_default.yaml", level_ids=[3, 1])
    assert [level.id for level in cfg.levels] == [3, 1]
    assert cfg.levels[0].difference == "detail"


def test_rung_must_be_one_to_five():
    raw = _load_mapping(Path("configs/odd_default.yaml"))
    raw["rung"]["hue"] = 6
    hooks = _load_hooks(Path("configs/odd_hooks.yaml"))
    with pytest.raises(ConfigError, match="rung.hue"):
        validate_odd(raw, hooks=hooks, config_path="configs/odd_default.yaml")
    raw = _load_mapping(Path("configs/odd_default.yaml"))
    raw["rung"]["detail"] = 0
    with pytest.raises(ConfigError, match="rung.detail"):
        validate_odd(raw, hooks=hooks, config_path="configs/odd_default.yaml")


def test_config_has_no_bloom_key():
    text = Path("configs/odd_default.yaml").read_text(encoding="utf-8").lower()
    assert "bloom" not in text
