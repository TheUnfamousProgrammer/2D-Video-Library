from pathlib import Path

import pytest

from fc_sat.config import ConfigError, _load_hooks, _load_mapping
from fc_sat.odd_config import (
    build_timeline,
    load_odd_config,
    timeline_frames,
    validate_caption,
    validate_odd,
)


def test_default_config_and_timeline():
    cfg = load_odd_config("configs/odd_default.yaml")
    assert cfg.generator == "odd"
    assert cfg.tier_name == "normal"
    assert [level.id for level in cfg.levels] == [1, 2, 3]
    assert [level.difference for level in cfg.levels] == ["hue", "tilt", "detail"]
    assert [level.count for level in cfg.levels] == [16, 25, 36]
    assert [level.size for level in cfg.levels] == [100, 96, 88]
    assert cfg.tier.hue_min_distance == 0.25
    assert cfg.tier.tilt_degrees == 22
    assert cfg.tier.cvd_min_distance == 0.12
    assert "bloom" not in cfg.to_public_dict()
    segments = build_timeline(cfg)
    assert sum(segment.duration_s for segment in segments) == pytest.approx(24.0)
    assert timeline_frames(cfg) == 1440
    assert segments[0].kind == "play" and segments[0].n_frames == 300
    play = next(segment for segment in segments if segment.kind == "play" and segment.level_id == 1)
    reveal = next(segment for segment in segments if segment.kind == "reveal" and segment.level_id == 1)
    assert play.end_frame == reveal.start_frame
    assert segments[-1].kind == "outro" and segments[-1].n_frames == 120


def test_easy_is_bigger_and_hard_adds_level_4():
    easy = load_odd_config("configs/odd_default.yaml", tier="easy")
    assert easy.tier.hue_min_distance == 0.35
    assert easy.tier.tilt_degrees == 32
    assert easy.tier.dot_fraction == 0.22
    assert easy.tier.cvd_min_distance >= 0.12
    hard = load_odd_config("configs/odd_default.yaml", tier="hard")
    assert [level.id for level in hard.levels] == [1, 2, 3, 4]
    assert hard.levels[-1].difference == "hue_subtle"
    assert hard.levels[-1].count == 36
    assert hard.levels[-1].timer == 8.0
    assert hard.tier.hue_min_distance == 0.25
    assert hard.tier.hue_subtle_cvd >= 0.06
    assert sum(segment.duration_s for segment in build_timeline(hard)) == pytest.approx(33.4)


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


def test_cvd_floor_cannot_be_relaxed(tmp_path: Path):
    raw = _load_mapping(Path("configs/odd_default.yaml"))
    raw["tiers"]["normal"]["cvd_min_distance"] = 0.05
    hooks = _load_hooks(Path("configs/odd_hooks.yaml"))
    with pytest.raises(ConfigError, match="cvd_min_distance"):
        validate_odd(raw, hooks=hooks, config_path="configs/odd_default.yaml")


def test_config_has_no_bloom_key():
    text = Path("configs/odd_default.yaml").read_text(encoding="utf-8").lower()
    assert "bloom" not in text
