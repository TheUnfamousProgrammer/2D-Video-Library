from pathlib import Path

import pytest
import yaml

from fc_sat.config import ConfigError, _load_mapping
from fc_sat.odd_config import (
    build_timeline,
    load_odd_config,
    timeline_frames,
    validate_caption,
    validate_odd,
)
from fc_sat.config import _load_hooks


def test_default_config_and_timeline():
    cfg = load_odd_config("configs/odd_default.yaml")
    assert cfg.generator == "odd"
    assert cfg.tier_name == "normal"
    assert [level.id for level in cfg.levels] == [1, 2, 3, 4]
    assert [level.difference for level in cfg.levels] == ["hue", "size", "spin", "pulse"]
    assert [level.count for level in cfg.levels] == [36, 64, 96, 128]
    assert cfg.tier.hue_min_distance == 0.20
    assert cfg.tier.size_ratio == 1.18
    assert cfg.tier.cvd_min_distance == 0.10
    segments = build_timeline(cfg)
    assert sum(segment.duration_s for segment in segments) == pytest.approx(34.9)
    assert timeline_frames(cfg) == 2094
    assert segments[0].kind == "play" and segments[0].n_frames == 300
    assert segments[-1].kind == "outro" and segments[-1].n_frames == 144


def test_quick_tier_keeps_two_levels():
    cfg = load_odd_config("configs/odd_default.yaml", tier="quick")
    assert [level.id for level in cfg.levels] == [1, 3]
    assert cfg.levels[0].timer == 3.0
    assert cfg.levels[1].timer == 4.0
    assert cfg.reveal_seconds == 1.0
    assert cfg.wipe_seconds == 0.2
    # 3 + 4 + 1 + 1 + 0.2 + 1.6
    assert sum(segment.duration_s for segment in build_timeline(cfg)) == pytest.approx(10.8)


def test_brutal_does_not_drop_the_cvd_floor():
    cfg = load_odd_config("configs/odd_default.yaml", tier="brutal")
    assert cfg.tier.hue_min_distance == 0.08
    assert cfg.tier.size_ratio == 1.10
    assert cfg.tier.pulse_odd_hz == 1.62
    assert cfg.tier.cvd_min_distance >= 0.10


def test_unknown_field_names_itself():
    raw = _load_mapping(Path("configs/odd_default.yaml"))
    raw["not_a_field"] = 1
    hooks = _load_hooks(Path("configs/odd_hooks.yaml"))
    captions = _load_hooks(Path("configs/odd_captions.yaml"))
    with pytest.raises(ConfigError, match="not_a_field"):
        validate_odd(raw, hooks=hooks, captions=captions, config_path="configs/odd_default.yaml")


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
    assert cfg.levels[0].difference == "spin"


def test_cvd_floor_cannot_be_relaxed(tmp_path: Path):
    raw = _load_mapping(Path("configs/odd_default.yaml"))
    raw["tiers"]["brutal"]["cvd_min_distance"] = 0.05
    path = tmp_path / "odd.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    hooks = _load_hooks(Path("configs/odd_hooks.yaml"))
    captions = _load_hooks(Path("configs/odd_captions.yaml"))
    with pytest.raises(ConfigError, match="cvd_min_distance"):
        validate_odd(raw, hooks=hooks, captions=captions, config_path=str(path), tier_name="brutal")
