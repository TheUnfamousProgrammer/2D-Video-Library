import pytest

from fc_sat.config import ConfigError, load_config, with_overrides


def test_default_config_loads():
    cfg = load_config("configs/default.yaml")
    assert cfg.cap == 1000
    assert cfg.ring_radius == 380
    assert cfg.fps == 60
    assert cfg.n_frames() == 1290
    assert abs(cfg.total_seconds - 21.5) < 1e-9
    assert cfg.throttle[-1] == (0.985, 1000)
    assert len(cfg.palette_stops) >= 3


def test_radius_min_out_of_range():
    cfg = load_config("configs/default.yaml")
    with pytest.raises(ConfigError, match="radius_min"):
        with_overrides(cfg, radius_min=100)
    with pytest.raises(ConfigError, match="radius_min"):
        with_overrides(cfg, radius_min=0)


def test_total_duration_cap():
    cfg = load_config("configs/default.yaml")
    with pytest.raises(ConfigError, match="30"):
        with_overrides(cfg, growth_seconds=40)
