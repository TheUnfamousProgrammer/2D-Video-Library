import pytest

from fc_sat.config import ConfigError
from fc_sat.hop_config import load_hop_config, validate_hop
from fc_sat.hop_render import HopRenderer, action_bounds, hook_alpha
from make_hop import unique_hop_variants, variant_name
from pathlib import Path


def test_hop_config_names_invalid_fields():
    cfg = load_hop_config("configs/hop_default.yaml")
    assert cfg.hook == "Guess the song"
    assert cfg.repeats == 1
    assert cfg.octave_shift == 1
    assert cfg.bloom_strength == pytest.approx(0.6)
    assert cfg.show_note_names is False
    palettes = {"sunset": cfg.palette_stops}
    with pytest.raises(ConfigError, match="repeats"):
        validate_hop({"repeats": 4, "palette": "sunset"}, palettes=palettes, hooks=cfg.hooks)
    with pytest.raises(ConfigError, match="palette"):
        validate_hop({"palette": "nope"}, palettes=palettes, hooks=cfg.hooks)
    with pytest.raises(ConfigError, match="audio.reverb_wet"):
        validate_hop(
            {"palette": "sunset", "audio": {"reverb_wet": 2}},
            palettes=palettes,
            hooks=cfg.hooks,
        )


def test_hook_alpha_opens_and_closes_the_loop():
    cfg = load_hop_config("configs/hop_default.yaml")
    renderer = HopRenderer(cfg)
    duration = renderer.dance.grid.duration
    n = renderer.dance.grid.n_frames
    assert hook_alpha(0.0, duration) == 1.0
    assert hook_alpha((n - 1) / 60.0, duration) >= 0.97
    assert hook_alpha(2.8, duration) == pytest.approx(0.5)
    assert hook_alpha(4.0, duration) == 0.0
    assert hook_alpha(duration, duration) == 1.0


def test_nothing_is_drawn_outside_the_action_box_before_bloom():
    cfg = load_hop_config("configs/hop_default.yaml")
    renderer = HopRenderer(cfg)
    signal = renderer.pre_bloom(0.0)
    x0, y0, x1, y1 = action_bounds(renderer.scale)
    assert signal[y0:y1, x0:x1].max() > 0
    assert float(signal[:y0].max()) == 0.0
    assert float(signal[y1:].max()) == 0.0
    assert float(signal[:, :x0].max()) == 0.0
    assert float(signal[:, x1:].max()) == 0.0
    later = renderer.pre_bloom(1.25)
    assert float(later[:y0].max()) == 0.0
    assert float(later[:, x1:].max()) == 0.0


def test_batch_tuples_are_unique():
    cfg = load_hop_config("configs/hop_default.yaml")
    names = ["sunset", "aurora", "candy", "ocean"]
    combos = unique_hop_variants(
        3,
        palette=cfg.palette,
        hook_index=0,
        octave=cfg.octave_shift,
        bpm=120,
        vary="palette,hook",
        palette_names=names,
        n_hooks=len(cfg.hooks),
    )
    assert len(combos) == 3
    assert len(set(combos)) == 3
    paths = [
        variant_name(Path("out/hop_batch.mp4"), "ode_to_joy", palette, hook)
        for palette, hook, _octave, _bpm in combos
    ]
    assert len(set(paths)) == 3
    assert paths[0].name == "hop_batch_ode_to_joy_sunset_0.mp4"
