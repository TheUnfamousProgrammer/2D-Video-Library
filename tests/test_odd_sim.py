from dataclasses import replace

import numpy as np
import pytest

from fc_sat.odd_config import load_odd_config
from fc_sat.odd_sim import reseed_loop, sample_drivers, simulate_show


def _small(tier="normal", level_ids=None, seed=7):
    cfg = load_odd_config("configs/odd_default.yaml", tier=tier, seed=seed, level_ids=level_ids)
    levels = tuple(replace(level, count=12, timer=1.2) for level in cfg.levels)
    return replace(cfg, levels=levels, warmup_seconds=0.25, reveal_seconds=0.3, wipe_seconds=0.2, outro_seconds=0.4)


def test_sim_is_deterministic():
    cfg = _small()
    first = simulate_show(cfg)
    second = simulate_show(cfg)
    for level in cfg.levels:
        assert first.levels[level.id].odd_index == second.levels[level.id].odd_index
        assert np.array_equal(first.levels[level.id].positions, second.levels[level.id].positions)
        assert np.allclose(first.levels[level.id].speeds, second.levels[level.id].speeds)


def test_one_odd_item_and_shared_draws():
    cfg = _small()
    show = simulate_show(cfg)
    for level in cfg.levels:
        sim = show.levels[level.id]
        assert 0 <= sim.odd_index < level.count
        drivers = sample_drivers(cfg, level, sim.seed)
        assert drivers["odd_index"] == sim.odd_index
        assert np.allclose(drivers["speeds"], sim.speeds)
        others = np.delete(sim.speeds, sim.odd_index)
        low, high = np.percentile(others, [cfg.speed_low_pct, cfg.speed_high_pct])
        assert low <= sim.speeds[sim.odd_index] <= high


def test_no_overlaps_and_middle_position():
    cfg = _small()
    show = simulate_show(cfg)
    for level in cfg.levels:
        sim = show.levels[level.id]
        assert sim.min_clearance >= -0.35
        span = 0.5 * (1.0 - cfg.position_middle)
        assert cfg.field_x0 + span * (cfg.field_x1 - cfg.field_x0) <= sim.mean_pos[0]
        assert sim.mean_pos[0] <= cfg.field_x1 - span * (cfg.field_x1 - cfg.field_x0)


def test_reseed_loop_logs_then_succeeds():
    def attempt(trial, k):
        if k < 3:
            return False, "forced"
        return True, trial

    payload, used, notes = reseed_loop(7, 20, attempt)
    assert payload == 7 + 3000
    assert used == 3
    assert len(notes) == 3


def test_preview_frame_is_half_size():
    from fc_sat.odd_render import OddRenderer

    cfg = _small(level_ids=[1])
    show = simulate_show(cfg)
    renderer = OddRenderer(cfg, show, preview=True)
    frame = renderer.render(0)
    assert renderer.width == 540 and renderer.height == 960 and renderer.fps == 30
    assert frame.shape == (960, 540, 3)


def test_rendered_text_boxes_stay_apart():
    from fc_sat.odd_layout import field_box, layout_failures
    from fc_sat.odd_render import OddRenderer

    cfg = _small()
    show = simulate_show(cfg)
    renderer = OddRenderer(cfg, show)
    field = field_box(cfg)
    safe = (cfg.safe_x[0], cfg.safe_y[0], cfg.safe_x[1], cfg.safe_y[1])
    indexes = [0, 8, renderer.n_frames // 2, renderer.n_frames - 1]
    reveal = next(segment for segment in show.timeline if segment.kind == "reveal")
    indexes.append(reveal.start_frame)
    for index in indexes:
        renderer.render(min(index, renderer.n_frames - 1))
        boxes = [(name, tuple(float(v) for v in box)) for name, box in renderer.boxes if name != "pill"]
        failures = layout_failures(boxes, field, safe)
        assert failures == [], failures
        for name, box in renderer.boxes:
            if name == "pill":
                assert box[0] >= cfg.field_x0 - 2
                assert box[2] <= cfg.field_x1 + 2
                assert box[1] >= cfg.field_y0 - 2
                assert box[3] <= cfg.field_y1 + 2


def test_reseed_loop_fails_after_20():
    def attempt(trial, k):
        return False, "still bad"

    with pytest.raises(RuntimeError, match="20 attempts"):
        reseed_loop(11, 20, attempt)
