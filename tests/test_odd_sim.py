import numpy as np

from fc_sat.odd_config import load_odd_config
from fc_sat.odd_sim import reseed_loop, simulate_show


def test_sim_is_deterministic():
    cfg = load_odd_config("configs/odd_default.yaml")
    first = simulate_show(cfg)
    second = simulate_show(cfg)
    for level in cfg.levels:
        assert first.levels[level.id].odd_index == second.levels[level.id].odd_index
        assert np.array_equal(first.levels[level.id].centers, second.levels[level.id].centers)


def test_one_odd_item_and_grid_cells():
    cfg = load_odd_config("configs/odd_default.yaml")
    show = simulate_show(cfg)
    previous = None
    for level in cfg.levels:
        sim = show.levels[level.id]
        assert 0 <= sim.odd_index < level.count
        assert level.grid in (5, 6, 7)
        assert len(sim.centers) == level.grid * level.grid
        half = level.size / 2.0
        for (cx, cy), box in zip(sim.centers, sim.cells):
            assert box[0] < cx < box[2]
            assert box[1] < cy < box[3]
            assert cx - half >= cfg.field_x0
            assert cx + half <= cfg.field_x1
            assert abs(cx - (box[0] + box[2]) / 2) < 1e-6
            assert abs(cy - (box[1] + box[3]) / 2) < 1e-6
        # Neighboring centers stay at least a cell apart, so the discs cannot overlap.
        deltas = np.diff(sim.centers[:: level.grid], axis=0)
        if len(deltas):
            assert np.min(np.linalg.norm(deltas, axis=1)) > level.size
        cell = (sim.row, sim.col)
        assert cell != previous
        previous = cell


def test_reseed_loop_fails_after_20():
    import pytest

    def attempt(trial, k):
        return False, "still bad"

    with pytest.raises(RuntimeError, match="20 attempts"):
        reseed_loop(11, 20, attempt)


def test_reseed_loop_logs_then_succeeds():
    def attempt(trial, k):
        if k < 3:
            return False, "forced"
        return True, trial

    payload, used, notes = reseed_loop(7, 20, attempt)
    assert payload == 7 + 3000
    assert used == 3
    assert len(notes) == 3
