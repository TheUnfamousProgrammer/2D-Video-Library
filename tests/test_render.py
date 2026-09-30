import numpy as np

from fc_sat.config import load_config
from fc_sat.render import Renderer, assert_layout_safe, ring_layout_box
from fc_sat.sim import simulate_once
from fc_sat.verify import ssim_u8


def test_layout_and_ring_box():
    cfg = load_config("configs/default.yaml")
    assert assert_layout_safe(cfg) == []
    box = ring_layout_box(cfg)
    assert box == (157.0, 497.0, 923.0, 1263.0)


def test_preview_loop_seam():
    cfg = load_config("configs/default.yaml")
    sim = simulate_once(cfg, speed=1500, gravity=900, seed_offset=0)
    renderer = Renderer(cfg, sim, preview=True)
    first = renderer.render(0)
    last = renderer.render(renderer.n_frames - 1)
    assert first.shape == (960, 540, 3)
    assert ssim_u8(first, last) >= 0.995
    assert np.array_equal(first, last)
