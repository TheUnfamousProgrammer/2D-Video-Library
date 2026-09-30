from pathlib import Path

import numpy as np

from fc_sat.morph_config import load_morph_config, morph_from_public
from fc_sat.morph_render import nearest_block
from fc_sat.verify import cell_oklab_error, detect_mode
from fc_sat.visual import vignette_bgr


def test_detect_mode_keeps_bounce_and_hop():
    assert detect_mode(Path("configs/default.yaml")) == "bounce"
    assert detect_mode(Path("configs/hop_default.yaml")) == "hop"
    assert detect_mode(Path("configs/morph_default.yaml")) == "morph"


def test_cell_average_of_a_nearest_upscale_matches():
    cfg = load_morph_config("configs/morph_default.yaml")
    grid = np.random.default_rng(4).integers(0, 255, size=(6, 8, 3), dtype=np.uint8)
    frame = vignette_bgr(cfg.width, cfg.height)
    block = nearest_block(grid, cfg.cell)
    y0, x0 = cfg.origin_y, cfg.origin_x
    frame[y0 : y0 + block.shape[0], x0 : x0 + block.shape[1]] = block
    error = cell_oklab_error(frame, grid, (x0, y0), cfg.cell, 8, 6)
    assert error < 1e-6
    rebuilt = morph_from_public(cfg.to_public_dict(), cfg.hooks)
    assert rebuilt.cols == cfg.cols
    assert rebuilt.recolor_strength == cfg.recolor_strength
