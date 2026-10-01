from pathlib import Path

import numpy as np

from fc_sat.odd_config import load_odd_config
from fc_sat.odd_diff import apply_look
from fc_sat.odd_render import OddRenderer
from fc_sat.odd_sim import simulate_show
from fc_sat.odd_verify import sequence_checks
from fc_sat.verify import detect_mode


def test_detect_mode_odd_and_leaves_bounce():
    assert detect_mode(Path("configs/odd_default.yaml")) == "odd"
    assert detect_mode(Path("configs/default.yaml")) == "bounce"


def test_reveal_starts_when_the_timer_ends():
    cfg = load_odd_config("configs/odd_default.yaml")
    checks = {item.name: item for item in sequence_checks(cfg)}
    assert checks["reveal at timer 0"].ok
    assert checks["label sequence"].detail == "LEVEL 1 LEVEL 2 LEVEL 3"


def test_rendered_text_stays_under_four_and_off_the_field():
    from fc_sat.odd_layout import field_box, layout_failures

    cfg = load_odd_config("configs/odd_default.yaml")
    show = simulate_show(cfg)
    renderer = OddRenderer(cfg, show)
    field = field_box(cfg)
    safe = (cfg.safe_x[0], cfg.safe_y[0], cfg.safe_x[1], cfg.safe_y[1])
    indexes = [0, renderer.n_frames // 2, renderer.n_frames - 1]
    reveal = next(segment for segment in show.timeline if segment.kind == "reveal")
    indexes.append(reveal.start_frame)
    for index in indexes:
        frame = renderer.render(index)
        assert len(renderer.boxes) <= 4
        assert layout_failures(renderer.boxes, field, safe) == []
        assert frame.shape == (1920, 1080, 3)
    colors = set()
    play = next(segment for segment in show.timeline if segment.kind == "play" and segment.level_id == 1)
    renderer.render_time(play.start_s + cfg.pop_seconds + 0.1)
    sim = show.levels[1]
    for index in range(sim.level.count):
        look = apply_look(
            "hue",
            cfg.tier,
            is_odd=index == sim.odd_index,
            base_lab=sim.base_lab,
            odd_lab=sim.odd_lab,
            size=sim.level.size,
        )
        colors.add(look["bgr"])
    assert len(colors) <= 2
    # A pixel just outside a disc is the flat background, not a halo.
    cx, cy = sim.centers[0]
    x = int(round(cx + sim.level.size / 2.0 + 6))
    y = int(round(cy))
    frame = renderer.render_time(play.start_s + 0.4)
    bg = np.array([31, 23, 20], dtype=np.int16)
    assert int(np.max(np.abs(frame[y, x].astype(np.int16) - bg))) <= 3
