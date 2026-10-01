import numpy as np

from fc_sat.color import oklab_delta
from fc_sat.odd_config import load_odd_config
from fc_sat.odd_diff import (
    apply_look,
    choose_odd_lab,
    measure_detail,
    measure_hue,
    measure_tilt,
    paint_synthetic,
)
from fc_sat.odd_layout import math_layout_failures
from fc_sat.odd_sim import grid_geometry


def _bg():
    return (31, 23, 20)  # #14171F in BGR


def test_hue_round_trip_and_cvd():
    cfg = load_odd_config("configs/odd_default.yaml")
    base = np.array([0.72, 0.10, 0.0])
    odd = choose_odd_lab(base, cfg.tier)
    assert odd["distance"] >= 0.25
    assert abs(odd["lightness_delta"]) >= 0.15
    assert min(odd["cvd"].values()) >= 0.12
    looks = []
    centers = []
    for index in range(8):
        looks.append(
            apply_look("hue", cfg.tier, is_odd=index == 3, base_lab=base, odd_lab=odd["lab"], size=80)
        )
        centers.append((80 + index * 90, 120))
    frame = paint_synthetic(800, 300, looks, centers, _bg())
    measured = measure_hue(frame, centers[3], [centers[i] for i in range(8) if i != 3])
    assert measured >= 0.70 * odd["distance"]
    colors = {look["bgr"] for look in looks}
    assert len(colors) <= 2


def test_subtle_hue_keeps_its_cvd_floor():
    cfg = load_odd_config("configs/odd_default.yaml", tier="hard")
    base = np.array([0.72, 0.0, 0.10])
    odd = choose_odd_lab(base, cfg.tier, subtle=True)
    assert odd["distance"] >= 0.10
    assert min(odd["cvd"].values()) >= 0.06


def test_tilt_round_trip():
    cfg = load_odd_config("configs/odd_default.yaml")
    base = np.array([0.72, 0.08, 0.04])
    level = cfg.levels[1]
    centers, cells, _cell = grid_geometry(cfg, level)
    looks = [
        apply_look("tilt", cfg.tier, is_odd=index == 6, base_lab=base, odd_lab=base, size=level.size)
        for index in range(level.count)
    ]
    assert looks[6]["angle"] == 22
    assert all(look["angle"] == 0 for index, look in enumerate(looks) if index != 6)
    frame = paint_synthetic(cfg.width, cfg.height, looks, [tuple(p) for p in centers], _bg(), corner_frac=cfg.corner_frac)
    stats = measure_tilt(frame, cells, 6, _bg())
    assert stats["delta"] >= 0.60 * cfg.tier.tilt_degrees


def test_detail_round_trip():
    cfg = load_odd_config("configs/odd_default.yaml")
    base = np.array([0.72, 0.05, -0.04])
    centers = [(120 + (index % 4) * 140, 160 + (index // 4) * 140) for index in range(8)]
    looks = [
        apply_look("detail", cfg.tier, is_odd=index == 2, base_lab=base, odd_lab=base, size=80)
        for index in range(8)
    ]
    assert looks[2]["dot"] is None
    assert looks[0]["dot"]["radius"] == pytest_dot(cfg, 80)
    frame = paint_synthetic(700, 500, looks, centers, _bg())
    stats = measure_detail(frame, centers[2], [centers[i] for i in range(8) if i != 2], looks[0]["dot"]["radius"])
    assert stats["ratio"] < 0.20


def pytest_dot(cfg, size):
    return cfg.tier.dot_fraction * (size / 2.0)


def test_layout_math_is_clear():
    cfg = load_odd_config("configs/odd_default.yaml")
    assert math_layout_failures(cfg) == []


def test_renderer_does_not_use_bloom():
    from pathlib import Path

    source = Path("fc_sat/odd_render.py").read_text(encoding="utf-8")
    assert "apply_bloom" not in source
    assert "vignette_bgr" not in source
    assert "GaussianBlur" not in source
    assert "raster_text_flat" in source
