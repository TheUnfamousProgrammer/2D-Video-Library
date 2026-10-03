import math

import numpy as np
import pytest

from fc_sat.color import oklab_delta
from fc_sat.odd_config import (
    DETAIL_OFFSET,
    HUE_DISTANCE,
    HUE_LIGHTNESS,
    SKIPPED_MEASURE,
    TILT_DEGREES,
    cvd_floor,
    load_odd_config,
)
from fc_sat.odd_diff import (
    apply_look,
    choose_odd_lab,
    measure_detail,
    measure_hue,
    measure_tilt,
    paint_synthetic,
)
from fc_sat.odd_layout import math_layout_failures
from fc_sat.odd_sim import grid_geometry, simulate_show


def _bg():
    return (31, 23, 20)  # #14171F in BGR


def test_hue_round_trip_and_cvd():
    cfg = load_odd_config("configs/odd_default.yaml")
    base = np.array([0.72, 0.10, 0.0])
    nominal = HUE_DISTANCE[cfg.tier.hue_rung]
    odd = choose_odd_lab(
        base,
        min_distance=nominal,
        min_lightness=HUE_LIGHTNESS,
        min_cvd=cvd_floor(nominal),
        chroma=cfg.tier.base_c,
    )
    assert odd["distance"] >= nominal
    assert abs(odd["lightness_delta"]) >= HUE_LIGHTNESS
    assert min(odd["cvd"].values()) >= cvd_floor(nominal)
    looks = []
    centers = []
    for index in range(8):
        looks.append(apply_look("hue", cfg.tier, is_odd=index == 3, base_lab=base, odd_lab=odd["lab"], size=80))
        centers.append((80 + index * 90, 120))
    frame = paint_synthetic(800, 300, looks, centers, _bg())
    measured = measure_hue(frame, centers[3], [centers[i] for i in range(8) if i != 3])
    assert measured >= 0.70 * nominal
    assert oklab_delta(base, odd["lab"]) == pytest.approx(odd["distance"])
    colors = {look["bgr"] for look in looks}
    assert len(colors) <= 2


def test_tilt_round_trip():
    cfg = load_odd_config("configs/odd_default.yaml")
    base = np.array([0.72, 0.08, 0.04])
    level = cfg.levels[1]
    centers, cells, _cell = grid_geometry(cfg, level)
    nominal = float(TILT_DEGREES[cfg.tier.tilt_rung])
    looks = [
        apply_look(
            "tilt",
            cfg.tier,
            is_odd=index == 6,
            base_lab=base,
            odd_lab=base,
            size=level.size,
            params={"tilt_degrees": nominal},
        )
        for index in range(level.count)
    ]
    assert looks[6]["angle"] == nominal
    assert all(look["angle"] == 0 for index, look in enumerate(looks) if index != 6)
    frame = paint_synthetic(cfg.width, cfg.height, looks, [tuple(p) for p in centers], _bg(), corner_frac=cfg.corner_frac)
    stats = measure_tilt(frame, cells, 6, _bg())
    assert stats["delta"] >= 0.60 * nominal


def test_tilt_rung_4_still_measures():
    """Rung 4 is the smallest tilt that must still measure. Rung 5 is skipped."""
    from fc_sat.odd_config import TILT_DEGREES

    cfg = load_odd_config("configs/odd_default.yaml")
    base = np.array([0.72, 0.08, 0.04])
    level = cfg.levels[1]
    centers, cells, _cell = grid_geometry(cfg, level)
    nominal = float(TILT_DEGREES[4])
    looks = [
        apply_look(
            "tilt",
            cfg.tier,
            is_odd=index == 6,
            base_lab=base,
            odd_lab=base,
            size=level.size,
            params={"tilt_degrees": nominal},
        )
        for index in range(level.count)
    ]
    frame = paint_synthetic(cfg.width, cfg.height, looks, [tuple(p) for p in centers], _bg(), corner_frac=cfg.corner_frac)
    stats = measure_tilt(frame, cells, 6, _bg())
    assert stats["delta"] >= 0.60 * nominal


def test_moved_detail_directions_and_measure():
    cfg = load_odd_config("configs/odd_default.yaml")
    show = simulate_show(cfg)
    sim = show.levels[3]
    again = simulate_show(cfg).levels[3]
    assert sim.diff_params["direction"] in range(8)
    assert sim.diff_params["direction"] == again.diff_params["direction"]
    assert sim.diff_params["offset"] == pytest.approx(DETAIL_OFFSET[cfg.tier.detail_rung])
    assert sim.diff_params["detail_mode"] == "moved"
    size = sim.level.size
    radius = size / 2.0
    base = sim.base_lab
    for direction in range(8):
        look = apply_look(
            "detail",
            cfg.tier,
            is_odd=True,
            base_lab=base,
            odd_lab=base,
            size=size,
            params={"offset": 0.22, "direction": direction, "detail_mode": "moved", "dot_fraction": cfg.dot_fraction},
        )
        ox = look["dot"]["ox"]
        oy = look["dot"]["oy"]
        assert math.hypot(ox, oy) == pytest.approx(0.22 * radius)
        angle = math.atan2(oy, ox) % (2.0 * math.pi)
        expect = (direction * math.pi / 4.0) % (2.0 * math.pi)
        assert min(abs(angle - expect), abs(angle - expect + 2.0 * math.pi), abs(angle - expect - 2.0 * math.pi)) < 1e-6
    centered = apply_look(
        "detail",
        cfg.tier,
        is_odd=False,
        base_lab=base,
        odd_lab=base,
        size=size,
        params=sim.diff_params,
    )
    assert centered["dot"]["ox"] == 0.0 and centered["dot"]["oy"] == 0.0
    assert centered["dot"]["radius"] == pytest.approx(cfg.dot_fraction * radius)
    centers, _cells, _cell = grid_geometry(cfg, sim.level)
    looks = [
        apply_look(
            "detail",
            cfg.tier,
            is_odd=index == sim.odd_index,
            base_lab=base,
            odd_lab=base,
            size=size,
            params=sim.diff_params,
        )
        for index in range(sim.level.count)
    ]
    frame = paint_synthetic(cfg.width, cfg.height, looks, [tuple(p) for p in centers], _bg())
    others = [tuple(centers[index]) for index in range(sim.level.count) if index != sim.odd_index]
    stats = measure_detail(frame, tuple(centers[sim.odd_index]), others, radius)
    nominal = float(sim.diff_params["offset"])
    assert stats["odd"] >= 0.60 * nominal
    assert stats["max_other"] < 0.25 * nominal


def test_rung_five_measurement_is_skipped():
    from fc_sat.odd_verify import score_frame

    cfg = load_odd_config("configs/odd_default.yaml")

    class _Level:
        difference = "hue"
        size = 90

    class _Sim:
        level = _Level()
        odd_index = 0
        centers = [(100.0, 100.0)]
        cells = []
        diff_params = {"rung": 5, "nominal": 0.06}

    ok, detail = score_frame(np.zeros((20, 20, 3), dtype=np.uint8), cfg, _Sim(), 1.0)
    assert ok
    assert detail == SKIPPED_MEASURE


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
