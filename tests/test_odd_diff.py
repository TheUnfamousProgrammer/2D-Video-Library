import math

import numpy as np
import pytest

from fc_sat.color import cvd_distance, lch_to_oklab, oklab_delta, rgb_u8_to_oklab
from fc_sat.odd_config import load_odd_config
from fc_sat.odd_diff import (
    apply_look,
    choose_odd_lab,
    measure_hue,
    measure_pulse,
    measure_size,
    measure_spin,
    paint_synthetic,
)
from fc_sat.odd_layout import math_layout_failures


def test_cvd_floor_and_neutral():
    cfg = load_odd_config("configs/odd_default.yaml")
    base = lch_to_oklab(0.72, 0.10, 0.4)
    chosen = choose_odd_lab(base, cfg.tier)
    assert chosen["distance"] >= 0.20
    assert abs(chosen["lightness_delta"]) >= 0.15
    for kind, value in chosen["cvd"].items():
        assert value >= 0.10, kind
    gray = np.array([0.7, 0.0, 0.0])
    for kind in ("protan", "deutan", "tritan"):
        assert cvd_distance(gray, gray, kind) == pytest.approx(0.0, abs=1e-6)
    brutal = load_odd_config("configs/odd_default.yaml", tier="brutal").tier
    hard = choose_odd_lab(base, brutal)
    assert min(hard["cvd"].values()) >= 0.10


def test_hue_and_size_round_trip():
    cfg = load_odd_config("configs/odd_default.yaml")
    base = lch_to_oklab(cfg.tier.base_l, cfg.tier.base_c, 1.1)
    chosen = choose_odd_lab(base, cfg.tier)
    bg = (26, 15, 11)
    looks = []
    others = []
    for index in range(13):
        odd = index == 0
        look = apply_look(
            "hue",
            cfg.tier,
            is_odd=odd,
            base_lab=base,
            odd_lab=chosen["lab"],
            radius=16,
            t=0.2,
            spin_phase=0.0,
            pulse_phase=0.0,
            exaggerate=0.0,
        )
        look["x"] = 80 + (index % 7) * 50
        look["y"] = 80 + (index // 7) * 50
        looks.append(look)
        if not odd:
            others.append((look["x"], look["y"]))
    frame = paint_synthetic(480, 200, looks, bg)
    frames = [frame] * 10
    odd_xy = [(looks[0]["x"], looks[0]["y"])] * 10
    other_xy = [others[:12]] * 10
    measured = measure_hue(frames, odd_xy, other_xy)
    assert measured >= 0.70 * chosen["distance"]

    size_looks = []
    for index in range(7):
        odd = index == 0
        look = apply_look(
            "size",
            cfg.tier,
            is_odd=odd,
            base_lab=base,
            odd_lab=base,
            radius=16,
            t=0.0,
            spin_phase=0.0,
            pulse_phase=0.0,
            exaggerate=0.0,
        )
        look["x"] = 60 + index * 56
        look["y"] = 80
        size_looks.append(look)
    size_frame = paint_synthetic(480, 180, size_looks, bg)
    bg_lab = rgb_u8_to_oklab(np.array([[11, 15, 26]], dtype=np.uint8))[0]
    measured_size = measure_size(
        size_frame,
        (size_looks[0]["x"], size_looks[0]["y"]),
        [(look["x"], look["y"]) for look in size_looks[1:]],
        window=16 * 1.18 * 1.6,
        bg_lab=bg_lab,
    )
    assert measured_size["ratio"] >= 1.2


def test_spin_and_pulse_round_trip_and_pause():
    cfg = load_odd_config("configs/odd_default.yaml")
    base = lch_to_oklab(0.72, 0.10, 0.2)
    bg = (26, 15, 11)
    fps = 60
    n = 90
    spin_frames = []
    odd_pts = []
    other_pts = []
    phases = np.linspace(0.1, 5.5, 6)
    for frame_index in range(n):
        t = frame_index / fps
        looks = []
        centers = []
        for index in range(6):
            look = apply_look(
                "spin",
                cfg.tier,
                is_odd=index == 0,
                base_lab=base,
                odd_lab=base,
                radius=16,
                t=t,
                spin_phase=float(phases[index]),
                pulse_phase=0.4,
                exaggerate=0.0,
            )
            look["x"] = 40.0
            look["y"] = 40.0 + index * 48.0
            looks.append(look)
            centers.append((look["x"], look["y"]))
        spin_frames.append(paint_synthetic(80, 320, looks, bg))
        odd_pts.append(centers[0])
        other_pts.append(centers[1])
    odd_spin = measure_spin(spin_frames, odd_pts, 16, fps, expect_sign=-1.0, nominal_rev_s=1.2)
    normal_spin = measure_spin(spin_frames, other_pts, 16, fps, expect_sign=1.0, nominal_rev_s=1.2)
    assert odd_spin["sign_fraction"] >= 0.95
    assert normal_spin["sign_fraction"] >= 0.95
    assert odd_spin["omega"] == pytest.approx(odd_spin["nominal"], rel=0.30)
    assert normal_spin["omega"] == pytest.approx(normal_spin["nominal"], rel=0.30)

    # One still frame: the odd tick is just another angle.
    still = spin_frames[20]
    means = []
    for index in range(6):
        y = int(40 + index * 48)
        crop = still[y - 16 : y + 17, 40 - 16 : 40 + 17]
        means.append(float(crop.mean()))
    assert min(means[1:]) - 8 <= means[0] <= max(means[1:]) + 8

    pulse_frames = []
    pts = []
    for frame_index in range(8 * fps):
        t = frame_index / fps
        looks = []
        for index in range(4):
            look = apply_look(
                "pulse",
                cfg.tier,
                is_odd=index == 0,
                base_lab=base,
                odd_lab=base,
                radius=10,
                t=t,
                spin_phase=0.0,
                pulse_phase=float(phases[index]),
                exaggerate=0.0,
            )
            look["x"] = 20.0
            look["y"] = 20.0 + index * 30.0
            looks.append(look)
        pulse_frames.append(paint_synthetic(48, 140, looks, bg))
        pts.append((20.0, 20.0))
    peak = measure_pulse(pulse_frames, pts, fps)
    others = []
    for index in range(1, 4):
        series = [(20.0, 20.0 + index * 30.0) for _ in range(8 * fps)]
        others.append(measure_pulse(pulse_frames, series, fps))
    assert peak == pytest.approx(1.85, abs=0.2)
    assert all(value == pytest.approx(1.50, abs=0.2) for value in others)

    still_pulse = pulse_frames[30]
    brightness = []
    for index in range(4):
        y = int(20 + index * 30)
        brightness.append(float(still_pulse[y - 4 : y + 5, 16:25].mean()))
    assert min(brightness[1:]) - 12 <= brightness[0] <= max(brightness[1:]) + 12


def test_layout_math_does_not_overlap():
    cfg = load_odd_config("configs/odd_default.yaml")
    assert math_layout_failures(cfg) == []


def test_exaggerated_size_is_1_36():
    cfg = load_odd_config("configs/odd_default.yaml")
    base = lch_to_oklab(0.72, 0.10, 0.0)
    look = apply_look(
        "size",
        cfg.tier,
        is_odd=True,
        base_lab=base,
        odd_lab=base,
        radius=16,
        t=0.0,
        spin_phase=0.0,
        pulse_phase=0.0,
        exaggerate=1.0,
    )
    assert look["radius"] == pytest.approx(16 * 1.36, rel=1e-6)
    plain = apply_look(
        "size",
        cfg.tier,
        is_odd=True,
        base_lab=base,
        odd_lab=base,
        radius=16,
        t=0.0,
        spin_phase=0.0,
        pulse_phase=0.0,
        exaggerate=0.0,
    )
    assert plain["radius"] == pytest.approx(16 * 1.18)
    assert oklab_delta(base, base) == 0.0
    assert math.isfinite(look["radius"])
