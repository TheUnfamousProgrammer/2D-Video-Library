"""Coinspin: rolling without slipping, the spin schedule, claims, and delivery rules."""

from __future__ import annotations

import math

import numpy as np
import pytest

from fc_sat.circlesquare_draw import make_canvas
from fc_sat.coinspin_audio import events_aligned, mix
from fc_sat.coinspin_claims import evaluate, load_claims
from fc_sat.coinspin_config import ConfigError, load_config
from fc_sat.coinspin_math import (
    CX,
    CY,
    DOUBLING_RATIOS,
    LAPS,
    TAU,
    count_spins,
    count_spins_inside,
    format_hms,
    rolling_pose,
    scene_at,
    sidereal_day,
    theta_at,
    trip_bonus,
)
from fc_sat.coinspin_post import lint_post
from fc_sat.coinspin_render import CoinRenderer
from fc_sat.coinspin_schedule import counter_at, screen_counts, spin_frames
from fc_sat.coinspin_text import lint_script, load_script, sub_text, top_lines
from fc_sat.coinspin_timeline import build_timeline
from fc_sat.coinspin_verify import contact_errors, layout_errors, spin_sync_errors
from fc_sat.verify import ssim_u8
from make_coinspin import main


@pytest.mark.parametrize("ratio", [1, 2, 3, 4, 10, 96])
def test_outside_roll_spins_ratio_plus_one(ratio):
    assert count_spins(ratio) == ratio + 1


@pytest.mark.parametrize("ratio", [2, 3, 5])
def test_inside_roll_spins_ratio_minus_one(ratio):
    assert count_spins_inside(ratio) == ratio - 1


def test_trip_bonus_is_one_at_every_doubling():
    assert trip_bonus() == 1
    assert [count_spins(ratio) - ratio for ratio in DOUBLING_RATIOS[:6]] == [1] * 6


def test_rolling_coin_touches_the_rim():
    for ratio in (1.0, 2.0, 3.0, 6.0):
        for phi in np.linspace(0.0, TAU, 9):
            pose = rolling_pose(ratio, float(phi), 0.0)
            gap = float(np.hypot(*(pose["rolling_center"] - pose["fixed_center"])))
            assert gap == pytest.approx(pose["fixed_px"] + pose["rolling_px"], abs=1e-9)


def test_spins_land_on_the_beat_grid():
    frames = spin_frames()
    assert frames == [96, 192, 264, 312, 360, 480, 576, 672, 768]
    assert all(frame % 6 == 0 for frame in frames)
    timeline = build_timeline()
    assert set(frames) <= set(timeline.bells)
    assert timeline.max_snap_error == 0.0


def test_arrow_points_up_on_each_spin_frame():
    for frame in spin_frames():
        if frame in (192, 360, 768):
            lap = next(lap for lap in LAPS if lap.end == frame)
            angle = (lap.ratio + 1.0) * TAU
        else:
            angle = theta_at(frame)
        assert math.cos(angle) == pytest.approx(1.0, abs=1e-9)


def test_counter_story():
    assert counter_at(0).value == 0
    assert counter_at(96).value == 1
    hold = counter_at(192)
    assert (hold.mode, hold.value, hold.gold) == ("hold", 2, True)
    assert counter_at(671).value == 2 and counter_at(672).value == 3
    assert counter_at(767).value == 3
    drop = counter_at(768)
    assert (drop.value, drop.gold) == (4, True)
    split = counter_at(900)
    assert (split.mode, split.value, split.plus) == ("formula", 3, 1)
    assert counter_at(1343).value == DOUBLING_RATIOS[-1]
    year = counter_at(1488)
    assert (year.mode, year.days, year.spins, year.gold) == ("sky", 365, 366, True)
    after = counter_at(1700)
    assert (after.value, after.plus) == (365, 1)
    assert counter_at(1823).value == 0


def test_sidereal_day_is_23h_56m_4s():
    seconds = sidereal_day(365.24219)
    assert seconds == pytest.approx(86164.0905, abs=1e-3)
    assert format_hms(seconds) == "23H 56M 4S"


def test_claims_pass_and_cover_every_counter_value():
    results = evaluate(load_claims())
    assert [claim_id for claim_id, ok, _ in results if not ok] == []
    assert load_claims().get("screen_counts").value == screen_counts()


def test_script_and_post_lint_clean():
    book = load_claims()
    script = load_script(book=book)
    assert lint_script(script, book) == []
    assert lint_post(book) == []
    assert top_lines(script, 400, "A") == ("1982 SAT:", "HOW MANY SPINS?")
    assert top_lines(script, 800, "A") == ("4. NOT EVEN", "AN OPTION.")
    assert sub_text(script, 1500) == "1 SPIN = 23H 56M 4S"
    assert sub_text(script, 1000) == "BIG COIN 12X WIDER"


def test_verify_gates_without_a_file():
    assert spin_sync_errors() == []
    assert contact_errors() == []
    assert layout_errors(step=7) == []


def test_loop_seam_and_frame_zero():
    renderer = CoinRenderer(width=270, height=480)
    assert ssim_u8(renderer.render(0), renderer.render(1823)) >= 0.99
    scene = scene_at(0)
    assert scene.rolling_center == pytest.approx((CX, CY - 2.0 * scene.rolling_px))


def test_skia_canvas_returns_rgb():
    canvas = make_canvas(4, 4)
    canvas.fill("#0E1117")
    assert tuple(int(value) for value in canvas.rgb()[0, 0]) == (14, 17, 23)


def test_config_pins_the_stage(tmp_path):
    cfg = load_config()
    text = cfg.path.read_text().replace("stage_radius: 400", "stage_radius: 380")
    moved = tmp_path / "coinspin.yaml"
    moved.write_text(text)
    with pytest.raises(ConfigError):
        load_config(moved)


def test_mix_is_loud_enough_and_hits_every_spin():
    timeline = build_timeline()
    full, _sfx, lufs, peak = mix(timeline.to_json(), seed=timeline.config.seed, n_frames=timeline.n_frames)
    assert abs(lufs + 14.0) <= 1.0
    assert peak <= -2.0
    assert events_aligned(full, timeline.spins + timeline.impacts) == []


def test_full_needs_approval():
    with pytest.raises(SystemExit, match="approved"):
        main(["full"])
