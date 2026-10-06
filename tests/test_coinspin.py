"""Coinspin: the drawn spins, the counter story, claims, layout, and delivery rules."""

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
    R1,
    ROAD_LENGTH,
    TAU,
    carry_spins,
    count_spins,
    count_spins_inside,
    film_lap_spins,
    gold_pose,
    quarter_frames,
    road_spins,
    sat_lap_spins,
    segment_spins,
    unroll_points,
    upright_frames,
)
from fc_sat.coinspin_post import lint_post
from fc_sat.coinspin_render import CoinRenderer
from fc_sat.coinspin_schedule import count_changes, counter_at, screen_counts
from fc_sat.coinspin_text import lint_script, load_script, top_lines
from fc_sat.coinspin_timeline import build_timeline
from fc_sat.coinspin_verify import continuity_errors, count_sync_errors, layout_errors, physics_errors
from fc_sat.verify import ssim_u8
from make_coinspin import main

UPRIGHTS = [288, 768, 1032, 1224, 1452, 1512, 1572, 1632]


def test_drawn_spins_match_the_story():
    assert film_lap_spins() == 2
    assert road_spins() == 1
    assert carry_spins() == 1
    assert sat_lap_spins() == 4
    assert round(segment_spins(12, 288)) == 1


@pytest.mark.parametrize("ratio", [1, 2, 3, 10])
def test_outside_roll_spins_ratio_plus_one(ratio):
    assert count_spins(ratio) == ratio + 1


def test_inside_roll_spins_ratio_minus_one():
    assert count_spins_inside(3) == 2


def test_faces_come_upright_on_the_designed_frames():
    assert upright_frames() == UPRIGHTS
    for frame in UPRIGHTS:
        turns = gold_pose(frame).theta / TAU
        assert turns == pytest.approx(round(turns), abs=1e-9)
    assert gold_pose(0).theta == 0.0
    assert all(frame % 6 == 0 for frame in UPRIGHTS)


def test_quarter_turns_before_the_sat_land_on_the_grid():
    early = [frame for frame in quarter_frames() if frame < 1344]
    assert early == [102, 150, 198, 480, 570, 660, 960, 984, 1008, 1152, 1176, 1200]


def test_counts_change_only_when_the_face_is_upright():
    assert count_changes() == UPRIGHTS
    assert counter_at(0).terms[0].text == "?"
    assert counter_at(287).terms[0].text == "?"
    assert counter_at(288).terms[0].text == "1"
    assert counter_at(768).terms[0].text == "2"
    assert [term.text for term in counter_at(1100).terms] == ["1", "?", "2"]
    assert [term.text for term in counter_at(1300).terms] == ["1", "1", "2"]
    assert [term.label for term in counter_at(1300).terms[:2]] == ["ROLLING", "TRIP AROUND"]
    assert counter_at(1571).terms[0].text == "2"
    assert counter_at(1572).terms[0].text == "3"
    assert counter_at(1632).terms[0].text == "4"
    assert [term.text for term in counter_at(1700).terms] == ["3", "1", "4"]
    assert counter_at(1823).terms[0].text == "?"
    assert screen_counts() == [1, 2, 3, 4]


def test_counter_never_shows_zero():
    for frame in range(0, 1824, 3):
        counter = counter_at(frame)
        assert all(term.text != "0" for term in counter.terms)


def test_road_is_the_grey_edge_laid_flat():
    right, left = unroll_points(1.0)
    assert right[-1][0] - left[-1][0] == pytest.approx(ROAD_LENGTH)
    assert np.allclose(right[:, 1], CY + R1)
    closed_right, _ = unroll_points(0.0)
    assert np.allclose(np.hypot(closed_right[:, 0] - CX, closed_right[:, 1] - CY), R1)


def test_claims_and_copy():
    book = load_claims()
    assert [claim_id for claim_id, ok, _ in evaluate(book) if not ok] == []
    script = load_script(book=book)
    assert lint_script(script, book) == []
    assert lint_post(book) == []
    assert top_lines(script, 0, "A") == ("ROLL THE GOLD", "COIN AROUND.", "HOW MANY SPINS?")
    assert top_lines(script, 300, "A") == ("HALFWAY THERE...", "ALREADY 1 SPIN!")
    assert top_lines(script, 1700, "A") == ("IT'S 4.", "EACH TRIP AROUND", "ADDS 1 SPIN.")


def test_verify_gates_without_a_file():
    assert physics_errors() == []
    assert count_sync_errors() == []
    assert continuity_errors() == []
    assert layout_errors(step=5) == []


def test_loop_seam():
    renderer = CoinRenderer(width=270, height=480)
    assert ssim_u8(renderer.render(0), renderer.render(1823)) >= 0.99


def test_skia_canvas_returns_rgb():
    canvas = make_canvas(4, 4)
    canvas.fill("#0E1117")
    assert tuple(int(value) for value in canvas.rgb()[0, 0]) == (14, 17, 23)


def test_config_pins_the_coins(tmp_path):
    cfg = load_config()
    moved = tmp_path / "coinspin.yaml"
    moved.write_text(cfg.path.read_text().replace("coin_radius: 120", "coin_radius: 110"))
    with pytest.raises(ConfigError):
        load_config(moved)


def test_mix_is_loud_enough_and_rings_every_upright():
    timeline = build_timeline()
    full, _sfx, lufs, peak = mix(timeline.to_json(), seed=timeline.config.seed, n_frames=timeline.n_frames)
    assert abs(lufs + 14.0) <= 1.0
    assert peak <= -2.0
    assert events_aligned(full, timeline.uprights + timeline.impacts) == []
    assert math.isfinite(lufs)


def test_full_needs_approval():
    with pytest.raises(SystemExit, match="approved"):
        main(["full"])
