"""Circlesquare: the Fourier square, the schedule, and the delivery rules."""

from __future__ import annotations

import math

import numpy as np
import pytest

from fc_sat.audio import SR, true_peak_db
from fc_sat.beatkit.claims import lint_text
from fc_sat.beatkit.delivery import DELIVERIES, require_delivery
from fc_sat.circlesquare_audio import events_aligned, mix, onset_sample
from fc_sat.circlesquare_claims import evaluate, load_claims
from fc_sat.circlesquare_math import (
    A,
    CX,
    CY,
    DRAW_CAP,
    R1,
    ZMAX,
    asymptotic_gap,
    camera_anchor,
    capped_circles,
    center_screen,
    coefficient,
    corner_screen,
    distance_to_boundary,
    first_k_within,
    fourier_compare,
    full_curve,
    gap,
    harmonic,
    project_points,
    use_exact_square,
    zoom_at,
)
from fc_sat.circlesquare_post import description, hashtags, lint_post, titles
from fc_sat.circlesquare_render import CircleRenderer, inside_safe, overlaps
from fc_sat.circlesquare_schedule import (
    add_frames,
    add_weight,
    build_schedule,
    counter_value,
    double_frames,
    doubling_blend,
)
from fc_sat.circlesquare_text import lint_script, load_script, top_lines
from fc_sat.circlesquare_timeline import build_timeline
from fc_sat.circlesquare_verify import geometry_errors, layout_errors, style_law_errors
from fc_sat.encode import encode_command
from fc_sat.polycircle_format import format_count, format_gap
from fc_sat.verify import ssim_u8
from make_circlesquare import main

GAP_TABLE = {
    1: 64.41,
    2: 33.78,
    3: 22.76,
    5: 13.73,
    9: 7.648,
    25: 2.756,
    57: 1.209,
    93: 0.7408,
    100: 0.689,
    138: 0.4993,
    140: 0.4921,
    150: 0.4593,
    200: 0.3445,
    400: 0.1722,
}


def test_fourier_matches_the_square_boundary():
    worst, stray = fourier_compare(2**20, 40)
    assert worst < 1e-6
    assert stray < 1e-8
    assert [harmonic(index) for index in range(1, 6)] == [1, -3, 5, -7, 9]
    assert all(harmonic(index) % 4 == 1 for index in range(1, 40))
    assert coefficient(1) == pytest.approx(R1 * np.exp(1j * np.pi / 4))


def test_gap_table_and_the_half_pixel_threshold():
    for circles, expected in GAP_TABLE.items():
        assert gap(circles) == pytest.approx(expected, rel=1.5e-3)
    assert gap(137) > 0.5
    assert gap(138) <= 0.5
    assert first_k_within(0.5) == 138
    for circles in (10, 25, 93, 138, 400, 2976):
        assert gap(circles) == pytest.approx(asymptotic_gap(circles), rel=0.002)
    # The brief's 0.01 px shortcut is true from about K = 11904, not at 2976.
    assert gap(2976) == pytest.approx(0.02315, rel=0.01)
    assert gap(11904) < 0.01


def test_add_and_doubling_schedule():
    schedule = build_schedule()
    adds = add_frames()
    doubles = double_frames()
    assert len(adds) == 92
    assert len(doubles) == 16
    assert adds[:8] == [24 * k for k in range(1, 9)]
    assert adds[8:24] == [192 + 12 * k for k in range(1, 17)]
    assert doubles[:8] == [960 + 24 * j for j in range(8)]
    assert doubles[8:] == [1152 + 24 * j for j in range(8)]
    assert schedule.k(0) == 1
    assert schedule.k(192) == 9
    assert schedule.k(384) == 25
    assert schedule.k(576) == 57
    assert schedule.k(672) == 89
    assert schedule.k(768) == 93
    assert schedule.k(954) == 93
    assert schedule.k(960) == 186
    assert schedule.k(1128) == 23808
    assert schedule.k(1152) == 47616
    assert schedule.k(1320) == 6094848
    assert schedule.k(1805) == 6094848
    assert schedule.k(1806) == 1
    assert schedule.k(1823) == 1
    for item in schedule.adds:
        assert counter_value(schedule, item.frame) == counter_value(schedule, item.frame - 1) + 1


def test_weight_ramps_do_not_jump():
    schedule = build_schedule()
    for item in schedule.adds:
        weights = [add_weight(schedule, frame, item.k_after) for frame in range(item.frame - 1, item.frame + 10)]
        assert weights[0] == 0.0
        assert weights[1] == 0.0
        assert weights[-1] == 1.0
        assert all(right + 1e-12 >= left for left, right in zip(weights, weights[1:]))
        assert max(right - left for left, right in zip(weights, weights[1:])) < 0.95
    first = schedule.doubles[0]
    assert doubling_blend(schedule, 954) is None
    assert doubling_blend(schedule, 960)[1] == pytest.approx(1.0)
    later = schedule.doubles[1]
    assert doubling_blend(schedule, later.frame - 8)[1] == pytest.approx(0.0)
    assert doubling_blend(schedule, later.frame)[1] == pytest.approx(1.0)


def test_camera_identity_and_corner_lock():
    point = np.array([[120.0, 400.0]])
    assert np.allclose(project_points(point, 1.0), point)
    locked = project_points(corner_screen().reshape(1, 2), ZMAX)
    assert np.allclose(locked[0], center_screen())
    assert np.allclose(center_screen(), [CX, CY])
    assert zoom_at(875) == 1.0
    assert zoom_at(948) == pytest.approx(ZMAX)
    assert zoom_at(954) == pytest.approx(ZMAX)
    assert zoom_at(1152) == pytest.approx(ZMAX)
    assert zoom_at(1280) == pytest.approx(1.0)
    # The anchor walks from the center toward the corner and stops there at Zmax.
    assert np.allclose(camera_anchor(1.0), center_screen())
    assert np.allclose(camera_anchor(ZMAX), corner_screen())


def test_render_shortcuts():
    assert use_exact_square(5952, 1.0)
    assert not use_exact_square(5952, 36.0)
    assert not use_exact_square(2976, 1.0)
    assert capped_circles(47616) == DRAW_CAP
    renderer = CircleRenderer()
    capped = renderer._shape(1152, 36.0)
    assert capped is not None
    assert np.allclose(capped, full_curve(DRAW_CAP, len(capped)))
    dense = full_curve(138, 65536)
    assert float(np.max(distance_to_boundary(dense))) <= 0.5 + 0.05


def test_loop_frame_matches_the_opening():
    renderer = CircleRenderer()
    seam = ssim_u8(renderer.render(0), renderer.render(1823))
    assert seam >= 0.99


def test_text_stays_in_the_safe_zone():
    assert layout_errors() == []
    renderer = CircleRenderer()
    assert renderer.bbox_fraction(0) >= 0.18
    assert renderer.design_stroke(0) >= 10
    assert renderer.design_text_size(0) >= 72
    boxes = [box for box in renderer.plan(0) if box.alpha > 0.2]
    assert len(boxes) <= 5
    for box in boxes:
        assert inside_safe(box, 1.0)
    for left, right in zip(boxes, boxes[1:]):
        assert not overlaps(left, right)
    script = load_script(book=load_claims())
    for lines in script.hooks.values():
        assert all(len(line) <= 16 for line in lines)


def test_claims_and_script_lint():
    book = load_claims()
    results = evaluate(book)
    assert results
    assert all(ok for _claim, ok, _detail in results)
    assert lint_script(load_script(book=book), book) == []
    assert lint_text("it takes 99 circles", list(book.claims), book, "probe")
    assert "e" not in format_gap(1.13e-5).lower()
    assert format_gap(1.13e-5) == "0.000011"
    assert format_count(6094848) == "6,094,848"
    assert format_gap(64.41) == "64"
    assert format_gap(0.7408) == "0.74"


def test_postkit_and_full_refusal():
    book = load_claims()
    assert lint_post(book) == []
    first = titles(book)[0][0]
    assert len(first) <= 60
    line = description(book).splitlines()[0]
    assert len(line) <= 100
    assert line.startswith("How many circles")
    assert hashtags().split()[0] == "#shorts"
    with pytest.raises(SystemExit, match="approved"):
        main(["full"])


def test_delivery_modes_and_the_encode_flags():
    assert (DELIVERIES["full"].width, DELIVERIES["full"].height, DELIVERIES["full"].fps, DELIVERIES["full"].frames) == (
        1080,
        1920,
        60,
        1824,
    )
    assert (DELIVERIES["preview"].width, DELIVERIES["preview"].height, DELIVERIES["preview"].fps) == (540, 960, 30)
    assert (DELIVERIES["hooks"].width, DELIVERIES["hooks"].height, DELIVERIES["hooks"].fps, DELIVERIES["hooks"].frames) == (
        540,
        960,
        30,
        105,
    )
    require_delivery("full", 1080, 1920, 60, 1824)
    with pytest.raises(SystemExit):
        require_delivery("full", 1080, 1920, 30, 1824)
    command = encode_command("ffmpeg", width=1080, height=1920, fps=60, output=__import__("pathlib").Path("out/x.mp4"), audio_path=None)
    assert "-framerate" in command
    assert "-fps_mode" in command
    assert command[command.index("-fps_mode") + 1] == "cfr"
    assert "-r" in command
    assert style_law_errors() == []


def test_hook_variants_share_the_clock():
    book = load_claims()
    script = load_script(book=book)
    for hook in ("A", "B", "D"):
        assert top_lines(script, 0, hook) == script.hooks[hook]
        assert top_lines(script, 1823, hook) == script.hooks[hook]
        assert top_lines(script, 768, hook) == top_lines(script, 768, "A")
    schedule = build_schedule()
    assert schedule.adds[0].frame == 24


@pytest.fixture(scope="module")
def music():
    timeline = build_timeline()
    full, _sfx, lufs, peak = mix(timeline.to_json(), seed=timeline.config.seed, n_frames=timeline.n_frames)
    return timeline, full, lufs, peak


def test_audio_length_loudness_and_the_kick(music):
    timeline, full, lufs, peak = music
    assert full.shape == (1824 * 800, 2)
    assert SR == 48000
    assert float(np.max(np.abs(full))) <= 1.0
    assert abs(lufs + 14) <= 1
    assert true_peak_db(full) <= -1
    assert onset_sample(full) <= 48
    tail = full[-18 * 800 :]
    assert float(np.max(np.abs(tail))) == 0.0
    missing = events_aligned(full, timeline.kicks[:8] + timeline.impacts + timeline.bells)
    assert missing == []
