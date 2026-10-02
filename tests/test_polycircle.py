"""Fast checks for the polycircle short."""

from __future__ import annotations

import pytest

from fc_sat.polycircle_config import lightness_spread, load_config
from fc_sat.polycircle_format import format_count, format_gap
from make_polycircle import main


def test_full_render_is_refused_without_approval():
    with pytest.raises(SystemExit, match="approved"):
        main(["full"])


def test_bar_colors_share_a_lightness():
    cfg = load_config()
    assert len(cfg.bar_colors) == 8
    assert lightness_spread(cfg.bar_colors) <= 0.03
    assert cfg.bpm == 150
    assert cfg.frames == 1824
    assert cfg.radius == 370


def test_big_numbers_and_gaps_have_no_float_noise():
    assert format_count(6_291_456) == "6,291,456"
    assert format_count(24576) == "24,576"
    assert format_count(4) == "4"
    assert format_gap(0.4905867426127042) == "0.49"
    assert format_gap(7.131694814809082) == "7.1"
    assert format_gap(1.783043044636785) == "1.8"
    assert format_gap(0.4457682202174773) == "0.45"
    assert format_gap(0.02786065944888616) == "0.028"
    assert format_gap(0.006965166683015056) == "0.0070"
    assert format_gap(4.613087689619988e-11) == "0.000000000046"
    assert "e" not in format_gap(4.613087689619988e-11)


def test_gap_formula_and_the_61_threshold():
    from fc_sat.polycircle_geometry import edge_length, first_n_within, gap

    assert first_n_within(370, 0.5) == 61
    assert gap(60, 370) == pytest.approx(0.5071, abs=5e-5)
    assert gap(61, 370) == pytest.approx(0.4906, abs=5e-5)
    assert gap(96, 370) == pytest.approx(0.1981, abs=5e-5)
    assert gap(96, 370) * 36 == pytest.approx(7.13, abs=5e-3)
    assert edge_length(96, 370) == pytest.approx(24.21, abs=5e-3)
    assert edge_length(96, 370) * 36 == pytest.approx(871.6, abs=0.1)
    expected = [7.13, 1.783, 0.4458, 0.1114, 0.02786, 0.00697, 0.00174, 0.00044, 0.00011]
    shown = ["7.1", "1.8", "0.45", "0.11", "0.028", "0.0070", "0.0017", "0.00044", "0.00011"]
    for k, (want, label) in enumerate(zip(expected, shown)):
        value = gap(96 * (2**k), 370) * 36
        assert value == pytest.approx(want, rel=0.02)
        assert format_gap(value) == label
    assert 96 * (2**16) == 6_291_456
    assert gap(6_291_456, 370) == pytest.approx(4.6e-11, rel=0.02)


def test_add_and_doubling_schedule():
    from fc_sat.polycircle_schedule import build_schedule

    schedule = build_schedule()
    assert schedule.max_snap_error <= 0.5
    assert len(schedule.adds) == 92
    assert len(schedule.doubles) == 16
    assert schedule.n(0) == 4
    assert schedule.n(192) == 12
    assert schedule.n(384) == 28
    assert schedule.n(576) == 60
    assert schedule.n(672) == 92
    assert schedule.n(768) == 96
    assert schedule.n(578) == 60
    assert schedule.n(579) == 61
    assert schedule.n(960) == 192
    assert schedule.n(1128) == 24_576
    assert schedule.n(1320) == 6_291_456
    assert [item.frame for item in schedule.doubles[:8]] == [960 + 24 * j for j in range(8)]
    assert [item.frame for item in schedule.doubles[8:]] == [1152 + 24 * j for j in range(8)]


def test_morph_stays_convex_and_monotone():
    import numpy as np

    from fc_sat.polycircle_geometry import (
        add_morph,
        double_morph,
        is_convex,
        monotone_angles,
        points_from,
    )

    center = np.array([540.0, 910.0])
    for edge in range(12):
        for step in range(7):
            angles, radii = add_morph(12, -0.4, 370.0, edge, step / 6)
            assert monotone_angles(angles)
            assert is_convex(points_from(center, angles, radii))
    for step in range(11):
        angles, radii = double_morph(96, -1.2, 370.0, step / 10)
        assert monotone_angles(angles)
        assert is_convex(points_from(center, angles, radii))


def test_culling_matches_the_full_polygon_on_a_small_case():
    from fc_sat.polycircle_geometry import culled_matches_full, vertices_in_window
    import numpy as np

    phi = -1.2
    assert culled_matches_full(16, phi, 370.0, np.array([540.0, 910.0]), None)
    window = (-1.7, -1.4)
    full, _ = vertices_in_window(32, phi, 370.0, None, cap=64)
    culled, _ = vertices_in_window(32, phi, 370.0, window, cap=64)
    assert len(culled) < len(full)
    assert culled_matches_full(32, phi, 370.0, np.array([540.0, 910.0]), window)


def test_camera_endpoints_and_rotation_freeze():
    import math

    import numpy as np

    from fc_sat.polycircle_geometry import PHI_F, project, rotation_angles, solve_rotation, zoom_at

    center = np.array([540.0, 910.0])
    top = np.array([540.0, 540.0])
    world = np.array([[100.0, 200.0]])
    assert project(world, center, top, 1.0)[0] == pytest.approx(world[0])
    assert project(top.reshape(1, 2), center, top, 36.0)[0] == pytest.approx(center)
    assert zoom_at(875) == 1
    assert zoom_at(948) == pytest.approx(36)
    assert zoom_at(1280) == pytest.approx(1)
    solution = solve_rotation()
    angles = rotation_angles(solution, 1824)
    assert angles[864] == pytest.approx(PHI_F, abs=1e-9)
    quarters = (angles[0] - PHI_F) / (math.pi / 2)
    assert quarters == pytest.approx(round(quarters), abs=1e-9)


def test_claims_recompute_and_reject_stray_digits():
    from fc_sat.polycircle_claims import evaluate, lint_text, load_claims
    from fc_sat.polycircle_text import lint_script, load_script

    book = load_claims()
    results = evaluate(book)
    failed = [claim_id for claim_id, ok, _detail in results if not ok]
    assert failed == []
    script = load_script(book=book)
    assert lint_script(script, book) == []
    assert lint_text("it takes 99 sides", [], book, "caption")
    assert lint_text("AT 61 SIDES", ["n_fool"], book, "caption") == []


def test_hook_variants_share_the_schedule():
    from fc_sat.polycircle_text import load_script, top_lines
    from fc_sat.polycircle_timeline import build_timeline

    timeline = build_timeline()
    script = load_script()
    assert timeline.schedule.adds[0].frame == 24
    for hook in ("A", "B", "C"):
        assert top_lines(script, 0, hook)
        assert top_lines(script, 0, hook) == top_lines(script, 1823, hook)


def test_layout_stays_in_the_safe_zone():
    from fc_sat.polycircle_render import PolyRenderer, inside_safe, overlaps

    renderer = PolyRenderer(width=540, height=960, hook="A")
    for frame in range(0, 1824, 3):
        boxes = renderer.plan(frame)
        assert len(boxes) <= 4
        for box in boxes:
            assert inside_safe(box, renderer.scale), (frame, box.text)
        for index, left in enumerate(boxes):
            for right in boxes[index + 1 :]:
                assert not overlaps(left, right), (frame, left.text, right.text)


def test_style_law_source():
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    forbidden = ("blur", "bloom", "glow", "gradient", "shadow")
    for name in ("fc_sat/polycircle_render.py", "fc_sat/polycircle_draw.py"):
        text = (root / name).read_text().lower()
        for word in forbidden:
            assert word not in text, f"{name} contains {word}"


def test_timeline_event_counts():
    from fc_sat.polycircle_timeline import build_timeline

    timeline = build_timeline()
    assert len(timeline.schedule.adds) == 92
    assert len(timeline.schedule.doubles) == 16
    assert timeline.kicks[0] == 0
    assert 744 not in timeline.kicks
    assert 768 in timeline.kicks
    assert timeline.impacts == [768, 960, 1632]
    assert 744 in timeline.arp
    assert all(not 745 <= frame < 768 for frame in timeline.kicks + timeline.hats + timeline.bass)
    payload = timeline.to_json()
    assert payload["adds"][0]["frame"] == 24
    assert payload["doublings"][-1]["n_after"] == 6_291_456


@pytest.fixture(scope="module")
def mixed():
    from fc_sat.polycircle_audio import mix
    from fc_sat.polycircle_timeline import build_timeline

    timeline = build_timeline()
    audio, sfx, lufs, peak = mix(timeline.to_json(), seed=timeline.config.seed, n_frames=timeline.n_frames)
    return timeline, audio, sfx, lufs, peak


def test_mix_length_loudness_onset_and_tail(mixed):
    import numpy as np

    from fc_sat.polycircle_audio import events_aligned, onset_sample

    timeline, audio, _sfx, lufs, peak = mixed
    assert len(audio) == 1824 * 800
    assert abs(float(lufs) + 14) <= 0.5
    assert float(peak) <= -1.0
    assert float(np.max(np.abs(audio))) <= 1.0
    assert onset_sample(audio) <= 48
    assert float(np.max(np.abs(audio[-18 * 800 :]))) == 0.0
    missing = events_aligned(audio, timeline.kicks + timeline.arp + timeline.bells)
    assert missing == []


def test_sidechain_ducks_and_releases():
    import numpy as np

    from fc_sat.polycircle_audio import sidechain

    bus = np.ones((48000, 2), dtype=np.float64)
    ducked = sidechain(bus, [0], start=0, end=24)
    assert ducked[0, 0] == pytest.approx(10 ** (-8 / 20), rel=1e-3)
    assert ducked[int(0.12 * 48000), 0] == pytest.approx(1.0, abs=0.02)


def test_postkit_lints_clean():
    from fc_sat.polycircle_claims import load_claims
    from fc_sat.polycircle_post import lint_post, titles

    book = load_claims()
    assert lint_post(book) == []
    for title, _claims in titles(book):
        assert len(title) <= 60
        assert not title.isupper()
