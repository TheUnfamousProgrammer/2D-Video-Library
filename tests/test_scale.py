"""Scale: the counter, the cited sizes, the camera, the score, and the layout."""

from __future__ import annotations

import math

import pytest

from fc_sat.scale_claims import display_errors, display_metres, evaluate, load_claims
from fc_sat.scale_format import metres
from fc_sat.scale_render import ScaleRenderer, counter_rows, inside_safe, overlaps
from fc_sat.scale_scene import build_scene, load_sizes
from fc_sat.scale_timeline import ARP_BUILD, build_timeline
from fc_sat.scale_world import SNAP, camera_at, check_order, focus_index


def test_metres_keeps_every_zero():
    assert metres(1.68e-15, 2) == "0.0000000000000017 m"
    assert metres(1.71, 2) == "1.7 m"
    assert metres(8848.86, 4) == "8,849 m"
    assert metres(8.7985e26, 2) == "880,000,000,000,000,000,000,000,000 m"


def test_display_lines_read_back_to_their_sizes():
    assert display_metres("12,756 KM") == pytest.approx(12_756_000.0)
    assert display_metres("1.39 MILLION KM") == pytest.approx(1.39e9)
    assert display_metres("UP TO 30 METRES") == pytest.approx(30.0)
    assert display_metres("ABOUT 900 SUNS WIDE") == pytest.approx(900 * 1_391_400_000.0)
    assert display_metres("HALF A MILLIMETRE") == pytest.approx(5e-4)
    assert display_errors(load_sizes()) == []


def test_a_wrong_display_line_is_caught():
    sizes = {"x": {"value": 12_742_000.0, "display": "1,274 KM"}}
    assert display_errors(sizes)


def test_every_size_is_cited_and_numeric():
    book = load_claims()
    assert all(ok for _, ok, _ in evaluate(book))
    for claim in book.claims.values():
        assert isinstance(claim.value, float)
        assert "provisional" not in claim.source.lower()


def test_objects_grow_and_land_in_order():
    scene = build_scene()
    assert check_order(scene.items) == []
    assert len(scene.items) == 28
    assert scene.items[0].land == 0
    assert scene.items[-1].land == 1632


def test_camera_frames_each_object_on_its_landing():
    placed = list(build_scene().placed)
    for index, p in enumerate(placed):
        camera = camera_at(placed, p.item.land)
        assert camera.scale == pytest.approx(p.scale, rel=1e-9)
        assert camera.x_m == pytest.approx(p.x_m, rel=1e-9, abs=1e-30)
        assert focus_index(placed, p.item.land) == index


def test_camera_only_zooms_out_and_loops():
    placed = list(build_scene().placed)
    previous = math.inf
    for frame in range(SNAP[0]):
        scale = camera_at(placed, frame).scale
        assert scale <= previous * (1 + 1e-12)
        previous = scale
    assert camera_at(placed, 1823) == camera_at(placed, 0)


def test_every_landing_rings_a_rising_bell():
    timeline = build_timeline()
    assert timeline.max_snap_error == 0.0
    assert timeline.bells == timeline.lands[1:]
    assert timeline.bell_degrees == sorted(timeline.bell_degrees)
    assert timeline.bell_degrees[-1] == 14
    assert 768 in timeline.impacts and 1632 in timeline.impacts


def test_arp_build_is_the_polycircle_build():
    from fc_sat.polycircle_timeline import build_timeline as poly_timeline

    adds = [item.frame for item in poly_timeline().schedule.adds if item.frame != 768]
    assert ARP_BUILD == adds


def test_long_counters_wrap_at_a_comma():
    rows = counter_rows("880,000,000,000,000,000,000,000,000 m", 820.0, 110.0, 64.0, 30.0)
    assert len(rows) == 2
    assert rows[0].endswith(",")
    assert "".join(rows) == "880,000,000,000,000,000,000,000,000 m"
    assert counter_rows("0.0000000000000017 m", 820.0, 110.0, 64.0, 30.0) == ("0.0000000000000017 m",)


@pytest.mark.parametrize("frame", [0, 96, 700, 768, 1104, 1344, 1500, 1560, 1640, 1720, 1810])
def test_text_stays_in_the_safe_zone_without_overlaps(frame):
    renderer = ScaleRenderer()
    boxes = renderer.plan(frame)
    assert all(inside_safe(box, 1.0) for box in boxes)
    assert not any(overlaps(a, b) for i, a in enumerate(boxes) for b in boxes[i + 1 :])


def test_small_render_has_the_right_shape():
    image = ScaleRenderer(width=270, height=480).render(1200)
    assert image.shape == (480, 270, 3)
