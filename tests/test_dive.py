"""Dive: cited values, the camera on both acts, the score, and the layout."""

from __future__ import annotations

import pytest
import yaml

from fc_sat.dive_render import DiveRenderer
from fc_sat.dive_scene import build_scene, counter_text, load_values
from fc_sat.dive_timeline import build_timeline
from fc_sat.dive_world import CLAIMS_PATH, SNAP, camera_at, check_order
from fc_sat.scale_claims import display_errors, display_metres, evaluate, load_claims
from fc_sat.scale_render import inside_safe, overlaps


def test_values_are_cited_and_read_back():
    assert all(ok for _, ok, _ in evaluate(load_claims(CLAIMS_PATH)))
    assert display_errors(load_values()) == []
    assert display_metres("12.3 KM INTO ROCK") == pytest.approx(12_300.0)
    assert display_metres("ABOUT 25.8 BILLION KM AWAY") == pytest.approx(2.58e13)
    assert not any("provisional" in str(v.get("source", "")) for v in yaml.safe_load(CLAIMS_PATH.read_text())["claims"].values())


def test_counter_signs():
    assert counter_text(0.0, 1, "down") == "0 m"
    assert counter_text(10935.0, 5, "down") == "-10,935 m"
    assert counter_text(8848.86, 4, "up") == "8,849 m"


def test_stops_go_deeper_then_higher():
    stops = list(build_scene().stops)
    assert check_order(stops) == []
    assert len(stops) == 28
    assert [s.land for s in stops if s.act == "up"][0] == 768


def test_camera_lands_and_only_pulls_back():
    stops = list(build_scene().stops)
    for index, stop in enumerate(stops):
        camera = camera_at(stops, stop.land)
        assert camera.span == pytest.approx(stop.span) and camera.focus == index
    previous = None
    for frame in range(SNAP[0]):
        camera = camera_at(stops, frame)
        if previous is not None and previous.act == camera.act:
            assert camera.span >= previous.span * (1 - 1e-12)
        previous = camera
    assert camera_at(stops, 1823) == camera_at(stops, 0)


def test_bells_fall_on_the_dive_and_rise_on_the_climb():
    timeline = build_timeline()
    assert timeline.bells == timeline.lands[1:]
    down = timeline.bell_degrees[:10]
    up = timeline.bell_degrees[11:]
    assert down == sorted(down, reverse=True)
    assert up == sorted(up)


@pytest.mark.parametrize("frame", [0, 96, 672, 768, 912, 1056, 1392, 1632, 1720, 1810])
def test_text_layout(frame):
    boxes = DiveRenderer().plan(frame)
    assert all(inside_safe(b, 1.0) for b in boxes)
    assert not any(overlaps(a, b) for i, a in enumerate(boxes) for b in boxes[i + 1 :])
