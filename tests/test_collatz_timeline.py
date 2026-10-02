"""Timeline snapping, the elastic bridge, and hook shifts."""

import pytest

from fc_sat.collatz_script import load_script
from fc_sat.collatz_timeline import (
    TimelineError,
    build_timeline,
    proof_end,
    retention_markdown,
    retention_rows,
    ride_step_time,
    s2_step_frames,
)


def test_hook_a_frames():
    timeline = build_timeline(hook="A")
    assert timeline.n_frames == 1956
    assert timeline.duration == pytest.approx(32.6)
    assert s2_step_frames(timeline) == [183, 246, 270, 294, 318, 342, 366, 390]
    assert ride_step_time(timeline, 77) == pytest.approx(19.7166666667)
    assert ride_step_time(timeline, 36) == pytest.approx(16.3)
    assert proof_end(timeline, 7) == pytest.approx(8.3333333333)
    assert proof_end(timeline, 9) == pytest.approx(10.0833333333)
    assert timeline.scene("S9").start == pytest.approx(31.6)
    assert timeline.duration <= 40


def test_elastic_bridge_and_cap():
    pushed = build_timeline(last_vo_end=31.5)
    assert pushed.scene("S9").start == pytest.approx(31.75)
    assert pushed.duration <= 40
    held = build_timeline(last_vo_end=30.0)
    assert held.scene("S9").start == pytest.approx(31.6)
    with pytest.raises(TimelineError):
        build_timeline(hook_end=4.0)


def test_hook_shift_is_capped_at_one_second():
    shifted = build_timeline(hook="B", hook_end=3.1)
    assert shifted.shifted_by == pytest.approx(0.5)
    assert shifted.scene("S2").start == pytest.approx(3.1)
    assert shifted.duration <= 40


def test_retention_map_uses_computed_times():
    timeline = build_timeline()
    text = retention_markdown(retention_rows(timeline, load_script()))
    assert "0.00s hook" in text
    assert "16.30s step 36" in text
    assert "19.72s peak" in text
    assert "31.60s loop bridge" in text
