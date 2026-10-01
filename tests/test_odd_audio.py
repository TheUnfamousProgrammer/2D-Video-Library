import numpy as np

from fc_sat.audio import SR
from fc_sat.odd_audio import event_list, synthesize_odd
from fc_sat.odd_config import build_timeline, load_odd_config, timeline_frames


def test_ticks_double_in_the_last_two_seconds():
    cfg = load_odd_config("configs/odd_default.yaml")
    events = event_list(cfg)
    ticks = [event["t"] for event in events if event["kind"] == "tick" and event["level"] == 1]
    assert ticks == [1.0, 2.0, 3.0, 3.5, 4.0, 4.5]
    kinds = {event["kind"] for event in events}
    assert "heart" not in kinds
    assert "riser" not in kinds
    reveal = next(segment for segment in build_timeline(cfg) if segment.kind == "reveal" and segment.level_id == 1)
    ding = next(event for event in events if event["kind"] == "ding" and event["level"] == 1)
    assert ding["t"] == reveal.start_s
    tone = next(event for event in events if event["kind"] == "tone" and event["level"] == 1)
    assert tone["t"] == pytest_approx_time(reveal.start_s - cfg.tone_seconds)


def pytest_approx_time(value):
    return value


def test_length_loudness_and_no_clip():
    cfg = load_odd_config("configs/odd_default.yaml")
    audio, lufs, true_peak = synthesize_odd(cfg)
    assert len(audio) == timeline_frames(cfg) * (SR // 60)
    assert abs(lufs + 14.0) <= 0.5
    assert true_peak <= -1.0 + 1e-6
    assert float(np.max(np.abs(audio))) < 1.0
