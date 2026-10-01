import numpy as np

from fc_sat.audio import SR
from fc_sat.odd_audio import event_list, mix_odd, synthesize_odd
from fc_sat.odd_config import build_timeline, load_odd_config, timeline_frames


def test_events_match_the_timeline_samples():
    cfg = load_odd_config("configs/odd_default.yaml")
    events = event_list(cfg)
    reveal = next(segment for segment in build_timeline(cfg) if segment.kind == "reveal" and segment.level_id == 1)
    ding = next(event for event in events if event["kind"] == "ding" and event["level"] == 1)
    assert ding["t"] == reveal.start_s + cfg.silence_at_reveal
    assert int(round(ding["t"] * SR)) == int(round((reveal.start_s + 0.15) * SR))
    ticks = [event for event in events if event["kind"] == "tick" and event["level"] == 1]
    assert [event["t"] for event in ticks] == [1.0, 2.0, 3.0, 4.0]


def test_gap_before_the_ding_is_quieter_than_the_ding():
    cfg = load_odd_config("configs/odd_default.yaml")
    dry = mix_odd(cfg)
    reveal = next(
        segment
        for segment in build_timeline(cfg)
        if segment.kind == "reveal" and segment.level_id == 1
    )
    gap0 = int(round((reveal.start_s + 0.05) * SR))
    gap1 = int(round((reveal.start_s + cfg.silence_at_reveal) * SR))
    ding1 = gap1 + int(0.04 * SR)
    gap = dry[gap0:gap1]
    ding = dry[gap1:ding1]
    gap_rms = float(np.sqrt(np.mean(gap ** 2)))
    ding_rms = float(np.sqrt(np.mean(ding ** 2)))
    assert ding_rms > gap_rms * 4


def test_length_loudness_and_no_clip():
    cfg = load_odd_config("configs/odd_default.yaml")
    audio, lufs, true_peak = synthesize_odd(cfg)
    assert len(audio) == timeline_frames(cfg) * 800
    assert abs(lufs + 14.0) <= 0.5
    assert true_peak <= -1.0 + 1e-6
    assert float(np.max(np.abs(audio))) < 1.0
