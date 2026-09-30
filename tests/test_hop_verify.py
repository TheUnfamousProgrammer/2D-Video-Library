import numpy as np

from fc_sat.hop_config import load_hop_config
from fc_sat.hop_render import HopRenderer, layout_failures
from fc_sat.verify import circular_peak, onset_score


def test_layout_boxes_stay_inside_the_safe_zone():
    cfg = load_hop_config("configs/hop_default.yaml")
    renderer = HopRenderer(cfg)
    assert layout_failures(cfg, renderer.dance) == []


def test_audio_onset_detector_finds_a_click():
    sr = 48000
    samples = np.zeros((sr, 2), dtype=np.float64)
    start = int(0.50 * sr)
    t = np.arange(1200) / sr
    click = np.sin(2 * np.pi * 523.0 * t) * np.exp(-t / 0.04)
    samples[start : start + len(click)] = click[:, None]
    score, hop = onset_score(samples, sr)
    found = circular_peak(score, hop, 0.50, 0.040)
    assert abs(found - 0.50) <= 0.020


def test_video_onset_detector_finds_the_bright_frame():
    series = np.zeros(60, dtype=np.float64)
    series[30:] = 40.0
    diff = np.empty_like(series)
    diff[0] = series[0] - series[-1]
    diff[1:] = np.diff(series)
    diff = np.maximum(diff, 0.0)
    found = circular_peak(diff, 1.0 / 60.0, 0.50, 0.040)
    assert abs(found - 0.50) <= 1.0 / 60.0 + 1e-9
