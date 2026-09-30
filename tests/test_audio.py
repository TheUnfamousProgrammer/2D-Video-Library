import hashlib

import numpy as np

from fc_sat.audio import audio_length, master, synthesize
from fc_sat.config import load_config
from fc_sat.sim import simulate_once


def test_audio_length_lufs_and_determinism():
    cfg = load_config("configs/default.yaml")
    sim = simulate_once(cfg, speed=1500, gravity=900, seed_offset=0)
    first, lufs, peak = synthesize(cfg, sim)
    second, lufs2, peak2 = synthesize(cfg, sim)
    assert first.shape == (audio_length(cfg.n_frames(), cfg.fps), 2)
    assert abs(lufs + 14) <= 0.5
    assert peak <= -1.0
    assert abs(lufs2 - lufs) < 1e-6
    assert abs(peak2 - peak) < 1e-6
    digest = hashlib.sha256(np.ascontiguousarray(first).tobytes()).hexdigest()
    again = hashlib.sha256(np.ascontiguousarray(second).tobytes()).hexdigest()
    assert digest == again
    tail = first[-int(0.3 * 48000) :]
    assert np.max(np.abs(tail)) == 0.0
    assert np.max(np.abs(first)) < 1.0


def test_overloud_mix_converges():
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 0.8, size=(48000 * 2, 2))
    audio, lufs, peak = master(noise)
    assert abs(lufs + 14) <= 0.5
    assert peak <= -1.0
    assert audio.shape == noise.shape
    assert np.max(np.abs(audio[-int(0.3 * 48000) :])) == 0.0
