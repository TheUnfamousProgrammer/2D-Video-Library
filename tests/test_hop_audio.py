import hashlib

import numpy as np
import pytest

from fc_sat.hop_audio import (
    equal_power_pan,
    kalimba_wave,
    mix_dry,
    mix_period,
    synthesize_hop,
    true_peak_fft,
    voice_levels,
)
from fc_sat.hop_choreo import build_choreography
from fc_sat.song import load_song

ODE = "songs/ode_to_joy.yaml"


def _dance():
    song = load_song(ODE)
    return build_choreography(song)


def test_dry_note_starts_at_its_sample_and_wraps():
    dance = _dance()
    note = 4
    dry = mix_dry(dance, only=note)
    start = dance.grid.onset_samples[note]
    freq = dance.song.frequency(dance.song.notes[note].midi)
    wave = kalimba_wave(freq)
    level = float(voice_levels(dance)[note])
    left, _right = equal_power_pan(
        dance.layout.pads[dance.pad_index(note)].x,
        dance.layout.span,
    )
    assert start > 0
    assert dry[start - 1, 0] == 0.0
    probe = wave[:200] * left * level
    assert np.allclose(dry[start : start + 200, 0], probe)
    assert len(wave) == 2.5 * 48000

    last = len(dance.song.notes) - 1
    wrapped = mix_dry(dance, only=last)
    start = dance.grid.onset_samples[last]
    freq = dance.song.frequency(dance.song.notes[last].midi)
    wave = kalimba_wave(freq)
    level = float(voice_levels(dance)[last])
    left, _right = equal_power_pan(
        dance.layout.pads[dance.pad_index(last)].x,
        dance.layout.span,
    )
    assert start + len(wave) > len(wrapped)
    folded = (start + np.arange(len(wave))) % len(wrapped)
    head = int(np.where(folded == 0)[0][0])
    assert np.allclose(wrapped[:80, 0], wave[head : head + 80] * left * level)


def test_hop_mix_is_deterministic_circular_and_loud():
    dance = _dance()
    first, lufs, peak = synthesize_hop(dance, seed=7)
    second, lufs2, peak2 = synthesize_hop(dance, seed=7)
    period = dance.grid.n_frames * 800
    assert first.shape == (period, 2)
    assert abs(lufs + 14) <= 0.5
    assert peak <= -1.0
    assert abs(lufs2 - lufs) < 1e-6
    assert hashlib.sha256(np.ascontiguousarray(first).tobytes()).hexdigest() == hashlib.sha256(
        np.ascontiguousarray(second).tobytes()
    ).hexdigest()
    assert np.max(np.abs(first[-int(0.3 * 48000) :])) > 0.0
    for channel in range(2):
        seam = abs(first[-1, channel] - first[0, channel])
        jumps = np.abs(np.diff(first[:, channel]))
        assert seam <= 3.0 * np.percentile(jumps, 99)
    rms = float(np.sqrt(np.mean(first[: int(0.030 * 48000)] ** 2)))
    assert 20 * np.log10(rms) > -40.0
    repeated, _l, _p = synthesize_hop(dance, seed=7, repeats=2)
    assert repeated.shape[0] == period * 2


def test_offset_rolls_without_changing_length():
    dance = _dance()
    base, _, _ = synthesize_hop(dance, seed=3, audio_offset_ms=0)
    # Compare the pre-master roll on a dry isolated path: 10 ms is 480 samples.
    dry = mix_dry(dance, only=0)
    shifted = mix_period(dance, seed=3, reverb_wet=0.0, audio_offset_ms=10)
    plain = mix_period(dance, seed=3, reverb_wet=0.0, audio_offset_ms=0)
    assert np.allclose(shifted, np.roll(plain, 480, axis=0))
    assert base.shape[0] == dry.shape[0]


def test_fft_loudness_loop_converges_on_overloud_noise():
    from fc_sat.audio import loudness_loop

    rng = np.random.default_rng(0)
    noise = rng.normal(0, 0.8, size=(48000 * 2, 2))
    audio, lufs, peak = loudness_loop(noise, 48000, peak_fn=true_peak_fft, zero_tail=0)
    assert abs(lufs + 14) <= 0.5
    assert peak <= -1.0
    assert audio.shape == noise.shape
    assert true_peak_fft(audio) <= 10 ** (-1 / 20) + 1e-6
    assert np.max(np.abs(audio[-100:])) > 0.0
