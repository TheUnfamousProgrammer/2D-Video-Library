"""Circular kalimba mix for Melody Hop.

The period is exactly ``N * 800`` samples. Note tails and the reverb wrap with
modulo arithmetic and an FFT circular convolution, so the loop has no fade and
no zeroed tail. True peak uses ``scipy.signal.resample`` at 4x, which treats
the buffer as periodic. The shared loudness loop then aims at -14 LUFS and
-1 dBTP.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.signal import lfilter, resample

from fc_sat.audio import SR, loudness_loop
from fc_sat.hop_choreo import Choreography

REFERENCE_HZ = 523.25
NOTE_SECONDS = 2.5
ATTACK_SECONDS = 0.003


def true_peak_fft(audio: np.ndarray) -> float:
    """Linear true peak from a periodic 4x FFT resample."""
    if audio.size == 0:
        return 0.0
    up = resample(np.asarray(audio, dtype=np.float64), audio.shape[0] * 4, axis=0)
    return float(np.max(np.abs(up)))


def kalimba_wave(freq: float, sr: int = SR) -> np.ndarray:
    """Sine plus a 2nd harmonic and a faster 5.4x partial. Cut at 2.5 s."""
    n = int(round(NOTE_SECONDS * sr))
    t = np.arange(n, dtype=np.float64) / sr
    attack_n = max(1, int(round(ATTACK_SECONDS * sr)))
    attack = np.ones(n, dtype=np.float64)
    u = np.arange(attack_n, dtype=np.float64) / attack_n
    attack[:attack_n] = 0.5 - 0.5 * np.cos(math.pi * u)
    tau = 0.55 * (freq / REFERENCE_HZ) ** -0.35
    env = np.exp(-t / tau) * attack
    partial = np.exp(-t / (tau / 6.0)) * attack
    wave = np.sin(2.0 * math.pi * freq * t) * env
    wave += 0.25 * np.sin(2.0 * math.pi * (2.0 * freq) * t) * env
    wave += 0.10 * np.sin(2.0 * math.pi * (5.4 * freq) * t) * partial
    return wave


def equal_power_pan(x: float, span: float) -> tuple[float, float]:
    """Pan from pad x, limited to +/-0.35, equal power about the frame center."""
    half = span / 2.0 if span else 1.0
    norm = min(1.0, max(-1.0, (x - 540.0) / half))
    pan = norm * 0.35
    theta = (pan + 1.0) * 0.25 * math.pi
    return math.cos(theta), math.sin(theta)


def voice_levels(choreo: Choreography) -> np.ndarray:
    speeds = np.array([choreo.takeoff_speed(i) for i in range(len(choreo.song.notes))], dtype=np.float64)
    peak = float(np.max(speeds)) if speeds.size else 1.0
    if peak <= 0:
        peak = 1.0
    return 0.75 + 0.25 * (speeds / peak)


def mix_dry(choreo: Choreography, sr: int = SR, only: int | None = None) -> np.ndarray:
    """Place kalimba notes on the circle. Sample ``n_k`` is the first sample of note k."""
    period = choreo.grid.n_frames * (sr // 60)
    bus = np.zeros((period, 2), dtype=np.float64)
    levels = voice_levels(choreo)
    indices = range(len(choreo.song.notes)) if only is None else (only,)
    for index in indices:
        note = choreo.song.notes[index]
        freq = choreo.song.frequency(note.midi)
        wave = kalimba_wave(freq, sr)
        left, right = equal_power_pan(choreo.layout.pads[choreo.pad_index(index)].x, choreo.layout.span)
        gain = float(levels[index])
        start = int(choreo.grid.onset_samples[index])
        positions = (start + np.arange(len(wave))) % period
        np.add.at(bus[:, 0], positions, wave * left * gain)
        np.add.at(bus[:, 1], positions, wave * right * gain)
    return bus


def reverb_ir(
    n: int,
    sr: int = SR,
    *,
    seed: int = 1,
    rt60: float = 1.4,
    predelay: float = 0.012,
    cutoff: float = 6000.0,
) -> np.ndarray:
    """Seeded stereo noise, RT60 decay, 12 ms pre-delay, one-pole low-pass, unit energy."""
    rng = np.random.default_rng(seed)
    tau = rt60 / (3.0 * math.log(10.0))
    pre = int(round(predelay * sr))
    pre = min(max(pre, 0), n - 1)
    envelope = np.zeros(n, dtype=np.float64)
    envelope[pre:] = np.exp(-np.arange(n - pre, dtype=np.float64) / sr / tau)
    coeff = math.exp(-2.0 * math.pi * cutoff / sr)
    kernel = np.array([1.0 - coeff])
    denom = np.array([1.0, -coeff])
    ir = np.zeros((n, 2), dtype=np.float64)
    for channel in range(2):
        noise = rng.standard_normal(n) * envelope
        ir[:, channel] = lfilter(kernel, denom, noise)
        energy = float(np.sqrt(np.sum(ir[:, channel] ** 2)))
        if energy > 0:
            ir[:, channel] /= energy
    return ir


def circular_convolve(signal: np.ndarray, impulse: np.ndarray) -> np.ndarray:
    """FFT circular convolution, one period in and one period out."""
    n = signal.shape[0]
    out = np.empty_like(signal)
    for channel in range(signal.shape[1]):
        spectrum = np.fft.rfft(signal[:, channel]) * np.fft.rfft(impulse[:, channel])
        out[:, channel] = np.fft.irfft(spectrum, n=n)
    return out


def mix_period(
    choreo: Choreography,
    *,
    sr: int = SR,
    seed: int = 1,
    reverb_wet: float = 0.22,
    rt60: float = 1.4,
    audio_offset_ms: float = 0.0,
) -> np.ndarray:
    """Dry notes plus circular reverb, mean removed, then a circular offset."""
    dry = mix_dry(choreo, sr)
    impulse = reverb_ir(len(dry), sr, seed=seed, rt60=rt60)
    wet = circular_convolve(dry, impulse)
    mixed = dry + float(reverb_wet) * wet
    mixed -= np.mean(mixed, axis=0, keepdims=True)
    shift = int(round(audio_offset_ms * 0.001 * sr))
    if shift:
        mixed = np.roll(mixed, shift, axis=0)
    return mixed


def synthesize_hop(
    choreo: Choreography,
    *,
    sr: int = SR,
    seed: int = 1,
    reverb_wet: float = 0.22,
    rt60: float = 1.4,
    audio_offset_ms: float = 0.0,
    repeats: int = 1,
) -> tuple[np.ndarray, float, float]:
    """Tile the circular period ``repeats`` times, then run the shared loudness loop."""
    if repeats < 1 or repeats > 3:
        raise ValueError(f"repeats must be from 1 to 3, got {repeats}")
    period = mix_period(
        choreo,
        sr=sr,
        seed=seed,
        reverb_wet=reverb_wet,
        rt60=rt60,
        audio_offset_ms=audio_offset_ms,
    )
    if repeats > 1:
        period = np.tile(period, (repeats, 1))
    return loudness_loop(period, sr, peak_fn=true_peak_fft, zero_tail=0)
