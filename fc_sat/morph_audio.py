"""Pixel Morph soundtrack. Notes are the bounce mallet; the master is bounce's too.

Arrival ticks are the analytic time each particle reaches ease 1. Each 100 ms
window keeps at most ``tick_density`` of them (seeded draw, not loudest-wins).
Pitch walks up the C major pentatonic on the way to B and back down on the
return, with a seeded wobble of one step. The whoosh is band-limited noise
under the ticks by ``whoosh_db`` (default -18) relative to a nominal tick gain
of 0.5, following the fraction of particles in flight. A pentatonic triad
chimes when B is fully formed. A quieter chime marks the return home; the
shared master then zeros the last 300 ms, so most of that final chime is
silence on purpose.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.signal import butter, sosfilt

from fc_sat.audio import FREQS, SR, audio_length, master, note_cache
from fc_sat.morph_config import MorphConfig
from fc_sat.morph_motion import arrival_tau, phase_bounds

_TICK_GAIN = 0.5
_WINDOW = 0.1


def choose_ticks(times: np.ndarray, limit: int, seed: int) -> np.ndarray:
    """Indices to keep so no 100 ms window holds more than ``limit`` ticks."""
    if times.size == 0:
        return np.zeros(0, dtype=np.int32)
    order = np.argsort(times, kind="mergesort")
    rng = np.random.default_rng(int(seed))
    kept: list[int] = []
    start = 0
    while start < len(order):
        window = int(math.floor(float(times[order[start]]) / _WINDOW + 1e-9))
        end = start + 1
        while end < len(order) and int(math.floor(float(times[order[end]]) / _WINDOW + 1e-9)) == window:
            end += 1
        group = order[start:end]
        if len(group) > limit:
            pick = rng.choice(group, size=int(limit), replace=False)
            pick = pick[np.argsort(times[pick], kind="mergesort")]
            kept.extend(int(i) for i in pick)
        else:
            kept.extend(int(i) for i in group)
        start = end
    return np.asarray(kept, dtype=np.int32)


def _place(bus: np.ndarray, start: int, wave: np.ndarray, left: float, right: float) -> None:
    if start >= len(bus) or not len(wave):
        return
    if start < 0:
        wave = wave[-start:]
        start = 0
    if not len(wave):
        return
    stop = min(len(bus), start + len(wave))
    bus[start:stop, 0] += wave[: stop - start] * left
    bus[start:stop, 1] += wave[: stop - start] * right


def _pan(x: float, origin_x: float, box_w: float) -> tuple[float, float]:
    half = box_w / 2.0 if box_w else 1.0
    norm = min(1.0, max(-1.0, (x - (origin_x + half)) / half))
    theta = (norm + 1.0) * 0.25 * math.pi
    return math.cos(theta), math.sin(theta)


def _shift(bus: np.ndarray, offset_ms: float, sr: int) -> np.ndarray:
    shift = int(round(offset_ms * 0.001 * sr))
    if shift == 0:
        return bus
    out = np.zeros_like(bus)
    if shift > 0:
        out[shift:] = bus[:-shift]
    else:
        out[:shift] = bus[-shift:]
    return out


def _chime(decay: float, gain: float) -> np.ndarray:
    n = int(SR * (decay + 0.05))
    t = np.arange(n, dtype=np.float64) / SR
    attack = np.minimum(t / 0.004, 1.0)
    env = np.exp(-t / decay) * attack
    wave = np.zeros(n, dtype=np.float64)
    for index, amp in ((0, 0.60), (2, 0.45), (4, 0.32)):
        wave += amp * np.sin(2.0 * math.pi * FREQS[index] * t)
    peak = float(np.max(np.abs(wave)))
    if peak > 0:
        wave /= peak
    return wave * gain


def _add_ticks(
    bus: np.ndarray,
    cfg: MorphConfig,
    times: np.ndarray,
    progress: np.ndarray,
    xs: np.ndarray,
    *,
    rising: bool,
    rng: np.random.Generator,
) -> None:
    chosen = choose_ticks(times, cfg.tick_density, int(rng.integers(0, 2**31 - 1)))
    if chosen.size == 0:
        return
    notes = note_cache()
    order = chosen[np.argsort(times[chosen], kind="mergesort")]
    box_w = cfg.cols * cfg.cell
    for index in order:
        tau = float(progress[index])
        base = int(round((tau if rising else 1.0 - tau) * 14.0))
        wobble = int(rng.integers(-1, 2))
        note = int(min(14, max(0, base + wobble)))
        gain = float(rng.uniform(0.35, 0.60))
        left, right = _pan(float(xs[index]), cfg.origin_x, box_w)
        start = int(round(float(times[index]) * SR))
        _place(bus, start, notes[note][1] * gain, left, right)


def _flight_envelope(
    cfg: MorphConfig,
    n_samples: int,
    delay_ab: np.ndarray,
    delay_ba: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Smoothed flight fraction and the whoosh center frequency, per sample."""
    hop = 0.01
    duration = n_samples / SR
    times = np.arange(0.0, duration, hop)
    fraction = np.zeros(times.size, dtype=np.float64)
    freq = np.full(times.size, 300.0, dtype=np.float64)
    bounds = phase_bounds(cfg.timeline())
    span = 1.0 - float(cfg.delay_frac)
    for name, delay, forward in (("morph_ab", delay_ab, True), ("morph_ba", delay_ba, False)):
        start, length = bounds[name]
        inside = (times >= start) & (times < start + length)
        if not np.any(inside):
            continue
        tau = np.clip((times[inside] - start) / length, 0.0, 1.0)
        progress = np.clip((tau[:, None] - delay[None, :]) / span, 0.0, 1.0)
        flying = (progress > 0.0) & (progress < 1.0)
        fraction[inside] = flying.mean(axis=1)
        sweep = 300.0 + tau * 1500.0
        freq[inside] = sweep if forward else 1800.0 - tau * 1500.0
    kernel = np.ones(10, dtype=np.float64) / 10.0
    smooth = np.convolve(fraction, kernel, mode="same")
    sample_t = np.arange(n_samples, dtype=np.float64) / SR
    envelope = np.interp(sample_t, times, smooth)
    centers = np.interp(sample_t, times, freq)
    return envelope, centers


def _whoosh(cfg: MorphConfig, n_samples: int, envelope: np.ndarray, centers: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng([int(cfg.seed), 1])
    noise = rng.standard_normal(n_samples)
    filtered = np.zeros(n_samples, dtype=np.float64)
    block = SR // 50
    zi = None
    for start in range(0, n_samples, block):
        stop = min(n_samples, start + block)
        if float(np.max(envelope[start:stop])) < 1e-4:
            zi = None
            continue
        center = float(np.median(centers[start:stop]))
        lo = max(40.0, center / 1.4)
        hi = min(SR / 2.0 - 200.0, center * 1.4)
        if hi <= lo + 10:
            continue
        sos = butter(2, [lo, hi], btype="band", fs=SR, output="sos")
        if zi is None:
            zi = np.zeros((sos.shape[0], 2), dtype=np.float64)
        chunk, zi = sosfilt(sos, noise[start:stop], zi=zi)
        filtered[start:stop] = chunk
    peak = float(np.max(np.abs(filtered))) if filtered.size else 0.0
    if peak <= 0:
        return np.zeros((n_samples, 2), dtype=np.float64)
    gain = _TICK_GAIN * (10.0 ** (float(cfg.whoosh_db) / 20.0))
    mono = filtered * (gain / peak) * envelope
    return np.column_stack((mono, mono))


def synthesize_morph(
    cfg: MorphConfig,
    *,
    delay_ab: np.ndarray,
    delay_ba: np.ndarray,
    x_b: np.ndarray,
    x_a: np.ndarray,
    n_frames: int,
    fps: int,
) -> tuple[np.ndarray, float, float]:
    """Stereo mix of length ``n_frames * 800``, then the bounce master."""
    n_samples = audio_length(n_frames, fps, SR)
    bus = np.zeros((n_samples, 2), dtype=np.float64)
    bounds = phase_bounds(cfg.timeline())
    rng = np.random.default_rng(int(cfg.seed))
    for name, delay, xs, rising in (
        ("morph_ab", delay_ab, x_b, True),
        ("morph_ba", delay_ba, x_a, False),
    ):
        start, duration = bounds[name]
        tau = arrival_tau(delay, cfg.delay_frac)
        times = start + tau * duration
        _add_ticks(bus, cfg, times, tau, xs, rising=rising, rng=rng)
    hold_b, _hold_b_len = bounds["hold_b"]
    hold_end, _hold_end_len = bounds["hold_a_end"]
    _place(bus, int(round(hold_b * SR)), _chime(1.2, 0.22), 0.707, 0.707)
    _place(bus, int(round(hold_end * SR)), _chime(0.6, 0.10), 0.707, 0.707)
    envelope, centers = _flight_envelope(cfg, n_samples, delay_ab, delay_ba)
    bus += _whoosh(cfg, n_samples, envelope, centers)
    bus = _shift(bus, cfg.audio_offset_ms, SR)
    if len(bus) != n_samples:
        fitted = np.zeros((n_samples, 2), dtype=np.float64)
        fitted[: min(n_samples, len(bus))] = bus[:n_samples]
        bus = fitted
    return master(bus, SR)
