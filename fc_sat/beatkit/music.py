"""Synthesizer primitives shared by the 150 bpm films.

Arrangements stay in each generator. These functions are the ones polycircle
already used: placement, drums, sidechain, filters, the loudness master, and
onset probes. Moving them must not change a single sample.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.signal import butter, sosfilt

from fc_sat.audio import SR, loudness_loop
from fc_sat.beatkit.grid import HOP

__all__ = [
    "HOP",
    "SR",
    "detect_onsets",
    "events_aligned",
    "kick_wave",
    "master",
    "onset_sample",
    "sidechain",
    "sync_rows",
]


def _place(bus: np.ndarray, start: int, wave: np.ndarray, gain: float = 1.0, pan: float = 0.0) -> None:
    if wave.size == 0 or gain == 0:
        return
    left = gain * (1.0 - max(pan, 0.0))
    right = gain * (1.0 + min(pan, 0.0))
    if start < 0:
        wave = wave[-start:]
        start = 0
    end = min(len(bus), start + len(wave))
    span = end - start
    if span <= 0:
        return
    bus[start:end, 0] += wave[:span] * left
    bus[start:end, 1] += wave[:span] * right


def _silence(bus: np.ndarray, frame0: int, frame1: int) -> None:
    bus[frame0 * HOP : frame1 * HOP] = 0.0


def _tone(freq: float, n: int, decay: float, sr: int = SR) -> np.ndarray:
    t = np.arange(n, dtype=np.float64) / sr
    return np.sin(2.0 * math.pi * freq * t) * np.exp(-t / decay)


def kick_wave(sr: int = SR) -> np.ndarray:
    n = int(0.220 * sr)
    t = np.arange(n, dtype=np.float64) / sr
    freq = 150.0 * ((45.0 / 150.0) ** (t / 0.220))
    phase = 2.0 * math.pi * np.cumsum(freq) / sr
    body = np.sin(phase) * np.exp(-t / 0.05)
    click = np.exp(-t / 0.0011)
    click[0] = 1.0
    return body * 0.9 + click * 0.55


def _noise(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal(n)


def _shape(noise: np.ndarray, sr: int, decay: float, low: float | None, high: float | None) -> np.ndarray:
    if low and high:
        sos = butter(2, [low, high], btype="band", fs=sr, output="sos")
    elif high:
        sos = butter(2, high, btype="low", fs=sr, output="sos")
    else:
        sos = butter(2, low or 30.0, btype="high", fs=sr, output="sos")
    wave = sosfilt(sos, noise)
    peak = float(np.max(np.abs(wave))) or 1.0
    t = np.arange(len(wave), dtype=np.float64) / sr
    return wave / peak * np.exp(-t / decay)


def sidechain(bus: np.ndarray, kicks: list[int], sr: int = SR, start: int = 960, end: int = 1344) -> np.ndarray:
    """Drop bass, pad, and arp by 8 dB on each kick, back over 120 ms."""
    env = np.ones(len(bus), dtype=np.float64)
    drop = 10.0 ** (-8.0 / 20.0)
    release = int(0.120 * sr)
    ramp = drop + (1.0 - drop) * (np.arange(release, dtype=np.float64) / max(1, release - 1))
    for frame in kicks:
        if not start <= int(frame) < end:
            continue
        sample = int(frame) * (sr // 60)
        stop = min(len(env), sample + release)
        if sample >= len(env):
            continue
        env[sample:stop] = np.minimum(env[sample:stop], ramp[: stop - sample])
    return bus * env[:, None]


def _sweep_lowpass(bus: np.ndarray, frame0: int, frame1: int, cut0: float, cut1: float, sr: int = SR) -> np.ndarray:
    out = np.array(bus, dtype=np.float64, copy=True)
    hop = sr // 60
    state = None
    for frame in range(frame0, frame1 + 1):
        u = (frame - frame0) / max(1, frame1 - frame0)
        cutoff = cut0 * ((cut1 / cut0) ** u)
        sos = butter(2, min(cutoff, sr * 0.45), btype="low", fs=sr, output="sos")
        sl = slice(frame * hop, min(len(out), (frame + 1) * hop))
        if state is None or state.shape[-1] != out.shape[1]:
            zi = np.zeros((sos.shape[0], 2, out.shape[1]), dtype=np.float64)
            state = zi
        filtered, state = sosfilt(sos, out[sl], axis=0, zi=state)
        out[sl] = filtered
    return out


def _supersaw(freq: float, n: int, rng: np.random.Generator, sr: int = SR) -> np.ndarray:
    t = np.arange(n, dtype=np.float64) / sr
    wave = np.zeros(n, dtype=np.float64)
    for offset in (-2, -1, 0, 1, 2):
        f = freq * (1.0 + 0.012 * offset)
        phase = float(rng.random()) * 2.0 * math.pi
        for partial in range(1, 6):
            wave += (1.0 / partial) * np.sin(2.0 * math.pi * f * partial * t + phase)
    peak = float(np.max(np.abs(wave))) or 1.0
    return wave / peak


def _highpass(audio: np.ndarray, sr: int, freq: float = 30.0) -> np.ndarray:
    sos = butter(2, freq, btype="high", fs=sr, output="sos")
    return sosfilt(sos, audio, axis=0)


def master(audio: np.ndarray, sr: int = SR, tail_frames: int = 18) -> tuple[np.ndarray, float, float]:
    x = np.array(audio, dtype=np.float64, copy=True)
    x -= np.mean(x, axis=0, keepdims=True)
    x = _highpass(x, sr, 30.0)
    fade = int(0.002 * sr)
    x[:fade] *= np.linspace(0.0, 1.0, fade)[:, None]
    tail = tail_frames * (sr // 60)
    if tail:
        x[-tail:] = 0.0
    # -2.3 dBTP on the wav leaves room for the AAC encode to stay at or under -1 dBTP.
    return loudness_loop(x, sr, ceiling=10.0 ** (-2.3 / 20.0))


def _riser(bus: np.ndarray, rng: np.random.Generator, frame0: int, frame1: int, sr: int) -> None:
    n = (frame1 - frame0) * HOP
    if n <= 8:
        return
    noise = _noise(n, rng)
    t = np.arange(n, dtype=np.float64) / sr
    dur = max(float(t[-1]), 1e-6)
    out = np.zeros(n, dtype=np.float64)
    hop = sr // 60
    state = None
    for frame in range(frame0, frame1):
        u = (frame - frame0) / max(1, frame1 - frame0)
        low = 200.0 * (2.0 ** (u * 4))
        high = min(sr * 0.45, low * 1.6)
        sos = butter(2, [low, high], btype="band", fs=sr, output="sos")
        sl = slice((frame - frame0) * hop, (frame - frame0 + 1) * hop)
        if state is None:
            state = np.zeros((sos.shape[0], 2), dtype=np.float64)
        filtered, state = sosfilt(sos, noise[sl], zi=state)
        out[sl] = filtered
    peak = float(np.max(np.abs(out))) or 1.0
    env = (t / dur) ** 1.3
    _place(bus, frame0 * HOP, out / peak * env, 0.08)


def _whoosh(bus: np.ndarray, rng: np.random.Generator, frame0: int, frame1: int, sr: int) -> None:
    _riser(bus, rng, frame0, frame1, sr)


def _reverse(bus: np.ndarray, rng: np.random.Generator, frame0: int, frame1: int, sr: int) -> None:
    n = (frame1 - frame0) * HOP
    burst = _shape(_noise(n, rng), sr, 0.4, 4000.0, None)
    t = np.arange(n, dtype=np.float64) / max(1, n)
    _place(bus, frame0 * HOP, burst * (t ** 2), 0.18)


def _tape(bus: np.ndarray, frame: int, sr: int) -> None:
    n = int(0.1 * sr)
    t = np.arange(n, dtype=np.float64) / sr
    freq = 1200.0 * ((60.0 / 1200.0) ** (t / 0.1))
    phase = 2.0 * math.pi * np.cumsum(freq) / sr
    env = np.linspace(1.0, 0.0, n)
    _place(bus, frame * HOP, np.sin(phase) * env, 0.2)


def sync_rows(events: dict, fps: int = 60, sr: int = SR) -> list[str]:
    rows = ["| kind | frame | seconds | sample |", "| --- | --- | --- | --- |"]
    for kind in ("kicks", "arp", "bells", "impacts", "ticks"):
        frames = events[kind]
        if kind == "ticks":
            frames = list(frames) + [events["chime"]]
            kind = "tick/chime"
        for frame in frames:
            rows.append(f"| {kind} | {frame} | {frame / fps:.3f} | {int(frame) * (sr // fps)} |")
    return rows


def events_aligned(audio: np.ndarray, frames: list[int], sr: int = SR, fps: int = 60, floor: float = 0.006) -> list[int]:
    """Scheduled frames whose transient is missing within one frame."""
    hop = sr // fps
    diff = np.abs(np.diff(audio[:, 0], prepend=audio[0, 0]))
    usable = diff[: len(diff) // hop * hop]
    strength = usable.reshape(-1, hop).max(axis=1)
    missing = []
    for frame in frames:
        window = strength[max(0, int(frame) - 1) : int(frame) + 2]
        if window.size == 0 or float(window.max()) < floor:
            missing.append(int(frame))
    return missing


def detect_onsets(audio: np.ndarray, sr: int = SR, fps: int = 60) -> list[int]:
    """Frame indices whose transient is a local peak above the noise floor."""
    hop = sr // fps
    diff = np.abs(np.diff(audio[:, 0], prepend=audio[0, 0]))
    usable = diff[: len(diff) // hop * hop]
    frames = usable.reshape(-1, hop).max(axis=1)
    floor = max(float(np.median(frames)) * 6.0, 0.004)
    hits = []
    for index, value in enumerate(frames):
        left = float(frames[index - 1]) if index else 0.0
        right = float(frames[index + 1]) if index + 1 < len(frames) else 0.0
        if value >= floor and value >= left and value >= right:
            hits.append(index)
    if frames.size and float(np.max(np.abs(audio[: hop // 2, 0]))) > 0.01 and 0 not in hits:
        hits.insert(0, 0)
    return hits


def onset_sample(audio: np.ndarray, sr: int = SR) -> int:
    """First sample of the opening kick that clears 5% of the first 50 ms peak."""
    window = audio[: int(0.05 * sr), 0]
    peak = float(np.max(np.abs(window))) or 1.0
    hits = np.where(np.abs(window) >= 0.05 * peak)[0]
    return int(hits[0]) if len(hits) else 10**9
