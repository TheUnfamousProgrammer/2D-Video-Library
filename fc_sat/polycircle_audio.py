"""Original A-minor arrangement. Every hit is a timeline event. No borrowed audio.

The oscillators, the master, and the onset probes live in ``fc_sat.beatkit.music``.
This module is only the polycircle score.
"""

from __future__ import annotations

import math

import numpy as np

from fc_sat.audio import SR
from fc_sat.beatkit.music import (
    HOP,
    _place,
    _reverse,
    _riser,
    _shape,
    _silence,
    _supersaw,
    _sweep_lowpass,
    _tape,
    _tone,
    _whoosh,
    detect_onsets,
    events_aligned,
    kick_wave,
    master,
    onset_sample,
    sidechain,
    sync_rows,
    _noise,
)

A2 = 110.0
A4 = 440.0
CHORDS = ((0, 3, 7), (5, 9, 12), (3, 7, 10), (7, 11, 14))
PENTA = tuple(A4 * 2 ** ((semi + 12 * octave) / 12.0) for octave in range(3) for semi in (0, 3, 5, 7, 10))

__all__ = [
    "detect_onsets",
    "events_aligned",
    "master",
    "mix",
    "onset_sample",
    "render_buses",
    "sidechain",
    "sync_rows",
]


def _arp_freq(events: dict, frame: int) -> float:
    # Walk the current bar's chord tones. Bar length is 96 frames at the default tempo.
    bar = min(18, frame // 96)
    tones = []
    for octave in range(3):
        for semi in CHORDS[bar % 4]:
            tones.append(A2 * (2.0 ** (semi / 12.0)) * (2.0 ** octave))
    arp_frames = events["arp"]
    index = arp_frames.index(frame) if frame in arp_frames else 0
    return tones[index % len(tones)]


def render_buses(events: dict, seed: int, n_frames: int, sr: int = SR) -> tuple[np.ndarray, np.ndarray]:
    """Beat bus and SFX bus, before the shared loudness loop."""
    n = n_frames * (sr // 60)
    rng = np.random.Generator(np.random.PCG64(seed))
    kick = np.zeros((n, 2), dtype=np.float64)
    drums = np.zeros((n, 2), dtype=np.float64)
    bass = np.zeros((n, 2), dtype=np.float64)
    pad = np.zeros((n, 2), dtype=np.float64)
    arp = np.zeros((n, 2), dtype=np.float64)
    sfx = np.zeros((n, 2), dtype=np.float64)
    kick_w = kick_wave(sr)
    for frame in events["kicks"]:
        _place(kick, int(frame) * HOP, kick_w, 0.85)
    for frame in events["claps"]:
        burst = _shape(_noise(int(0.12 * sr), rng), sr, 0.03, 900.0, 2500.0)
        _place(drums, int(frame) * HOP, burst, 0.28)
    for frame in events["hats"]:
        burst = _shape(_noise(int(0.04 * sr), rng), sr, 0.012, 7000.0, None)
        _place(drums, int(frame) * HOP, burst, 0.12, pan=0.15)
    for frame in events["open_hats"]:
        burst = _shape(_noise(int(0.18 * sr), rng), sr, 0.05, 6000.0, None)
        _place(drums, int(frame) * HOP, burst, 0.16)
    for frame in events["snares"]:
        burst = _shape(_noise(int(0.08 * sr), rng), sr, 0.02, 180.0, 4000.0)
        _place(drums, int(frame) * HOP, burst, 0.22)
    for frame in events["bass"]:
        bar = min(18, int(frame) // 96)
        root = A2 * (2.0 ** (CHORDS[bar % 4][0] / 12.0))
        n = int(0.18 * sr)
        t = np.arange(n, dtype=np.float64) / sr
        saw = 2.0 * ((t * root) % 1.0) - 1.0
        sub = np.sin(2.0 * math.pi * (root * 0.5) * t)
        mixed = np.tanh(2.2 * (0.65 * saw + 0.85 * sub))
        mixed *= np.exp(-t / 0.12)
        mixed = _shape(mixed, sr, 0.2, None, 280.0)
        _place(bass, int(frame) * HOP, mixed, 0.34)
    # Pedal: the bass event at frame 1728 is already placed; lengthen it.
    if 1728 in events["bass"]:
        n = int(1.1 * sr)
        t = np.arange(n, dtype=np.float64) / sr
        root = A2 * (2.0 ** (CHORDS[(1728 // 96) % 4][0] / 12.0))
        pedal = np.tanh(1.8 * np.sin(2.0 * math.pi * root * 0.5 * t)) * np.exp(-t / 0.6)
        _place(bass, 1728 * HOP, pedal, 0.22)
    hop = sr // 60
    for bar, chord_name_index in enumerate(range(19)):
        if bar < 4 or bar in (15, 16):
            # Pad enters at bar 5 (index 4). Breakdown bars 16-17 (index 15, 16) keep a soft pad.
            pass
        start = bar * 96
        if start >= n_frames:
            break
        stop = min(n_frames, start + 96)
        if bar < 4:
            continue
        length = (stop - start) * hop
        tone = np.zeros(length, dtype=np.float64)
        for semi in CHORDS[bar % 4]:
            tone += _supersaw(A2 * (2.0 ** (semi / 12.0)), length, rng)
        tone = _shape(tone, sr, 2.0, None, 1400.0)
        env = np.ones(length, dtype=np.float64)
        edge = min(int(0.02 * sr), length // 4)
        if edge > 1:
            env[:edge] *= np.linspace(0.0, 1.0, edge)
            env[-edge:] *= np.linspace(1.0, 0.0, edge)
        gain = 0.045 if bar in (15, 16) else 0.07
        _place(pad, start * hop, tone * env, gain)
    arp_set = set(events["arp"])
    for frame in events["arp"]:
        freq = _arp_freq(events, int(frame))
        soft = 1440 <= int(frame) < 1632
        n = int((0.09 if soft else 0.06) * sr)
        t = np.arange(n, dtype=np.float64) / sr
        pluck = (np.sin(2.0 * math.pi * freq * t) + 0.25 * np.sin(2.0 * math.pi * freq * 2 * t)) * np.exp(-t / (0.05 if soft else 0.03))
        click_n = min(len(pluck), int(0.003 * sr))
        pluck[:click_n] += 0.65 * np.exp(-np.arange(click_n) / (0.0007 * sr))
        _place(arp, int(frame) * HOP, pluck, 0.22 if soft else 0.2)
    for frame, degree in zip(events["bells"], events["bell_degrees"]):
        freq = PENTA[min(len(PENTA) - 1, int(degree))]
        n = int(0.45 * sr)
        t = np.arange(n, dtype=np.float64) / sr
        bell = (np.sin(2.0 * math.pi * freq * t) + 0.35 * np.sin(2.0 * math.pi * freq * 2.4 * t)) * np.exp(-t / 0.18)
        thump = np.sin(2.0 * math.pi * 90.0 * t) * np.exp(-t / 0.05)
        _place(sfx, int(frame) * HOP, bell + 0.35 * thump, 0.22)
    for frame in events["impacts"]:
        n = int(0.4 * sr)
        t = np.arange(n, dtype=np.float64) / sr
        boom = np.sin(2.0 * math.pi * 55.0 * t) * np.exp(-t / 0.12)
        burst = _shape(_noise(n, rng), sr, 0.04, 200.0, 5000.0)
        _place(sfx, int(frame) * HOP, boom * 0.8 + burst * 0.25, 0.55)
    for frame in events["ticks"]:
        blip = _tone(1800.0, int(0.03 * sr), 0.008)
        _place(sfx, int(frame) * HOP, blip, 0.12)
    chime = _tone(1760.0, int(0.35 * sr), 0.12) + 0.4 * _tone(2217.0, int(0.35 * sr), 0.1)
    _place(sfx, int(events["chime"]) * HOP, chime[: int(0.35 * sr)], 0.2)
    _riser(sfx, rng, 384, 743, sr)
    _riser(sfx, rng, 1152, 1439, sr)
    _whoosh(sfx, rng, 876, 948, sr)
    _reverse(drums, rng, 672, 743, sr)
    _tape(sfx, 1800, sr)
    for bus in (drums, bass, pad, arp, sfx):
        _silence(bus, 744, 768)
    _silence(kick, 744, 768)
    # The lone arp note at 744 is the exception to the silent gap.
    if 744 in arp_set:
        freq = _arp_freq(events, 744)
        n = int(0.08 * sr)
        t = np.arange(n, dtype=np.float64) / sr
        pluck = np.sin(2.0 * math.pi * freq * t) * np.exp(-t / 0.04)
        _place(arp, 744 * HOP, pluck, 0.9)
    ducked = sidechain(bass + pad + arp, events["kicks"], sr)
    musical = drums + ducked
    musical = _sweep_lowpass(musical, 876, 959, 8000.0, 400.0, sr)
    beat = kick + musical
    _silence(beat, 1806, n_frames)
    _silence(sfx, 1806, n_frames)
    return beat, sfx


def mix(events: dict, *, seed: int = 7, n_frames: int = 1824, sr: int = SR) -> tuple[np.ndarray, np.ndarray, float, float]:
    beat, sfx = render_buses(events, seed, n_frames, sr)
    full, lufs, peak = master(beat + sfx, sr, tail_frames=18)
    return full, sfx, lufs, peak
