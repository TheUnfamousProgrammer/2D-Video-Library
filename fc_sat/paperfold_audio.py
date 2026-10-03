"""Paperfold arrangement. Events come from the timeline. The beatkit primitives
are reused and not modified.

The mix is A minor, Am F C G, one chord per bar. A paper whoomp and an arp
note land on every fold. The last 18 frames are left for the master to zero.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fc_sat.audio import write_wav
from fc_sat.beatkit.music import (
    HOP,
    SR,
    _noise,
    _place,
    _riser,
    _shape,
    _supersaw,
    _sweep_lowpass,
    _tape,
    _tone,
    _whoosh,
    kick_wave,
    master,
    sidechain,
)
from fc_sat.paperfold_timeline import CHORDS, MINOR, Timeline, build_timeline

ROOT = Path(__file__).resolve().parents[1]


def _whoomp(rng: np.random.Generator, sr: int = SR) -> np.ndarray:
    n = int(0.15 * sr)
    burst = _shape(_noise(n, rng), sr, 0.08, None, 6000.0)
    thump = _tone(60.0, n, 0.05, sr)
    return burst * 0.7 + thump * 0.8


def _sat(wave: np.ndarray) -> np.ndarray:
    return np.tanh(wave * 1.4)


def _note(freq: float, n: int, sr: int = SR) -> np.ndarray:
    return _supersaw(freq, n, np.random.default_rng(int(freq) or 1), sr)


def _freq(degree: int) -> float:
    semi = MINOR[degree % len(MINOR)] + 12 * (degree // len(MINOR))
    return 220.0 * (2.0 ** (semi / 12.0))


def mix(timeline: Timeline | None = None) -> tuple[np.ndarray, np.ndarray, float, float]:
    timeline = timeline or build_timeline()
    n = timeline.n_frames * HOP
    music = np.zeros((n, 2), np.float64)
    sfx = np.zeros((n, 2), np.float64)
    rng = np.random.default_rng(7)
    kick = kick_wave()
    hat = _shape(_noise(int(0.04 * SR), rng), SR, 0.02, 6000.0, None)
    clap = _shape(_noise(int(0.12 * SR), rng), SR, 0.06, 800.0, 5000.0)
    whoomp = _whoomp(rng)
    for frame in timeline.kicks:
        _place(music, frame * HOP, kick, 0.85)
    for frame in timeline.hats:
        _place(music, frame * HOP, hat, 0.18)
    for frame in timeline.claps:
        _place(music, frame * HOP, clap, 0.35)
    for frame in timeline.bass:
        bar = frame // 96
        root = CHORDS[bar % 4][0]
        freq = 55.0 * (2.0 ** (root / 12.0))
        _place(music, frame * HOP, _sat(_tone(freq, int(0.18 * SR), 0.12)), 0.55)
    for frame in timeline.snares:
        _place(music, frame * HOP, clap, 0.22)
    for frame, degree in zip(timeline.arp, timeline.arp_degrees):
        if 552 <= frame < 576 and frame != 552:
            continue
        _place(sfx, frame * HOP, whoomp, 0.7)
        _place(music, frame * HOP, _note(_freq(degree), int(0.18 * SR)), 0.16)
    for index, frame in enumerate(timeline.bells):
        _place(music, frame * HOP, _tone(880.0 * (2.0 ** (index / 24.0)), int(0.4 * SR), 0.3), 0.12)
    for frame in timeline.ticks:
        _place(sfx, frame * HOP, _tone(1400.0, int(0.05 * SR), 0.02), 0.2)
    _place(sfx, timeline.pickup * HOP, _tone(660.0, int(0.2 * SR), 0.08), 0.25)
    _place(sfx, timeline.thump * HOP, _tone(80.0, int(0.2 * SR), 0.1), 0.4)
    for frame in timeline.impacts:
        _place(sfx, frame * HOP, _tone(50.0, int(0.4 * SR), 0.2), 0.8)
        chord = CHORDS[(frame // 96) % 4]
        stab = np.zeros(int(0.3 * SR))
        for semi in chord:
            stab += _tone(220.0 * (2.0 ** (semi / 12.0)), len(stab), 0.15)
        _place(music, frame * HOP, stab / 3.0, 0.4)
    _riser(music, rng, 384, 576, SR)
    _whoosh(sfx, rng, 560, 590, SR)
    _riser(music, rng, 720, 863, SR)
    _tape(sfx, timeline.tape, SR)
    for bar in range(4, 19):
        start = bar * 96
        if start >= 1800:
            break
        chord = CHORDS[bar % 4]
        pad = np.zeros(96 * HOP)
        for semi in chord:
            pad += _tone(110.0 * (2.0 ** (semi / 12.0)), len(pad), 0.8)
        _place(music, start * HOP, pad / 3.0, 0.08)
    # Bars 8-9 close the filter. Frame 864 is left open: the filter snaps there.
    music = _sweep_lowpass(music, 672, 863, 8000.0, 400.0, SR)
    # Pitch dip on the 792 stamp: a short downward tone on the sfx bus.
    dip = _tone(400.0, int(0.2 * SR), 0.1)
    t = np.linspace(1.0, 0.5, len(dip))
    _place(sfx, 792 * HOP, dip * t, 0.2)
    music = sidechain(music, timeline.kicks, SR, start=864, end=1152)
    if music.shape[0] != n:
        music = music[:n]
        sfx = sfx[:n]
    full, lufs, peak = master(music + sfx, tail_frames=18)
    return full, sfx, float(lufs), float(peak)


def render_audio(out_dir: Path | None = None) -> tuple[Path, Path, float, float]:
    out_dir = out_dir or (ROOT / "out")
    out_dir.mkdir(parents=True, exist_ok=True)
    full, sfx, lufs, peak = mix()
    music_path = out_dir / "paperfold.wav"
    sfx_path = out_dir / "paperfold.sfx.wav"
    write_wav(str(music_path), full, SR)
    sfx_master, _, _ = master(sfx, tail_frames=18)
    write_wav(str(sfx_path), sfx_master, SR)
    return music_path, sfx_path, lufs, peak
