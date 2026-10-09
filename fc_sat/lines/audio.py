"""Mix for the lines short: narration, cue-locked SFX (reusing the 37% kit), and a bouncy bed."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

from fc_sat.beatkit.music import master
from fc_sat.lines.cues import Cues
from fc_sat.rule37.audio import SR, Sfx, _bp, _env, _epiano, _hat, _kick, _lp, _midi, _place, _sweep, _t

BPM = 108.0
BEAT = 60.0 / BPM


class LineSfx(Sfx):
    def beep(self, v=0):
        n = int(0.09 * SR)
        f = 2300 + 300 * v
        sq = np.sign(np.sin(2 * np.pi * f * _t(n)))
        return 0.12 * _lp(sq, 6000) * _env(n, 0.001, 0.05)

    def bwomp(self, v=0):
        n = int(0.5 * SR)
        x = _sweep(260, 90, n)
        return 0.45 * np.tanh(2.5 * x) * _env(n, 0.005, 0.18)

    def thunder(self, v=0):
        n = int(1.4 * SR)
        rumble = _lp(self.noise(n), 180, 2) * 6
        return 0.5 * rumble * _env(n, 0.08, 0.5)

    def zap(self, v=0):
        n = int(0.3 * SR)
        crack = _bp(self.noise(n), 1500, 9000) * _env(n, 0.001, 0.04)
        return 0.55 * crack + 0.4 * self.thunder()[:n]

    def car(self, v=0):
        return self._whoosh(0.32, 250, 1800, 0.55)

    def car_fast(self, v=0):
        return self._whoosh(0.16, 600, 4000, 0.5)

    def printer(self, v=0):
        n = int(1.1 * SR)
        t = _t(n)
        clicks = (np.sin(2 * np.pi * 38 * t) > 0.6).astype(float)
        buzz = np.sin(2 * np.pi * 180 * t) * 0.3
        return 0.14 * (_bp(self.noise(n), 2000, 6000) * clicks + buzz) * np.clip(t / 0.02, 0, 1) * np.clip((1.1 - t) / 0.05, 0, 1)


def music(cues: Cues, n: int) -> np.ndarray:
    c = cues.t
    rng = np.random.default_rng(5)
    bus = np.zeros((n, 2))
    road = np.zeros((n, 2))
    # C - Am - F - G, staccato plucks; the road section swaps to a driving bass pulse
    prog = [[60, 64, 67, 72], [57, 60, 64, 69], [53, 57, 60, 65], [55, 59, 62, 67]]
    total = n / SR
    b = 0
    while b * BEAT < total:
        t0 = b * BEAT
        ch = prog[(b // 4) % 4]
        for k in range(2):
            m = ch[(b * 2 + k) % 4] + 12
            _place(bus, t0 + k * BEAT / 2, _epiano(_midi(m), 0.18, 0.035), 1.0, -0.25 if k else 0.25)
        if b % 4 == 0:
            _place(bus, t0, _epiano(_midi(ch[0] - 24), 4 * BEAT * 0.9, 0.11))
        if b % 2 == 0:
            _place(bus, t0, _kick(), 0.8)
        _place(bus, t0 + BEAT / 2, _hat(rng), 1.0, 0.2)
        # road: eighth-note bass + kick every beat
        for k in range(2):
            _place(road, t0 + k * BEAT / 2, _epiano(_midi(ch[0] - 24), 0.2, 0.12))
        _place(road, t0, _kick(), 0.9)
        _place(road, t0 + BEAT / 2, _hat(rng), 1.0, -0.2)
        b += 1
    tt = np.arange(n) / SR
    road_on = (tt >= c["traffic"] - 0.1) & (tt < c["fix"])
    g_bus = (~road_on).astype(float)
    g_bus[(tt >= c["one1"] - 0.03) & (tt < c["one1"] + 0.35)] = 0.0
    g_road = road_on.astype(float)
    g_road[(tt >= c["receipts"] - 0.25) & (tt < c["fix"])] = 0.35
    return bus * g_bus[:, None] + road * g_road[:, None]


def render_mix(cues: Cues, vo_wav: Path, out_wav: Path) -> None:
    n = int(round(cues.n_frames / 60 * SR))
    vo, sr = sf.read(vo_wav)
    assert sr == SR
    vo = vo if vo.ndim == 1 else vo.mean(axis=1)
    vo = vo / (np.abs(vo).max() + 1e-9) * 0.9
    voice = np.zeros((n, 2))
    _place(voice, 0.0, vo[:n], 1.0)
    fx = np.zeros((n, 2))
    gen = LineSfx()
    gains = {"tick": 0.6, "tick_low": 0.6, "beep": 0.9, "pop": 0.8, "car": 0.7, "car_fast": 0.7}
    for t, kind, v in cues.sfx:
        pan = {"car": -0.4, "car_fast": -0.4}.get(kind, 0.0)
        _place(fx, t, getattr(gen, kind)(v), gains.get(kind, 1.0) * 0.8, pan)
    bed = music(cues, n)
    env = _lp(np.abs(voice[:, 0]), 6.0, 1)
    env = np.clip(env / (np.percentile(env, 95) + 1e-9), 0, 1)
    bed *= (1 - 0.55 * env)[:, None]
    mix = voice + fx * 0.42 + bed * 0.42
    out, lufs, tp = master(mix)
    sf.write(out_wav, out.astype(np.float32), SR, subtype="FLOAT")
    print(f"mix {n / SR:.2f}s  {lufs:.1f} LUFS  {tp:.1f} dBTP")
