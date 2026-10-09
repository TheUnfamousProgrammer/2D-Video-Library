"""Mix for the voice short. The bed loses its bass while the script talks about recordings."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

from fc_sat.beatkit.music import master
from fc_sat.rule37.audio import SR, Sfx, _bp, _env, _epiano, _hat, _kick, _lp, _midi, _place, _sweep, _t
from fc_sat.voice.cues import Cues

BPM = 96.0
BEAT = 60.0 / BPM


class VoiceSfx(Sfx):
    def hum(self, v=0):
        n = int(0.9 * SR)
        t = _t(n)
        return 0.35 * np.sin(2 * np.pi * 70 * t) * np.sin(np.pi * t / 0.9) ** 2

    def stamp(self, v=0.6):
        n = int(0.25 * SR)
        thud = _sweep(180, 60, n) * _env(n, 0.002, 0.06)
        snap = _bp(self.noise(n), 1500, 6000) * _env(n, 0.001, 0.015)
        return (0.5 + 0.4 * v) * (0.8 * thud + 0.4 * snap)

    def tap(self, v=0):
        return self.ping(1800, 0.08, 0.25)

    def scratch(self, v=0):
        m = int(0.35 * SR)
        t = _t(m)
        return 0.6 * _bp(self.noise(m), 700, 3200) * (0.5 + 0.5 * np.sin(2 * np.pi * (6 + 22 * t) * t)) * np.sin(np.pi * t / 0.35)

    def air(self, v=0):
        return self._whoosh(0.7, 2500, 9000, 0.35)

    def thump(self, v=0):
        n = int(0.6 * SR)
        return 0.9 * _sweep(90, 40, n) * _env(n, 0.003, 0.2)

    def sub(self, v=0):
        n = int(0.9 * SR)
        return 0.8 * _sweep(60, 42, n) * _env(n, 0.01, 0.35)

    def cut(self, v=0):
        n = int(0.15 * SR)
        return 0.4 * _bp(self.noise(n), 3000, 9000) * _env(n, 0.001, 0.03)

    def boing(self, v=0):
        n = int(0.45 * SR)
        t = _t(n)
        return 0.3 * np.sin(2 * np.pi * (500 + 300 * np.sin(2 * np.pi * 10 * t)) * t) * _env(n, 0.005, 0.15)

    def toggle(self, v=0):
        return self.ping(1200, 0.06, 0.3) + np.pad(self.ping(1600, 0.06, 0.25), (int(0.04 * SR), 0))[: int(0.06 * SR)]


def _hp(x: np.ndarray, f: float) -> np.ndarray:
    return sosfilt(butter(4, f, btype="high", fs=SR, output="sos"), x, axis=0)


def music(cues: Cues, n: int) -> np.ndarray:
    c = cues.t
    rng = np.random.default_rng(9)
    bus = np.zeros((n, 2))
    prog = [[57, 60, 64, 67], [53, 57, 60, 64], [48, 52, 55, 59], [55, 59, 62, 65]]  # Am7 Fmaj7 Cmaj7 G7
    b = 0
    while b * BEAT < n / SR:
        t0 = b * BEAT
        ch = prog[(b // 4) % 4]
        if b % 4 == 0:
            for m in ch:
                _place(bus, t0, _epiano(_midi(m), 4 * BEAT * 0.95, 0.045))
            _place(bus, t0, _epiano(_midi(ch[0] - 24), 4 * BEAT * 0.9, 0.16))  # deep bass note
        for k in range(2):
            m = ch[(b * 2 + k) % 4] + 12
            _place(bus, t0 + k * BEAT / 2, _epiano(_midi(m), 0.3, 0.028), 1.0, 0.3 if k else -0.3)
        if b % 2 == 0:
            _place(bus, t0, _kick(), 0.85)
        _place(bus, t0 + BEAT / 2, _hat(rng), 1.0, 0.2)
        b += 1
    thin = _hp(bus, 700.0)
    tt = np.arange(n) / SR
    # thin from "No bass boost" until the filter toggle, and again on "unfiltered"
    k = np.clip((tt - c["no"]) / 0.15, 0, 1) * (1 - np.clip((tt - c["filter"]) / 0.15, 0, 1))
    k = np.maximum(k, np.clip((tt - c["unfiltered"]) / 0.1, 0, 1) * (1 - np.clip((tt - c["so4"]) / 0.3, 0, 1)))
    out = bus * (1 - k)[:, None] + thin * 1.4 * k[:, None]
    out[(tt >= c["yeah"] - 0.05) & (tt < c["yeah"] + 0.45)] = 0.0  # freeze-frame silence for the scratch
    out[(tt >= c["plot"] - 0.05) & (tt < c["plot"] + 0.3)] *= 0.2
    return out


def render_mix(cues: Cues, vo_wav: Path, out_wav: Path) -> None:
    n = int(round(cues.n_frames / 60 * SR))
    vo, sr = sf.read(vo_wav)
    assert sr == SR
    vo = vo if vo.ndim == 1 else vo.mean(axis=1)
    vo = vo / (np.abs(vo).max() + 1e-9) * 0.9
    voice = np.zeros((n, 2))
    _place(voice, 0.0, vo[:n], 1.0)
    fx = np.zeros((n, 2))
    gen = VoiceSfx()
    gains = {"tick": 0.6, "pop": 0.8, "sub": 0.9}
    for t, kind, v in cues.sfx:
        _place(fx, t, getattr(gen, kind)(v), gains.get(kind, 1.0) * 0.8)
    bed = music(cues, n)
    env = _lp(np.abs(voice[:, 0]), 6.0, 1)
    env = np.clip(env / (np.percentile(env, 95) + 1e-9), 0, 1)
    bed *= (1 - 0.55 * env)[:, None]
    mix = voice + fx * 0.42 + bed * 0.42
    out, lufs, tp = master(mix)
    sf.write(out_wav, out.astype(np.float32), SR, subtype="FLOAT")
    print(f"mix {n / SR:.2f}s  {lufs:.1f} LUFS  {tp:.1f} dBTP")
