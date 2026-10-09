"""Mix: narration on top, cue-locked SFX, and a lo-fi bed that ducks under the voice."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

from fc_sat.beatkit.music import master
from fc_sat.rule37.cues import Cues

SR = 48000
BPM = 100.0
BEAT = 60.0 / BPM


def _t(n: int) -> np.ndarray:
    return np.arange(n) / SR


def _env(n: int, attack: float, decay: float) -> np.ndarray:
    t = _t(n)
    a = np.clip(t / max(attack, 1e-4), 0, 1)
    return a * np.exp(-np.maximum(t - attack, 0) / decay)


def _bp(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return sosfilt(butter(2, [lo, hi], btype="band", fs=SR, output="sos"), x)


def _lp(x: np.ndarray, f: float, order: int = 2) -> np.ndarray:
    return sosfilt(butter(order, f, btype="low", fs=SR, output="sos"), x, axis=0)


def _sweep(f0: float, f1: float, n: int) -> np.ndarray:
    f = f0 * (f1 / f0) ** (np.arange(n) / n)
    return np.sin(2 * np.pi * np.cumsum(f) / SR)


class Sfx:
    def __init__(self, seed: int = 37) -> None:
        self.rng = np.random.default_rng(seed)

    def noise(self, n: int) -> np.ndarray:
        return self.rng.standard_normal(n)

    def _whoosh(self, dur=0.3, lo=500, hi=4500, g=0.5):
        n = int(dur * SR)
        x = _bp(self.noise(n), lo, hi)
        return g * x * np.sin(np.pi * np.linspace(0, 1, n)) ** 2 / 3

    def whoosh(self, v=0):
        return self._whoosh()

    def swipe(self, v):
        return self._whoosh(0.2, 900 + 600 * v, 6000, 0.55 - 0.25 * v)

    def slam(self, v=0):
        n = int(0.9 * SR)
        boom = _sweep(110, 38, n) * _env(n, 0.003, 0.28)
        hit = _bp(self.noise(n), 150, 3000) * _env(n, 0.001, 0.05)
        return 0.9 * boom + 0.35 * hit

    def ping(self, f, dur=0.5, g=0.3):
        n = int(dur * SR)
        t = _t(n)
        return g * (np.sin(2 * np.pi * f * t) + 0.3 * np.sin(2 * np.pi * 2.01 * f * t)) * _env(n, 0.002, dur / 4)

    def sparkle(self, v=0):
        out = np.zeros(int(0.9 * SR))
        for k, f in enumerate((2093, 2637, 3136, 3951, 3520)):
            p = self.ping(f, 0.5, 0.12)
            s = int(k * 0.07 * SR)
            out[s:s + len(p)] += p[: len(out) - s]
        return out

    def bonk(self, v=0):
        n = int(0.22 * SR)
        x = _sweep(420, 160, n)
        return 0.5 * np.tanh(3 * x) * _env(n, 0.002, 0.06)

    def pop(self, v):
        n = int(0.09 * SR)
        f0 = 500 + 700 * v
        return 0.4 * _sweep(f0 * 1.6, f0, n) * _env(n, 0.001, 0.025)

    def scribble(self, v=0):
        n = int(0.5 * SR)
        x = _bp(self.noise(n), 2500, 7000)
        am = 0.5 + 0.5 * np.sin(2 * np.pi * 13 * _t(n)) ** 2
        return 0.25 * x * am * np.sin(np.pi * np.linspace(0, 1, n))

    def step(self, v):
        n = int(0.06 * SR)
        return 0.25 * np.sin(2 * np.pi * 700 * _t(n)) * _env(n, 0.001, 0.012)

    def gate(self, v=0):
        n = int(0.8 * SR)
        t = _t(n)
        clank = sum(a * np.sin(2 * np.pi * f * t) for f, a in ((220, 1), (563, 0.7), (1187, 0.5), (1730, 0.35)))
        return 0.3 * clank * _env(n, 0.001, 0.16) + 0.36 * self.slam()[:n]

    def lock(self, v=0):
        a = self.ping(1800, 0.05, 0.4)
        out = np.zeros(int(0.12 * SR))
        out[: len(a)] += a
        out[int(0.05 * SR): int(0.05 * SR) + len(a)] += a * 0.8
        return out

    def grass(self, v=0):
        n = int(0.5 * SR)
        t = _t(n)
        boing = np.sin(2 * np.pi * (300 + 200 * t / 0.5) * t + 3 * np.sin(2 * np.pi * 9 * t)) * _env(n, 0.01, 0.2)
        return 0.3 * boing + self._whoosh(0.5, 2000, 8000, 0.4)

    def grass_back(self, v=0):
        return self.grass()[::-1] * 0.6

    def tick(self, v):
        n = int(0.04 * SR)
        return 0.22 * np.sin(2 * np.pi * (1100 + 1300 * v) * _t(n)) * _env(n, 0.0008, 0.01)

    def tick_low(self, v):
        n = int(0.04 * SR)
        return 0.14 * np.sin(2 * np.pi * (500 + 400 * v) * _t(n)) * _env(n, 0.0008, 0.01)

    def tick_soft(self, v):
        return self.tick(v) * 0.4

    def record(self, v):
        return self.ping(1400 + 1800 * v, 0.25, 0.22)

    def ding(self, v=0):
        n = int(1.6 * SR)
        t = _t(n)
        out = np.zeros(n)
        for f in (523.25, 659.25, 783.99, 1046.5):
            for ratio, a in ((1, 1), (2.0, 0.4), (3.01, 0.2), (4.2, 0.1)):
                out += a * np.sin(2 * np.pi * f * ratio * t) * np.exp(-t / (0.5 / ratio))
        return 0.08 * out * np.clip(t / 0.003, 0, 1)

    def fade_down(self, v=0):
        n = int(0.45 * SR)
        return 0.18 * _sweep(700, 300, n) * _env(n, 0.01, 0.15)

    def wobble(self, v=0):
        n = int(0.9 * SR)
        t = _t(n)
        return 0.16 * np.sin(2 * np.pi * 440 * t + 6 * np.sin(2 * np.pi * 7 * t)) * np.sin(np.pi * t / 0.9)

    def crown(self, v=0):
        out = self.sparkle()
        d = self.ding()[: len(out)] * 0.6
        out[: len(d)] += d
        return out

    def riser(self, v=0):
        n = int(0.85 * SR)
        x = _bp(self.noise(n), 800, 6000) * np.linspace(0, 1, n) ** 2 * 0.12
        return x + 0.12 * _sweep(300, 1200, n) * np.linspace(0, 1, n)

    def slide(self, v=0):
        return self._whoosh(1.2, 300, 2500, 0.45)

    def confetti(self, v=0):
        out = self.sparkle()
        for k in range(6):
            p = self.pop(self.rng.uniform(0.3, 1.0))
            s = int(self.rng.uniform(0, 0.25) * SR)
            out[s:s + len(p)] += p * 0.7
        return out

    def drop(self, v=0):
        n = int(1.0 * SR)
        boom = _sweep(70, 30, n) * _env(n, 0.005, 0.4)
        # record scratch
        m = int(0.32 * SR)
        t = _t(m)
        scratch = _bp(self.noise(m), 700, 3000) * (0.5 + 0.5 * np.sin(2 * np.pi * (6 + 18 * t) * t)) * np.sin(np.pi * t / 0.32)
        out = 0.9 * boom
        out[:m] += 0.45 * scratch
        return out

    def sad(self, v=0):
        notes = [(233.08, 0.2), (220.0, 0.2), (207.65, 0.2), (196.0, 0.6)]
        out = []
        for k, (f, d) in enumerate(notes):
            n = int(d * SR)
            t = _t(n)
            vib = 4 * np.sin(2 * np.pi * 6 * t) * (k == 3)
            ph = 2 * np.pi * np.cumsum(f + vib) / SR
            saw = sum(np.sin(h * ph) / h for h in range(1, 9))
            env = np.clip(t / 0.02, 0, 1) * np.clip((d - t) / 0.06, 0, 1)
            out.append(saw * env)
        x = np.concatenate(out)
        return 0.28 * _lp(x, 1400)

    def crack(self, v=0):
        n = int(0.25 * SR)
        out = 0.4 * _bp(self.noise(n), 1500, 8000) * _env(n, 0.001, 0.03)
        b = self.bonk()
        out[: len(b)] += 0.15 * b
        return out

    def settle(self, v=0):
        return self.ping(1318.5, 0.6, 0.12)

    def whoosh_soft(self, v=0):
        return self._whoosh(0.35, 400, 3000, 0.35)


def _place(bus: np.ndarray, t: float, wave: np.ndarray, gain: float = 1.0, pan: float = 0.0) -> None:
    s = int(round(t * SR))
    if s < 0:
        wave = wave[-s:]
        s = 0
    e = min(len(bus), s + len(wave))
    if e <= s:
        return
    bus[s:e, 0] += wave[: e - s] * gain * (1 - max(pan, 0))
    bus[s:e, 1] += wave[: e - s] * gain * (1 + min(pan, 0))


def _epiano(f: float, dur: float, g: float) -> np.ndarray:
    n = int(dur * SR)
    t = _t(n)
    x = np.sin(2 * np.pi * f * t + 0.8 * np.sin(2 * np.pi * f * t) * np.exp(-t / 0.3))
    x += 0.15 * np.sin(2 * np.pi * 2 * f * t)
    return g * x * _env(n, 0.004, dur / 3) * np.clip((dur - t) / 0.05, 0, 1)


def _kick() -> np.ndarray:
    n = int(0.25 * SR)
    return 0.7 * _sweep(120, 45, n) * _env(n, 0.002, 0.08)


def _hat(rng) -> np.ndarray:
    n = int(0.05 * SR)
    return 0.08 * sosfilt(butter(2, 7000, btype="high", fs=SR, output="sos"), rng.standard_normal(n)) * _env(n, 0.0005, 0.012)


def _midi(m: float) -> float:
    return 440.0 * 2 ** ((m - 69) / 12)


def music(cues: Cues, n: int) -> np.ndarray:
    c = cues.t
    rng = np.random.default_rng(3)
    bright = np.zeros((n, 2))
    dark = np.zeros((n, 2))
    # Fmaj7 - Em7 - Dm7 - Cmaj7, one chord per bar
    prog = [[53, 57, 60, 64], [52, 55, 59, 62], [50, 53, 57, 60], [48, 52, 55, 59]]
    minor = [[45, 48, 52, 55], [41, 45, 48, 52], [38, 41, 45, 48], [40, 44, 47, 50]]
    bar = 4 * BEAT
    total = n / SR
    beat_i = 0
    while beat_i * BEAT < total:
        t0 = beat_i * BEAT
        chord = prog[(beat_i // 4) % 4]
        if beat_i % 4 == 0:
            for m in chord:
                _place(bright, t0, _epiano(_midi(m), bar * 0.95, 0.05), 1.0, 0.0)
            _place(bright, t0, _epiano(_midi(chord[0] - 12), bar * 0.9, 0.09))
        for k in range(2):  # eighth-note arpeggio
            m = chord[(beat_i * 2 + k) % 4] + 12
            _place(bright, t0 + k * BEAT / 2, _epiano(_midi(m), 0.35, 0.03), 1.0, 0.3 if k else -0.3)
        if beat_i % 2 == 0:
            _place(bright, t0, _kick(), 0.8)
        _place(bright, t0 + BEAT / 2, _hat(rng), 1.0, 0.2)
        mch = minor[(beat_i // 4) % 4]
        if beat_i % 4 == 0:
            for m in mch:
                _place(dark, t0, _epiano(_midi(m), bar * 0.98, 0.06))
            _place(dark, t0, _epiano(_midi(mch[0] - 12), bar * 0.95, 0.1))
        beat_i += 1
    dark = _lp(dark, 900)
    # sections
    tt = np.arange(n) / SR
    g_b = np.ones(n)
    g_d = np.zeros(n)
    on, ev = c["on"] - 0.03, c["even"] - 0.15
    g_b[(tt >= on) & (tt < ev)] = 0.0  # silence under the 37% slam
    stop0, stop1 = c["but2"], c["also"] - 0.2
    g_b[tt >= stop0] = 0.0
    g_d = np.clip((tt - stop1) / 0.6, 0, 1) * (tt < c["so2"])
    g_b = np.maximum(g_b, np.clip((tt - c["so2"]) / 0.4, 0, 1))
    out = bright * g_b[:, None] + dark * g_d[:, None]
    # tape stop on the catch
    s0, L = int(stop0 * SR), int(0.45 * SR)
    seg_src = bright[s0 - L: s0 + L]
    pos = np.cumsum(np.linspace(1, 0, L))  # read head slows to zero
    idx = np.clip(pos.astype(int), 0, len(seg_src) - 1)
    out[s0:s0 + L] += seg_src[idx] * np.linspace(1, 0, L)[:, None] ** 0.5
    return out


def render_mix(cues: Cues, vo_wav: Path, out_wav: Path) -> None:
    n = int(round(cues.n_frames / 60 * SR))
    vo, sr = sf.read(vo_wav)
    assert sr == SR
    vo = vo if vo.ndim == 1 else vo.mean(axis=1)
    vo = vo / (np.abs(vo).max() + 1e-9) * 0.9
    voice = np.zeros((n, 2))
    lead = 0.0
    _place(voice, lead, vo[: n], 1.0)
    fx = np.zeros((n, 2))
    gen = Sfx()
    gains = {"tick": 0.7, "tick_low": 0.7, "tick_soft": 0.7, "record": 0.8, "swipe": 1.0, "pop": 0.8}
    for t, kind, v in cues.sfx:
        wave = getattr(gen, kind)(v)
        pan = 0.0
        if kind in ("swipe",):
            pan = -0.35
        _place(fx, t, wave, gains.get(kind, 1.0) * 0.8, pan)
    bed = music(cues, n)
    # duck the bed under the voice
    env = np.abs(voice[:, 0])
    env = _lp(env, 6.0, 1)
    env = np.clip(env / (np.percentile(env, 95) + 1e-9), 0, 1)
    duck = 1 - 0.55 * env
    bed *= duck[:, None]
    mix = voice * 1.0 + fx * 0.42 + bed * 0.42
    out, lufs, tp = master(mix)
    sf.write(out_wav, out.astype(np.float32), SR, subtype="FLOAT")
    print(f"mix {n / SR:.2f}s  {lufs:.1f} LUFS  {tp:.1f} dBTP")
