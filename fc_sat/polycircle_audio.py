"""Original A-minor arrangement. Every hit is a timeline event. No borrowed audio."""

from __future__ import annotations

import math

import numpy as np
from scipy.signal import butter, sosfilt

from fc_sat.audio import SR, loudness_loop, true_peak_db

HOP = SR // 60
A2 = 110.0
A4 = 440.0
CHORDS = ((0, 3, 7), (5, 9, 12), (3, 7, 10), (7, 11, 14))
PENTA = tuple(A4 * 2 ** ((semi + 12 * octave) / 12.0) for octave in range(3) for semi in (0, 3, 5, 7, 10))


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


def mix(events: dict, *, seed: int = 7, n_frames: int = 1824, sr: int = SR) -> tuple[np.ndarray, np.ndarray, float, float]:
    beat, sfx = render_buses(events, seed, n_frames, sr)
    full, lufs, peak = master(beat + sfx, sr, tail_frames=18)
    return full, sfx, lufs, peak


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
