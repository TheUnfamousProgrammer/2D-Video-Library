"""Original 150 bpm score. Every hit is synthesized and lands on the frame grid."""

from __future__ import annotations

import math

import numpy as np
from scipy.signal import butter, lfilter, sosfilt, sosfiltfilt

from fc_sat.audio import SR, loudness_loop
from fc_sat.beatkit.grid import HOP

N_FRAMES = 1824
CHORDS = ("Am", "F", "C", "G")
ROOTS = {"Am": 55.00, "F": 43.6535, "C": 65.4064, "G": 48.9994}
PENTA = (69, 72, 74, 76, 79)
MOTIFS = (
    (0, -1, 2, 4, -1, 2, 0, -1, 3, -1, 4, 2, -1, 1, 0, -1),
    (1, -1, 3, -1, 0, 1, -1, 4, -1, 2, 3, -1, 1, 0, -1, 2),
    (2, 0, -1, 4, -1, 3, 2, -1, 0, -1, 1, 4, -1, 2, 0, -1),
    (4, -1, 0, 2, -1, 4, -1, 1, 3, -1, 0, -1, 2, 4, -1, 0),
)


def samples(n_frames: int = N_FRAMES) -> int:
    return int(n_frames) * HOP


def silence() -> np.ndarray:
    return np.zeros((samples(), 2), dtype=np.float64)


def _place(bus: np.ndarray, start: int, wave: np.ndarray, gain: float = 1.0, pan: float = 0.0) -> None:
    if wave.size == 0 or gain == 0.0:
        return
    start = int(start)
    if start < 0:
        wave = wave[-start:]
        start = 0
    end = min(len(bus), start + len(wave))
    span = end - start
    if span <= 0:
        return
    left = gain * (1.0 - max(pan, 0.0))
    right = gain * (1.0 + min(pan, 0.0))
    bus[start:end, 0] += wave[:span] * left
    bus[start:end, 1] += wave[:span] * right


def _midi(note: float) -> float:
    return 440.0 * 2.0 ** ((note - 69) / 12.0)


def _kick(muffled: bool = False) -> np.ndarray:
    n = int(0.16 * SR)
    t = np.arange(n, dtype=np.float64) / SR
    body = np.sin(2.0 * math.pi * 55.0 * t) * np.exp(-t / 0.055)
    if muffled:
        body = sosfilt(butter(2, 200.0, btype="low", fs=SR, output="sos"), body)
    click = np.exp(-t / 0.0011)
    click[0] = 1.0
    return body * 0.85 + click * 0.8


def _clap(rng: np.random.Generator) -> np.ndarray:
    n = int(0.12 * SR)
    burst = int(0.010 * SR)
    wave = np.zeros(n, dtype=np.float64)
    sos = butter(2, [900.0, 2400.0], btype="band", fs=SR, output="sos")
    for hit in range(3):
        noise = rng.standard_normal(burst)
        shaped = sosfilt(sos, noise)
        peak = np.max(np.abs(shaped)) or 1.0
        env = np.linspace(0.0, 1.0, min(48, burst)) 
        shaped[: len(env)] *= env
        shaped *= np.exp(-np.arange(burst) / (0.012 * SR))
        start = hit * burst
        wave[start : start + burst] += shaped / peak
    return wave


def _hat(rng: np.random.Generator, velocity: float) -> np.ndarray:
    n = int(0.025 * SR)
    noise = rng.standard_normal(n)
    wave = sosfilt(butter(2, 7000.0, btype="high", fs=SR, output="sos"), noise)
    peak = np.max(np.abs(wave)) or 1.0
    env = np.linspace(0.0, 1.0, 24)
    wave[: len(env)] *= env
    wave *= np.exp(-np.arange(n) / (0.008 * SR))
    return wave / peak * velocity


def _snare(rng: np.random.Generator) -> np.ndarray:
    n = int(0.09 * SR)
    noise = rng.standard_normal(n)
    wave = sosfilt(butter(2, [180.0, 4000.0], btype="band", fs=SR, output="sos"), noise)
    peak = np.max(np.abs(wave)) or 1.0
    tone = np.sin(2.0 * math.pi * 180.0 * np.arange(n) / SR) * np.exp(-np.arange(n) / (0.03 * SR))
    env = np.linspace(0.0, 1.0, 16)
    wave[: len(env)] *= env
    return (wave / peak + 0.35 * tone) * np.exp(-np.arange(n) / (0.04 * SR))


def _cowbell(note: float) -> np.ndarray:
    n = int(0.12 * SR)
    t = np.arange(n, dtype=np.float64) / SR
    low = _midi(note)
    high = low * (800.0 / 540.0)
    square = np.sign(np.sin(2.0 * math.pi * low * t)) + 0.7 * np.sign(np.sin(2.0 * math.pi * high * t + 0.3))
    wave = sosfilt(butter(2, [700.0, 2500.0], btype="band", fs=SR, output="sos"), square)
    peak = np.max(np.abs(wave)) or 1.0
    return wave / peak * np.exp(-t / 0.045)


def _pad_note(freqs: tuple[float, ...], n: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n, dtype=np.float64) / SR
    wave = np.zeros(n, dtype=np.float64)
    for freq in freqs:
        for cents in (-8.0, 0.0, 8.0):
            partial_sum = np.zeros(n, dtype=np.float64)
            f = freq * 2.0 ** (cents / 1200.0)
            phase = float(rng.random()) * 2.0 * math.pi
            for partial in range(1, 7):
                partial_sum += (1.0 / partial) * np.sin(2.0 * math.pi * f * partial * t + phase)
            wave += partial_sum
    peak = np.max(np.abs(wave)) or 1.0
    env = np.ones(n)
    attack = min(n, int(0.04 * SR))
    release = min(n // 3, int(0.08 * SR))
    env[:attack] = np.linspace(0.0, 1.0, attack)
    if release:
        env[-release:] *= np.linspace(1.0, 0.0, release)
    wave = sosfilt(butter(2, 1800.0, btype="low", fs=SR, output="sos"), wave / peak)
    return wave * env


def _bass(root: float, dip: tuple[int, int] | None = None) -> np.ndarray:
    n = int(0.95 * SR)
    t = np.arange(n, dtype=np.float64) / SR
    glide = np.clip(t / 0.055, 0.0, 1.0)
    freq = 90.0 * (root / 90.0) ** glide
    if dip is not None:
        start, length = dip
        end = min(n, start + length)
        if end > start:
            bend = np.sin(np.linspace(0.0, math.pi, end - start))
            freq[start:end] *= 2.0 ** (-3.0 * bend / 12.0)
    phase = 2.0 * math.pi * np.cumsum(freq) / SR
    env = np.exp(-t / 0.5)
    return np.tanh(3.0 * np.sin(phase) * env)


def _pluck(note: float) -> np.ndarray:
    n = int(0.18 * SR)
    t = np.arange(n, dtype=np.float64) / SR
    freq = _midi(note)
    wave = np.sin(2.0 * math.pi * freq * t) * np.exp(-t / 0.05)
    wave += 0.25 * np.sin(2.0 * math.pi * freq * 2 * t) * np.exp(-t / 0.03)
    return wave


def _scratch(rng: np.random.Generator, pitch: float) -> np.ndarray:
    n = int(0.060 * SR)
    noise = rng.standard_normal(n)
    low = max(80.0, 2000.0 * pitch)
    high = min(SR * 0.45, 6000.0 * pitch)
    if high <= low + 50:
        high = low + 200
    wave = sosfilt(butter(2, [low, high], btype="band", fs=SR, output="sos"), noise)
    peak = np.max(np.abs(wave)) or 1.0
    env = np.linspace(0.0, 1.0, 24)
    wave[: len(env)] *= env
    wave *= np.exp(-np.arange(n) / (0.018 * SR))
    return wave / peak


def _whoosh(rng: np.random.Generator, n: int, upward: bool) -> np.ndarray:
    noise = rng.standard_normal(n)
    out = np.zeros(n, dtype=np.float64)
    hop = HOP
    state = None
    frames = max(1, n // hop)
    for frame in range(frames):
        u = frame / max(1, frames - 1)
        if not upward:
            u = 1.0 - u
        center = 300.0 * (2.0 ** (u * 4.0))
        low = max(40.0, center * 0.7)
        high = min(SR * 0.45, center * 1.5)
        sos = butter(2, [low, high], btype="band", fs=SR, output="sos")
        sl = slice(frame * hop, min(n, (frame + 1) * hop))
        if state is None or state.shape[0] != sos.shape[0]:
            state = np.zeros((sos.shape[0], 2), dtype=np.float64)
        filtered, state = sosfilt(sos, noise[sl], zi=state)
        out[sl] = filtered[:, 0] if filtered.ndim > 1 else filtered
    peak = np.max(np.abs(out)) or 1.0
    env = np.sin(np.linspace(0.0, math.pi, n))
    return out / peak * env


def _impact(rng: np.random.Generator) -> np.ndarray:
    n = int(0.45 * SR)
    t = np.arange(n, dtype=np.float64) / SR
    boom = np.sin(2.0 * math.pi * 50.0 * t) * np.exp(-t / 0.18)
    noise = rng.standard_normal(int(0.08 * SR))
    burst = sosfilt(butter(2, [200.0, 5000.0], btype="band", fs=SR, output="sos"), noise)
    peak = np.max(np.abs(burst)) or 1.0
    burst = burst / peak * np.exp(-np.arange(len(burst)) / (0.02 * SR))
    wave = boom
    wave[: len(burst)] += burst * 0.85
    return wave


def _riser(rng: np.random.Generator, n: int) -> np.ndarray:
    noise = _whoosh(rng, n, True)
    t = np.arange(n, dtype=np.float64) / SR
    freq = 110.0 * (2.0 ** (t / max(t[-1], 1e-6) * 2.0))
    saw = np.tanh(2.0 * np.sin(2.0 * math.pi * np.cumsum(freq) / SR))
    env = (t / max(t[-1], 1e-6)) ** 1.4
    return (noise * 0.7 + saw * 0.3) * env


def _chord_tones(name: str) -> tuple[float, ...]:
    roots = {"Am": (45, 52, 57, 60), "F": (41, 48, 53, 57), "C": (48, 55, 60, 64), "G": (43, 50, 55, 59)}
    return tuple(_midi(note) for note in roots[name])


def _sweep_lowpass(bus: np.ndarray, frame0: int, frame1: int, cut0: float, cut1: float) -> None:
    start = frame0 * HOP
    end = min(len(bus), frame1 * HOP)
    original = bus[start:end].copy()
    state = None
    for frame in range(frame0, frame1):
        u = (frame - frame0) / max(1, frame1 - frame0 - 1)
        cutoff = min(SR * 0.45, cut0 * ((cut1 / max(cut0, 1.0)) ** u))
        sos = butter(2, max(40.0, cutoff), btype="low", fs=SR, output="sos")
        sl = slice(frame * HOP, min(len(bus), (frame + 1) * HOP))
        if state is None or state.shape[0] != sos.shape[0]:
            state = np.zeros((sos.shape[0], 2, bus.shape[1]), dtype=np.float64)
        filtered, state = sosfilt(sos, bus[sl], axis=0, zi=state)
        bus[sl] = filtered
    fade = min(int(0.01 * SR), max(1, (end - start) // 4))
    ramp = np.linspace(0.0, 1.0, fade)[:, None]
    bus[start : start + fade] = original[:fade] * (1.0 - ramp) + bus[start : start + fade] * ramp
    bus[end - fade : end] = original[-fade:] * ramp + bus[end - fade : end] * (1.0 - ramp)


def _comb(mono: np.ndarray, delay: int, feedback: float) -> np.ndarray:
    coeff = np.zeros(delay + 1, dtype=np.float64)
    coeff[0] = 1.0
    coeff[delay] = -feedback
    return lfilter([1.0], coeff, mono)


def _reverb(mono: np.ndarray) -> np.ndarray:
    wet = np.zeros_like(mono)
    for delay_s in (0.0297, 0.0371, 0.0411, 0.0437):
        delay = int(delay_s * SR)
        feedback = 10.0 ** (-3.0 * delay / (1.2 * SR))
        wet += _comb(mono, delay, feedback)
    peak = np.max(np.abs(wet)) or 1.0
    return wet / peak


def sidechain(bus: np.ndarray, kicks: list[int]) -> np.ndarray:
    """Drop 6 dB over 10 ms when a kick lands, then release over 120 ms."""
    env = np.ones(len(bus), dtype=np.float64)
    floor = 10.0 ** (-6.0 / 20.0)
    attack = int(0.010 * SR)
    release = int(0.120 * SR)
    for frame in kicks:
        sample = int(frame) * HOP
        if sample >= len(env):
            continue
        attack_end = min(len(env), sample + attack)
        env[sample:attack_end] = np.minimum(env[sample:attack_end], np.linspace(1.0, floor, max(1, attack_end - sample)))
        release_end = min(len(env), attack_end + release)
        if release_end > attack_end:
            env[attack_end:release_end] = np.minimum(
                env[attack_end:release_end], np.linspace(floor, 1.0, release_end - attack_end)
            )
    return bus * env[:, None]


def _kick_frames() -> list[int]:
    frames = [0, 48, 96, 144]
    frames += list(range(192, 744, 24))
    frames += list(range(768, 1344, 24))
    frames += list(range(1440, 1800, 24))
    return frames


def _compress(audio: np.ndarray) -> np.ndarray:
    power = np.mean(audio * audio, axis=1) + 1e-12
    attack = math.exp(-1.0 / (0.040 * SR))
    release = math.exp(-1.0 / (0.250 * SR))
    env = 0.0
    gains = np.ones(len(audio), dtype=np.float64)
    threshold = 10.0 ** (-12.0 / 20.0)
    for index, value in enumerate(np.sqrt(power)):
        coeff = attack if value > env else release
        env = coeff * env + (1.0 - coeff) * value
        if env > threshold:
            desired = threshold * (env / threshold) ** 0.5
            gains[index] = desired / env
    return audio * gains[:, None]


def _tape_stop(audio: np.ndarray) -> None:
    start = 1800 * HOP
    n = int(0.1 * SR)
    src = audio[start : start + int(0.2 * SR)].copy()
    if len(src) < 8:
        audio[start:] = 0.0
        return
    speed = np.cos(np.linspace(0.0, math.pi / 2.0, n)) ** 2
    pos = np.cumsum(speed)
    pos *= min(len(src) - 2, int(0.05 * SR)) / max(float(pos[-1]), 1e-9)
    index = np.clip(pos.astype(np.int32), 0, len(src) - 2)
    frac = (pos - index)[:, None]
    out = src[index] * (1.0 - frac) + src[index + 1] * frac
    out *= np.linspace(1.0, 0.0, n)[:, None]
    audio[start : start + n] = out
    audio[start + n :] = 0.0


# AAC spreads a transient about 50 ms. The measured silent beat is frames 744–767,
# so the wav goes quiet earlier and the drop waits a few milliseconds.
_SILENCE_PRE = int(0.070 * SR)
_SILENCE_POST = int(0.030 * SR)


def _cut_silence(audio: np.ndarray) -> None:
    fade = int(0.005 * SR)
    start = 744 * HOP - _SILENCE_PRE
    end = 768 * HOP + _SILENCE_POST
    if start - fade > 0:
        audio[start - fade : start] *= np.linspace(1.0, 0.0, fade)[:, None]
    audio[start:end] = 0.0
    if end + fade <= len(audio):
        audio[end : end + fade] *= np.linspace(0.0, 1.0, fade)[:, None]


def _master(audio: np.ndarray) -> tuple[np.ndarray, float, float]:
    x = np.array(audio, dtype=np.float64, copy=True)
    x -= np.mean(x, axis=0, keepdims=True)
    x = sosfilt(butter(2, 30.0, btype="high", fs=SR, output="sos"), x, axis=0)
    x = _compress(x)
    mastered, lufs, peak = loudness_loop(x, SR, ceiling=10.0 ** (-2.3 / 20.0))
    mastered = np.array(mastered[: samples()], dtype=np.float64, copy=True)
    if len(mastered) < samples():
        padded = np.zeros((samples(), 2), dtype=np.float64)
        padded[: len(mastered)] = mastered
        mastered = padded
    _cut_silence(mastered)
    mastered[1806 * HOP :] = 0.0
    return mastered, float(lufs), float(peak)


def arrangement(fit, plan, seed: int = 7) -> dict[str, np.ndarray]:
    """Instrument stems, the full mix, and the effects-only mix, before the master."""
    rng = np.random.Generator(np.random.PCG64(int(seed) + 4001))
    drums = silence()
    bass = silence()
    lead = silence()
    harmony = silence()
    fx = silence()
    scratches = silence()
    whooshes = silence()
    impact = silence()
    kicks = _kick_frames()
    kick_wave = _kick(False)
    muffled = _kick(True)
    clap_wave = _clap(rng)
    for frame in kicks:
        wave = muffled if frame < 192 else kick_wave
        _place(drums, frame * HOP, wave, 0.62 if frame >= 768 else 0.34, 0.0)
    for frame in range(192, 1800, 96):
        if 744 <= frame < 768 or 1344 <= frame < 1440:
            continue
        for offset in (24, 72):
            if frame + offset >= 1794 or 744 <= frame + offset < 768:
                continue
            _place(drums, (frame + offset) * HOP, clap_wave, 0.32, 0.08)
    hat_frames = list(range(192, 384, 12)) + list(range(384, 744, 6)) + list(range(768, 1344, 6)) + list(range(1440, 1800, 6))
    for frame in hat_frames:
        roll = frame % 96 >= 72 and frame >= 768 and not 1344 <= frame < 1440
        if roll and (frame - 768) % 3 != 0 and frame < 1440:
            continue
        velocity = 0.45 + 0.55 * float(rng.random())
        pan = 0.3 if frame % 12 == 0 else -0.3
        _place(drums, frame * HOP, _hat(rng, velocity), 0.11 if frame < 768 else 0.16, pan)
    snare = _snare(rng)
    frame = 576
    while frame < 744:
        _place(drums, frame * HOP, snare, 0.16 + 0.14 * ((frame - 576) / 168.0), 0.0)
        gap = max(3, int(round(12 - 9 * ((frame - 576) / 167.0))))
        frame += gap
    for bar in range(19):
        name = CHORDS[bar % 4]
        start = bar * 96
        if start < 192 or start >= 1800 or 1344 <= start < 1440:
            if not (1344 <= start < 1440):
                pass
        pad_gain = 0.07 if start < 192 else 0.11
        if start < 1800 and not (744 <= start < 768):
            _place(harmony, start * HOP, _pad_note(_chord_tones(name), 96 * HOP, rng), pad_gain, 0.0)
        if 768 <= start < 1344 or 1440 <= start < 1800:
            dip = None
            if start == 864:
                dip = ((940 - 864) * HOP, 8 * HOP)
            _place(bass, start * HOP, _bass(ROOTS[name], dip), 0.72, 0.0)
            if 960 <= start < 1344 and bar + 1 < 19:
                nxt = ROOTS[CHORDS[(bar + 1) % 4]]
                slide = _bass((ROOTS[name] + nxt) * 0.5)
                _place(bass, (start + 48) * HOP, slide, 0.28, 0.0)
        if start < 192 or start >= 1794:
            continue
        motif = MOTIFS[bar % 4]
        octave = 12 if 960 <= start < 1344 else 0
        gain = 0.05 if 1344 <= start < 1440 else 0.16
        for slot, degree in enumerate(motif):
            if degree < 0:
                continue
            hit = start + slot * 6
            if hit >= 1794 or 744 <= hit < 768:
                continue
            if 1344 <= start < 1440 and slot % 2:
                continue
            _place(lead, hit * HOP, _cowbell(PENTA[degree] + octave), gain, 0.0)
    _sweep_lowpass(lead, 192, 744, 600.0, 3000.0)
    for index, frame in enumerate(range(0, 192, 12)):
        _place(fx, frame * HOP, _pluck(PENTA[index % 5] - 12), 0.16, 0.0)
    vinyl = rng.standard_normal(192 * HOP)
    vinyl = sosfilt(butter(1, 800.0, btype="low", fs=SR, output="sos"), vinyl)
    peak = np.max(np.abs(vinyl)) or 1.0
    _place(fx, 0, vinyl / peak, 10.0 ** (-42.0 / 20.0), 0.0)
    for _ in range(40):
        click = np.zeros(int(0.004 * SR))
        click[0] = 1.0
        click *= np.exp(-np.arange(len(click)) / 8.0)
        _place(fx, int(rng.integers(0, 192 * HOP)), click, 0.04, float(rng.uniform(-0.4, 0.4)))
    _schedule_scratches(scratches, fit, plan, rng)
    riser_n = (744 - 384) * HOP
    _place(fx, 384 * HOP, _riser(rng, riser_n), 0.12, 0.0)
    impact_wave = _impact(rng)
    _place(impact, 768 * HOP, impact_wave, 0.9, 0.0)
    _place(fx, 768 * HOP, impact_wave, 0.9, 0.0)
    for frame0, frame1, upward in ((876, 940, True), (960, 1056, False), (1440, 1536, True), (1536, 1632, False)):
        whoosh = _whoosh(rng, (frame1 - frame0) * HOP, upward)
        _place(whooshes, frame0 * HOP, whoosh, 0.16, 0.0)
        _place(fx, frame0 * HOP, whoosh, 0.16, 0.0)
    for frame in range(1344, 1440, 12):
        tick = np.sin(2.0 * math.pi * 1800.0 * np.arange(int(0.02 * SR)) / SR)
        tick *= np.exp(-np.arange(len(tick)) / (0.005 * SR))
        _place(fx, frame * HOP, tick, 0.05, 0.0)
    stabbed = _pad_note(_chord_tones("F"), int(0.4 * SR), rng)
    _place(harmony, 1632 * HOP, stabbed, 0.28, 0.0)
    send = lead[:, 0] + drums[:, 0] * 0.0
    clap_send = silence()
    for frame in range(192, 1800, 96):
        if 1344 <= frame < 1440:
            continue
        for offset in (24, 72):
            if 744 <= frame + offset < 768 or frame + offset >= 1794:
                continue
            _place(clap_send, (frame + offset) * HOP, clap_wave, 1.0, 0.0)
    reverb_source = lead[:, 0] + clap_send[:, 0] + scratches[:, 0]
    room = _reverb(reverb_source)
    wet = np.stack([room, np.roll(room, int(0.007 * SR))], axis=1)
    lead *= 0.85
    scratches *= 0.85
    drums_claps = clap_send * 0.15
    wet *= 0.15
    bass = sidechain(bass, kicks)
    harmony = sidechain(harmony, kicks)
    lead = sidechain(lead, kicks)
    mix = drums + bass + lead + harmony + fx + scratches + whooshes + wet + drums_claps * 0.0
    # claps are already inside drums; the wet return carries their room.
    _ = send
    _sweep_lowpass(mix, 864, 960, 400.0, 400.0)
    sfx = scratches + whooshes + impact
    _cut_silence(mix)
    _cut_silence(sfx)
    _tape_stop(mix)
    _tape_stop(sfx)
    return {
        "drums": drums,
        "bass": bass,
        "cowbell": lead,
        "pad": harmony,
        "fx": fx,
        "scratches": scratches,
        "whooshes": whooshes,
        "impact": impact,
        "mix": mix,
        "sfx": sfx,
    }


def _schedule_scratches(bus: np.ndarray, fit, plan, rng: np.random.Generator) -> None:
    if len(fit.x0) == 0:
        return
    appear = np.searchsorted(plan.counts[:1800], np.arange(1, len(fit.x0) + 1), side="left")
    gap = int(0.100 * SR / 14.0)
    grid_w = 254.0
    grouped: dict[int, list[int]] = {}
    for index, frame in enumerate(appear):
        frame = int(frame)
        if frame >= 1800 or 744 <= frame < 768:
            continue
        grouped.setdefault(frame, []).append(index)
    recent: list[int] = []
    for frame in sorted(grouped):
        chosen = grouped[frame][:14]
        for slot, index in enumerate(chosen):
            sample = frame * HOP + slot * gap
            recent = [item for item in recent if item > sample - int(0.1 * SR)]
            if len(recent) >= 14:
                continue
            recent.append(sample)
            gain = 0.09 / math.sqrt(max(1, len(chosen)))
            if frame >= 768:
                gain *= 0.45
            x = float(fit.x0[index] + fit.x1[index]) * 0.5
            angle = abs(float(np.arctan2(fit.y1[index] - fit.y0[index], fit.x1[index] - fit.x0[index])))
            pitch = 0.85 + 0.35 * (angle / math.pi)
            if int(fit.ink[index]) < 0:
                pitch *= 0.9
            pan = max(-0.9, min(0.9, (x / grid_w - 0.5) * 1.6))
            _place(bus, sample, _scratch(rng, pitch), gain, pan)


def master_mix(audio: np.ndarray) -> tuple[np.ndarray, float, float]:
    return _master(audio)


def _finish_edges(audio: np.ndarray) -> np.ndarray:
    audio = np.array(audio, dtype=np.float64, copy=True)
    _cut_silence(audio)
    audio[1806 * HOP :] = 0.0
    return audio


def _master_best_effort(audio: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Effects-only mixes can be too peaky for -14 LUFS. Keep the peak ceiling and the silence."""
    try:
        return _master(audio)
    except RuntimeError:
        x = np.array(audio, dtype=np.float64, copy=True)
        x -= np.mean(x, axis=0, keepdims=True)
        x = sosfilt(butter(2, 30.0, btype="high", fs=SR, output="sos"), x, axis=0)
        peak = float(np.max(np.abs(x))) or 1.0
        x *= (10.0 ** (-2.3 / 20.0)) / peak
        x = _finish_edges(x)
        meter_lufs = float("nan")
        try:
            import pyloudnorm as pyln

            meter_lufs = float(pyln.Meter(SR).integrated_loudness(x))
        except Exception:
            meter_lufs = -20.0
        from fc_sat.audio import true_peak_db

        return x, meter_lufs, true_peak_db(x)


def render_score(fit, plan, seed: int = 7) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, float, float]:
    stems = arrangement(fit, plan, seed)
    mix, lufs, peak = _master(stems["mix"])
    sfx, _, _ = _master_best_effort(stems["sfx"])
    stems["mix"] = mix
    stems["sfx"] = sfx
    return stems, mix, sfx, lufs, peak
