"""Mallet synthesis, density control, and the LUFS / true-peak master loop.

Notes are built only from the sim event log. Pitch walks up a C major pentatonic
as the count grows, with a seeded +/-1 step so the riser is not a staircase.
Wall-hit tails are silenced at the start of HOLD ("no notes"); the thud, implode
sweep, and reset sparkle are separate one-shots.

Master order is fixed: DC removal, 12 kHz low-pass, 5 ms edge fades, then a hard
zero on the last 300 ms. Only after that does the loudness loop run, so the
fade and the silent tail are part of every LUFS and true-peak measurement.
Each iteration (at most 5) gains toward -14 LUFS, soft-limits, enforces a
-1 dBTP ceiling on a 4x oversampled peak, and measures again. The next pass
corrects whatever gain the limiter removed.
"""

from __future__ import annotations

import math

import numpy as np
import pyloudnorm as pyln
from scipy.signal import butter, resample_poly, sosfilt, sosfiltfilt

from fc_sat.config import Config
from fc_sat.sim import SimResult

SR = 48000
C4 = 261.6255653005986
PENTATONIC = (0, 2, 4, 7, 9)
SEMITONES = tuple(octv * 12 + step for octv in range(3) for step in PENTATONIC)
FREQS = tuple(C4 * (2.0 ** (semi / 12.0)) for semi in SEMITONES)
CEILING = 10.0 ** (-1.0 / 20.0)  # -1 dBTP
MAX_VOICES = 24
MAX_ONSETS_PER_WINDOW = 14
ONSET_WINDOW = 0.100


def audio_length(n_frames: int, fps: int, sr: int = SR) -> int:
    if sr % fps == 0:
        return n_frames * (sr // fps)
    return int(round(n_frames / fps * sr))


def _note_cache(sr: int = SR) -> list[list[np.ndarray]]:
    """15 notes x 3 velocity layers. Partials at 4.0x and 9.2x decay faster."""
    cache: list[list[np.ndarray]] = []
    for index, freq in enumerate(FREQS):
        layers = []
        base_decay = 0.400 + (0.220 - 0.400) * (index / 14.0)
        for layer, brightness, decay_scale in ((0, 0.55, 1.0), (1, 0.82, 0.86), (2, 1.0, 0.74)):
            decay = base_decay * decay_scale
            n = int(sr * (decay + 0.03))
            t = np.arange(n, dtype=np.float64) / sr
            attack = np.minimum(t / 0.002, 1.0)
            env = np.exp(-t / decay) * attack
            env_p = np.exp(-t / (decay * 0.32)) * attack
            env_q = np.exp(-t / (decay * 0.16)) * attack
            wave = np.sin(2 * math.pi * freq * t) * env
            wave += brightness * 0.34 * np.sin(2 * math.pi * freq * 4.0 * t) * env_p
            wave += brightness * 0.12 * np.sin(2 * math.pi * freq * 9.2 * t) * env_q
            peak = float(np.max(np.abs(wave)))
            if peak > 0:
                wave /= peak
            layers.append(wave.astype(np.float64))
        cache.append(layers)
    return cache


_CACHE: list[list[np.ndarray]] | None = None


def note_cache() -> list[list[np.ndarray]]:
    global _CACHE
    if _CACHE is None:
        _CACHE = _note_cache()
    return _CACHE


def note_index(count: int, cap: int, wobble: int) -> int:
    if cap <= 1 or count <= 1:
        base = 0
    else:
        base = int(math.floor(math.log(count) / math.log(cap) * 14.0))
    return int(min(14, max(0, base + wobble)))


def _impact_gain(speed: float, max_speed: float) -> float:
    x = min(1.0, max(0.0, speed / max_speed))
    return 0.16 * (0.22 + 0.78 * (x ** 0.65))


def _layer(speed: float) -> int:
    if speed < 700.0:
        return 0
    if speed < 1100.0:
        return 1
    return 2


def _pan(x: float, center_x: float, radius: float) -> tuple[float, float]:
    norm = (x - center_x) / radius if radius else 0.0
    norm = min(1.0, max(-1.0, norm))
    pan = norm * 0.35
    theta = (pan + 1.0) * 0.25 * math.pi
    return math.cos(theta), math.sin(theta)


def select_onsets(events: np.ndarray, limit: int = MAX_ONSETS_PER_WINDOW, window: float = ONSET_WINDOW) -> np.ndarray:
    """Keep the loudest impacts so no 100 ms window holds more than `limit` onsets."""
    if events.size == 0:
        return events
    order = np.argsort(events["time"], kind="mergesort")
    chosen: list[int] = []
    window_ids: list[int] = []
    times = events["time"]
    speeds = events["impact_speed"]
    for raw in order:
        ei = int(raw)
        t = float(times[ei])
        speed = float(speeds[ei])
        if window_ids:
            window_ids = [j for j in window_ids if float(times[j]) >= t - window]
        if len(window_ids) < limit:
            window_ids.append(ei)
            chosen.append(ei)
            continue
        quietest = min(window_ids, key=lambda j: (float(speeds[j]), -float(times[j])))
        if speed > float(speeds[quietest]):
            window_ids.remove(quietest)
            chosen.remove(quietest)
            window_ids.append(ei)
            chosen.append(ei)
    if not chosen:
        return events[:0]
    chosen.sort(key=lambda j: (float(times[j]), int(events["ball_id"][j])))
    return events[np.array(chosen, dtype=np.int64)]


def _allocate(events: np.ndarray, durations: np.ndarray, gains: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (keep_mask, concurrent_counts) after stealing the quietest voice past 24."""
    n = len(events)
    keep = np.ones(n, dtype=bool)
    concurrent = np.ones(n, dtype=np.int32)
    active: list[tuple[float, float, int]] = []
    times = events["time"]
    for i in range(n):
        t = float(times[i])
        active = [item for item in active if item[0] > t]
        gain = float(gains[i])
        if len(active) >= MAX_VOICES:
            quiet_at = min(range(len(active)), key=lambda k: active[k][1])
            if active[quiet_at][1] <= gain:
                keep[active[quiet_at][2]] = False
                active.pop(quiet_at)
            else:
                keep[i] = False
                continue
        active.append((t + float(durations[i]), gain, i))
        concurrent[i] = len(active)
    return keep, concurrent


def _place(bus: np.ndarray, start: int, wave: np.ndarray, left: float, right: float) -> None:
    if start >= len(bus) or start < -len(wave):
        return
    if start < 0:
        wave = wave[-start:]
        start = 0
    end = min(len(bus), start + len(wave))
    span = end - start
    if span <= 0:
        return
    bus[start:end, 0] += wave[:span] * left
    bus[start:end, 1] += wave[:span] * right


def _silence_notes_at_hold(bus: np.ndarray, tg: float, sr: int) -> None:
    i0 = int(round(tg * sr))
    if i0 >= len(bus):
        return
    fade = min(int(0.020 * sr), len(bus) - i0)
    if fade > 0:
        bus[i0 : i0 + fade] *= np.linspace(1.0, 0.0, fade, dtype=np.float64)[:, None]
        i0 += fade
    if i0 < len(bus):
        bus[i0:] = 0.0


def mix_events(cfg: Config, result: SimResult, n_samples: int, sr: int = SR) -> np.ndarray:
    """Mallet bus plus milestone chimes. No samples past the start of HOLD."""
    bus = np.zeros((n_samples, 2), dtype=np.float64)
    offset = cfg.audio_offset_ms / 1000.0
    events = result.events
    if events.size:
        shifted = events.copy()
        shifted["time"] = events["time"] + offset
        shifted = shifted[(shifted["time"] >= 0.0) & (shifted["time"] < cfg.growth_seconds)]
        chosen = select_onsets(shifted)
    else:
        chosen = events
    cache = note_cache()
    rng = np.random.Generator(np.random.PCG64(cfg.seed + 7919))
    if chosen.size:
        durations = np.zeros(len(chosen), dtype=np.float64)
        gains = np.zeros(len(chosen), dtype=np.float64)
        waves: list[np.ndarray] = []
        pans: list[tuple[float, float]] = []
        for event in chosen:
            wobble = int(rng.integers(-1, 2))
            idx = note_index(int(event["count_at_time"]), cfg.cap, wobble)
            layer = _layer(float(event["impact_speed"]))
            wave = cache[idx][layer]
            waves.append(wave)
            durations[len(waves) - 1] = len(wave) / sr
            gains[len(waves) - 1] = _impact_gain(float(event["impact_speed"]), cfg.max_speed)
            pans.append(_pan(float(event["x"]), cfg.ring_cx, cfg.ring_radius))
        keep, concurrent = _allocate(chosen, durations, gains)
        for i, event in enumerate(chosen):
            if not keep[i]:
                continue
            scale = 1.0 / math.sqrt(float(concurrent[i]))
            gain = float(gains[i]) * scale
            left, right = pans[i]
            start = int(round(float(event["time"]) * sr))
            _place(bus, start, waves[i], left * gain, right * gain)

    # Soft chime on milestones. These are sparse, so they skip the onset limiter.
    chime = cache[7][0]
    for row in np.atleast_2d(result.milestones):
        if row.size < 2:
            continue
        when = float(row[0]) + offset
        if when < 0 or when >= cfg.growth_seconds:
            continue
        _place(bus, int(round(when * sr)), chime, 0.045, 0.045)
    _silence_notes_at_hold(bus, cfg.growth_seconds + max(offset, 0.0), sr)
    return bus


def _add_extras(cfg: Config, n_samples: int, sr: int, bus: np.ndarray) -> None:
    tg = cfg.growth_seconds
    rng = np.random.Generator(np.random.PCG64(cfg.seed + 17))

    def put(mono: np.ndarray, start: int, gain: float = 1.0) -> None:
        _place(bus, start, mono, gain, gain)

    # Rising filtered noise + sine from 0.75*Tg to Tg.
    t0 = 0.75 * tg
    t1 = tg
    i0 = int(round(t0 * sr))
    i1 = min(n_samples, int(round(t1 * sr)))
    length = i1 - i0
    if length > 32:
        t = np.arange(length, dtype=np.float64) / sr
        dur = max(t[-1], 1e-6)
        f0, f1 = 180.0, 2200.0
        phase = 2 * math.pi * (f0 * t + (f1 - f0) * (t ** 2) / (2 * dur))
        sine = np.sin(phase)
        noise = rng.standard_normal(length)
        sos = butter(2, [250.0, 5000.0], btype="band", fs=sr, output="sos")
        filtered = sosfilt(sos, noise)
        peak = float(np.max(np.abs(filtered))) or 1.0
        env = (t / dur) ** 1.4
        fade = min(int(0.04 * sr), length // 5)
        env[-fade:] *= np.linspace(1.0, 0.0, fade)
        mono = (0.60 * sine + 0.40 * filtered / peak) * env * 0.10
        put(mono, i0)

    # Low thud + soft click at HOLD start. Not a pitched mallet note.
    hold = tg
    thud_n = int(0.40 * sr)
    t = np.arange(thud_n, dtype=np.float64) / sr
    body = np.sin(2 * math.pi * 55.0 * t) * np.exp(-t / 0.10)
    click = np.sin(2 * math.pi * 1400.0 * t) * np.exp(-t / 0.004)
    put(body * 0.50 + click * 0.10, int(round(hold * sr)), 0.55)

    # Descending sweep across IMPLODE. Envelope is zero at both ends.
    imp0 = tg + cfg.hold_seconds
    imp1 = imp0 + cfg.implode_seconds
    j0 = int(round(imp0 * sr))
    j1 = min(n_samples, int(round(imp1 * sr)))
    length = j1 - j0
    if length > 32:
        t = np.arange(length, dtype=np.float64) / sr
        dur = max(float(t[-1]), 1e-6)
        u = t / dur
        freq = 720.0 * ((70.0 / 720.0) ** u)
        phase = 2 * math.pi * np.cumsum(freq) / sr
        env = np.sin(math.pi * u) ** 2
        put(np.sin(phase) * env * 0.12, j0)

    # Sparkle when the ball reappears, finished well before the silent tail.
    reset = tg + cfg.hold_seconds + cfg.implode_seconds + cfg.beat_seconds
    spark_n = int(0.12 * sr)
    t = np.arange(spark_n, dtype=np.float64) / sr
    spark = (
        np.sin(2 * math.pi * 1760.0 * t) * 0.65
        + np.sin(2 * math.pi * 2637.0 * t) * 0.35
    ) * np.exp(-t / 0.045)
    put(spark, int(round(reset * sr)), 0.07)


def true_peak_linear(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    up = resample_poly(audio, 4, 1, axis=0)
    return float(np.max(np.abs(up)))


def true_peak_db(audio: np.ndarray) -> float:
    peak = true_peak_linear(audio)
    if peak <= 1e-12:
        return -120.0
    return 20.0 * math.log10(peak)


def _lowpass(audio: np.ndarray, sr: int) -> np.ndarray:
    sos = butter(4, 12000.0, btype="low", fs=sr, output="sos")
    return sosfiltfilt(sos, audio, axis=0)


def _fade_edges(audio: np.ndarray, sr: int, seconds: float = 0.005) -> None:
    n = min(int(round(seconds * sr)), len(audio) // 2)
    if n <= 1:
        return
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float64)
    audio[:n] *= ramp[:, None]
    audio[-n:] *= ramp[::-1, None]


def _soft_limit(audio: np.ndarray, ceiling: float, peak_fn=true_peak_linear) -> np.ndarray:
    """Compress only the peaks above 85% of the ceiling. Quieter samples stay put."""
    knee = ceiling * 0.85
    room = ceiling - knee
    ax = np.abs(audio)
    over = ax > knee
    if not np.any(over):
        y = audio
    else:
        y = audio.copy()
        excess = ax[over] - knee
        compressed = knee + room * (1.0 - np.exp(-excess / room))
        y[over] = np.sign(audio[over]) * compressed
    peak = float(peak_fn(y))
    if peak > ceiling:
        y = y * (ceiling / peak) * 0.999
    return y


def peak_db(audio: np.ndarray, peak_fn=true_peak_linear) -> float:
    peak = float(peak_fn(audio))
    if peak <= 1e-12:
        return -120.0
    return 20.0 * math.log10(peak)


def loudness_loop(
    audio: np.ndarray,
    sr: int = SR,
    *,
    peak_fn=true_peak_linear,
    zero_tail: int = 0,
    ceiling: float = CEILING,
) -> tuple[np.ndarray, float, float]:
    """At most 5 iterations: gain toward -14 LUFS, soft-knee, then the ceiling, re-measure.

    ``peak_fn`` returns a linear true-peak. ``zero_tail`` samples are forced to
    zero after every limiter pass so a silent ending stays part of the measurement.
    ``ceiling`` defaults to -1 dBTP. A lower ceiling leaves room for an AAC encode.
    """
    x = np.array(audio, dtype=np.float64, copy=True)
    if x.ndim != 2 or x.shape[1] != 2:
        raise RuntimeError("loudness loop expects stereo audio shaped (samples, 2)")
    meter = pyln.Meter(sr)
    last_lufs = float("nan")
    last_tp = float("nan")
    tail = min(int(zero_tail), len(x))
    limit_db = 20.0 * math.log10(ceiling)
    for _ in range(5):
        lufs = float(meter.integrated_loudness(x))
        if not math.isfinite(lufs):
            raise RuntimeError("integrated loudness is not finite; the mix is silent or invalid")
        gain = 10.0 ** ((-14.0 - lufs) / 20.0)
        x *= gain
        x = _soft_limit(x, ceiling, peak_fn)
        if tail:
            x[-tail:] = 0.0
        last_lufs = float(meter.integrated_loudness(x))
        last_tp = peak_db(x, peak_fn)
        if math.isfinite(last_lufs) and abs(last_lufs + 14.0) <= 0.5 and last_tp <= limit_db + 1e-6:
            return x, last_lufs, last_tp
    raise RuntimeError(
        f"loudness loop failed after 5 iterations: LUFS={last_lufs:.3f} true_peak={last_tp:.3f} dBTP"
    )


def finish_broadcast(audio: np.ndarray, sr: int = SR) -> tuple[np.ndarray, float, float]:
    """DC removal, 5 ms edge fades, then the shared loudness loop. The ending stays audible."""
    x = np.array(audio, dtype=np.float64, copy=True)
    if x.ndim != 2 or x.shape[1] != 2:
        raise RuntimeError("finish_broadcast expects stereo audio shaped (samples, 2)")
    x -= np.mean(x, axis=0, keepdims=True)
    _fade_edges(x, sr, 0.005)
    return loudness_loop(x, sr, peak_fn=true_peak_linear, zero_tail=0)


def master(audio: np.ndarray, sr: int = SR) -> tuple[np.ndarray, float, float]:
    """Fades and the zero tail happen before the loudness loop. See module docstring."""
    x = np.array(audio, dtype=np.float64, copy=True)
    if x.ndim != 2 or x.shape[1] != 2:
        raise RuntimeError("master expects stereo audio shaped (samples, 2)")
    x -= np.mean(x, axis=0, keepdims=True)
    if len(x) > 64:
        x = _lowpass(x, sr)
    _fade_edges(x, sr, 0.005)
    tail = int(round(0.300 * sr))
    tail = min(tail, len(x))
    if tail:
        x[-tail:] = 0.0
    return loudness_loop(x, sr, peak_fn=true_peak_linear, zero_tail=tail)


def synthesize(cfg: Config, result: SimResult, *, n_frames: int | None = None, fps: int | None = None) -> tuple[np.ndarray, float, float]:
    fps = cfg.fps if fps is None else fps
    n_frames = cfg.n_frames(preview=False) if n_frames is None else n_frames
    n_samples = audio_length(n_frames, fps, SR)
    bus = mix_events(cfg, result, n_samples, SR)
    _add_extras(cfg, n_samples, SR, bus)
    if len(bus) < n_samples:
        padded = np.zeros((n_samples, 2), dtype=np.float64)
        padded[: len(bus)] = bus
        bus = padded
    elif len(bus) > n_samples:
        bus = bus[:n_samples]
    return master(bus, SR)


def write_wav(path: str, audio: np.ndarray, sr: int = SR) -> None:
    """32-bit float WAV so the limiter ceiling is not quantized away before AAC."""
    from scipy.io import wavfile

    wavfile.write(path, sr, np.asarray(audio, dtype=np.float32))
