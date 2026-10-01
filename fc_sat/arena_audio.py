"""Synthesized Flag Arena bed and sound effects. No samples and no music files."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfiltfilt

from fc_sat.arena_config import ArenaConfig
from fc_sat.arena_render import Timeline, build_timeline, event_video_times, video_time
from fc_sat.arena_sim import SimResult
from fc_sat.audio import SR, audio_length, finish_broadcast


@dataclass(frozen=True)
class ArenaMix:
    full: np.ndarray
    sfx_only: np.ndarray
    lufs: float
    true_peak: float
    n_samples: int


def render_arena_mix(
    cfg: ArenaConfig,
    result: SimResult,
    n_frames: int,
    *,
    fps: int,
    timeline: Timeline | None = None,
    voice: np.ndarray | None = None,
    music: bool = True,
) -> ArenaMix:
    timeline = build_timeline(result, cfg) if timeline is None else timeline
    n_samples = audio_length(n_frames, fps, SR)
    bed = _bed(cfg, timeline, n_samples) if music else np.zeros((n_samples, 2))
    sfx = _sfx(cfg, result, timeline, n_samples)
    _duck_bed(cfg, timeline, bed, sfx, voice)
    mix = bed + sfx
    if voice is not None and len(voice):
        mix = mix + _fit(voice, n_samples)
        sfx = sfx + _fit(voice, n_samples)
    full, lufs, peak = _broadcast(_fit(mix, n_samples))
    full = _aac_headroom(full)
    sfx_full, _, _ = _broadcast(_fit(sfx, n_samples))
    sfx_full = _aac_headroom(sfx_full)
    import pyloudnorm as pyln

    from fc_sat.audio import true_peak_db

    lufs = float(pyln.Meter(SR).integrated_loudness(full))
    peak = true_peak_db(full)
    return ArenaMix(full=full, sfx_only=sfx_full, lufs=lufs, true_peak=peak, n_samples=n_samples)


def _compress(audio: np.ndarray, thresh: float) -> np.ndarray:
    """Clip transients so the shared loudness loop can reach -14 LUFS at -1 dBTP."""
    peak = float(np.max(np.abs(audio))) or 1.0
    clipped = np.clip(audio / peak, -thresh, thresh)
    clipped /= float(np.max(np.abs(clipped))) or 1.0
    return clipped * 0.5


def _aac_headroom(audio: np.ndarray) -> np.ndarray:
    """Pull 4x oversampled peaks down, then low-pass so AAC stays under -1 dBTP.

    Hard-clipped harmonics reconstruct above the ceiling in the native AAC
    encoder. A 12 kHz low-pass keeps that overshoot off the decoded file.
    """
    from scipy.signal import butter, resample_poly, sosfiltfilt

    ceiling = 10.0 ** (-9.2 / 20.0)
    up = resample_poly(audio, 4, 1, axis=0)
    up = np.clip(up, -ceiling, ceiling)
    down = resample_poly(up, 1, 4, axis=0)
    if len(down) < len(audio):
        padded = np.zeros_like(audio)
        padded[: len(down)] = down
        down = padded
    else:
        down = down[: len(audio)]
    sos = butter(4, 12000.0, btype="low", fs=SR, output="sos")
    return sosfiltfilt(sos, down, axis=0)


def _broadcast(audio: np.ndarray):
    error: RuntimeError | None = None
    for thresh in (0.16, 0.12, 0.08, 0.05):
        try:
            return finish_broadcast(_compress(audio, thresh), SR)
        except RuntimeError as exc:
            error = exc
    if error is not None:
        raise error
    raise RuntimeError("broadcast finish did not run")


def _fit(audio: np.ndarray, n_samples: int) -> np.ndarray:
    if audio.ndim == 1:
        audio = np.column_stack([audio, audio])
    out = np.zeros((n_samples, 2), dtype=np.float64)
    n = min(n_samples, len(audio))
    out[:n] = audio[:n]
    return out


def _bpm(cfg: ArenaConfig, timeline: Timeline, time: float) -> float:
    span = max(timeline.winner_video, 1e-3)
    u = min(1.0, max(0.0, time / span))
    return cfg.bpm[0] + (cfg.bpm[1] - cfg.bpm[0]) * u


def _beat_times(cfg: ArenaConfig, timeline: Timeline) -> list[float]:
    beats = [0.0]
    t = 0.0
    acc = 0.0
    target = 1.0
    step = 0.0005
    limit = timeline.duration
    while t < limit and len(beats) < 2000:
        acc += _bpm(cfg, timeline, t) / 60.0 * step
        t += step
        if acc >= target:
            beats.append(min(t, limit))
            target += 1.0
    return beats


def _tone(freq0: float, freq1: float, seconds: float, sr: int, decay: float, shape: str = "sine") -> np.ndarray:
    n = max(1, int(seconds * sr))
    t = np.arange(n, dtype=np.float64) / sr
    u = t / max(t[-1], 1e-6)
    freq = freq0 + (freq1 - freq0) * u
    phase = 2.0 * math.pi * np.cumsum(freq) / sr
    if shape == "saw":
        wave = 2.0 * ((phase / (2.0 * math.pi)) % 1.0) - 1.0
    elif shape == "square":
        wave = np.sign(np.sin(phase))
    else:
        wave = np.sin(phase)
    return wave * np.exp(-t / max(decay, 1e-4))


def _put(bus: np.ndarray, mono: np.ndarray, start: int, pan: float = 0.0, gain: float = 1.0) -> None:
    if start >= len(bus) or not len(mono):
        return
    if start < 0:
        mono = mono[-start:]
        start = 0
    n = min(len(mono), len(bus) - start)
    if n <= 0:
        return
    angle = (max(-1.0, min(1.0, pan)) + 1.0) * math.pi * 0.25
    left = math.cos(angle) * gain
    right = math.sin(angle) * gain
    bus[start : start + n, 0] += mono[:n] * left
    bus[start : start + n, 1] += mono[:n] * right


def _bed(cfg: ArenaConfig, timeline: Timeline, n_samples: int) -> np.ndarray:
    bus = np.zeros((n_samples, 2), dtype=np.float64)
    beats = _beat_times(cfg, timeline)
    minor = (220.0, 261.63, 329.63, 392.0)
    for index, when in enumerate(beats):
        start = int(round(when * SR))
        if index % 4 in (0, 2):
            _put(bus, _tone(140.0, 42.0, 0.18, SR, 0.08), start, gain=0.62)
            _put(bus, _tone(minor[0] / 2.0, minor[0] / 2.0, 0.25, SR, 0.12), start, gain=0.22)
        else:
            noise = np.random.Generator(np.random.PCG64(index + 3)).standard_normal(int(0.04 * SR))
            _put(bus, noise * np.exp(-np.arange(len(noise)) / (0.01 * SR)), start, gain=0.18)
        step = 2 if when < timeline.winner_video - 10.0 else 1
        for sub in range(step):
            hat_at = start + int(sub * (beats[index + 1] - when) * SR / 2) if index + 1 < len(beats) else start
            hat = np.random.Generator(np.random.PCG64(10_000 + index * 4 + sub)).standard_normal(int(0.03 * SR))
            _put(bus, hat * np.exp(-np.arange(len(hat)) / (0.006 * SR)), hat_at, gain=0.05)
        note = minor[index % len(minor)]
        _put(bus, _tone(note, note, 0.12, SR, 0.05), start + int(0.08 * SR), gain=0.07)
        if index % 2 == 1:
            _put(bus, _tone(cfg.cowbell_hz[0], cfg.cowbell_hz[0], 0.06, SR, 0.015, "square"), start, gain=0.04)
            _put(bus, _tone(cfg.cowbell_hz[1], cfg.cowbell_hz[1], 0.05, SR, 0.012, "square"), start + 40, gain=0.03)
    return bus


def _span_rate(timeline: Timeline, video_t: float) -> float:
    for span in timeline.spans:
        if span.video0 - 1e-6 <= video_t <= span.video1 + 1e-6:
            return span.rate if span.rate > 0 else 1.0
    return 1.0


def _stretch(mono: np.ndarray, rate: float) -> np.ndarray:
    if rate >= 0.999:
        return mono
    n = max(1, int(round(len(mono) / rate)))
    src = np.linspace(0.0, len(mono) - 1, n)
    return np.interp(src, np.arange(len(mono)), mono)


def _sfx(cfg: ArenaConfig, result: SimResult, timeline: Timeline, n_samples: int) -> np.ndarray:
    bus = np.zeros((n_samples, 2), dtype=np.float64)
    onsets: list[int] = []
    stinger_at = -1e9
    stinger_i = 0

    def allow(sample: int) -> bool:
        recent = [item for item in onsets if sample - item <= int(0.1 * SR)]
        if len(recent) >= cfg.max_onsets_per_100ms:
            return False
        if len(onsets) >= cfg.max_voices * 8:
            return True
        return True

    def play(mono: np.ndarray, video_t: float, pan: float = 0.0, gain: float = 0.3, rate: float | None = None) -> None:
        sample = int(round(video_t * SR))
        if sample < 0 or sample >= n_samples or not allow(sample):
            return
        used = _span_rate(timeline, video_t) if rate is None else rate
        _put(bus, _stretch(mono, used), sample, pan=pan, gain=gain)
        onsets.append(sample)

    for item in result.elims:
        for when in event_video_times(timeline, item.time):
            play(_tone(cfg.whistle_hz[0], cfg.whistle_hz[1], cfg.whistle_seconds, SR, 0.35), when, gain=0.16)
            if when - stinger_at < cfg.stinger_gap:
                play(_tone(cfg.ding_hz[0], cfg.ding_hz[1], cfg.ding_ms / 1000.0, SR, 0.04), when, gain=0.2)
                continue
            stinger_at = when
            kind = stinger_i % 3
            stinger_i += 1
            if kind == 0:
                note = _tone(392.0, 370.0, 0.18, SR, 0.08, "saw")
                note = np.concatenate([note, _tone(330.0, 311.0, 0.18, SR, 0.08, "saw"), _tone(262.0, 240.0, 0.22, SR, 0.09, "saw")])
                play(note, when, gain=0.2)
            elif kind == 1:
                play(_tone(520.0, 180.0, 0.16, SR, 0.05), when, gain=0.22)
            else:
                play(_mix(_tone(cfg.sub_hz[0], cfg.sub_hz[1], 0.28, SR, 0.1), _noise(0.04, 0.3)), when, gain=0.28)
    for impulse in result.impulses:
        if impulse.source == "dash":
            play(_tone(cfg.windup_hz[0], cfg.windup_hz[1], 0.12, SR, 0.04), video_time(timeline, impulse.time), pan=0.0, gain=0.08)
        elif impulse.source in {"ball", "boss"} and impulse.closing >= 200:
            pan = max(-1.0, min(1.0, (impulse.x - cfg.center[0]) / 500.0))
            gain = 0.12 + 0.2 * min(1.0, impulse.closing / 1400.0)
            play(_mix(_tone(140.0, 55.0, 0.08, SR, 0.03), _noise(0.03, 0.2)), video_time(timeline, impulse.time), pan=pan, gain=gain)
    if result.cameo_exit is not None:
        play(_tone(90.0, 40.0, 0.35, SR, 0.12), video_time(timeline, max(0.0, cfg.cameo_enter - 1.2)), gain=0.16)
    third = _when_alive(result, len(cfg.countries), 3)
    if third is not None:
        play(_horn(cfg), video_time(timeline, third), gain=0.24)
    if result.t_win is not None:
        play(_horn(cfg), timeline.winner_video, gain=0.28)
        play(_mix(_tone(523.25, 523.25, 0.4, SR, 0.15), _tone(659.25, 659.25, 0.45, SR, 0.16), _tone(783.99, 783.99, 0.5, SR, 0.18)), timeline.winner_video, gain=0.16)
        cheer = _noise(1.2, 1.0)
        play(cheer, timeline.celebration_video, gain=0.08)
    return bus


def _mix(*parts: np.ndarray) -> np.ndarray:
    n = max(len(part) for part in parts)
    out = np.zeros(n, dtype=np.float64)
    for part in parts:
        out[: len(part)] += part
    return out


def _noise(seconds: float, decay: float) -> np.ndarray:
    n = int(seconds * SR)
    rng = np.random.Generator(np.random.PCG64(int(seconds * 1000) + 9))
    env = np.exp(-np.arange(n) / (decay * SR))
    return rng.standard_normal(n) * env


def _horn(cfg: ArenaConfig) -> np.ndarray:
    a = _tone(cfg.airhorn_hz[0], cfg.airhorn_hz[0], 0.45, SR, 0.2, "saw")
    b = _tone(cfg.airhorn_hz[1], cfg.airhorn_hz[1], 0.45, SR, 0.2, "saw")
    n = min(len(a), len(b))
    return a[:n] + b[:n]


def _when_alive(result: SimResult, count: int, target: int) -> float | None:
    for index, item in enumerate(result.elims):
        if count - index - 1 == target:
            return item.time
    return None


def _duck_bed(cfg: ArenaConfig, timeline: Timeline, bed: np.ndarray, sfx: np.ndarray, voice: np.ndarray | None) -> None:
    if not len(bed):
        return
    env = np.maximum(np.abs(sfx[:, 0]), np.abs(sfx[:, 1]))
    kernel = max(1, int(0.03 * SR))
    smooth = np.convolve(env, np.ones(kernel) / kernel, mode="same")
    depth = 1.0 - 10.0 ** (-cfg.sidechain_db / 20.0)
    bed *= (1.0 - depth * np.clip(smooth / 0.25, 0.0, 1.0))[:, None]
    if voice is not None and len(voice):
        fitted = _fit(voice, len(bed))
        venv = np.maximum(np.abs(fitted[:, 0]), np.abs(fitted[:, 1]))
        voice_depth = 1.0 - 10.0 ** (-cfg.voice_duck_db / 20.0)
        bed *= (1.0 - voice_depth * np.clip(venv / 0.2, 0.0, 1.0))[:, None]
    regions = [span for span in timeline.spans if span.phase in {"slow", "replay"} and span.video1 > span.video0]
    if regions:
        sos = butter(2, min(cfg.slowmo_lpf, SR * 0.45), btype="low", fs=SR, output="sos")
        low = sosfiltfilt(sos, bed, axis=0)
        gain = 10.0 ** (-cfg.slowmo_duck_db / 20.0)
        for span in regions:
            i0 = max(0, int(span.video0 * SR))
            i1 = min(len(bed), int(span.video1 * SR))
            if i1 - i0 > 32:
                bed[i0:i1] = low[i0:i1] * gain
    drop0 = int((timeline.winner_video - cfg.pre_drop) * SR)
    drop1 = int(timeline.winner_video * SR)
    if 0 <= drop0 < drop1 <= len(bed):
        bed[drop0:drop1] = 0.0
