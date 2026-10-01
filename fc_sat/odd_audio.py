"""Synthesized Odd One Out mix. No samples, no music bed.

Events are placed at ``round(t * 48000)``. The riser ends when the timer ends.
The next 0.15 s stays quiet, then the reveal ding plays. ``finish_broadcast``
removes DC, fades the edges, and runs the shared loudness loop.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.signal import butter, sosfilt

from fc_sat.audio import SR, finish_broadcast
from fc_sat.odd_config import OddConfig, build_timeline, timeline_frames


def event_list(cfg: OddConfig) -> list[dict]:
    """Sample-accurate event times. The mixer and the sync test share this list."""
    levels = {level.id: level for level in cfg.levels}
    events: list[dict] = []
    for segment in build_timeline(cfg):
        if segment.kind == "play":
            level = levels[segment.level_id]
            events.append({"kind": "start", "t": segment.start_s, "level": level.id})
            events.append({"kind": "riser", "t": segment.start_s, "dur": segment.duration_s, "level": level.id})
            second = 1
            while second < segment.duration_s - 1e-6:
                events.append({"kind": "tick", "t": segment.start_s + second, "level": level.id})
                second += 1
            heart = max(0.0, segment.duration_s - cfg.heartbeat_window)
            events.append(
                {
                    "kind": "heart",
                    "t": segment.start_s + heart,
                    "dur": min(cfg.heartbeat_window, segment.duration_s),
                    "level": level.id,
                }
            )
            events.append({"kind": "buzz", "t": max(segment.start_s, segment.end_s - cfg.buzz_seconds), "level": level.id})
        elif segment.kind == "reveal":
            events.append({"kind": "pop", "t": segment.start_s, "level": segment.level_id})
            events.append(
                {
                    "kind": "ding",
                    "t": segment.start_s + cfg.silence_at_reveal,
                    "level": segment.level_id,
                }
            )
        elif segment.kind == "outro":
            events.append({"kind": "chord", "t": segment.start_s, "level": 0})
    return events


def _place(bus: np.ndarray, mono: np.ndarray, start: int, gain: float = 1.0) -> None:
    if start >= len(bus) or mono.size == 0 or gain == 0.0:
        return
    if start < 0:
        mono = mono[-start:]
        start = 0
    end = min(len(bus), start + len(mono))
    if end <= start:
        return
    sl = mono[: end - start] * gain
    bus[start:end, 0] += sl
    bus[start:end, 1] += sl


def _env_noise(rng: np.random.Generator, n: int, sr: int, low: float, high: float) -> np.ndarray:
    noise = rng.standard_normal(n)
    sos = butter(2, [low, high], btype="band", fs=sr, output="sos")
    filtered = sosfilt(sos, noise)
    peak = float(np.max(np.abs(filtered))) or 1.0
    return filtered / peak


def _whoosh(rng: np.random.Generator, sr: int) -> np.ndarray:
    n = int(0.20 * sr)
    tone = _env_noise(rng, n, sr, 280.0, 2200.0)
    env = np.linspace(0.0, 1.0, n) ** 1.3
    fade = max(1, n // 5)
    env[-fade:] *= np.linspace(1.0, 0.0, fade)
    return tone * env


def _blip(sr: int, freq: float) -> np.ndarray:
    n = int(0.045 * sr)
    t = np.arange(n, dtype=np.float64) / sr
    return np.sin(2.0 * math.pi * freq * t) * np.exp(-t / 0.012)


def _wood(rng: np.random.Generator, sr: int, freq: float) -> np.ndarray:
    n = int(0.055 * sr)
    t = np.arange(n, dtype=np.float64) / sr
    body = np.sin(2.0 * math.pi * freq * t) * np.exp(-t / 0.016)
    click = _env_noise(rng, n, sr, 800.0, 2800.0) * np.exp(-t / 0.006)
    return body * 0.7 + click * 0.25


def _riser(rng: np.random.Generator, sr: int, dur: float) -> np.ndarray:
    n = max(8, int(round(dur * sr)))
    t = np.arange(n, dtype=np.float64) / sr
    span = max(float(t[-1]), 1e-6)
    f0, f1 = 160.0, 1500.0
    phase = 2.0 * math.pi * (f0 * t + (f1 - f0) * (t ** 2) / (2.0 * span))
    saw = 2.0 * ((phase / (2.0 * math.pi)) % 1.0) - 1.0
    noise = _env_noise(rng, n, sr, 300.0, 4000.0)
    env = (t / span) ** 1.6
    fade = min(n // 8, int(0.04 * sr))
    if fade > 1:
        env[-fade:] *= np.linspace(1.0, 0.0, fade)
    return (0.22 * saw + 0.18 * noise) * env


def _buzz(sr: int, dur: float) -> np.ndarray:
    n = max(8, int(round(dur * sr)))
    t = np.arange(n, dtype=np.float64) / sr
    span = max(float(t[-1]), 1e-6)
    env = np.sin(math.pi * np.clip(t / span, 0.0, 1.0)) ** 2
    return np.sin(2.0 * math.pi * 155.0 * t) * env


def _heartbeat(sr: int) -> np.ndarray:
    n = int(0.11 * sr)
    t = np.arange(n, dtype=np.float64) / sr
    thump = np.sin(2.0 * math.pi * 58.0 * t) * np.exp(-t / 0.045)
    click = np.sin(2.0 * math.pi * 190.0 * t) * np.exp(-t / 0.008)
    return thump * 0.8 + click * 0.15


def _heartbeat_samples(t0: float, dur: float, bpm0: float, bpm1: float, sr: int) -> list[int]:
    n = int(round(dur * sr))
    start = int(round(t0 * sr))
    phase = 1.0
    hits = []
    for index in range(n):
        u = index / max(n - 1, 1)
        bpm = bpm0 + (bpm1 - bpm0) * u
        phase += (bpm / 60.0) / sr
        if phase >= 1.0:
            phase -= 1.0
            hit = start + index
            if hit + int(0.12 * sr) <= start + n:
                hits.append(hit)
    return hits


def _ding(rng: np.random.Generator, sr: int) -> np.ndarray:
    n = int(0.50 * sr)
    t = np.arange(n, dtype=np.float64) / sr
    tone = (
        np.sin(2.0 * math.pi * 1320.0 * t)
        + 0.65 * np.sin(2.0 * math.pi * 1980.0 * t)
    ) * np.exp(-t / 0.16)
    spark = np.sin(2.0 * math.pi * 3520.0 * t) * np.exp(-t / 0.045) * 0.35
    whoosh = _env_noise(rng, n, sr, 140.0, 500.0) * np.exp(-t / 0.10) * 0.35
    return tone + spark + whoosh


def _pop(sr: int) -> np.ndarray:
    n = int(0.04 * sr)
    t = np.arange(n, dtype=np.float64) / sr
    return np.sin(2.0 * math.pi * 620.0 * t) * np.exp(-t / 0.008)


def _chord(sr: int) -> np.ndarray:
    n = int(1.5 * sr)
    t = np.arange(n, dtype=np.float64) / sr
    tone = np.zeros(n, dtype=np.float64)
    for freq, amp in ((261.63, 0.55), (329.63, 0.40), (392.00, 0.35), (523.25, 0.22)):
        tone += amp * np.sin(2.0 * math.pi * freq * t)
    return tone * np.exp(-t / 0.50)


def mix_odd(cfg: OddConfig, n_frames: int | None = None) -> np.ndarray:
    """Dry stereo mix, before the loudness loop. Length is ``n_frames * 800``."""
    frames = timeline_frames(cfg) if n_frames is None else int(n_frames)
    if cfg.fps != 60:
        raise RuntimeError("odd audio expects 60 fps so each frame is 800 samples")
    n_samples = frames * (SR // cfg.fps)
    bus = np.zeros((n_samples, 2), dtype=np.float64)
    rng = np.random.Generator(np.random.PCG64(cfg.seed))
    for event in event_list(cfg):
        start = int(round(event["t"] * SR))
        kind = event["kind"]
        level = int(event.get("level") or 1)
        if kind == "start":
            _place(bus, _whoosh(rng, SR), start, 0.22)
            _place(bus, _blip(SR, 660.0 + 70.0 * (level - 1)), start, 0.18)
        elif kind == "tick":
            freq = 210.0 + 28.0 * (level - 1)
            _place(bus, _wood(rng, SR, freq), start, 0.20)
        elif kind == "riser":
            _place(bus, _riser(rng, SR, float(event["dur"])), start, 0.07)
        elif kind == "heart":
            for sample in _heartbeat_samples(
                event["t"],
                float(event["dur"]),
                cfg.heartbeat_bpm[0],
                cfg.heartbeat_bpm[1],
                SR,
            ):
                _place(bus, _heartbeat(SR), sample, 0.28)
        elif kind == "buzz":
            _place(bus, _buzz(SR, cfg.buzz_seconds), start, 0.10)
        elif kind == "pop":
            _place(bus, _pop(SR), start, 0.16)
        elif kind == "ding":
            _place(bus, _ding(rng, SR), start, 0.22)
        elif kind == "chord":
            _place(bus, _chord(SR), start, 0.16)
    return bus


def synthesize_odd(cfg: OddConfig, n_frames: int | None = None) -> tuple[np.ndarray, float, float]:
    dry = mix_odd(cfg, n_frames)
    mastered, lufs, true_peak = finish_broadcast(dry, SR)
    if len(mastered) != len(dry):
        raise RuntimeError(f"audio length changed from {len(dry)} to {len(mastered)}")
    return mastered, lufs, true_peak
