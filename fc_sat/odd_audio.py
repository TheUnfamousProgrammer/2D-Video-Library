"""Synthesized Odd One Out mix. No samples, no music bed, no riser, no heartbeat.

Events sit at ``round(t * 48000)``. ``finish_broadcast`` removes DC, fades the
edges, and runs the shared loudness loop.
"""

from __future__ import annotations

import math

import numpy as np

from fc_sat.audio import SR, finish_broadcast
from fc_sat.odd_config import OddConfig, build_timeline, timeline_frames


def event_list(cfg: OddConfig) -> list[dict]:
    """Sample-accurate event times. The mixer and the sync test share this list."""
    events: list[dict] = []
    levels = {level.id: level for level in cfg.levels}
    for segment in build_timeline(cfg):
        if segment.kind == "play":
            level = levels[segment.level_id]
            events.append({"kind": "blip", "t": segment.start_s, "level": level.id})
            t = 1.0
            end = segment.duration_s
            while t < end - 1e-6:
                events.append({"kind": "tick", "t": segment.start_s + t, "level": level.id})
                step = 0.5 if t + 1e-9 >= end - cfg.double_tick_window else 1.0
                t += step
            events.append(
                {
                    "kind": "tone",
                    "t": max(segment.start_s, segment.end_s - cfg.tone_seconds),
                    "level": level.id,
                }
            )
        elif segment.kind == "reveal":
            events.append({"kind": "ding", "t": segment.start_s, "level": segment.level_id})
        elif segment.kind == "outro":
            events.append({"kind": "chord", "t": segment.start_s, "level": 0})
    return events


def _place(bus: np.ndarray, mono: np.ndarray, start: int, gain: float) -> None:
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


def _tone(freq: float, seconds: float, sr: int, decay: float) -> np.ndarray:
    n = max(1, int(round(seconds * sr)))
    t = np.arange(n, dtype=np.float64) / sr
    return np.sin(2.0 * math.pi * freq * t) * np.exp(-t / decay)


def mix_odd(cfg: OddConfig, n_frames: int | None = None) -> np.ndarray:
    frames = timeline_frames(cfg) if n_frames is None else int(n_frames)
    bus = np.zeros((frames * (SR // cfg.fps), 2), dtype=np.float64)
    for event in event_list(cfg):
        start = int(round(event["t"] * SR))
        level = int(event["level"])
        kind = event["kind"]
        if kind == "blip":
            _place(bus, _tone(660.0 + 40.0 * (level - 1), 0.09, SR, 0.04), start, 0.16)
        elif kind == "tick":
            _place(bus, _tone(520.0 + 70.0 * (level - 1), 0.045, SR, 0.018), start, 0.14)
        elif kind == "tone":
            _place(bus, _tone(196.0, cfg.tone_seconds, SR, 0.12), start, 0.12)
        elif kind == "ding":
            ding = _tone(1320.0, 0.35, SR, 0.12)
            spark = _tone(1980.0, 0.35, SR, 0.10)
            _place(bus, ding + 0.7 * spark, start, 0.18)
        elif kind == "chord":
            chord = _tone(261.63, 1.4, SR, 0.45)
            chord += 0.8 * _tone(329.63, 1.4, SR, 0.45)
            chord += 0.7 * _tone(392.00, 1.4, SR, 0.45)
            chord += 0.45 * _tone(523.25, 1.4, SR, 0.40)
            _place(bus, chord, start, 0.12)
    return bus


def synthesize_odd(cfg: OddConfig, n_frames: int | None = None) -> tuple[np.ndarray, float, float]:
    frames = timeline_frames(cfg) if n_frames is None else int(n_frames)
    dry = mix_odd(cfg, frames)
    mastered, lufs, true_peak = finish_broadcast(dry)
    if len(mastered) != frames * (SR // cfg.fps):
        raise RuntimeError("loudness loop changed the audio length")
    return mastered, lufs, true_peak
