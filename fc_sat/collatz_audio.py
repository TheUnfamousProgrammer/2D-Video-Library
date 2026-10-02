"""Collatz score. Notes, pad, and hits are synthesized. The master uses the shared loudness loop."""

from __future__ import annotations

import math

import numpy as np
from scipy.signal import butter, sosfilt

from fc_sat.audio import SR, loudness_loop
from fc_sat.collatz_math import peak, trajectory

C3 = 130.8127826502993
PENT = (0, 2, 4, 7, 9)


def pitch_index(value: int) -> int:
    if value < 1:
        return 0
    return max(0, min(14, int(round(math.log2(value)))))


def frequency(value: int) -> float:
    index = pitch_index(value)
    semi = (index // 5) * 12 + PENT[index % 5]
    return C3 * (2.0 ** (semi / 12.0))


def pan_gains(pan: float) -> tuple[float, float]:
    pan = max(-1.0, min(1.0, pan))
    angle = (pan + 1.0) * 0.25 * math.pi
    return math.cos(angle), math.sin(angle)


def _place(bus: np.ndarray, wave: np.ndarray, start: int, pan: float, gain: float) -> None:
    if start >= len(bus) or wave.size == 0:
        return
    left, right = pan_gains(pan)
    end = min(len(bus), start + len(wave))
    chunk = wave[: end - start] * gain
    bus[start:end, 0] += chunk * left
    bus[start:end, 1] += chunk * right


def pluck(freq: float, sr: int = SR) -> np.ndarray:
    n = int(sr * 0.40)
    t = np.arange(n, dtype=np.float64) / sr
    attack = np.minimum(t / 0.002, 1.0)
    env = np.exp(-t / 0.120) * attack
    wave = np.zeros(n, dtype=np.float64)
    for partial, amp in ((1, 1.0), (2, 0.5), (3, 0.3), (4, 0.15)):
        wave += amp * np.sin(2 * math.pi * freq * partial * t)
    wave *= env
    peak_amp = float(np.max(np.abs(wave))) or 1.0
    return wave / peak_amp * 0.9


def bell(freq: float, sr: int = SR) -> np.ndarray:
    n = int(sr * 0.55)
    t = np.arange(n, dtype=np.float64) / sr
    attack = np.minimum(t / 0.002, 1.0)
    wave = np.sin(2 * math.pi * freq * t) * np.exp(-t / 0.220)
    wave += 0.3 * np.sin(2 * math.pi * freq * 2.76 * t) * np.exp(-t / 0.060)
    wave += 0.15 * np.sin(2 * math.pi * freq * 5.4 * t) * np.exp(-t / 0.040)
    wave *= attack
    peak_amp = float(np.max(np.abs(wave))) or 1.0
    return wave / peak_amp * 0.6


def _tone(freq: float, seconds: float, sr: int, decay: float) -> np.ndarray:
    n = int(sr * seconds)
    t = np.arange(n, dtype=np.float64) / sr
    return np.sin(2 * math.pi * freq * t) * np.exp(-t / decay)


def duck(bus: np.ndarray, voice: np.ndarray | None, sr: int, db: float) -> np.ndarray:
    if voice is None or not np.any(voice):
        return bus
    mag = np.mean(np.abs(voice), axis=1)
    attack = math.exp(-1.0 / (0.010 * sr))
    release = math.exp(-1.0 / (0.250 * sr))
    env = np.zeros(len(mag), dtype=np.float64)
    level = 0.0
    for index, sample in enumerate(mag):
        coef = attack if sample > level else release
        level = coef * level + (1.0 - coef) * sample
        env[index] = level
    peak_env = float(env.max()) or 1.0
    env /= peak_env
    depth = 10.0 ** (-db / 20.0)
    gain = 1.0 - env * (1.0 - depth)
    return bus * gain[:, None]


def finish(bus: np.ndarray, sr: int = SR) -> tuple[np.ndarray, float, float]:
    import math

    import pyloudnorm as pyln

    from fc_sat.audio import _soft_limit, true_peak_db

    audio = np.array(bus, dtype=np.float64, copy=True)
    audio -= np.mean(audio, axis=0, keepdims=True)
    n = min(len(audio) // 2, int(round(0.010 * sr)))
    if n > 1:
        audio[-n:, :] *= np.linspace(1.0, 0.0, n)[:, None]
    audio, _lufs, _tp = loudness_loop(audio, sr)
    # AAC reconstruction sits about 1 dB above the wav true peak. Hold the wav
    # to -2.0 dBTP so the encoded file stays at or under -1 dBTP.
    audio = _soft_limit(audio, 10.0 ** (-2.0 / 20.0))
    lufs = float(pyln.Meter(sr).integrated_loudness(audio))
    tp = true_peak_db(audio)
    if not math.isfinite(lufs) or abs(lufs + 14.0) > 1.0 or tp > -1.0:
        raise RuntimeError(f"master out of spec: LUFS={lufs:.2f} true_peak={tp:.2f}")
    return audio, lufs, tp


def _sample(seconds: float, sr: int) -> int:
    return int(round(seconds * sr))


def step_events(timeline) -> list[tuple[float, int, str]]:
    """(time, value_after, 'up'|'down') for every drawn step in S2, S3, and S5."""
    events = []
    scene = timeline.scene("S2")
    values = trajectory(6)
    for index, offset in enumerate(scene.spec["step_offsets"]):
        if index >= len(values) - 1:
            break
        kind = "down" if values[index] % 2 == 0 else "up"
        events.append((scene.rel(float(offset), timeline.fps), values[index + 1], kind))
    scene = timeline.scene("S3")
    for run in scene.spec["runs"]:
        n = int(run["n"])
        values = trajectory(n)
        start = scene.rel(float(run["offset"]), timeline.fps)
        step = int(scene.spec["frames_per_step"]) / timeline.fps
        for index in range(len(values) - 1):
            kind = "down" if values[index] % 2 == 0 else "up"
            events.append((start + index * step, values[index + 1], kind))
    scene = timeline.scene("S5")
    values = trajectory(27)
    step = int(scene.spec["frames_per_step"]) / timeline.fps
    for index in range(len(values) - 1):
        kind = "down" if values[index] % 2 == 0 else "up"
        events.append((scene.start + index * step, values[index + 1], kind))
    return events


def mix(timeline, voice: np.ndarray | None = None, sr: int = SR) -> tuple[np.ndarray, float, float, list[dict]]:
    n = timeline.n_frames * (sr // timeline.fps)
    notes = np.zeros((n, 2), dtype=np.float64)
    music = np.zeros((n, 2), dtype=np.float64)
    sync: list[dict] = []
    events = step_events(timeline)
    grouped: dict[str, list] = {}
    for when, value, kind in events:
        scene = timeline.at(min(when, timeline.duration - 1e-4))
        grouped.setdefault(scene.id, []).append((when, value, kind))
    for scene_id, rows in grouped.items():
        for index, (when, value, kind) in enumerate(rows):
            pan = -0.3 + 0.6 * (index / max(1, len(rows) - 1))
            wave = pluck(frequency(value), sr) if kind == "up" else bell(frequency(value), sr)
            _place(notes, wave, _sample(when, sr), pan, 0.22)
            sync.append({"time": when, "kind": kind, "value": value})

    s4 = timeline.scene("S4")
    s5 = timeline.scene("S5")
    # Pad through the opening scenes, then a glide up to the peak and down after it.
    pad_end = s4.start
    t = np.arange(_sample(pad_end, sr), dtype=np.float64) / sr
    for freq, gain in ((C3 / 2, 0.5), (C3 / 2 * 1.5, 0.35), (C3, 0.4)):
        music[: len(t), 0] += np.sin(2 * math.pi * freq * t) * gain
        music[: len(t), 1] += np.sin(2 * math.pi * freq * t) * gain
    music[: len(t)] *= 10 ** (-30 / 20) / 1.25

    peak_t = s5.start + 77 * int(s5.spec["frames_per_step"]) / timeline.fps
    glide_n = _sample(peak_t, sr) - _sample(s5.start, sr)
    if glide_n > 0:
        glide = np.arange(glide_n, dtype=np.float64) / sr
        semis = 7.0 * glide / max(glide[-1], 1e-6)
        for base in (C3 / 2, C3 / 2 * 1.5, C3):
            phase = 2 * math.pi * np.cumsum(base * (2 ** (semis / 12.0))) / sr
            tone = np.sin(phase) * (10 ** (-30 / 20))
            start = _sample(s5.start, sr)
            music[start : start + glide_n, 0] += tone
            music[start : start + glide_n, 1] += tone
    down_n = _sample(s5.end, sr) - _sample(peak_t, sr)
    if down_n > 0:
        glide = np.arange(down_n, dtype=np.float64) / sr
        semis = 7.0 * (1 - glide / max(glide[-1], 1e-6))
        for base in (C3 / 2, C3 / 2 * 1.5, C3):
            phase = 2 * math.pi * np.cumsum(base * (2 ** (semis / 12.0))) / sr
            tone = np.sin(phase) * (10 ** (-32 / 20))
            start = _sample(peak_t, sr)
            music[start : start + down_n, 0] += tone
            music[start : start + down_n, 1] += tone

    for when in (s4.start + 0.50, s4.start + 0.70, s4.start + 1.40, s4.start + 1.60):
        _place(music, _tone(55, 0.18, sr, 0.05), _sample(when, sr), 0.0, 0.20)
        sync.append({"time": when, "kind": "heartbeat"})

    riser_n = _sample(peak_t, sr) - _sample(s5.start, sr)
    if riser_n > 8:
        rng = np.random.default_rng(timeline.spec.get("seed", 1000) if False else 1000)
        noise = rng.standard_normal(riser_n)
        chunks = []
        size = 2048
        for start in range(0, riser_n, size):
            u = start / riser_n
            center = 200 * ((2500 / 200) ** u)
            low = max(40.0, center * 0.7)
            high = min(sr / 2 - 100, center * 1.3)
            sos = butter(2, [low, high], btype="band", fs=sr, output="sos")
            chunks.append(sosfilt(sos, noise[start : start + size]))
        band = np.concatenate(chunks)
        level = 10 ** ((-34 + 12 * np.linspace(0, 1, len(band))) / 20)
        tone = band * level
        tone *= 0.15 / (float(np.max(np.abs(tone))) or 1)
        start = _sample(s5.start, sr)
        music[start : start + len(tone), 0] += tone
        music[start : start + len(tone), 1] += tone

    _place(music, _tone(60, 0.7, sr, 0.18), _sample(peak_t, sr), 0.0, 0.45)
    burst = np.random.default_rng(7).standard_normal(int(0.15 * sr)) * np.exp(-np.arange(int(0.15 * sr)) / (0.04 * sr))
    _place(music, burst, _sample(peak_t, sr), 0.0, 0.12)
    sync.append({"time": peak_t, "kind": "peak"})

    def chord(freqs: list[float], when: float, seconds: float, gain: float) -> None:
        for freq in freqs:
            _place(music, _tone(freq, seconds, sr, seconds * 0.45), _sample(when, sr), 0.0, gain)
        sync.append({"time": when, "kind": "chord"})

    c4 = C3 * 2
    e4 = c4 * 2 ** (4 / 12)
    g4 = c4 * 2 ** (7 / 12)
    d4 = c4 * 2 ** (2 / 12)
    chord([e4], timeline.scene("S2").rel(timeline.scene("S2").spec["step_offsets"][-1], timeline.fps), 0.8, 0.18)
    # second arrival is the end of 9
    from fc_sat.collatz_timeline import proof_end

    chord([g4], proof_end(timeline, 9), 0.8, 0.18)
    chord([c4, e4, g4, d4], timeline.scene("S5").end, 1.2, 0.12)
    resolve = timeline.scene("S7").rel(float(timeline.scene("S7").spec["stamp_offset"]), timeline.fps)
    chord([c4, e4, g4, d4], resolve, 1.4, 0.14)
    hit = timeline.scene("S8").rel(float(timeline.scene("S8").spec["hit_offset"]), timeline.fps)
    chord([c4, d4, g4], hit, 3.2, 0.10)

    # Funnel ticks, capped at 24 per 100 ms, pitched from each line's peak.
    draw = timeline.scene("S6").rel(float(timeline.scene("S6").spec["relayout_s"]), timeline.fps)
    span = timeline.scene("S6").end - draw
    click = _tone(1800, 0.02, sr, 0.008) * 0.3
    window = 0
    in_window = 0
    for start_n in range(1, 1001):
        if start_n == 27:
            continue
        u = (start_n / 1000) ** (1 / 3)  # inverse of ease-in, so early starts are spaced out
        when = draw + u * span
        sample = _sample(when, sr)
        bucket = sample // int(0.1 * sr)
        if bucket != window:
            window = bucket
            in_window = 0
        if in_window >= 24:
            continue
        in_window += 1
        _place(
            music,
            click * (0.4 + 0.6 * pitch_index(peak(start_n)) / 14),
            sample,
            -0.2 + 0.4 * (start_n / 1000),
            0.05,
        )

    if voice is not None:
        stem = np.zeros((n, 2), dtype=np.float64)
        stem[: min(n, len(voice))] = voice[: min(n, len(voice))]
    else:
        stem = None
    mixed = duck(music, stem, sr, 7) + duck(notes, stem, sr, 4)
    if stem is not None:
        mixed += stem
    audio, lufs, true_peak = finish(mixed, sr)
    if len(audio) != n:
        fitted = np.zeros((n, 2), dtype=np.float64)
        fitted[: min(n, len(audio))] = audio[: min(n, len(audio))]
        audio = fitted
    return audio, lufs, true_peak, sync
