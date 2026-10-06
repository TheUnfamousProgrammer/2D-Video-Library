"""The scale short's score: the polycircle music engine playing the scale timeline."""

from __future__ import annotations

import numpy as np

from fc_sat.audio import SR
from fc_sat.polycircle_audio import (
    detect_onsets,
    events_aligned,
    master,
    onset_sample,
    render_buses,
    sidechain,
    sync_rows,
)

# The shared master stops the wav at -2.3 dBTP. Bell-heavy mixes can still overshoot
# -1 dBTP after the AAC encode, so this mix keeps a fixed trim.
_AAC_TRIM_DB = 0.55

__all__ = [
    "detect_onsets",
    "events_aligned",
    "master",
    "mix",
    "onset_sample",
    "sidechain",
    "sync_rows",
    "trim_for_aac",
]


def trim_for_aac(audio: np.ndarray, lufs: float, peak: float) -> tuple[np.ndarray, float, float]:
    gain = 10.0 ** (-_AAC_TRIM_DB / 20.0)
    return np.asarray(audio, dtype=np.float64) * gain, lufs - _AAC_TRIM_DB, peak - _AAC_TRIM_DB


def mix(events: dict, *, seed: int = 29, n_frames: int = 1824, sr: int = SR):
    beat, sfx = render_buses(events, seed, n_frames, sr)
    full, lufs, peak = master(beat + sfx, sr, tail_frames=18)
    full, lufs, peak = trim_for_aac(full, lufs, peak)
    return full, sfx, lufs, peak
