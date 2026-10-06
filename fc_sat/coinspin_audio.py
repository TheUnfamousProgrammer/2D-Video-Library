"""A-minor score for the coinspin short.

The oscillators, the drums, and the loudness master are the polycircle music
engine. This module hands that engine the coinspin timeline: every quarter turn
of the arrow is a pluck and every finished spin is a bell.
"""

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

# The shared master stops the wav at -2.3 dBTP. The AAC encode can still
# overshoot -1 dBTP on bell-heavy mixes, so this mix keeps a fixed trim.
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
    """Pull a fixed trim so the AAC file stays at or under -1 dBTP."""
    gain = 10.0 ** (-_AAC_TRIM_DB / 20.0)
    return np.asarray(audio, dtype=np.float64) * gain, lufs - _AAC_TRIM_DB, peak - _AAC_TRIM_DB


def mix(events: dict, *, seed: int = 13, n_frames: int = 1824, sr: int = SR):
    beat, sfx = render_buses(events, seed, n_frames, sr)
    full, lufs, peak = master(beat + sfx, sr, tail_frames=18)
    full, lufs, peak = trim_for_aac(full, lufs, peak)
    return full, sfx, lufs, peak
