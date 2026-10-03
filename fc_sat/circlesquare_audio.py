"""A-minor score for the circlesquare short.

The oscillators, the drums, and the loudness master are the polycircle music
engine. This module only hands that engine the circlesquare timeline.
"""

from __future__ import annotations

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

__all__ = [
    "detect_onsets",
    "events_aligned",
    "master",
    "mix",
    "onset_sample",
    "sidechain",
    "sync_rows",
]


def mix(events: dict, *, seed: int = 11, n_frames: int = 1824, sr: int = SR):
    beat, sfx = render_buses(events, seed, n_frames, sr)
    full, lufs, peak = master(beat + sfx, sr, tail_frames=18)
    return full, sfx, lufs, peak
