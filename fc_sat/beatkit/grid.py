"""Frame-exact musical grid. At 150 bpm and 60 fps every beat is 24 frames."""

from __future__ import annotations

BPM = 150.0
FPS = 60
SR = 48000
N_FRAMES = 1824
FRAMES_PER_BEAT = 24
FRAMES_PER_BAR = 96
FRAMES_PER_EIGHTH = 12
FRAMES_PER_SIXTEENTH = 6
HOP = SR // FPS  # 800 samples per frame


def snap_frame(beats: float, bpm: float, fps: int) -> tuple[int, float]:
    """Nearest frame to a beat position, and the absolute snap error in frames."""
    exact = float(beats) * (60.0 / float(bpm)) * int(fps)
    frame = int(round(exact))
    return frame, abs(exact - frame)
