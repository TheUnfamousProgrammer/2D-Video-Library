"""Three silent motion tests. Each is 3 seconds at 540x960, 30 fps.

The diorama is drawn at 1080x1920 and scaled down, so the layout matches the stills.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from fc_sat.encode import pipe_raw_bgr
from fc_sat.paperfold_scenes import HANDOFF_FRAMES, handoff_start_frame
from fc_sat.paperfold_schedule import fold_frame
from fc_sat.paperfold_stills import DioramaStill

ROOT = Path(__file__).resolve().parents[1]
PREVIEW_W = 540
PREVIEW_H = 960
FPS = 30
FRAMES = 90


def sample_master(start: int, end: int, count: int = FRAMES) -> list[int]:
    if count <= 1:
        return [int(start)]
    return [int(round(start + (end - start) * i / (count - 1))) for i in range(count)]


def clip_ranges() -> dict[str, tuple[int, int]]:
    fold8 = fold_frame(8)
    fold15 = fold_frame(15)
    fold16 = fold_frame(16)
    handoff_end = handoff_start_frame(17) + HANDOFF_FRAMES
    return {
        "hook_to_fold8": (0, fold8),
        "fold15_to_16": (fold15, fold16 + 6),
        "street_to_city": (fold16, handoff_end),
    }


def motion_energy(frames: list[np.ndarray]) -> list[float]:
    energy = []
    for prev, frame in zip(frames, frames[1:]):
        energy.append(float(np.mean(np.abs(frame.astype(np.float32) - prev.astype(np.float32)))))
    return energy


def _resize(frame: np.ndarray) -> np.ndarray:
    return cv2.resize(frame, (PREVIEW_W, PREVIEW_H), interpolation=cv2.INTER_AREA)


def render_clips(out_dir: Path | None = None) -> dict[str, list[float]]:
    out_dir = out_dir or (ROOT / "out" / "motion")
    out_dir.mkdir(parents=True, exist_ok=True)
    drawer = DioramaStill("A")
    report = ["# Motion energy", "", "Mean absolute difference between consecutive 540x960 frames.", ""]
    series: dict[str, list[float]] = {}
    for name, (start, end) in clip_ranges().items():
        masters = sample_master(start, end)
        frames = [_resize(drawer.render_animated(frame)) for frame in masters]
        energy = motion_energy(frames)
        series[name] = energy
        path = out_dir / f"{name}.mp4"
        pipe_raw_bgr(
            frames,
            path,
            width=PREVIEW_W,
            height=PREVIEW_H,
            fps=FPS,
            audio_path=None,
            preset="veryfast",
            n_frames=len(frames),
        )
        spikes = _spikes(energy)
        report.append(f"## {name}")
        report.append("")
        report.append(f"Master frames {start}–{end}, {len(frames)} frames, silent.")
        report.append("")
        report.append("Frame energy (output frame 1 is the first difference):")
        report.append("")
        report.append(" ".join(f"{value:.3f}" for value in energy))
        report.append("")
        if spikes:
            report.append("Spikes (more than twice the median, and above 1.5): " + ", ".join(spikes))
        else:
            report.append("No frame jumps more than twice the median.")
        report.append("")
        print(f"wrote {path} ({len(frames)} frames)", flush=True)
    text = "\n".join(report) + "\n"
    (out_dir / "motion_energy.md").write_text(text)
    (ROOT / "out" / "motion_energy.md").write_text(text)
    return series


def _spikes(energy: list[float]) -> list[str]:
    if not energy:
        return []
    ordered = sorted(energy)
    median = ordered[len(ordered) // 2]
    found = []
    for index, value in enumerate(energy, start=1):
        if value > max(1.5, median * 2.0):
            found.append(f"frame {index}={value:.2f}")
    return found
