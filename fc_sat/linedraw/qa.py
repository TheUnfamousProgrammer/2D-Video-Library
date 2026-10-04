"""Stills, contact sheet, spectrogram, and the loudness timeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from scipy.signal import spectrogram

from fc_sat.audio import SR
from fc_sat.linedraw.choreo import N_FRAMES


STILLS = (0, 190, 500, 700, 743, 774, 940, 1100, 1400, 1490, 1600)


def write_stills(renderer, folder: Path, frames: tuple[int, ...] = STILLS) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for frame in frames:
        image = renderer.render(frame)
        path = folder / f"frame_{frame:04d}.png"
        Image.fromarray(image[:, :, ::-1], mode="RGB").save(path)
        paths.append(path)
        print(f"wrote {path}", flush=True)
    return paths


def contact_sheet(renderer, path: Path, step: int = 30) -> None:
    frames = list(range(0, N_FRAMES, step))
    thumbs = []
    for frame in frames:
        image = renderer.render(frame)
        rgb = Image.fromarray(image[:, :, ::-1], mode="RGB")
        rgb.thumbnail((180, 320))
        thumbs.append(rgb)
    columns = 8
    cell_w, cell_h = thumbs[0].size
    rows = (len(thumbs) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_w, rows * cell_h), (14, 17, 23))
    for index, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((index % columns) * cell_w, (index // columns) * cell_h))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    print(f"wrote {path}", flush=True)


def spectrogram_image(audio: np.ndarray, path: Path) -> None:
    mono = audio.mean(axis=1)
    _freqs, _times, power = spectrogram(mono, SR, nperseg=2048, noverlap=1536)
    db = 10.0 * np.log10(power + 1e-12)
    view = np.clip((db - (db.max() - 78.0)) / 78.0, 0.0, 1.0)
    view = np.flipud(view)
    rgb = np.zeros((*view.shape, 3), dtype=np.uint8)
    rgb[..., 0] = np.clip(view * 255, 0, 255)
    rgb[..., 1] = np.clip(np.maximum(0.0, view - 0.25) * 340, 0, 255)
    rgb[..., 2] = np.clip((1.0 - view) * 90 + view * 40, 0, 255)
    image = Image.fromarray(rgb, mode="RGB")
    image = image.resize((960, 540), Image.Resampling.BILINEAR)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def loudness_image(audio: np.ndarray, path: Path) -> None:
    mono = audio.mean(axis=1)
    win = int(0.4 * SR)
    hop = int(0.1 * SR)
    levels = []
    for start in range(0, max(1, len(mono) - win), hop):
        rms = float(np.sqrt(np.mean(mono[start : start + win] ** 2) + 1e-12))
        levels.append(20.0 * np.log10(rms))
    levels_arr = np.array(levels if levels else [-80.0])
    width, height = 960, 360
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:] = (14, 17, 23)
    lo, hi = -60.0, 0.0
    for index, level in enumerate(levels_arr):
        x = int(index / max(1, len(levels_arr) - 1) * (width - 1))
        y = int((1.0 - (np.clip(level, lo, hi) - lo) / (hi - lo)) * (height - 1))
        image[y:, x] = (232, 196, 126)
    Image.fromarray(image, mode="RGB").save(path)


def record_luma(renderer, path: Path) -> np.ndarray:
    values = np.empty(N_FRAMES, dtype=np.float64)
    for frame in range(N_FRAMES):
        image = renderer.render(frame)
        rgb = image[:, :, ::-1].astype(np.float64)
        values[frame] = (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]).mean() / 255.0
        if frame and frame % 120 == 0:
            print(f"luma {frame}/{N_FRAMES}", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, values)
    return values
