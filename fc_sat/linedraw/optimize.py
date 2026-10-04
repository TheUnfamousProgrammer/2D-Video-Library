"""Greedy straight-line fit. Dark ink subtracts, light ink adds, and only a positive gain is kept."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.draw import line_aa
from skimage.metrics import structural_similarity

GRID_W = 254
GRID_H = 368
INK = 0.06
CANDIDATES = 150
PLANNED = 36_000
REJECT_LIMIT = 2_000
LENGTH_START = 300.0
LENGTH_END = 40.0


def l_max_at(kept: int, planned: int = PLANNED) -> float:
    """Long chaotic strokes at the start, short detail strokes at the planned count."""
    span = max(int(planned), 1)
    u = min(max(int(kept), 0), span) / span
    return LENGTH_START + (LENGTH_END - LENGTH_START) * u


def likeness(canvas: np.ndarray, target: np.ndarray) -> float:
    blurred_canvas = gaussian_filter(np.clip(canvas, 0.0, 1.0), 1.5)
    blurred_target = gaussian_filter(np.clip(target, 0.0, 1.0), 1.5)
    score = structural_similarity(blurred_canvas, blurred_target, data_range=1.0)
    return float(score)


def find_l_aha(counts: np.ndarray, scores: np.ndarray) -> int:
    """First logged count whose likeness reaches 60% of the final logged likeness."""
    if len(counts) == 0:
        return 0
    goal = 0.60 * float(scores[-1])
    for count, score in zip(counts, scores):
        if float(score) >= goal:
            return int(count)
    return int(counts[-1])


def _line_gain(canvas: np.ndarray, target: np.ndarray, weight: np.ndarray, x0: float, y0: float, x1: float, y1: float, ink: float) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    height, width = canvas.shape
    rr, cc, val = line_aa(int(round(y0)), int(round(x0)), int(round(y1)), int(round(x1)))
    mask = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
    if not np.any(mask):
        empty = np.empty(0, dtype=np.int32)
        return -1.0, -1.0, empty, empty, np.empty(0, dtype=np.float64)
    rr = rr[mask]
    cc = cc[mask]
    coverage = val[mask].astype(np.float64)
    delta = ink * coverage
    residual = target[rr, cc] - canvas[rr, cc]
    local = weight[rr, cc]
    cross = float(np.dot(local, delta * residual))
    base = float(np.dot(local, delta * delta))
    # d = s * a * coverage. s = -1 is the first value, s = +1 is the second.
    return -2.0 * cross - base, 2.0 * cross - base, rr, cc, coverage


@dataclass
class Fit:
    x0: np.ndarray
    y0: np.ndarray
    x1: np.ndarray
    y1: np.ndarray
    ink: np.ndarray
    step: np.ndarray
    thrown: np.ndarray
    likeness_counts: np.ndarray
    likeness: np.ndarray
    canvas: np.ndarray
    seconds: float
    reject_x0: np.ndarray
    reject_y0: np.ndarray
    reject_x1: np.ndarray
    reject_y1: np.ndarray
    candidates: int
    planned: int


def optimize(
    target: np.ndarray,
    weight: np.ndarray,
    *,
    seed: int,
    max_kept: int = PLANNED,
    candidates: int = CANDIDATES,
    planned: int = PLANNED,
    reject_limit: int = REJECT_LIMIT,
    record_likeness: bool = True,
    store_rejects: int = 0,
) -> Fit:
    """Throw random lines and keep the single best positive-gain stroke each step."""
    if target.shape != weight.shape:
        raise ValueError("target and importance map shapes differ")
    height, width = target.shape
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    canvas = np.full((height, width), 0.5, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    xs0: list[float] = []
    ys0: list[float] = []
    xs1: list[float] = []
    ys1: list[float] = []
    inks: list[int] = []
    steps: list[int] = []
    thrown_at: list[int] = []
    like_counts: list[int] = []
    like_scores: list[float] = []
    rej0: list[float] = []
    rej1: list[float] = []
    rej2: list[float] = []
    rej3: list[float] = []
    kept = 0
    thrown = 0
    streak = 0
    step = 0
    started = time.perf_counter()
    max_steps = max(int(max_kept) * 6, int(reject_limit) + 1)
    while kept < int(max_kept) and streak < int(reject_limit) and step < max_steps:
        length_cap = l_max_at(kept, planned)
        centers_x = rng.random(candidates) * width
        centers_y = rng.random(candidates) * height
        angles = rng.random(candidates) * np.pi
        lengths = length_cap * (0.35 + 0.65 * rng.random(candidates))
        half_x = np.cos(angles) * lengths * 0.5
        half_y = np.sin(angles) * lengths * 0.5
        best_gain = 0.0
        best_index = -1
        best_ink = 0
        best_pixels: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        for index in range(candidates):
            gain_dark, gain_light, rr, cc, coverage = _line_gain(
                canvas,
                target,
                weight,
                float(centers_x[index] - half_x[index]),
                float(centers_y[index] - half_y[index]),
                float(centers_x[index] + half_x[index]),
                float(centers_y[index] + half_y[index]),
                INK,
            )
            if gain_dark > best_gain:
                best_gain = gain_dark
                best_index = index
                best_ink = -1
                best_pixels = (rr, cc, coverage)
            if gain_light > best_gain:
                best_gain = gain_light
                best_index = index
                best_ink = 1
                best_pixels = (rr, cc, coverage)
        thrown += int(candidates)
        step += 1
        if best_index < 0 or best_pixels is None or best_gain <= 0.0:
            streak += 1
            continue
        streak = 0
        rr, cc, coverage = best_pixels
        canvas[rr, cc] += best_ink * INK * coverage
        xs0.append(float(centers_x[best_index] - half_x[best_index]))
        ys0.append(float(centers_y[best_index] - half_y[best_index]))
        xs1.append(float(centers_x[best_index] + half_x[best_index]))
        ys1.append(float(centers_y[best_index] + half_y[best_index]))
        inks.append(int(best_ink))
        steps.append(step - 1)
        thrown_at.append(thrown)
        if store_rejects and kept < store_rejects:
            spare = [i for i in range(min(3, candidates)) if i != best_index]
            while len(spare) < 3 and len(spare) < candidates:
                spare.append((best_index + len(spare) + 1) % candidates)
            for spare_index in spare[:3]:
                rej0.append(float(centers_x[spare_index] - half_x[spare_index]))
                rej1.append(float(centers_y[spare_index] - half_y[spare_index]))
                rej2.append(float(centers_x[spare_index] + half_x[spare_index]))
                rej3.append(float(centers_y[spare_index] + half_y[spare_index]))
        kept += 1
        if record_likeness and kept % 100 == 0:
            score = likeness(canvas, target)
            like_counts.append(kept)
            like_scores.append(score)
            elapsed = time.perf_counter() - started
            rate = kept / max(elapsed, 1e-6)
            remaining = max(0, int(max_kept) - kept)
            print(
                f"kept {kept:,} thrown {thrown:,} likeness {score * 100:.1f}% eta {remaining / rate / 60.0:.1f} min",
                flush=True,
            )
    elapsed = time.perf_counter() - started
    print(f"optimize kept {kept:,} of {thrown:,} thrown in {elapsed:.1f}s (reject streak {streak})", flush=True)
    return Fit(
        x0=np.asarray(xs0, dtype=np.float32),
        y0=np.asarray(ys0, dtype=np.float32),
        x1=np.asarray(xs1, dtype=np.float32),
        y1=np.asarray(ys1, dtype=np.float32),
        ink=np.asarray(inks, dtype=np.int8),
        step=np.asarray(steps, dtype=np.int32),
        thrown=np.asarray(thrown_at, dtype=np.int32),
        likeness_counts=np.asarray(like_counts, dtype=np.int32),
        likeness=np.asarray(like_scores, dtype=np.float64),
        canvas=canvas,
        seconds=elapsed,
        reject_x0=np.asarray(rej0, dtype=np.float32),
        reject_y0=np.asarray(rej1, dtype=np.float32),
        reject_x1=np.asarray(rej2, dtype=np.float32),
        reject_y1=np.asarray(rej3, dtype=np.float32),
        candidates=int(candidates),
        planned=int(planned),
    )


def replay(fit_x0, fit_y0, fit_x1, fit_y1, fit_ink, count: int, shape: tuple[int, int]) -> np.ndarray:
    """Rebuild the canvas from the first ``count`` kept lines. Matches a persistent apply."""
    canvas = np.full(shape, 0.5, dtype=np.float64)
    height, width = shape
    n = min(int(count), len(fit_x0))
    for index in range(n):
        rr, cc, val = line_aa(
            int(round(float(fit_y0[index]))),
            int(round(float(fit_x0[index]))),
            int(round(float(fit_y1[index]))),
            int(round(float(fit_x1[index]))),
        )
        mask = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
        if not np.any(mask):
            continue
        canvas[rr[mask], cc[mask]] += int(fit_ink[index]) * INK * val[mask]
    return canvas


def save_fit(fit: Fit, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        x0=fit.x0,
        y0=fit.y0,
        x1=fit.x1,
        y1=fit.y1,
        ink=fit.ink,
        step=fit.step,
        thrown=fit.thrown,
        likeness_counts=fit.likeness_counts,
        likeness=fit.likeness,
        seconds=np.float64(fit.seconds),
        reject_x0=fit.reject_x0,
        reject_y0=fit.reject_y0,
        reject_x1=fit.reject_x1,
        reject_y1=fit.reject_y1,
        candidates=np.int32(fit.candidates),
        planned=np.int32(fit.planned),
    )


def load_fit(path) -> Fit:
    data = np.load(path)
    return Fit(
        x0=data["x0"],
        y0=data["y0"],
        x1=data["x1"],
        y1=data["y1"],
        ink=data["ink"],
        step=data["step"],
        thrown=data["thrown"],
        likeness_counts=data["likeness_counts"],
        likeness=data["likeness"],
        canvas=np.empty((0, 0)),
        seconds=float(data["seconds"]),
        reject_x0=data["reject_x0"],
        reject_y0=data["reject_y0"],
        reject_x1=data["reject_x1"],
        reject_y1=data["reject_y1"],
        candidates=int(data["candidates"]),
        planned=int(data["planned"]),
    )
