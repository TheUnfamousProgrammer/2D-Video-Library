"""Exact color-and-space assignment of A cells onto B cells."""

from __future__ import annotations

import hashlib
import struct
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from .color import rgb_u8_to_oklab

# float64 cost matrix. Past this the solver is refused before the array exists.
_MEMORY_LIMIT_BYTES = 1_000_000_000


class AssignmentError(ValueError):
    """The particle matching cannot be solved."""


@dataclass(frozen=True)
class Assignment:
    permutation: np.ndarray
    mean_error: float
    p95_error: float
    solve_seconds: float
    cache_key: str
    from_cache: bool


def cache_key(grid_a: np.ndarray, grid_b: np.ndarray, cols: int, rows: int, spatial_weight: float) -> str:
    """sha256 of the prepared pixels, the grid, and the spatial weight.

    Hook text, timing, seed, and recolor strength are intentionally absent.
    """
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(grid_a).tobytes())
    digest.update(np.ascontiguousarray(grid_b).tobytes())
    digest.update(struct.pack("<iid", int(cols), int(rows), float(spatial_weight)))
    return digest.hexdigest()


def assign(
    grid_a: np.ndarray,
    grid_b: np.ndarray,
    spatial_weight: float,
    *,
    cache_dir: Path | None = Path(".cache"),
) -> Assignment:
    """Return a permutation ``perm[i] = j`` minimizing color plus spatial cost.

    Cost is ``||Lab_A[i] - Lab_B[j]||^2 + spatial_weight * (d_ij / diag)^2``.
    ``d_ij`` is the distance between cell centers in grid units and ``diag``
    is the diagonal of the ``cols x rows`` rectangle.
    """
    rows, cols = grid_a.shape[:2]
    if grid_b.shape[:2] != (rows, cols):
        raise AssignmentError("prepared images must share the grid shape")
    n = cols * rows
    bytes_needed = n * n * 8
    if bytes_needed > _MEMORY_LIMIT_BYTES:
        raise AssignmentError(
            f"assignment matrix would be {bytes_needed / 1e9:.2f} GB, above the 1 GB limit. Use a smaller grid."
        )
    key = cache_key(grid_a, grid_b, cols, rows, spatial_weight)
    if cache_dir is not None:
        path = Path(cache_dir) / f"morph_{key}.npz"
        if path.is_file():
            loaded = np.load(path)
            return Assignment(
                permutation=np.asarray(loaded["permutation"], dtype=np.int32),
                mean_error=float(loaded["mean_error"]),
                p95_error=float(loaded["p95_error"]),
                solve_seconds=float(loaded["solve_seconds"]),
                cache_key=key,
                from_cache=True,
            )
    lab_a = rgb_u8_to_oklab(grid_a).reshape(n, 3)
    lab_b = rgb_u8_to_oklab(grid_b).reshape(n, 3)
    cost = _cost(lab_a, lab_b, cols, rows, spatial_weight)
    started = time.perf_counter()
    row_ind, col_ind = linear_sum_assignment(cost)
    elapsed = time.perf_counter() - started
    permutation = np.empty(n, dtype=np.int32)
    permutation[row_ind] = col_ind.astype(np.int32)
    matched = np.linalg.norm(lab_a - lab_b[permutation], axis=1)
    mean_error = float(np.mean(matched))
    p95_error = float(np.percentile(matched, 95))
    result = Assignment(permutation, mean_error, p95_error, elapsed, key, False)
    if cache_dir is not None:
        path = Path(cache_dir) / f"morph_{key}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp.npz")
        np.savez(
            temporary,
            permutation=permutation,
            mean_error=np.float64(mean_error),
            p95_error=np.float64(p95_error),
            solve_seconds=np.float64(elapsed),
        )
        temporary.replace(path)
    return result


def _cost(lab_a: np.ndarray, lab_b: np.ndarray, cols: int, rows: int, spatial_weight: float) -> np.ndarray:
    a2 = np.sum(lab_a * lab_a, axis=1)
    b2 = np.sum(lab_b * lab_b, axis=1)
    color = a2[:, None] + b2[None, :] - 2.0 * (lab_a @ lab_b.T)
    ys, xs = np.mgrid[0:rows, 0:cols]
    px = (xs.ravel().astype(np.float64) + 0.5)
    py = (ys.ravel().astype(np.float64) + 0.5)
    p2 = px * px + py * py
    spatial = p2[:, None] + p2[None, :] - 2.0 * (px[:, None] * px[None, :] + py[:, None] * py[None, :])
    np.maximum(spatial, 0.0, out=spatial)
    diag = float(np.hypot(cols, rows))
    if diag <= 0:
        raise AssignmentError("grid diagonal must be positive")
    weight = float(spatial_weight)
    if weight:
        color += weight * (spatial / (diag * diag))
    return np.ascontiguousarray(color, dtype=np.float64)
