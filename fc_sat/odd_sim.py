"""Static grids for Odd One Out.

Every item sits on its cell center. The only thing that differs is the level's
difference. Consecutive levels never share a cell. A failed draw reseeds with
``seed + 1000*k`` up to 20 times.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from fc_sat.color import lch_to_oklab
from fc_sat.odd_config import HUE_LIGHTNESS, LevelSpec, OddConfig, active_rung, build_timeline, cvd_floor, nominal_value
from fc_sat.odd_diff import DIR_NAMES, choose_odd_lab


class ConstraintFailure(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def make_rng(seed: int, level_id: int, stream: int) -> np.random.Generator:
    sequence = np.random.SeedSequence([int(seed) & 0xFFFFFFFF, int(level_id), int(stream)])
    return np.random.Generator(np.random.PCG64(sequence))


def reseed_loop(seed: int, max_attempts: int, attempt_fn):
    """Call ``attempt_fn(trial_seed, k)`` until it returns a truthy ok flag."""
    notes: list[str] = []
    for k in range(max_attempts):
        trial = int(seed) + 1000 * k
        ok, payload = attempt_fn(trial, k)
        if ok:
            return payload, k, notes
        notes.append(str(payload))
    raise RuntimeError(f"constraints failed after {max_attempts} attempts. Last notes: {notes[-3:]}")


def grid_geometry(cfg: OddConfig, level: LevelSpec) -> tuple[np.ndarray, list[tuple[float, float, float, float]], float]:
    """Centers (n, 2), cell boxes, and the cell size. Row-major, no jitter."""
    n = level.grid
    span_x = (cfg.field_x1 - cfg.field_x0) / n
    span_y = (cfg.field_y1 - cfg.field_y0) / n
    centers = np.empty((n * n, 2), dtype=np.float64)
    cells: list[tuple[float, float, float, float]] = []
    index = 0
    for row in range(n):
        for col in range(n):
            x0 = cfg.field_x0 + col * span_x
            y0 = cfg.field_y0 + row * span_y
            centers[index] = (x0 + span_x / 2.0, y0 + span_y / 2.0)
            cells.append((x0, y0, x0 + span_x, y0 + span_y))
            index += 1
    return centers, cells, float(min(span_x, span_y))


def location_phrase(row: int, col: int, grid: int) -> str:
    def band(index: int, low: str, mid: str, high: str) -> str:
        frac = (index + 0.5) / grid
        if frac < 1.0 / 3.0:
            return low
        if frac > 2.0 / 3.0:
            return high
        return mid

    vertical = band(row, "upper", "middle", "lower")
    horizontal = band(col, "left", "center", "right")
    if vertical == "middle" and horizontal == "center":
        return "center"
    if horizontal == "center":
        return vertical
    if vertical == "middle":
        return horizontal
    return f"{vertical} {horizontal}"


@dataclass
class LevelSim:
    level: LevelSpec
    seed: int
    odd_index: int
    row: int
    col: int
    centers: np.ndarray
    cells: list
    cell_size: float
    base_lab: np.ndarray
    odd_lab: np.ndarray
    diff_params: dict
    reseeds: list = field(default_factory=list)

    @property
    def phrase(self) -> str:
        return location_phrase(self.row, self.col, self.level.grid)


@dataclass
class Show:
    cfg: OddConfig
    levels: dict[int, LevelSim]
    timeline: tuple


def _simulate_once(cfg: OddConfig, level: LevelSpec, seed: int, forbidden: tuple[int, int] | None) -> LevelSim:
    centers, cells, cell_size = grid_geometry(cfg, level)
    choice = make_rng(seed, level.id, 1)
    color = make_rng(seed, level.id, 2)
    odd_index = int(choice.integers(0, level.count))
    row, col = divmod(odd_index, level.grid)
    if forbidden is not None and (row, col) == forbidden and level.grid > forbidden[0] and level.grid > forbidden[1]:
        raise ConstraintFailure(f"odd cell row {row} col {col} repeats the previous level")
    # A cell only repeats when both levels can address it.
    if forbidden is not None and row == forbidden[0] and col == forbidden[1]:
        if row < level.grid and col < level.grid and forbidden[0] < level.grid and forbidden[1] < level.grid:
            raise ConstraintFailure(f"odd cell row {row} col {col} repeats the previous level")
    hue = float(color.uniform(0.0, 2.0 * np.pi))
    base = lch_to_oklab(cfg.tier.base_l, cfg.tier.base_c, hue)
    rung = active_rung(cfg.tier, level)
    nominal = nominal_value(level.difference, rung)
    odd_info = None
    if level.difference == "hue":
        odd_info = choose_odd_lab(
            base,
            min_distance=nominal,
            min_lightness=HUE_LIGHTNESS,
            min_cvd=cvd_floor(nominal),
            chroma=cfg.tier.base_c,
        )
        odd_lab = odd_info["lab"]
    else:
        odd_lab = np.array(base, dtype=np.float64)
    params = {
        "kind": level.difference,
        "rung": rung,
        "nominal": nominal,
        "base_lab": [float(v) for v in base],
        "odd_lab": [float(v) for v in odd_lab],
        "base_hue": hue,
        "row": row,
        "col": col,
        "phrase": location_phrase(row, col, level.grid),
        "size": level.size,
    }
    if odd_info is not None:
        params.update(
            {
                "distance": odd_info["distance"],
                "lightness_delta": odd_info["lightness_delta"],
                "hue_offset_rad": odd_info["hue_offset_rad"],
                "cvd": odd_info["cvd"],
                "rejected_offsets": odd_info["rejected_offsets"],
                "requested_distance": odd_info["requested_distance"],
                "cvd_floor": cvd_floor(nominal),
            }
        )
    if level.difference == "tilt":
        params["tilt_degrees"] = nominal
    if level.difference == "detail":
        direction = int(make_rng(seed, level.id, 3).integers(0, 8))
        params["offset"] = nominal
        params["direction"] = direction
        params["direction_name"] = DIR_NAMES[direction]
        params["dot_fraction"] = cfg.dot_fraction
        params["detail_mode"] = cfg.detail_mode
        params["dot_radius"] = cfg.dot_fraction * (level.size / 2.0)
    half = level.size / 2.0
    for cx, cy in centers:
        if cx - half < cfg.field_x0 or cx + half > cfg.field_x1 or cy - half < cfg.field_y0 or cy + half > cfg.field_y1:
            raise ConstraintFailure("an item leaves the field")
    return LevelSim(
        level=level,
        seed=seed,
        odd_index=odd_index,
        row=row,
        col=col,
        centers=centers,
        cells=cells,
        cell_size=cell_size,
        base_lab=np.array(base, dtype=np.float64),
        odd_lab=np.array(odd_lab, dtype=np.float64),
        diff_params=params,
    )


def simulate_level(
    cfg: OddConfig,
    level: LevelSpec,
    seed: int,
    forbidden: tuple[int, int] | None = None,
) -> LevelSim:
    print(f"sim level {level.id} {level.difference} grid={level.grid} seed={seed}", flush=True)

    def attempt(trial: int, k: int):
        try:
            return True, _simulate_once(cfg, level, trial, forbidden)
        except ConstraintFailure as exc:
            print(f"  reseed k={k} seed={trial}: {exc.reason}", flush=True)
            return False, exc.reason

    result, _k, notes = reseed_loop(seed, cfg.max_attempts, attempt)
    result.reseeds = notes
    print(
        f"  level {level.id} rung {result.diff_params['rung']} "
        f"odd cell row {result.row} col {result.col} ({result.phrase}) "
        f"reseeds={len(notes)}",
        flush=True,
    )
    return result


def simulate_show(cfg: OddConfig, *, seed: int | None = None) -> Show:
    used = cfg.seed if seed is None else int(seed)
    levels: dict[int, LevelSim] = {}
    forbidden: tuple[int, int] | None = None
    for level in cfg.levels:
        sim = simulate_level(cfg, level, used, forbidden)
        levels[level.id] = sim
        forbidden = (sim.row, sim.col)
    return Show(cfg=cfg, levels=levels, timeline=build_timeline(cfg))


def save_show(path, show: Show) -> None:
    payload = {"level_ids": np.array([level.id for level in show.cfg.levels], dtype=np.int32)}
    for level_id, sim in show.levels.items():
        key = str(level_id)
        payload[f"centers_{key}"] = sim.centers
        payload[f"odd_{key}"] = np.array([sim.odd_index, sim.row, sim.col], dtype=np.int32)
        payload[f"size_{key}"] = np.array([sim.level.size], dtype=np.float64)
    np.savez_compressed(path, **payload)
