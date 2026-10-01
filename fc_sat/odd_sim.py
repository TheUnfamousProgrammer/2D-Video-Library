"""Odd One Out fields.

Every item, including the odd one, is drawn from the same streams. The odd
index comes from its own stream and does not change speeds, headings, or phases.
A failed anti-bias or overlap check reseeds with ``seed + 1000 * k`` and stops
after ``max_attempts`` instead of relaxing the rule.

Motion law, identical for every item:
  speed ~ Normal(mean, spread * mean), clipped to clip * mean
  heading0 ~ Uniform(0, 2π), omega0 = 0
  each substep: omega += Normal(0, angular_std * sqrt(dt)), clipped to ±omega_clip
  heading += omega * dt
  velocity = speed * (cos heading, sin heading), y grows downward
  walls reflect; centers closer than repulse * r are pushed apart
    (threshold = repulse/2 * (ri + rj), which is repulse * r when radii match)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from fc_sat.color import lch_to_oklab
from fc_sat.odd_config import LevelSpec, OddConfig, build_timeline
from fc_sat.odd_diff import choose_odd_lab

STREAMS = {"layout": 1, "motion": 2, "spin": 3, "pulse": 4, "odd": 5}


class ConstraintFailure(RuntimeError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def make_rng(seed: int, level_id: int, name: str) -> np.random.Generator:
    sequence = np.random.SeedSequence([int(seed) & 0xFFFFFFFF, int(level_id), STREAMS[name]])
    return np.random.Generator(np.random.PCG64(sequence))


def reseed_loop(seed: int, max_attempts: int, attempt_fn):
    """Call ``attempt_fn(trial_seed, k)`` until it returns a result or 20 attempts fail.

    ``attempt_fn`` returns ``(ok, payload_or_reason)``. On success the payload is returned.
    """
    notes: list[dict] = []
    for k in range(max_attempts):
        trial = int(seed) + 1000 * k
        ok, payload = attempt_fn(trial, k)
        if ok:
            return payload, k, notes
        notes.append({"attempt": k, "seed": trial, "reason": str(payload)})
    last = notes[-1]["reason"] if notes else "no attempt ran"
    raise RuntimeError(
        f"constraints failed after {max_attempts} attempts; last seed {seed + 1000 * (max_attempts - 1)}; {last}"
    )


def item_bounds(cfg: OddConfig, radius: float) -> tuple[float, float, float, float, float]:
    inset = cfg.margin + float(radius)
    x0 = cfg.field_x0 + inset
    x1 = cfg.field_x1 - inset
    y0 = cfg.field_y0 + inset
    y1 = cfg.field_y1 - inset
    corner = max(0.0, cfg.corner_radius - inset)
    return x0, y0, x1, y1, corner


def _reflect_walls(pos: np.ndarray, heading: np.ndarray, radii: np.ndarray, cfg: OddConfig) -> tuple[np.ndarray, np.ndarray]:
    inset = cfg.margin + radii
    x0 = cfg.field_x0 + inset
    x1 = cfg.field_x1 - inset
    y0 = cfg.field_y0 + inset
    y1 = cfg.field_y1 - inset
    x = pos[:, 0].copy()
    y = pos[:, 1].copy()
    h = heading.copy()
    hit = x < x0
    x = np.where(hit, 2.0 * x0 - x, x)
    h = np.where(hit, math.pi - h, h)
    hit = x > x1
    x = np.where(hit, 2.0 * x1 - x, x)
    h = np.where(hit, math.pi - h, h)
    hit = y < y0
    y = np.where(hit, 2.0 * y0 - y, y)
    h = np.where(hit, -h, h)
    hit = y > y1
    y = np.where(hit, 2.0 * y1 - y, y)
    h = np.where(hit, -h, h)
    # Corner arcs only matter when the inset is smaller than the field corner.
    if float(np.max(cfg.corner_radius - inset)) > 0.5:
        cr = np.maximum(0.0, cfg.corner_radius - inset)
        centers = (
            (x0 + cr, y0 + cr, -1.0, -1.0),
            (x1 - cr, y0 + cr, 1.0, -1.0),
            (x0 + cr, y1 - cr, -1.0, 1.0),
            (x1 - cr, y1 - cr, 1.0, 1.0),
        )
        vx = np.cos(h)
        vy = np.sin(h)
        for cx, cy, sx, sy in centers:
            dx = x - cx
            dy = y - cy
            outside = (sx * dx > 0.0) & (sy * dy > 0.0) & (cr > 0.5)
            dist = np.sqrt(dx * dx + dy * dy)
            far = outside & (dist > cr)
            if not np.any(far):
                continue
            scale = np.ones_like(dist)
            scale[far] = cr[far] / np.maximum(dist[far], 1e-6)
            x = np.where(far, cx + dx * scale, x)
            y = np.where(far, cy + dy * scale, y)
            nx = np.zeros_like(dist)
            ny = np.zeros_like(dist)
            nx[far] = dx[far] / np.maximum(dist[far], 1e-6)
            ny[far] = dy[far] / np.maximum(dist[far], 1e-6)
            outward = far & (vx * nx + vy * ny > 0.0)
            dot = vx * nx + vy * ny
            vx = np.where(outward, vx - 2.0 * dot * nx, vx)
            vy = np.where(outward, vy - 2.0 * dot * ny, vy)
        h = np.arctan2(vy, vx)
    pos = np.column_stack([x, y])
    return pos, h


def _repel(pos: np.ndarray, radii: np.ndarray, factor: float, passes: int) -> np.ndarray:
    half = factor / 2.0
    out = pos.copy()
    n = len(out)
    eye = np.eye(n, dtype=bool)
    for _ in range(passes):
        delta = out[:, None, :] - out[None, :, :]
        dist = np.sqrt(np.sum(delta * delta, axis=-1))
        dist[eye] = 1e9
        thresh = half * (radii[:, None] + radii[None, :])
        gap = np.maximum(thresh - dist, 0.0)
        gap[eye] = 0.0
        direction = delta / dist[..., None]
        out = out + (direction * (gap * 0.5)[..., None]).sum(axis=1)
    return out


def _min_clearance(positions: np.ndarray, radii: np.ndarray) -> float:
    limit = radii[:, None] + radii[None, :]
    worst = 1e9
    eye = np.eye(len(radii), dtype=bool)
    for frame in positions:
        delta = frame[:, None, :] - frame[None, :, :]
        dist = np.sqrt(np.sum(delta * delta, axis=-1))
        dist[eye] = 1e9
        worst = min(worst, float((dist - limit).min()))
    return worst


def _place(n: int, bounds: tuple[float, float, float, float], rng: np.random.Generator) -> np.ndarray:
    x0, y0, x1, y1 = bounds
    cols = max(1, int(math.ceil(math.sqrt(n * max(x1 - x0, 1.0) / max(y1 - y0, 1.0)))))
    rows = int(math.ceil(n / cols))
    xs = np.linspace(x0, x1, cols) if cols > 1 else np.array([(x0 + x1) * 0.5])
    ys = np.linspace(y0, y1, rows) if rows > 1 else np.array([(y0 + y1) * 0.5])
    grid = np.array([(xs[col % cols], ys[col // cols]) for col in range(n)], dtype=np.float64)
    cell = min((x1 - x0) / max(cols, 1), (y1 - y0) / max(rows, 1))
    jitter = rng.uniform(-1.0, 1.0, size=(n, 2)) * cell * 0.12
    pos = grid + jitter
    pos[:, 0] = np.clip(pos[:, 0], x0, x1)
    pos[:, 1] = np.clip(pos[:, 1], y0, y1)
    return pos


def sample_drivers(cfg: OddConfig, level: LevelSpec, seed: int) -> dict:
    """Draw layout, motion, phases, and the odd index. The odd index is last and separate."""
    n = level.count
    layout = make_rng(seed, level.id, "layout")
    motion = make_rng(seed, level.id, "motion")
    spin = make_rng(seed, level.id, "spin")
    pulse = make_rng(seed, level.id, "pulse")
    odd_rng = make_rng(seed, level.id, "odd")
    hue = float(layout.uniform(0.0, 2.0 * math.pi))
    base = lch_to_oklab(cfg.tier.base_l, cfg.tier.base_c, hue)
    radii = np.full(n, cfg.radius, dtype=np.float64)
    odd_index = int(odd_rng.integers(0, n))
    if level.difference == "size":
        radii[odd_index] = cfg.radius * cfg.tier.size_ratio
    lo = cfg.speed_clip[0] * cfg.speed_mean
    hi = cfg.speed_clip[1] * cfg.speed_mean
    speeds = np.clip(
        motion.normal(cfg.speed_mean, cfg.speed_mean * cfg.speed_spread, size=n),
        lo,
        hi,
    )
    headings = motion.uniform(0.0, 2.0 * math.pi, size=n)
    bounds = item_bounds(cfg, float(radii.max()))
    pos = _place(n, bounds[:4], layout)
    return {
        "hue": hue,
        "base_lab": base,
        "radii": radii,
        "odd_index": odd_index,
        "speeds": speeds.astype(np.float64),
        "headings": headings.astype(np.float64),
        "spin_phase": spin.uniform(0.0, 2.0 * math.pi, size=n),
        "pulse_phase": pulse.uniform(0.0, 2.0 * math.pi, size=n),
        "pos": pos,
        "motion": motion,
    }


def _simulate_once(cfg: OddConfig, level: LevelSpec, seed: int) -> "LevelSim":
    drawn = sample_drivers(cfg, level, seed)
    n = level.count
    radii = drawn["radii"]
    pos = drawn["pos"]
    heading = drawn["headings"]
    omega = np.zeros(n, dtype=np.float64)
    motion = drawn["motion"]
    pos = _repel(pos, radii, cfg.repulse, 24)
    pos, heading = _reflect_walls(pos, heading, radii, cfg)
    odd_info = None
    if level.difference == "hue":
        odd_info = choose_odd_lab(drawn["base_lab"], cfg.tier)
        odd_lab = odd_info["lab"]
    else:
        odd_lab = np.array(drawn["base_lab"], dtype=np.float64, copy=True)
    t_start = -cfg.warmup_seconds
    extra = cfg.wipe_seconds
    t_end = level.timer + cfg.reveal_seconds + extra
    fps = cfg.fps
    n_frames = int(round((t_end - t_start) * fps))
    dt = 1.0 / (fps * cfg.substeps)
    positions = np.empty((n_frames + 1, n, 2), dtype=np.float64)
    positions[0] = pos
    noise_std = cfg.angular_std * math.sqrt(dt)
    for frame in range(n_frames):
        for _ in range(cfg.substeps):
            omega = np.clip(omega + motion.normal(0.0, noise_std, size=n), -cfg.omega_clip, cfg.omega_clip)
            heading = heading + omega * dt
            step = np.column_stack([np.cos(heading), np.sin(heading)]) * drawn["speeds"][:, None] * dt
            pos = pos + step
            pos, heading = _reflect_walls(pos, heading, radii, cfg)
            pos = _repel(pos, radii, cfg.repulse, 2)
            pos, heading = _reflect_walls(pos, heading, radii, cfg)
        positions[frame + 1] = pos
    clearance = _min_clearance(positions, radii)
    if clearance < -0.35:
        raise ConstraintFailure(f"min center clearance {clearance:.3f}px (need >= ri+rj)")
    i0 = int(round((0.0 - t_start) * fps))
    i1 = int(round((level.timer - t_start) * fps))
    mean_pos = positions[i0 : i1 + 1, drawn["odd_index"]].mean(axis=0)
    span = 0.5 * (1.0 - cfg.position_middle)
    x_lo = cfg.field_x0 + span * (cfg.field_x1 - cfg.field_x0)
    x_hi = cfg.field_x1 - span * (cfg.field_x1 - cfg.field_x0)
    y_lo = cfg.field_y0 + span * (cfg.field_y1 - cfg.field_y0)
    y_hi = cfg.field_y1 - span * (cfg.field_y1 - cfg.field_y0)
    if not (x_lo <= mean_pos[0] <= x_hi and y_lo <= mean_pos[1] <= y_hi):
        raise ConstraintFailure(
            f"odd mean position ({mean_pos[0]:.1f},{mean_pos[1]:.1f}) outside the middle {cfg.position_middle:.0%}"
        )
    others = np.delete(drawn["speeds"], drawn["odd_index"])
    low, high = np.percentile(others, [cfg.speed_low_pct, cfg.speed_high_pct])
    odd_speed = float(drawn["speeds"][drawn["odd_index"]])
    if not (low - 1e-6 <= odd_speed <= high + 1e-6):
        raise ConstraintFailure(
            f"odd speed {odd_speed:.2f} outside p{cfg.speed_low_pct:.0f}..p{cfg.speed_high_pct:.0f} "
            f"[{low:.2f}, {high:.2f}]"
        )
    playable = item_bounds(cfg, cfg.radius)
    # HUD sits outside the field. An item that stays in its inset cannot sit under it.
    hud_top = max(cfg.timer_y + cfg.timer_h, cfg.caption_y + cfg.caption_px * 0.5)
    if float(positions[:, :, 1].min()) <= hud_top:
        raise ConstraintFailure("an item center reached the HUD band above the field")
    if float(positions[:, :, 1].max()) >= cfg.pip_y - cfg.pip_r - cfg.radius:
        raise ConstraintFailure("an item center reached the pip row")
    diff_params = {
        "kind": level.difference,
        "base_lab": [float(v) for v in drawn["base_lab"]],
        "odd_lab": [float(v) for v in odd_lab],
        "base_hue": drawn["hue"],
    }
    if odd_info is not None:
        diff_params.update(
            {
                "distance": odd_info["distance"],
                "lightness_delta": odd_info["lightness_delta"],
                "hue_offset_rad": odd_info["hue_offset_rad"],
                "cvd": odd_info["cvd"],
                "rejected_offsets": odd_info["rejected_offsets"],
                "requested_distance": odd_info["requested_distance"],
            }
        )
    if level.difference == "size":
        diff_params["size_ratio"] = cfg.tier.size_ratio
        diff_params["odd_radius"] = float(radii[drawn["odd_index"]])
    if level.difference == "spin":
        diff_params["rev_s"] = cfg.tier.spin_rev_s
        diff_params["odd_direction"] = "counter-clockwise"
        diff_params["normal_direction"] = "clockwise"
    if level.difference == "pulse":
        diff_params["normal_hz"] = cfg.tier.pulse_normal_hz
        diff_params["odd_hz"] = cfg.tier.pulse_odd_hz
        diff_params["amplitude"] = cfg.tier.pulse_amplitude
        diff_params["phase"] = "per item, shared clock; not one shared brightness"
    return LevelSim(
        level=level,
        seed=seed,
        odd_index=drawn["odd_index"],
        radii=radii,
        base_lab=np.array(drawn["base_lab"], dtype=np.float64),
        odd_lab=np.array(odd_lab, dtype=np.float64),
        spin_phase=np.array(drawn["spin_phase"], dtype=np.float64),
        pulse_phase=np.array(drawn["pulse_phase"], dtype=np.float64),
        speeds=np.array(drawn["speeds"], dtype=np.float64),
        headings0=np.array(drawn["headings"], dtype=np.float64),
        positions=positions,
        t0=t_start,
        fps=fps,
        min_clearance=clearance,
        mean_pos=mean_pos,
        speed_band=(float(low), float(high)),
        odd_speed=odd_speed,
        diff_params=diff_params,
        playable=playable,
        reseeds=[],
    )


@dataclass
class LevelSim:
    level: LevelSpec
    seed: int
    odd_index: int
    radii: np.ndarray
    base_lab: np.ndarray
    odd_lab: np.ndarray
    spin_phase: np.ndarray
    pulse_phase: np.ndarray
    speeds: np.ndarray
    headings0: np.ndarray
    positions: np.ndarray
    t0: float
    fps: int
    min_clearance: float
    mean_pos: np.ndarray
    speed_band: tuple[float, float]
    odd_speed: float
    diff_params: dict
    playable: tuple[float, float, float, float, float]
    reseeds: list = field(default_factory=list)

    def at(self, sim_t: float) -> np.ndarray:
        u = (sim_t - self.t0) * self.fps
        i0 = int(math.floor(u))
        frac = float(u - i0)
        i0 = min(max(i0, 0), len(self.positions) - 2)
        return (1.0 - frac) * self.positions[i0] + frac * self.positions[i0 + 1]


@dataclass
class Show:
    cfg: OddConfig
    levels: dict[int, LevelSim]
    timeline: tuple


def simulate_level(cfg: OddConfig, level: LevelSpec, seed: int) -> LevelSim:
    started = time.perf_counter()
    print(f"sim level {level.id} {level.difference} n={level.count} seed={seed}", flush=True)

    def attempt(trial: int, k: int):
        try:
            return True, _simulate_once(cfg, level, trial)
        except ConstraintFailure as exc:
            print(f"  reseed k={k} seed={trial}: {exc.reason}", flush=True)
            return False, exc.reason

    result, _k, notes = reseed_loop(seed, cfg.max_attempts, attempt)
    result.reseeds = notes
    elapsed = time.perf_counter() - started
    print(
        f"  level {level.id} odd={result.odd_index} clearance={result.min_clearance:.2f}px "
        f"reseeds={len(notes)} in {elapsed:.2f}s",
        flush=True,
    )
    return result


def simulate_show(cfg: OddConfig, *, seed: int | None = None) -> Show:
    used = cfg.seed if seed is None else int(seed)
    levels = {level.id: simulate_level(cfg, level, used) for level in cfg.levels}
    return Show(cfg=cfg, levels=levels, timeline=build_timeline(cfg))


def location_phrase(x: float, y: float, cfg: OddConfig) -> str:
    fx = (x - cfg.field_x0) / (cfg.field_x1 - cfg.field_x0)
    fy = (y - cfg.field_y0) / (cfg.field_y1 - cfg.field_y0)
    col = "left" if fx < 0.33 else "right" if fx > 0.67 else "center"
    row = "upper" if fy < 0.33 else "lower" if fy > 0.67 else "middle"
    if row == "middle" and col == "center":
        return "center"
    if col == "center":
        return row
    if row == "middle":
        return col
    return f"{row} {col}"


def reveal_position(sim: LevelSim, timer: float) -> np.ndarray:
    return sim.at(timer)[sim.odd_index]
