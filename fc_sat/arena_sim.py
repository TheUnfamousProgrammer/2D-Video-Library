"""Top-down Flag Arena solver.

No gravity, no wander, no sweeper, no meteors. Every impulse is tagged
ball, dash, boss, or clash. Fixed step 1/240, PCG64 streams for spawn, AI,
and effects.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from fc_sat.arena_config import ArenaConfig

IMPULSE_SOURCES = frozenset({"ball", "dash", "boss", "clash"})
BOSS_ID = -1
_PHASE_COOL = 0
_PHASE_WIND = 1
_PHASE_DASH = 2


@dataclass(frozen=True)
class Impulse:
    time: float
    source: str
    a: int
    b: int
    closing: float
    x: float
    y: float

    def __post_init__(self) -> None:
        if self.source not in IMPULSE_SOURCES:
            raise ValueError(f"impulse source {self.source!r} is not logged")


@dataclass(frozen=True)
class Elim:
    time: float
    index: int
    cause: str
    killer: int | None = None
    closing: float = 0.0
    combo: str = ""
    revenge: bool = False


@dataclass(frozen=True)
class SimResult:
    seed: int
    backend: str
    elims: tuple[Elim, ...]
    t_win: float | None
    winner: int | None
    duel_start: float | None
    cameo_exit: float | None
    trace: dict | None = None
    impulses: tuple[Impulse, ...] = ()
    metrics: dict = field(default_factory=dict)
    meteors: tuple = ()
    near_misses: tuple = ()

    @property
    def placements(self) -> tuple[int, ...]:
        order = [item.index for item in self.elims]
        if self.winner is not None and self.winner not in order:
            order.append(self.winner)
        return tuple(order)


def active_backend() -> str:
    return "numpy"


def rounded_rect_sd(x: float, y: float, cx: float, cy: float, hw: float, hh: float, cr: float) -> float:
    ax = abs(x - cx)
    ay = abs(y - cy)
    qx = ax - (hw - cr)
    qy = ay - (hh - cr)
    outside = math.hypot(max(qx, 0.0), max(qy, 0.0))
    inside = min(max(qx, qy), 0.0)
    return outside + inside - cr


def _sd_array(pos: np.ndarray, cx: float, cy: float, hw: float, hh: float, cr: float) -> np.ndarray:
    ax = np.abs(pos[:, 0] - cx)
    ay = np.abs(pos[:, 1] - cy)
    qx = ax - (hw - cr)
    qy = ay - (hh - cr)
    outside = np.hypot(np.maximum(qx, 0.0), np.maximum(qy, 0.0))
    inside = np.minimum(np.maximum(qx, qy), 0.0)
    return outside + inside - cr


def corner_radius(hw: float, hh: float, frac: float) -> float:
    return frac * min(hw, hh)


def resolve_collision(
    va: np.ndarray,
    vb: np.ndarray,
    ma: float,
    mb: float,
    restitution: float,
    normal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Equal-mass head-on contacts keep momentum and scale separation by restitution."""
    length = float(np.linalg.norm(normal))
    if length < 1e-8 or ma <= 0 or mb <= 0:
        return va.copy(), vb.copy(), 0.0
    normal = normal / length
    closing = float(np.dot(va - vb, normal))
    if closing <= 0.0:
        return va.copy(), vb.copy(), 0.0
    impulse = (1.0 + restitution) * closing / (1.0 / ma + 1.0 / mb)
    return va - (impulse / ma) * normal, vb + (impulse / mb) * normal, closing


def damp_factor(damping: float, elapsed: float) -> float:
    return math.exp(-damping * elapsed)


def boss_allowed(alive: int, minimum: int) -> bool:
    return alive >= minimum


def combo_name(streak: int) -> str:
    if streak >= 4:
        return "rampage"
    if streak == 3:
        return "triple"
    if streak == 2:
        return "double"
    return ""


def attribute_cause(last_hit_age: float, last_hit_kind: str, inside_old_platform: bool, *, hit_window: float) -> str:
    if last_hit_kind in {"ball", "boss"} and last_hit_age <= hit_window:
        return last_hit_kind
    if inside_old_platform:
        return "storm"
    return "self"


def dash_travel(speed: float, damping: float, seconds: float) -> float:
    if damping <= 1e-8:
        return speed * seconds
    return (speed / damping) * (1.0 - math.exp(-damping * seconds))


def _sample_pair(frames: tuple[tuple[float, float], ...], time: float) -> float:
    if time <= frames[0][0]:
        return frames[0][1]
    if time >= frames[-1][0]:
        return frames[-1][1]
    for (t0, v0), (t1, v1) in zip(frames, frames[1:]):
        if t0 <= time <= t1:
            if t1 <= t0:
                return v1
            u = (time - t0) / (t1 - t0)
            return v0 + (v1 - v0) * u
    return frames[-1][1]


def _sample_platform(frames: tuple[tuple[float, float, float], ...], time: float) -> tuple[float, float]:
    if time <= frames[0][0]:
        return frames[0][1], frames[0][2]
    if time >= frames[-1][0]:
        return frames[-1][1], frames[-1][2]
    for (t0, hw0, hh0), (t1, hw1, hh1) in zip(frames, frames[1:]):
        if t0 <= time <= t1:
            if t1 <= t0:
                return hw1, hh1
            u = (time - t0) / (t1 - t0)
            return hw0 + (hw1 - hw0) * u, hh0 + (hh1 - hh0) * u
    return frames[-1][1], frames[-1][2]


def _edge_behind(ax: float, ay: float, tx: float, ty: float, cx: float, cy: float, hw: float, hh: float, cr: float) -> float:
    """Distance from the target to the platform box along the attacker ray.

    The ring-out test still uses the rounded rectangle. This score uses the
    box so target selection stays inside the sim budget.
    """
    del cr
    dx = tx - ax
    dy = ty - ay
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return 0.0
    dx /= length
    dy /= length
    lx = tx - cx
    ly = ty - cy
    hits = []
    if dx > 1e-8:
        hits.append((hw - lx) / dx)
    elif dx < -1e-8:
        hits.append((-hw - lx) / dx)
    if dy > 1e-8:
        hits.append((hh - ly) / dy)
    elif dy < -1e-8:
        hits.append((-hh - ly) / dy)
    ahead = [item for item in hits if item > 0.0]
    return min(ahead) if ahead else 0.0


def _dist_to_segment(x: float, y: float, a: tuple[float, float], b: tuple[float, float]) -> float:
    abx = b[0] - a[0]
    aby = b[1] - a[1]
    denom = abx * abx + aby * aby
    if denom < 1e-8:
        return math.hypot(x - a[0], y - a[1])
    u = max(0.0, min(1.0, ((x - a[0]) * abx + (y - a[1]) * aby) / denom))
    return math.hypot(x - (a[0] + abx * u), y - (a[1] + aby * u))


def _spawn(cfg: ArenaConfig, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    count = len(cfg.countries)
    cx, cy = cfg.center
    # Scatter across the real opening floor, not the spec 410 x 460 box.
    # hw0 is the design size; the tuned floor starts much wider.
    start_hw = cfg.platform_keyframes[0][1]
    start_hh = cfg.platform_keyframes[0][2]
    hw = start_hw * cfg.disk_frac
    hh = start_hh * cfg.disk_frac
    cr = corner_radius(hw, hh, cfg.corner_frac)
    full_cr = corner_radius(start_hw, start_hh, cfg.corner_frac)
    order = rng.permutation(count)
    clash = (int(order[0]), int(order[1]))
    half = cfg.clash_gap * 0.5
    pos = np.zeros((count, 2), dtype=np.float64)
    pos[clash[0]] = (cx, cy - half)
    pos[clash[1]] = (cx, cy + half)
    spacing = max(cfg.ball_radius * 2.0 + 2.0, 150.0)
    placed = [tuple(pos[clash[0]]), tuple(pos[clash[1]])]
    filled = 2
    candidates: list[tuple[float, float]] = []
    row = 0
    y = cy - hh + spacing
    while y < cy + hh:
        x = cx - hw + (spacing * 0.5 if row % 2 else spacing)
        while x < cx + hw:
            point = (float(x), float(y))
            clear = rounded_rect_sd(point[0], point[1], cx, cy, hw, hh, cr) <= -cfg.ball_radius
            clear = clear and _dist_to_segment(point[0], point[1], placed[0], placed[1]) >= spacing
            clear = clear and all(math.hypot(point[0] - px, point[1] - py) >= spacing for px, py in placed)
            if clear:
                candidates.append(point)
            x += spacing
        y += spacing * 0.86602540378
        row += 1
    if len(candidates) < count - 2:
        raise RuntimeError(f"spawn has {len(candidates)} slots for {count - 2} balls")
    rng.shuffle(candidates)
    picked = [candidates.pop()]
    while len(picked) < count - 2 and candidates:
        best_i = 0
        best_d = -1.0
        for index, point in enumerate(candidates):
            distance = min(math.hypot(point[0] - px, point[1] - py) for px, py in picked)
            if distance > best_d:
                best_d = distance
                best_i = index
        picked.append(candidates.pop(best_i))
    for point in picked:
        slot = int(order[filled])
        pos[slot] = point
        filled += 1
    for index in range(count):
        if rounded_rect_sd(float(pos[index, 0]), float(pos[index, 1]), cx, cy, start_hw, start_hh, full_cr) > 0.0:
            raise RuntimeError("spawn placed a ball outside the platform")
    vel = np.zeros((count, 2), dtype=np.float64)
    direction = pos[clash[1]] - pos[clash[0]]
    direction /= max(float(np.linalg.norm(direction)), 1e-6)
    vel[clash[0]] = direction * cfg.clash_speed
    vel[clash[1]] = -direction * cfg.clash_speed
    for index in range(count):
        if index in clash:
            continue
        speed = float(rng.uniform(cfg.speed_min, cfg.speed_max))
        angle = float(rng.uniform(0.0, math.tau))
        vel[index] = (math.cos(angle) * speed, math.sin(angle) * speed)
    cooldown_until = np.zeros(count, dtype=np.float64)
    for index in range(count):
        low, high = cfg.clash_cooldown if index in clash else cfg.initial_cooldown
        cooldown_until[index] = float(rng.uniform(low, high))
    return pos, vel, cooldown_until, np.array(clash, dtype=np.int64)


_PAIR_I, _PAIR_J = np.triu_indices(33, 1)


def _collide(
    pos: np.ndarray,
    vel: np.ndarray,
    omega: np.ndarray,
    mass: np.ndarray,
    radius: np.ndarray,
    active: np.ndarray,
    iu: np.ndarray,
    ju: np.ndarray,
    *,
    ball_e: float,
    boss_e: float,
    boss_slot: int,
    iters: int,
) -> list[tuple[int, int, float, float, float]]:
    hits: list[tuple[int, int, float, float, float]] = []
    seen: set[tuple[int, int]] = set()
    inv_mass = None
    for _ in range(iters):
        dx = pos[ju, 0] - pos[iu, 0]
        dy = pos[ju, 1] - pos[iu, 1]
        dist2 = dx * dx + dy * dy
        radsum = radius[iu] + radius[ju]
        mask = active[iu] & active[ju] & (dist2 < radsum * radsum) & (dist2 > 1e-8)
        if not mask.any():
            break
        sel = np.flatnonzero(mask)
        if inv_mass is None:
            inv_mass = 1.0 / mass
        touched = False
        for k in sel:
            i = int(iu[k])
            j = int(ju[k])
            dist = math.sqrt(float(dist2[k]))
            nx = float(dx[k]) / dist
            ny = float(dy[k]) / dist
            gap = float(radsum[k]) - dist
            share = float(inv_mass[i] + inv_mass[j])
            corr = gap * 0.8 / share
            pos[i, 0] -= corr * inv_mass[i] * nx
            pos[i, 1] -= corr * inv_mass[i] * ny
            pos[j, 0] += corr * inv_mass[j] * nx
            pos[j, 1] += corr * inv_mass[j] * ny
            rvx = float(vel[i, 0] - vel[j, 0])
            rvy = float(vel[i, 1] - vel[j, 1])
            closing = rvx * nx + rvy * ny
            if closing <= 1e-4:
                continue
            touched = True
            e = boss_e if i == boss_slot or j == boss_slot else ball_e
            jmag = (1.0 + e) * closing / share
            ix = jmag * nx
            iy = jmag * ny
            vel[i, 0] -= ix * inv_mass[i]
            vel[i, 1] -= iy * inv_mass[i]
            vel[j, 0] += ix * inv_mass[j]
            vel[j, 1] += iy * inv_mass[j]
            tx = rvx - closing * nx
            ty = rvy - closing * ny
            tang = math.hypot(tx, ty)
            omega[i] += tang / max(float(radius[i]), 1.0)
            omega[j] -= tang / max(float(radius[j]), 1.0)
            pair = (i, j)
            if pair not in seen:
                seen.add(pair)
                hits.append((i, j, closing, float(pos[i, 0]), float(pos[i, 1])))
        if not touched and len(sel) < 2:
            break
    return hits


def _pick_target(
    index: int,
    pos: np.ndarray,
    alive: np.ndarray,
    kos: np.ndarray,
    hit_by: list[list[tuple[float, int]]],
    time: float,
    cfg: ArenaConfig,
    rng: np.random.Generator,
    cx: float,
    cy: float,
    hw: float,
    hh: float,
    cr: float,
) -> int:
    best = -1
    best_score = -1e9
    count = len(cfg.countries)
    for other in range(count):
        if other == index or not alive[other]:
            continue
        dist = math.hypot(float(pos[other, 0] - pos[index, 0]), float(pos[other, 1] - pos[index, 1]))
        behind = _edge_behind(
            float(pos[index, 0]),
            float(pos[index, 1]),
            float(pos[other, 0]),
            float(pos[other, 1]),
            cx,
            cy,
            hw,
            hh,
            cr,
        )
        edge = 1.0 - min(1.0, max(0.0, behind / cfg.edge_range))
        near = 1.0 - min(1.0, max(0.0, dist / cfg.near_range))
        revenge = 1.0 if any(time - when <= cfg.revenge_window and who == other for when, who in hit_by[index]) else 0.0
        leader = 1.0 if kos[other] >= cfg.leader_kos else 0.0
        score = (
            cfg.score_edge * edge
            + cfg.score_near * near
            + cfg.score_revenge * revenge
            + cfg.score_leader * leader
            + float(rng.normal(0.0, cfg.noise))
        )
        if score > best_score:
            best_score = score
            best = other
    return best


def _cooldown_duration(cfg: ArenaConfig, rng: np.random.Generator, time: float, alive: int) -> float:
    aggression = (1.0 + min(time / max(cfg.aggression_time, 1e-3), 1.0)) * cfg.aggression_scale
    target = _sample_pair(cfg.n_target, time)
    control = 1.0 + cfg.controller_gain * (alive - target)
    control = min(cfg.controller_clamp[1], max(cfg.controller_clamp[0], control))
    rolled = float(rng.uniform(cfg.cooldown[0], cfg.cooldown[1]))
    return max(cfg.cooldown_min, rolled / max(aggression * control, 1e-3))


def _dash_safe(cfg: ArenaConfig, pos: np.ndarray, direction: np.ndarray, cx: float, cy: float, hw: float, hh: float, cr: float) -> bool:
    travel = dash_travel(cfg.dash_speed, cfg.damping, cfg.dash_time)
    end = pos + direction * travel
    limit = -cfg.dash_margin_radii * cfg.ball_radius
    return rounded_rect_sd(float(end[0]), float(end[1]), cx, cy, hw, hh, cr) <= limit


def simulate(
    cfg: ArenaConfig,
    seed: int,
    *,
    horizon: float | None = None,
    record_trace: bool = False,
) -> SimResult:
    started = time.perf_counter()
    limit = cfg.horizon if horizon is None else horizon
    dt = cfg.dt
    steps = max(1, int(math.ceil(limit / dt)))
    count = len(cfg.countries)
    boss_slot = count
    spawn_rng = np.random.Generator(np.random.PCG64(seed))
    ai_rng = np.random.Generator(np.random.PCG64(seed + 10007))
    effect_rng = np.random.Generator(np.random.PCG64(seed + 20011))
    pos_c, vel_c, cool_until, _clash = _spawn(cfg, spawn_rng)
    pos = np.zeros((count + 1, 2), dtype=np.float64)
    vel = np.zeros((count + 1, 2), dtype=np.float64)
    pos[:count] = pos_c
    vel[:count] = vel_c
    mass = np.ones(count + 1, dtype=np.float64)
    mass[boss_slot] = cfg.cameo_mass
    radius = np.full(count + 1, cfg.ball_radius, dtype=np.float64)
    radius[boss_slot] = cfg.cameo_radius
    omega = np.zeros(count + 1, dtype=np.float64)
    alive = np.ones(count, dtype=bool)
    phase = np.zeros(count, dtype=np.int8)
    phase_until = cool_until.copy()
    windup_started = np.full(count, -1.0)
    dash_windups: list[float] = []
    target = np.full(count, -1, dtype=np.int16)
    retargets = np.zeros(count, dtype=np.int8)
    facing = np.zeros(count, dtype=np.float64)
    kos = np.zeros(count, dtype=np.int16)
    streak = np.zeros(count, dtype=np.int16)
    last_ko_at = np.full(count, -1e9, dtype=np.float64)
    last_hit_at = np.full(count, -1e9, dtype=np.float64)
    last_hit_by = np.full(count, -2, dtype=np.int16)
    last_hit_kind = [""] * count
    last_closing = np.zeros(count, dtype=np.float64)
    hit_by: list[list[tuple[float, int]]] = [[] for _ in range(count)]
    was_near_edge = np.zeros(count, dtype=bool)
    out_at = np.full(count, -1.0, dtype=np.float64)

    cx, cy = cfg.center
    storm = 0.0
    hw, hh = _sample_platform(cfg.platform_keyframes, 0.0)
    cr = corner_radius(hw, hh, cfg.corner_frac)
    platform_hist: list[tuple[float, float, float]] = []
    impulses: list[Impulse] = []
    impacts: list[tuple[float, float]] = []
    elims: list[Elim] = []
    eliminated: set[int] = set()
    mid = 0.5 * (pos[_clash[0]] + pos[_clash[1]])
    closing = float(np.linalg.norm(vel[_clash[0]] - vel[_clash[1]]))
    impulses.append(Impulse(0.0, "clash", int(_clash[0]), int(_clash[1]), closing, float(mid[0]), float(mid[1])))

    boss_on = False
    boss_skipped = False
    boss_phase = "idle"
    boss_until = 0.0
    boss_charges = 0
    boss_kos = 0
    cameo_exit = None
    boss_aim = np.zeros(2, dtype=np.float64)

    speed_sum: dict[int, float] = {}
    speed_n: dict[int, int] = {}
    heap_acc = 0.0
    heap_n = 0
    neighbor_acc = 0.0
    neighbor_n = 0
    escapes: list[float] = []
    duel_start = None
    t_win = None
    winner = None
    semi_start = None
    tie = False
    both_finalists_hit = False

    trace = None
    if record_trace:
        trace = {
            "xy": np.zeros((steps, count, 2), dtype=np.float32),
            "angle": np.zeros((steps, count), dtype=np.float32),
            "alive": np.zeros((steps, count), dtype=bool),
            "phase": np.zeros((steps, count), dtype=np.int8),
            "facing": np.zeros((steps, count), dtype=np.float32),
            "hw": np.zeros(steps, dtype=np.float32),
            "hh": np.zeros(steps, dtype=np.float32),
            "boss_xy": np.zeros((steps, 2), dtype=np.float32),
            "boss_on": np.zeros(steps, dtype=bool),
        }

    def log(source: str, a: int, b: int, closing_speed: float, at: np.ndarray, when: float) -> None:
        impulses.append(Impulse(when, source, a, b, closing_speed, float(at[0]), float(at[1])))

    def public_id(slot: int) -> int:
        return BOSS_ID if slot == boss_slot else slot

    target_alive = np.interp(
        np.arange(steps) * dt,
        np.array([item[0] for item in cfg.n_target], dtype=np.float64),
        np.array([item[1] for item in cfg.n_target], dtype=np.float64),
    )
    pair_i, pair_j = np.triu_indices(count + 1, 1)
    next_ai = 0.0

    for step in range(steps):
        now = step * dt
        if t_win is not None and now > t_win + 0.8:
            steps = step
            break
        alive_n = int(alive.sum())
        rate = 1.0 + cfg.shrink_gain * (alive_n - float(target_alive[step]))
        rate = min(2.0, max(0.0, rate))
        # Hold the floor while four names are on screen, then close it
        # once the final has been readable, so the duel does not stall.
        speed_cap = cfg.hw_speed_max
        if semi_start is not None and alive_n >= 4 and now < semi_start + 2.0:
            rate = 0.0
        if duel_start is not None and alive_n <= 2 and now > duel_start + 2.2:
            speed_cap = cfg.hw_speed_max * 3.2
        proposed = storm + rate * dt
        hw2, hh2 = _sample_platform(cfg.platform_keyframes, proposed)
        dhw = abs(hw2 - hw)
        if dhw > speed_cap * dt and dhw > 1e-8:
            proposed = storm + rate * dt * (speed_cap * dt) / dhw
            hw2, hh2 = _sample_platform(cfg.platform_keyframes, proposed)
        storm = proposed
        hw, hh = hw2, hh2
        cr = corner_radius(hw, hh, cfg.corner_frac)
        platform_hist.append((hw, hh, cr))

        if not boss_skipped and not boss_on and boss_phase == "idle" and now >= cfg.cameo_enter:
            if boss_allowed(alive_n, cfg.cameo_min_alive):
                angle = float(effect_rng.uniform(0.0, math.tau))
                lo, hi = 0.0, max(hw, hh) * 1.4
                for _ in range(20):
                    mid_r = 0.5 * (lo + hi)
                    sx = cx + math.cos(angle) * mid_r
                    sy = cy + math.sin(angle) * mid_r
                    if rounded_rect_sd(sx, sy, cx, cy, hw, hh, cr) > 0.0:
                        hi = mid_r
                    else:
                        lo = mid_r
                pos[boss_slot] = (cx + math.cos(angle) * lo, cy + math.sin(angle) * lo)
                inward = np.array([cx, cy]) - pos[boss_slot]
                inward /= max(float(np.linalg.norm(inward)), 1e-6)
                vel[boss_slot] = inward * cfg.cameo_speed
                log("boss", BOSS_ID, BOSS_ID, cfg.cameo_speed, pos[boss_slot], now)
                boss_on = True
                boss_phase = "enter"
                boss_until = now + 0.8
            else:
                boss_skipped = True

        if boss_on:
            if boss_kos >= cfg.cameo_ko_exit and boss_phase != "exit":
                boss_phase = "exit"
                boss_until = now
            if boss_phase == "enter" and now >= boss_until:
                boss_phase = "windup"
                boss_until = now + cfg.cameo_windup
                boss_aim = _densest(pos, alive, count)
            elif boss_phase == "windup" and now >= boss_until:
                aim = boss_aim - pos[boss_slot]
                if boss_charges >= cfg.cameo_charges:
                    aim = _outward_array(pos[boss_slot : boss_slot + 1], cx, cy, hw, hh, cr)[0]
                    boss_phase = "exit"
                else:
                    boss_phase = "charge"
                norm = float(np.linalg.norm(aim))
                if norm < 1e-6:
                    aim = np.array([1.0, 0.0])
                    norm = 1.0
                vel[boss_slot] = aim / norm * cfg.cameo_charge_speed
                log("boss", BOSS_ID, BOSS_ID, cfg.cameo_charge_speed, pos[boss_slot], now)
                if boss_phase == "charge":
                    boss_charges += 1
                    boss_until = now + 0.45
                else:
                    boss_until = now + 2.0
            elif boss_phase == "charge" and now >= boss_until:
                boss_phase = "windup"
                boss_until = now + cfg.cameo_windup
                boss_aim = _densest(pos, alive, count)

        if step % 4 == 0:
            for index in range(count):
                if phase[index] != _PHASE_WIND or not alive[index]:
                    continue
                victim = int(target[index])
                if victim < 0 or not alive[victim]:
                    continue
                facing[index] = math.atan2(
                    float(pos[victim, 1] - pos[index, 1]),
                    float(pos[victim, 0] - pos[index, 0]),
                )

        if now + 1e-9 >= next_ai:
            next_ai = now + 10.0
            for index in range(count):
                if not alive[index]:
                    continue
                if phase[index] == _PHASE_COOL and now >= phase_until[index]:
                    chosen = _pick_target(index, pos, alive, kos, hit_by, now, cfg, ai_rng, cx, cy, hw, hh, cr)
                    if chosen < 0:
                        phase_until[index] = now + cfg.fizzle_cooldown
                    else:
                        target[index] = chosen
                        retargets[index] = 0
                        phase[index] = _PHASE_WIND
                        phase_until[index] = now + cfg.windup
                        windup_started[index] = now
                        vel[index] *= cfg.windup_speed
                elif phase[index] == _PHASE_WIND:
                    victim = int(target[index])
                    if victim < 0 or not alive[victim]:
                        _retarget_or_fizzle(index, pos, alive, kos, hit_by, now, cfg, ai_rng, cx, cy, hw, hh, cr, phase, phase_until, target, retargets, vel, windup_started)
                    else:
                        aimx = float(pos[victim, 0] - pos[index, 0])
                        aimy = float(pos[victim, 1] - pos[index, 1])
                        facing[index] = math.atan2(aimy, aimx)
                        if now >= phase_until[index]:
                            predx = float(pos[victim, 0] + vel[victim, 0] * cfg.dash_predict)
                            predy = float(pos[victim, 1] + vel[victim, 1] * cfg.dash_predict)
                            dx = predx - float(pos[index, 0])
                            dy = predy - float(pos[index, 1])
                            length = math.hypot(dx, dy)
                            if length < 1e-6:
                                direction = np.array([1.0, 0.0])
                            else:
                                direction = np.array([dx / length, dy / length])
                            if _dash_safe(cfg, pos[index], direction, cx, cy, hw, hh, cr):
                                vel[index, 0] = direction[0] * cfg.dash_speed
                                vel[index, 1] = direction[1] * cfg.dash_speed
                                held = now - float(windup_started[index])
                                dash_windups.append(held)
                                log("dash", index, BOSS_ID, cfg.dash_speed, pos[index], now)
                                phase[index] = _PHASE_DASH
                                phase_until[index] = now + cfg.dash_time
                            else:
                                _retarget_or_fizzle(index, pos, alive, kos, hit_by, now, cfg, ai_rng, cx, cy, hw, hh, cr, phase, phase_until, target, retargets, vel, windup_started)
                elif phase[index] == _PHASE_DASH and now >= phase_until[index]:
                    phase[index] = _PHASE_COOL
                    phase_until[index] = now + _cooldown_duration(cfg, ai_rng, now, alive_n)
                if alive[index]:
                    next_ai = min(next_ai, float(phase_until[index]))

        active = np.zeros(count + 1, dtype=bool)
        active[:count] = alive
        active[boss_slot] = boss_on
        sd = _sd_array(pos[:count], cx, cy, hw, hh, cr)
        margin = -cfg.edge_margin_radii * cfg.ball_radius
        near = alive & (sd > margin) & (phase != _PHASE_WIND) & (phase != _PHASE_DASH)
        if np.any(near):
            outward = _outward_array(pos[:count], cx, cy, hw, hh, cr)
            outward_speed = vel[:count, 0] * outward[:, 0] + vel[:count, 1] * outward[:, 1]
            brake = near & (outward_speed > cfg.edge_outward_speed)
            vel[:count, 0] -= outward[:, 0] * cfg.edge_accel * dt * brake
            vel[:count, 1] -= outward[:, 1] * cfg.edge_accel * dt * brake

        if boss_on and boss_phase == "exit":
            if rounded_rect_sd(float(pos[boss_slot, 0]), float(pos[boss_slot, 1]), cx, cy, hw, hh, cr) > 0.0:
                boss_on = False
                boss_phase = "gone"
                cameo_exit = now

        damp = math.exp(-cfg.damping * dt)
        vel *= damp
        sp2 = vel[:, 0] * vel[:, 0] + vel[:, 1] * vel[:, 1]
        limit2 = cfg.speed_clamp * cfg.speed_clamp
        if float(sp2.max()) > limit2:
            too_fast = sp2 > limit2
            scale = cfg.speed_clamp / np.sqrt(sp2[too_fast])
            vel[too_fast, 0] *= scale
            vel[too_fast, 1] *= scale
        pos += vel * dt
        omega *= math.exp(-1.5 * dt)

        hit_slots: set[int] = set()
        for a, b, closing_speed, hx, hy in _collide(
            pos,
            vel,
            omega,
            mass,
            radius,
            active,
            pair_i,
            pair_j,
            ball_e=cfg.restitution,
            boss_e=cfg.cameo_restitution,
            boss_slot=boss_slot,
            iters=cfg.solver_iters,
        ):
            kind = "boss" if a == boss_slot or b == boss_slot else "ball"
            log(kind, public_id(a), public_id(b), closing_speed, np.array([hx, hy]), now)
            impacts.append((now, closing_speed))
            for slot, other in ((a, b), (b, a)):
                if slot == boss_slot or slot >= count or not alive[slot]:
                    continue
                hit_slots.add(slot)
                last_hit_at[slot] = now
                last_closing[slot] = closing_speed
                if other == boss_slot:
                    last_hit_by[slot] = BOSS_ID
                    last_hit_kind[slot] = "boss"
                else:
                    last_hit_by[slot] = other
                    last_hit_kind[slot] = "ball"
                    hit_by[slot].append((now, other))
                if phase[slot] == _PHASE_DASH:
                    phase[slot] = _PHASE_COOL
                    phase_until[slot] = now + _cooldown_duration(cfg, ai_rng, now, alive_n)
                    next_ai = min(next_ai, float(phase_until[slot]))
                margin = cfg.escape_margin_radii * cfg.ball_radius
                current = rounded_rect_sd(float(pos[slot, 0]), float(pos[slot, 1]), cx, cy, hw, hh, cr)
                if -margin <= current <= 0.0:
                    escapes.append(now)
            if a != boss_slot and b != boss_slot and alive_n <= 2:
                both_finalists_hit = True

        sd = _sd_array(pos[:count], cx, cy, hw, hh, cr)
        back = int(round(cfg.storm_window / dt))
        old = platform_hist[max(0, len(platform_hist) - 1 - back)]
        crossed = []
        for index in range(count):
            if not alive[index]:
                continue
            if sd[index] > -cfg.comeback_margin_radii * cfg.ball_radius:
                was_near_edge[index] = True
            if sd[index] <= 0.0:
                continue
            inside_old = rounded_rect_sd(float(pos[index, 0]), float(pos[index, 1]), cx, cy, old[0], old[1], old[2]) <= 0.0
            age = now - float(last_hit_at[index])
            cause = attribute_cause(age, last_hit_kind[index], inside_old, hit_window=cfg.hit_window)
            killer = None
            combo = ""
            revenge = False
            if cause == "ball":
                killer = int(last_hit_by[index])
                if was_near_edge[killer]:
                    pass
                if now - float(last_ko_at[killer]) <= cfg.combo_window:
                    streak[killer] += 1
                else:
                    streak[killer] = 1
                last_ko_at[killer] = now
                kos[killer] += 1
                combo = combo_name(int(streak[killer]))
                revenge = any(who == index and now - when <= cfg.revenge_window for when, who in hit_by[killer])
            elif cause == "boss":
                boss_kos += 1
            alive[index] = False
            out_at[index] = now
            crossed.append((float(sd[index]), index, cause, killer, float(last_closing[index]), combo, revenge))
        crossed.sort(key=lambda item: (-item[0], item[1]))
        for _depth, index, cause, killer, closing_speed, combo, revenge in crossed:
            if index in eliminated:
                continue
            eliminated.add(index)
            elims.append(Elim(now, index, cause, killer, closing_speed, combo, revenge))
            remaining = count - len(elims)
            if remaining == 4 and semi_start is None:
                semi_start = now
            if remaining == 2 and duel_start is None:
                duel_start = now
            if remaining == 1 and winner is None:
                left = [ball for ball in range(count) if ball not in eliminated]
                winner = left[0] if left else None
                t_win = now
            if remaining == 0:
                winner = None
                t_win = None
                if len(elims) >= 2 and abs(elims[-1].time - elims[-2].time) <= cfg.simultaneous:
                    tie = True

        if (
            not tie
            and len(elims) >= 2
            and winner is None
            and abs(elims[-1].time - elims[-2].time) <= cfg.simultaneous
            and count - len(elims) == 0
        ):
            tie = True

        if alive_n and step % 8 == 0:
            speeds = np.sqrt(vel[:count][alive, 0] ** 2 + vel[:count][alive, 1] ** 2)
            bucket = int(now // cfg.kinetic_window)
            speed_sum[bucket] = speed_sum.get(bucket, 0.0) + float(np.sum(speeds))
            speed_n[bucket] = speed_n.get(bucket, 0) + int(speeds.size)
            bottom = cy + hh / 3.0
            heap_acc += float(np.mean(pos[:count][alive, 1] > bottom))
            heap_n += 1
            alive_idx = np.flatnonzero(alive)
            if len(alive_idx) > 1:
                delta = pos[alive_idx][:, None, :] - pos[alive_idx][None, :, :]
                dist = np.linalg.norm(delta, axis=2)
                touch = dist < (cfg.ball_radius * 2.0 + 0.5)
                np.fill_diagonal(touch, False)
                neighbor_acc += float(np.mean(np.sum(touch, axis=1)))
            neighbor_n += 1

        if trace is not None and step < len(trace["xy"]):
            trace["xy"][step] = pos[:count]
            trace["angle"][step] = omega[:count]
            trace["alive"][step] = alive
            trace["phase"][step] = phase
            trace["facing"][step] = facing
            trace["hw"][step] = hw
            trace["hh"][step] = hh
            trace["boss_xy"][step] = pos[boss_slot]
            trace["boss_on"][step] = boss_on

    if trace is not None:
        for key, value in list(trace.items()):
            trace[key] = value[:steps]

    causes = {"ball": 0, "boss": 0, "storm": 0, "self": 0}
    for item in elims:
        causes[item.cause] = causes.get(item.cause, 0) + 1
    total_elims = max(1, len(elims))
    kinetic_fail, kinetic_means = _kinetic_fail(speed_sum, speed_n, cfg, t_win if t_win is not None else limit)
    first_hard = next((when for when, speed in impacts if speed >= cfg.first_impact_speed), None)
    hard_impacts = [when for when, speed in impacts if speed >= cfg.dead_impact]
    final_cause = elims[-1].cause if elims and winner is not None else ""
    comebacks = 0
    seen_come = set()
    for item in elims:
        if item.killer is not None and was_near_edge[item.killer] and item.killer not in seen_come:
            seen_come.add(item.killer)
            comebacks += 1
    late_escapes = sum(1 for when in escapes if t_win is not None and when >= t_win - cfg.escape_window)
    metrics = {
        "causes": causes,
        "cause_ball": causes.get("ball", 0) / total_elims if elims else 0.0,
        "cause_storm": causes.get("storm", 0) / total_elims if elims else 0.0,
        "cause_self": causes.get("self", 0) / total_elims if elims else 0.0,
        "first_impact": first_hard,
        "impacts": impacts,
        "kinetic_ok": not kinetic_fail,
        "kinetic_means": kinetic_means,
        "heap": heap_acc / heap_n if heap_n else 0.0,
        "neighbors": neighbor_acc / neighbor_n if neighbor_n else 0.0,
        "tie": tie,
        "final_cause": final_cause,
        "both_finalists_hit": both_finalists_hit,
        "winner_kos": int(kos[winner]) if winner is not None else 0,
        "near_escapes": late_escapes,
        "comebacks": comebacks,
        "combos": sum(1 for item in elims if item.combo),
        "revenges": sum(1 for item in elims if item.revenge),
        "boss_kos": boss_kos,
        "boss_skipped": boss_skipped,
        "dead_ok": _dead_ok(hard_impacts, duel_start if duel_start is not None else (t_win or limit), cfg.dead_time),
        "dash_windup_min": min(dash_windups) if dash_windups else cfg.windup,
        "seconds": time.perf_counter() - started,
        "was_near_edge": was_near_edge,
    }
    for item in impulses:
        if item.source not in IMPULSE_SOURCES:
            raise RuntimeError(f"unlogged impulse source {item.source}")
    return SimResult(
        seed=seed,
        backend="numpy",
        elims=tuple(elims),
        t_win=t_win,
        winner=winner,
        duel_start=duel_start,
        cameo_exit=cameo_exit,
        trace=trace,
        impulses=tuple(impulses),
        metrics=metrics,
    )


def _retarget_or_fizzle(index, pos, alive, kos, hit_by, now, cfg, rng, cx, cy, hw, hh, cr, phase, phase_until, target, retargets, vel, windup_started) -> None:
    if retargets[index] < 1:
        retargets[index] = 1
        chosen = _pick_target(index, pos, alive, kos, hit_by, now, cfg, rng, cx, cy, hw, hh, cr)
        if chosen >= 0:
            target[index] = chosen
            phase[index] = _PHASE_WIND
            phase_until[index] = now + cfg.windup
            windup_started[index] = now
            vel[index] *= cfg.windup_speed
            return
    phase[index] = _PHASE_COOL
    phase_until[index] = now + cfg.fizzle_cooldown
    target[index] = -1


def _densest(pos: np.ndarray, alive: np.ndarray, count: int) -> np.ndarray:
    best = None
    best_n = -1
    idx = np.flatnonzero(alive[:count])
    if len(idx) == 0:
        return pos[0].copy()
    for slot in idx:
        dist = np.linalg.norm(pos[idx] - pos[slot], axis=1)
        score = int(np.sum(dist < 180.0))
        if score > best_n:
            best_n = score
            best = pos[slot].copy()
    return best if best is not None else pos[idx[0]].copy()


def _outward_array(pos: np.ndarray, cx: float, cy: float, hw: float, hh: float, cr: float) -> np.ndarray:
    local_x = pos[:, 0] - cx
    local_y = pos[:, 1] - cy
    ax = np.abs(local_x)
    ay = np.abs(local_y)
    corner = (ax > hw - cr) & (ay > hh - cr)
    ox = np.sign(local_x)
    oy = np.sign(local_y)
    ox = np.where(corner, ox, np.where(ax >= ay, ox, 0.0))
    oy = np.where(corner, oy, np.where(ay > ax, oy, 0.0))
    out = np.column_stack([ox, oy])
    length = np.maximum(np.sqrt(ox * ox + oy * oy), 1e-8)
    out[:, 0] /= length
    out[:, 1] /= length
    return out


def _kinetic_fail(speed_sum: dict[int, float], speed_n: dict[int, int], cfg: ArenaConfig, until: float) -> tuple[bool, list[float]]:
    end = until - cfg.kinetic_tail
    if end <= cfg.kinetic_window:
        return False, []
    last_bucket = int(end // cfg.kinetic_window)
    means: list[float] = []
    failed = False
    for bucket in range(last_bucket):
        count = speed_n.get(bucket, 0)
        if count <= 0:
            means.append(0.0)
            failed = True
            continue
        mean = speed_sum[bucket] / count
        means.append(mean)
        if mean < cfg.kinetic_min:
            failed = True
    return failed, means


def _dead_ok(times: list[float], until: float, gap: float) -> bool:
    if until <= 0:
        return True
    marks = [0.0] + [item for item in times if item <= until] + [until]
    return all(b - a <= gap + 1e-6 for a, b in zip(marks, marks[1:]))


def alive_at(elims: tuple[Elim, ...], count: int, time: float) -> int:
    gone = sum(1 for item in elims if item.time <= time + 1e-9)
    return count - gone


def evaluate_gates(result: SimResult, cfg: ArenaConfig) -> dict[str, bool]:
    count = len(cfg.countries)
    metrics = result.metrics or {}
    gates: dict[str, bool] = {}
    for when, low, high in cfg.gates:
        alive = alive_at(result.elims, count, when)
        gates[f"alive@{int(when)}"] = low <= alive <= high
    gates["winner_window"] = result.t_win is not None and cfg.win_window[0] <= result.t_win <= cfg.win_window[1]
    duel = 0.0 if result.t_win is None or result.duel_start is None else result.t_win - result.duel_start
    gates["final_duel"] = result.duel_start is not None and duel >= cfg.final_duel_min
    first = result.elims[0].time if result.elims else None
    gates["first_ko"] = first is not None and cfg.first_ko[0] <= first <= cfg.first_ko[1]
    impact = metrics.get("first_impact")
    gates["first_impact"] = impact is not None and cfg.first_impact[0] <= impact <= cfg.first_impact[1]
    gates["cause_ball"] = float(metrics.get("cause_ball", 0.0)) >= cfg.cause_ball_min
    gates["cause_storm"] = float(metrics.get("cause_storm", 1.0)) <= cfg.cause_storm_max
    gates["cause_self"] = float(metrics.get("cause_self", 1.0)) <= cfg.cause_self_max
    gates["dead_time"] = bool(metrics.get("dead_ok", False))
    gates["kinetic"] = bool(metrics.get("kinetic_ok", False))
    gates["heap"] = float(metrics.get("heap", 1.0)) <= cfg.heap_bottom
    gates["neighbors"] = float(metrics.get("neighbors", 99.0)) <= cfg.heap_neighbors
    gates["final_hit"] = metrics.get("final_cause") == "ball"
    gates["no_tie"] = not bool(metrics.get("tie", False)) and result.winner is not None
    return gates


def drama_components(result: SimResult, cfg: ArenaConfig, previous_winner: str | None = None) -> dict[str, float]:
    metrics = result.metrics or {}
    escapes = min(cfg.escape_cap, int(metrics.get("near_escapes", 0)))
    comebacks = min(cfg.comeback_cap, int(metrics.get("comebacks", 0)))
    combos = min(cfg.combo_cap, int(metrics.get("combos", 0)))
    revenges = min(cfg.revenge_cap, int(metrics.get("revenges", 0)))
    duel = 0.0 if result.t_win is None or result.duel_start is None else min(cfg.duel_cap, result.t_win - result.duel_start)
    mutual = cfg.mutual_bonus if metrics.get("both_finalists_hit") else 0.0
    winner_bonus = cfg.winner_kos_bonus if int(metrics.get("winner_kos", 0)) >= 2 else 0.0
    winner_iso = ""
    if result.winner is not None and 0 <= result.winner < len(cfg.countries):
        winner_iso = cfg.countries[result.winner].iso2
    penalty = cfg.repeat_penalty if previous_winner and winner_iso == previous_winner else 0.0
    score = (
        escapes * cfg.escape_weight
        + comebacks * cfg.comeback_weight
        + combos * cfg.combo_weight
        + revenges * cfg.revenge_weight
        + duel * cfg.duel_weight
        + mutual
        + winner_bonus
        - penalty
    )
    return {
        "escapes": float(escapes),
        "comebacks": float(comebacks),
        "combos": float(combos),
        "revenges": float(revenges),
        "duel": float(duel),
        "mutual": float(mutual),
        "winner_kos": float(winner_bonus),
        "repeat_penalty": float(penalty),
        "score": float(score),
    }


def state_hash(result: SimResult) -> str:
    payload = {
        "elims": [(round(item.time, 4), item.index, item.cause, item.killer) for item in result.elims],
        "winner": result.winner,
        "impulses": len(result.impulses),
    }
    if result.trace is not None and len(result.trace["xy"]):
        payload["end"] = np.round(result.trace["xy"][-1], 2).tolist()
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def sim_fingerprint(cfg: ArenaConfig) -> str:
    payload = {
        "cast": [country.iso2 for country in cfg.countries],
        "damping": cfg.damping,
        "dash_speed": cfg.dash_speed,
        "aggression_scale": cfg.aggression_scale,
        "keyframes": cfg.platform_keyframes,
        "radius": cfg.ball_radius,
        "restitution": cfg.restitution,
        "v": 2,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def gate_pass_rates(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {}
    names = rows[0]["gates"].keys()
    return {name: sum(1 for row in rows if row["gates"][name]) / len(rows) for name in names}


def format_search_failure(rows: list[dict[str, Any]], minimum: float) -> str:
    passed = sum(1 for row in rows if row["passed"])
    rate = passed / len(rows) if rows else 0.0
    lines = [f"gate pass rate {rate:.1%} is under {minimum:.0%} ({passed}/{len(rows)})"]
    for name, value in gate_pass_rates(rows).items():
        lines.append(f"  {name}: {value:.1%}")
    lines.append(
        "Tune aggression (ai.aggression_scale), damping (physics.damping), "
        "dash speed (ai.dash_speed), or platform.keyframes in configs/arena_default.yaml. "
        "Gates were not relaxed."
    )
    return "\n".join(lines)


def _search_one(job: tuple) -> dict[str, Any]:
    cfg, seed, previous = job
    result = simulate(cfg, seed, record_trace=False)
    gates = evaluate_gates(result, cfg)
    parts = drama_components(result, cfg, previous_winner=previous)
    winner = None if result.winner is None else cfg.countries[result.winner].iso2
    return {
        "seed": seed,
        "passed": all(gates.values()),
        "gates": gates,
        "components": parts,
        "winner": winner,
        "t_win": result.t_win,
        "seconds": float(result.metrics.get("seconds", 0.0)),
    }


def search_seeds(
    cfg: ArenaConfig,
    count: int,
    *,
    workers: int = 1,
    start: int | None = None,
    previous_winner: str | None = None,
) -> list[dict[str, Any]]:
    origin = cfg.seed if start is None else start
    jobs = [(cfg, origin + offset, previous_winner) for offset in range(count)]
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    if workers <= 1 or count < 2:
        for index, job in enumerate(jobs, start=1):
            rows.append(_search_one(job))
            if index == 1 or index % 5 == 0 or time.perf_counter() - started > 20:
                print(f"search: {index}/{count} in {time.perf_counter() - started:.1f}s", flush=True)
                started = time.perf_counter()
        return rows
    import multiprocessing

    context = multiprocessing.get_context("spawn")
    done = 0
    last = time.perf_counter()
    with context.Pool(workers) as pool:
        for row in pool.imap_unordered(_search_one, jobs, chunksize=1):
            rows.append(row)
            done += 1
            if done == 1 or done % 5 == 0 or time.perf_counter() - last >= 20.0:
                print(f"search: {done}/{count}", flush=True)
                last = time.perf_counter()
    rows.sort(key=lambda item: item["seed"])
    return rows


def read_previous_winner(path: Path | None = None) -> str | None:
    history = Path(".cache/arena_history.json") if path is None else path
    if not history.is_file():
        return None
    data = json.loads(history.read_text(encoding="utf-8"))
    winner = data.get("winner")
    return str(winner) if winner else None
