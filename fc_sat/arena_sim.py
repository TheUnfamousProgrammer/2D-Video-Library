"""Deterministic Flag Arena simulation, pacing gates, and drama search.

pymunk is used when it imports. Otherwise a numpy circle-impulse solver runs.
The two backends are not required to match each other. Within one backend, the
same seed and config repeat.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from fc_sat.arena_config import ArenaConfig
from fc_sat.render import ease_in_out_cubic

try:
    import pymunk
except ImportError:  # pragma: no cover - exercised only when pymunk is absent
    pymunk = None

TAU = math.tau
HORIZON = 36.0


def active_backend() -> str:
    return "pymunk" if pymunk is not None else "numpy"


@dataclass(frozen=True)
class Elim:
    time: float
    index: int
    cause: str


@dataclass(frozen=True)
class Meteor:
    time: float
    x: float
    y: float


@dataclass
class SimResult:
    seed: int
    backend: str
    elims: tuple[Elim, ...]
    meteors: tuple[Meteor, ...]
    near_misses: tuple[tuple[int, float], ...]
    t_win: float | None
    winner: int | None
    duel_start: float | None
    cameo_exit: float | None
    trace: dict[str, np.ndarray] | None

    @property
    def placements(self) -> tuple[int, ...]:
        ordered = [item.index for item in self.elims]
        if self.winner is not None:
            ordered.append(self.winner)
        return tuple(ordered)


def zone_radius(cfg: ArenaConfig, time: float) -> float:
    frames = cfg.zone_keyframes
    if time <= frames[0][0]:
        return frames[0][1]
    if time >= frames[-1][0]:
        return frames[-1][1]
    for (t0, r0), (t1, r1) in zip(frames, frames[1:]):
        if t0 <= time <= t1:
            if t1 <= t0 or r0 == r1:
                return r0
            u = (time - t0) / (t1 - t0)
            return r0 + (r1 - r0) * ease_in_out_cubic(u)
    return frames[-1][1]


def _sweeper_hold(cfg: ArenaConfig) -> float:
    """Omega stays at the start value until six seconds before the ramp ends."""
    return max(cfg.sweeper_start, cfg.sweeper_omega_end_time - 6.0)


def sweeper_omega(cfg: ArenaConfig, time: float) -> float:
    if time < cfg.sweeper_start:
        return 0.0
    hold = _sweeper_hold(cfg)
    w0 = cfg.sweeper_omega_start
    w1 = cfg.sweeper_omega_end
    if time <= hold:
        return w0
    t1 = cfg.sweeper_omega_end_time
    if time >= t1 or t1 <= hold:
        return w1
    u = (time - hold) / (t1 - hold)
    return w0 + (w1 - w0) * u


def sweeper_angle(cfg: ArenaConfig, time: float) -> float:
    t0 = cfg.sweeper_start
    if time <= t0:
        return 0.0
    hold = _sweeper_hold(cfg)
    w0 = cfg.sweeper_omega_start
    w1 = cfg.sweeper_omega_end
    t1 = cfg.sweeper_omega_end_time
    if time <= hold:
        return w0 * (time - t0)
    base = w0 * (hold - t0)
    span = max(1e-6, t1 - hold)
    if time <= t1:
        dt = time - hold
        return base + w0 * dt + (w1 - w0) * dt * dt / (2.0 * span)
    mid = base + w0 * span + (w1 - w0) * span / 2.0
    return mid + w1 * (time - t1)


def is_out(distance: float, zone: float, radius: float, margin_radii: float) -> bool:
    return distance > zone + margin_radii * radius


def alive_at(elims: tuple[Elim, ...], count: int, time: float) -> int:
    gone = sum(1 for item in elims if item.time <= time + 1e-9)
    return count - gone


def evaluate_gates(result: SimResult, cfg: ArenaConfig) -> dict[str, bool]:
    count = len(cfg.countries)
    gates: dict[str, bool] = {}
    first = result.elims[0].time if result.elims else math.inf
    gates["no_early_elim"] = first >= cfg.no_elim_before - 1e-9
    for time, low, high in cfg.gates:
        alive = alive_at(result.elims, count, time)
        gates[f"alive@{time:g}"] = low <= alive <= high
    won = result.t_win is not None and cfg.win_window[0] <= result.t_win <= cfg.win_window[1]
    gates["winner_window"] = won
    duel = 0.0 if result.t_win is None or result.duel_start is None else result.t_win - result.duel_start
    gates["final_duel"] = won and duel + 1e-9 >= cfg.final_duel_min
    return gates


def drama_components(
    result: SimResult,
    cfg: ArenaConfig,
    *,
    previous_winner: str | None = None,
) -> dict[str, float]:
    t_win = result.t_win if result.t_win is not None else HORIZON
    near = sum(1 for _index, when in result.near_misses if t_win - cfg.near_miss_window <= when <= t_win)
    near_score = cfg.near_miss_weight * min(near, cfg.near_miss_cap)
    duel = 0.0 if result.t_win is None or result.duel_start is None else max(0.0, result.t_win - result.duel_start)
    duel_score = cfg.duel_weight * min(duel, cfg.duel_cap)
    late = sum(1 for item in result.elims if t_win - cfg.late_elim_window <= item.time <= t_win)
    late_score = cfg.late_elim_weight if late >= cfg.late_elim_min else 0.0
    doubles = 0
    for earlier, later in zip(result.elims, result.elims[1:]):
        if later.time - earlier.time <= cfg.double_window:
            doubles += 1
    double_score = cfg.double_weight * min(doubles, cfg.double_cap)
    meteor_elims = sum(1 for item in result.elims if item.cause == "meteor")
    low, high = cfg.meteor_elim_range
    meteor_score = cfg.meteor_elim_weight if low <= meteor_elims <= high else 0.0
    winner_code = None if result.winner is None else cfg.countries[result.winner].iso2
    penalty = cfg.repeat_penalty if previous_winner and winner_code == previous_winner else 0.0
    total = near_score + duel_score + late_score + double_score + meteor_score - penalty
    return {
        "near_misses": float(near),
        "near_score": near_score,
        "duel": duel,
        "duel_score": duel_score,
        "late_elims": float(late),
        "late_score": late_score,
        "doubles": float(doubles),
        "double_score": double_score,
        "meteor_elims": float(meteor_elims),
        "meteor_score": meteor_score,
        "repeat_penalty": penalty,
        "score": total,
    }


def state_hash(result: SimResult) -> str:
    blob = json.dumps(
        {
            "elims": [(round(item.time, 5), item.index, item.cause) for item in result.elims],
            "meteors": [(round(item.time, 5), round(item.x, 3), round(item.y, 3)) for item in result.meteors],
            "winner": result.winner,
            "t_win": None if result.t_win is None else round(result.t_win, 5),
        },
        separators=(",", ":"),
    ).encode()
    if result.trace is not None:
        xy = np.ascontiguousarray(np.round(result.trace["xy"], 3))
        blob += xy.tobytes()
    return hashlib.sha256(blob).hexdigest()


def _poisson(rng: np.random.Generator, count: int, disk: float, min_dist: float) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    tries = 0
    min_sq = min_dist * min_dist
    while len(points) < count and tries < 40000:
        tries += 1
        ang = float(rng.random() * TAU)
        rad = disk * math.sqrt(float(rng.random()))
        x = rad * math.cos(ang)
        y = rad * math.sin(ang)
        if all((x - px) * (x - px) + (y - py) * (y - py) >= min_sq for px, py in points):
            points.append((x, y))
    if len(points) < count:
        raise RuntimeError(f"poisson disk placed {len(points)} of {count}")
    return points


def _schedule(cfg: ArenaConfig, rng: np.random.Generator, horizon: float) -> tuple[list[Meteor], float]:
    meteors: list[Meteor] = []
    cx, cy = cfg.center
    time = cfg.meteor_start
    first = True
    while time < horizon:
        if not first:
            time += float(rng.uniform(cfg.meteor_gap[0], cfg.meteor_gap[1]))
        first = False
        if time >= horizon:
            break
        zone = zone_radius(cfg, time)
        ang = float(rng.random() * TAU)
        rad = zone * math.sqrt(float(rng.random())) * 0.9
        meteors.append(Meteor(time, cx + rad * math.cos(ang), cy + rad * math.sin(ang)))
    cameo_ang = float(rng.random() * TAU)
    return meteors, cameo_ang


def _spawn_vel(cfg: ArenaConfig, rng: np.random.Generator, x: float, y: float) -> tuple[float, float]:
    cx, cy = cfg.center
    inward = math.atan2(cy - y, cx - x)
    jitter = math.radians(float(rng.uniform(-cfg.inward_deg, cfg.inward_deg)))
    speed = float(rng.uniform(cfg.speed_min, cfg.speed_max))
    ang = inward + jitter
    return speed * math.cos(ang), speed * math.sin(ang)


@dataclass
class _Ball:
    index: int
    x: float
    y: float
    vx: float
    vy: float
    angle: float
    omega: float
    radius: float
    mass: float
    factor: float
    wander: tuple[float, float]
    wander_at: float
    alive: bool
    cameo: bool
    danger: bool


def _damp(cfg: ArenaConfig) -> float:
    return math.exp(-cfg.damping)


def simulate(
    cfg: ArenaConfig,
    seed: int | None = None,
    *,
    horizon: float = HORIZON,
    record_trace: bool = False,
    backend: str | None = None,
) -> SimResult:
    chosen = active_backend() if backend is None else backend
    if chosen == "pymunk":
        if pymunk is None:
            raise RuntimeError("pymunk is not installed")
        return _simulate_pymunk(cfg, seed, horizon=horizon, record_trace=record_trace)
    if chosen != "numpy":
        raise RuntimeError(f"unknown physics backend {chosen!r}")
    return _simulate_numpy(cfg, seed, horizon=horizon, record_trace=record_trace)


def _build_balls(cfg: ArenaConfig, seed: int) -> tuple[list[_Ball], list[Meteor], float, np.random.Generator]:
    rng = np.random.default_rng(np.random.PCG64(seed))
    cx, cy = cfg.center
    points = _poisson(rng, len(cfg.countries), cfg.disk_radius, cfg.ball_radius * 2.0)
    balls: list[_Ball] = []
    for index, (lx, ly) in enumerate(points):
        x = cx + lx
        y = cy + ly
        vx, vy = _spawn_vel(cfg, rng, x, y)
        factor = float(rng.uniform(cfg.edge_factor[0], cfg.edge_factor[1]))
        wander = _wander_vec(cfg, rng)
        balls.append(
            _Ball(
                index=index,
                x=x,
                y=y,
                vx=vx,
                vy=vy,
                angle=0.0,
                omega=0.0,
                radius=cfg.ball_radius,
                mass=cfg.ball_mass,
                factor=factor,
                wander=wander,
                wander_at=float(rng.uniform(cfg.wander_period[0], cfg.wander_period[1])),
                alive=True,
                cameo=False,
                danger=False,
            )
        )
    meteors, cameo_ang = _schedule(cfg, rng, HORIZON)
    return balls, meteors, cameo_ang, rng


def _wander_vec(cfg: ArenaConfig, rng: np.random.Generator) -> tuple[float, float]:
    ang = float(rng.random() * TAU)
    return cfg.wander_accel * math.cos(ang), cfg.wander_accel * math.sin(ang)


def _edge_accel(cfg: ArenaConfig, ball: _Ball, zone: float) -> tuple[float, float]:
    cx, cy = cfg.center
    dx = ball.x - cx
    dy = ball.y - cy
    dist = math.hypot(dx, dy)
    if dist < 1e-6:
        return 0.0, 0.0
    margin = zone - dist
    if margin >= cfg.edge_margin:
        return 0.0, 0.0
    if margin >= 0:
        scale = 1.0 - margin / cfg.edge_margin
    else:
        scale = 1.0
    accel = cfg.edge_accel * ball.factor * scale
    return -accel * dx / dist, -accel * dy / dist


def _cause(
    index: int,
    time: float,
    meteors: list[Meteor],
    sweeper_touch: set[int],
    ball_touch: set[int],
    meteor_touch: set[int],
) -> str:
    for meteor in reversed(meteors):
        if meteor.time > time:
            continue
        if time - meteor.time <= 0.35 and index in meteor_touch:
            return "meteor"
        break
    if index in sweeper_touch:
        return "sweeper"
    if index in ball_touch:
        return "ball"
    return "self"


def _simulate_numpy(
    cfg: ArenaConfig,
    seed: int | None,
    *,
    horizon: float,
    record_trace: bool,
) -> SimResult:
    used = cfg.seed if seed is None else seed
    balls, meteors, cameo_ang, rng = _build_balls(cfg, used)
    return _integrate(
        cfg,
        used,
        balls,
        meteors,
        cameo_ang,
        rng,
        horizon=horizon,
        record_trace=record_trace,
        backend="numpy",
        stepper=_numpy_step,
    )


def _numpy_step(
    cfg: ArenaConfig,
    balls: list[_Ball],
    dt: float,
    zone: float,
    time: float,
    omega_scale: float = 1.0,
    edge_scale: float = 1.0,
) -> tuple[set[int], set[int]]:
    sweeper_touch: set[int] = set()
    ball_touch: set[int] = set()
    alive = [ball for ball in balls if ball.alive]
    for ball in alive:
        ax, ay = ball.wander
        ex, ey = (0.0, 0.0) if ball.cameo else _edge_accel(cfg, ball, zone)
        ex *= edge_scale
        ey *= edge_scale
        ball.vx += (ax + ex) * dt
        ball.vy += (ay + ey) * dt
    decay = _damp(cfg) ** dt
    for ball in alive:
        ball.vx *= decay
        ball.vy *= decay
        ball.x += ball.vx * dt
        ball.y += ball.vy * dt
        ball.angle += ball.omega * dt
    for i, a in enumerate(alive):
        for b in alive[i + 1 :]:
            dx = b.x - a.x
            dy = b.y - a.y
            dist = math.hypot(dx, dy)
            limit = a.radius + b.radius
            if dist >= limit or dist < 1e-8:
                continue
            nx = dx / dist
            ny = dy / dist
            overlap = limit - dist
            share_a = b.mass / (a.mass + b.mass)
            share_b = a.mass / (a.mass + b.mass)
            a.x -= nx * overlap * share_a
            a.y -= ny * overlap * share_a
            b.x += nx * overlap * share_b
            b.y += ny * overlap * share_b
            rel = (b.vx - a.vx) * nx + (b.vy - a.vy) * ny
            if rel < 0:
                impulse = -(1.0 + cfg.restitution) * rel / (1.0 / a.mass + 1.0 / b.mass)
                a.vx -= impulse * nx / a.mass
                a.vy -= impulse * ny / a.mass
                b.vx += impulse * nx / b.mass
                b.vy += impulse * ny / b.mass
                if not a.cameo:
                    ball_touch.add(a.index)
                if not b.cameo:
                    ball_touch.add(b.index)
    if time >= cfg.sweeper_start:
        angle = sweeper_angle(cfg, time)
        length = max(1.0, zone - cfg.sweeper_inset)
        omega = sweeper_omega(cfg, time) * omega_scale
        half = cfg.sweeper_thickness * 0.5
        ux, uy = math.cos(angle), math.sin(angle)
        px, py = -uy, ux
        cx, cy = cfg.center
        for ball in alive:
            along = (ball.x - cx) * ux + (ball.y - cy) * uy
            if along < 0 or along > length:
                continue
            side = (ball.x - cx) * px + (ball.y - cy) * py
            limit = ball.radius + half
            if abs(side) >= limit:
                continue
            sign = 1.0 if side >= 0 else -1.0
            push = limit - abs(side)
            ball.x += px * sign * push
            ball.y += py * sign * push
            bar_vx = -omega * along * uy
            bar_vy = omega * along * ux
            rel = (ball.vx - bar_vx) * px * sign + (ball.vy - bar_vy) * py * sign
            if rel < 0:
                impulse = -(1.0 + cfg.restitution) * rel * ball.mass
                ball.vx += impulse * px * sign / ball.mass
                ball.vy += impulse * py * sign / ball.mass
            if not ball.cameo:
                sweeper_touch.add(ball.index)
    return sweeper_touch, ball_touch


def _simulate_pymunk(
    cfg: ArenaConfig,
    seed: int | None,
    *,
    horizon: float,
    record_trace: bool,
) -> SimResult:
    used = cfg.seed if seed is None else seed
    balls, meteors, cameo_ang, rng = _build_balls(cfg, used)
    return _integrate(
        cfg,
        used,
        balls,
        meteors,
        cameo_ang,
        rng,
        horizon=horizon,
        record_trace=record_trace,
        backend="pymunk",
        stepper=_PymunkWorld(cfg, balls),
    )


class _PymunkWorld:
    def __init__(self, cfg: ArenaConfig, balls: list[_Ball]) -> None:
        assert pymunk is not None
        self.cfg = cfg
        self.balls = balls
        self.space = pymunk.Space()
        self.space.gravity = (0.0, 0.0)
        self.space.damping = _damp(cfg)
        self.bodies: dict[int, Any] = {}
        self.shapes: dict[int, Any] = {}
        self.sweeper_body = None
        self.sweeper_shape = None
        self.sweeper_touch: set[int] = set()
        self.ball_touch: set[int] = set()
        for ball in balls:
            self._add(ball)
        self._add_sweeper()
        self.space.on_collision(1, 1, begin=self._balls_hit)
        self.space.on_collision(1, 2, begin=self._bar_hit)

    def _add(self, ball: _Ball) -> None:
        moment = pymunk.moment_for_circle(ball.mass, 0, ball.radius)
        body = pymunk.Body(ball.mass, moment)
        body.position = (ball.x, ball.y)
        body.velocity = (ball.vx, ball.vy)
        shape = pymunk.Circle(body, ball.radius)
        shape.elasticity = self.cfg.restitution
        shape.friction = 0.0
        shape.collision_type = 1
        shape.ball_index = ball.index
        self.space.add(body, shape)
        self.bodies[ball.index] = body
        self.shapes[ball.index] = shape

    def _add_sweeper(self) -> None:
        cfg = self.cfg
        body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        body.position = cfg.center
        length = max(1.0, cfg.floor_radius - cfg.sweeper_inset)
        shape = pymunk.Segment(body, (0, 0), (length, 0), cfg.sweeper_thickness * 0.5)
        shape.elasticity = cfg.restitution
        shape.friction = 0.0
        shape.collision_type = 2
        self.space.add(body, shape)
        self.sweeper_body = body
        self.sweeper_shape = shape

    def _balls_hit(self, arbiter, space, data) -> None:
        for shape in arbiter.shapes:
            index = getattr(shape, "ball_index", None)
            if index is not None and index >= 0:
                self.ball_touch.add(index)

    def _bar_hit(self, arbiter, space, data) -> None:
        for shape in arbiter.shapes:
            index = getattr(shape, "ball_index", None)
            if index is not None and index >= 0:
                self.sweeper_touch.add(index)

    def add_cameo(self, ball: _Ball) -> None:
        self._add(ball)

    def remove(self, index: int) -> None:
        body = self.bodies.pop(index, None)
        shape = self.shapes.pop(index, None)
        if body is not None and shape is not None:
            self.space.remove(body, shape)

    def step(
        self,
        cfg: ArenaConfig,
        balls: list[_Ball],
        dt: float,
        zone: float,
        time: float,
        omega_scale: float = 1.0,
        edge_scale: float = 1.0,
    ) -> tuple[set[int], set[int]]:
        self.sweeper_touch = set()
        self.ball_touch = set()
        length = max(1.0, zone - cfg.sweeper_inset)
        body = self.sweeper_body
        shape = self.sweeper_shape
        shape.sensor = time < cfg.sweeper_start
        if time >= cfg.sweeper_start:
            body.angle = sweeper_angle(cfg, time)
            body.angular_velocity = sweeper_omega(cfg, time) * omega_scale
            shape.unsafe_set_endpoints((0, 0), (length, 0))
        else:
            body.angle = 0.0
            body.angular_velocity = 0.0
            shape.unsafe_set_endpoints((0, 0), (1, 0))
        self.space.reindex_shapes_for_body(body)
        for ball in balls:
            if not ball.alive or ball.index not in self.bodies:
                continue
            ax, ay = ball.wander
            ex, ey = (0.0, 0.0) if ball.cameo else _edge_accel(cfg, ball, zone)
            ex *= edge_scale
            ey *= edge_scale
            phys = self.bodies[ball.index]
            phys.force = ((ax + ex) * ball.mass, (ay + ey) * ball.mass)
        self.space.step(dt)
        for ball in balls:
            if ball.index not in self.bodies:
                continue
            phys = self.bodies[ball.index]
            ball.x, ball.y = phys.position
            ball.vx, ball.vy = phys.velocity
            ball.angle = phys.angle
            ball.omega = phys.angular_velocity
        return set(self.sweeper_touch), set(self.ball_touch)


def _integrate(
    cfg: ArenaConfig,
    seed: int,
    balls: list[_Ball],
    meteors: list[Meteor],
    cameo_ang: float,
    rng: np.random.Generator,
    *,
    horizon: float,
    record_trace: bool,
    backend: str,
    stepper,
) -> SimResult:
    dt = cfg.dt
    steps = int(math.ceil(horizon / dt))
    cx, cy = cfg.center
    elims: list[Elim] = []
    near: list[tuple[int, float]] = []
    meteor_touch: set[int] = set()
    meteor_index = 0
    cameo: _Ball | None = None
    cameo_entered = False
    cameo_exit: float | None = None
    t_win: float | None = None
    winner: int | None = None
    duel_start: float | None = None
    trace_xy = [] if record_trace else None
    trace_ang = [] if record_trace else None
    trace_alive = [] if record_trace else None
    trace_cameo = [] if record_trace else None
    country_count = len(cfg.countries)
    for step in range(steps):
        time = step * dt
        zone = zone_radius(cfg, time)
        for ball in balls:
            if not ball.alive or ball.cameo:
                continue
            if time >= ball.wander_at:
                ball.wander = _wander_vec(cfg, rng)
                ball.wander_at = time + float(rng.uniform(cfg.wander_period[0], cfg.wander_period[1]))
        if cameo is None and time >= cfg.cameo_enter:
            zone_now = zone_radius(cfg, cfg.cameo_enter)
            dist = zone_now + cfg.cameo_radius + 30.0
            x = cx + dist * math.cos(cameo_ang)
            y = cy + dist * math.sin(cameo_ang)
            inward = math.atan2(cy - y, cx - x)
            speed = 900.0
            cameo = _Ball(
                index=-1,
                x=x,
                y=y,
                vx=speed * math.cos(inward),
                vy=speed * math.sin(inward),
                angle=0.0,
                omega=0.0,
                radius=cfg.cameo_radius,
                mass=cfg.cameo_mass,
                factor=1.0,
                wander=(0.0, 0.0),
                wander_at=math.inf,
                alive=True,
                cameo=True,
                danger=False,
            )
            balls.append(cameo)
            if hasattr(stepper, "add_cameo"):
                stepper.add_cameo(cameo)
        if cameo is not None and cameo.alive:
            dx = cameo.x - cx
            dy = cameo.y - cy
            dist = math.hypot(dx, dy) or 1.0
            if dist < zone:
                cameo_entered = True
            if time >= cfg.cameo_enter + 1.0:
                cameo.wander = (700.0 * dx / dist, 700.0 * dy / dist)
            if time >= cfg.cameo_exit - 0.6:
                need = zone + cameo.radius + 40.0 - dist
                launch = max(700.0, need / 0.35)
                cameo.vx = launch * dx / dist
                cameo.vy = launch * dy / dist
                cameo.wander = (0.0, 0.0)
                if backend == "pymunk" and cameo.index in stepper.bodies:
                    stepper.bodies[cameo.index].velocity = (cameo.vx, cameo.vy)
            outside = cameo_entered and dist > zone + cameo.radius + 5.0
            if time >= cfg.cameo_exit or outside:
                cameo.alive = False
                cameo_exit = time
                if hasattr(stepper, "remove"):
                    stepper.remove(cameo.index)
        grace = duel_start is not None and winner is None and time < duel_start + cfg.final_duel_min + 0.4
        omega_scale = 0.25 if grace else 1.0
        edge_scale = 3.0 if grace else 1.0
        while meteor_index < len(meteors) and meteors[meteor_index].time <= time:
            meteor = meteors[meteor_index]
            meteor_index += 1
            if grace:
                continue
            meteor_touch = set()
            for ball in balls:
                if not ball.alive or ball.cameo:
                    continue
                dx = ball.x - meteor.x
                dy = ball.y - meteor.y
                dist = math.hypot(dx, dy)
                if dist >= cfg.meteor_radius:
                    continue
                if dist < 1e-6:
                    dx, dy, dist = ball.x - cx, ball.y - cy, math.hypot(ball.x - cx, ball.y - cy) or 1.0
                falloff = 1.0 - dist / cfg.meteor_radius
                dv = cfg.meteor_impulse * falloff
                ball.vx += dv * dx / dist
                ball.vy += dv * dy / dist
                meteor_touch.add(ball.index)
                if backend == "pymunk" and ball.index in stepper.bodies:
                    stepper.bodies[ball.index].velocity = (ball.vx, ball.vy)
        if hasattr(stepper, "step"):
            sweeper_touch, ball_touch = stepper.step(cfg, balls, dt, zone, time, omega_scale, edge_scale)
        else:
            sweeper_touch, ball_touch = stepper(cfg, balls, dt, zone, time, omega_scale, edge_scale)
        doomed: list[tuple[float, _Ball]] = []
        for ball in list(balls):
            if not ball.alive or ball.cameo:
                continue
            dist = math.hypot(ball.x - cx, ball.y - cy)
            margin = zone + cfg.out_margin_radii * ball.radius - dist
            limit = cfg.near_miss_margin_radii * ball.radius
            if margin < limit:
                ball.danger = True
            elif ball.danger:
                ball.danger = False
                near.append((ball.index, time))
            if is_out(dist, zone, ball.radius, cfg.out_margin_radii):
                doomed.append((dist, ball))
        alive_now = sum(1 for ball in balls if ball.alive and not ball.cameo)
        protect = winner is None and (duel_start is None or time < duel_start + cfg.final_duel_min)
        can_kill = max(0, alive_now - 2) if protect else len(doomed)
        doomed.sort(key=lambda item: item[0])
        split = max(0, len(doomed) - can_kill)
        for dist, ball in doomed[:split]:
            dx = ball.x - cx
            dy = ball.y - cy
            norm = dist or 1.0
            keep = max(1.0, zone + cfg.out_margin_radii * ball.radius - 4.0)
            ball.x = cx + keep * dx / norm
            ball.y = cy + keep * dy / norm
            ball.vx = -80.0 * dx / norm
            ball.vy = -80.0 * dy / norm
            if backend == "pymunk" and hasattr(stepper, "bodies") and ball.index in stepper.bodies:
                stepper.bodies[ball.index].position = (ball.x, ball.y)
                stepper.bodies[ball.index].velocity = (ball.vx, ball.vy)
        for _dist, ball in doomed[split:]:
            ball.alive = False
            elims.append(Elim(time, ball.index, _cause(ball.index, time, meteors, sweeper_touch, ball_touch, meteor_touch)))
            if hasattr(stepper, "remove"):
                stepper.remove(ball.index)
        country_alive = [ball.index for ball in balls if ball.alive and not ball.cameo]
        if len(country_alive) == 2 and duel_start is None:
            duel_start = time
        if len(country_alive) == 1 and winner is None:
            winner = country_alive[0]
            t_win = time
        if record_trace:
            xy = np.zeros((country_count, 2), dtype=np.float64)
            ang = np.zeros(country_count, dtype=np.float64)
            alive = np.zeros(country_count, dtype=np.bool_)
            for ball in balls:
                if ball.cameo or ball.index < 0 or ball.index >= country_count:
                    continue
                xy[ball.index] = (ball.x, ball.y)
                ang[ball.index] = ball.angle
                alive[ball.index] = ball.alive
            trace_xy.append(xy)
            trace_ang.append(ang)
            trace_alive.append(alive)
            if cameo is None:
                trace_cameo.append((0.0, 0.0, 0.0))
            else:
                trace_cameo.append((cameo.x, cameo.y, 1.0 if cameo.alive else 0.0))
        if t_win is not None and time > t_win + 1.0 and time > cfg.win_window[1]:
            break
    elims.sort(key=lambda item: (item.time, item.index))
    trace = None
    if record_trace:
        cameo_rows = np.asarray(trace_cameo, dtype=np.float64)
        trace = {
            "xy": np.stack(trace_xy),
            "angle": np.stack(trace_ang),
            "alive": np.stack(trace_alive),
            "cameo_xy": cameo_rows[:, :2],
            "cameo_alive": cameo_rows[:, 2] > 0.5,
        }
    return SimResult(
        seed=seed,
        backend=backend,
        elims=tuple(elims),
        meteors=tuple(meteors),
        near_misses=tuple(near),
        t_win=t_win,
        winner=winner,
        duel_start=duel_start,
        cameo_exit=cameo_exit,
        trace=trace,
    )


def sim_fingerprint(cfg: ArenaConfig, seed: int) -> str:
    payload = {
        "seed": seed,
        "cast": [country.iso2 for country in cfg.countries],
        "ball_radius": cfg.ball_radius,
        "ball_mass": cfg.ball_mass,
        "restitution": cfg.restitution,
        "damping": cfg.damping,
        "zone": cfg.zone_keyframes,
        "spawn": [
            cfg.disk_radius,
            cfg.speed_min,
            cfg.speed_max,
            cfg.inward_deg,
            cfg.wander_accel,
            cfg.edge_accel,
            cfg.edge_margin,
        ],
        "sweeper": [
            cfg.sweeper_start,
            cfg.sweeper_omega_start,
            cfg.sweeper_omega_end,
            cfg.sweeper_omega_end_time,
        ],
        "meteor": [cfg.meteor_start, cfg.meteor_gap, cfg.meteor_impulse, cfg.meteor_radius],
        "cameo": [cfg.cameo_enter, cfg.cameo_exit, cfg.cameo_radius, cfg.cameo_mass],
        "backend": active_backend(),
    }
    raw = json.dumps(payload, sort_keys=True, default=list).encode()
    return hashlib.sha256(raw).hexdigest()


def cache_path(cfg: ArenaConfig, seed: int) -> Path:
    return Path(".cache") / f"arena_{sim_fingerprint(cfg, seed)[:20]}.json"


def result_summary(result: SimResult, cfg: ArenaConfig, *, previous_winner: str | None = None) -> dict[str, Any]:
    gates = evaluate_gates(result, cfg)
    parts = drama_components(result, cfg, previous_winner=previous_winner)
    winner = None if result.winner is None else cfg.countries[result.winner].iso2
    return {
        "seed": result.seed,
        "backend": result.backend,
        "passed": all(gates.values()),
        "gates": gates,
        "winner": winner,
        "t_win": result.t_win,
        "placements": [cfg.countries[index].iso2 for index in result.placements],
        "elims": [{"time": item.time, "iso2": cfg.countries[item.index].iso2, "cause": item.cause} for item in result.elims],
        "components": parts,
        "cameo_exit": result.cameo_exit,
    }


def load_cached(cfg: ArenaConfig, seed: int) -> dict[str, Any] | None:
    path = cache_path(cfg, seed)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def store_cached(cfg: ArenaConfig, summary: dict[str, Any]) -> None:
    path = cache_path(cfg, int(summary["seed"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary), encoding="utf-8")
    os.replace(temporary, path)


def _search_one(payload: tuple[ArenaConfig, int, str | None]) -> dict[str, Any]:
    cfg, seed, previous = payload
    cached = load_cached(cfg, seed)
    if cached is not None and cached.get("backend") == active_backend():
        if previous:
            parts = dict(cached["components"])
            winner = cached.get("winner")
            penalty = cfg.repeat_penalty if winner == previous else 0.0
            parts["repeat_penalty"] = penalty
            parts["score"] = (
                parts["near_score"]
                + parts["duel_score"]
                + parts["late_score"]
                + parts["double_score"]
                + parts["meteor_score"]
                - penalty
            )
            cached["components"] = parts
        return cached
    summary = result_summary(simulate(cfg, seed), cfg, previous_winner=previous)
    store_cached(cfg, summary)
    return summary


def read_previous_winner(path: Path = Path(".cache/arena_history.json")) -> str | None:
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    winner = data.get("winner")
    return str(winner) if winner else None


def search_seeds(
    cfg: ArenaConfig,
    count: int,
    *,
    workers: int = 1,
    start: int | None = None,
    previous_winner: str | None = None,
) -> list[dict[str, Any]]:
    origin = cfg.seed if start is None else start
    seeds = [origin + offset for offset in range(count)]
    jobs = [(cfg, seed, previous_winner) for seed in seeds]
    if workers <= 1 or count < 2:
        return [_search_one(job) for job in jobs]
    import multiprocessing

    context = multiprocessing.get_context("spawn")
    with context.Pool(workers) as pool:
        return list(pool.map(_search_one, jobs, chunksize=1))


def gate_pass_rates(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {}
    names = rows[0]["gates"].keys()
    return {name: sum(1 for row in rows if row["gates"][name]) / len(rows) for name in names}


def format_search_failure(rows: list[dict[str, Any]], minimum: float) -> str:
    passed = sum(1 for row in rows if row["passed"])
    rate = passed / len(rows) if rows else 0.0
    lines = [
        f"gate pass rate {rate:.1%} is under {minimum:.0%} ({passed}/{len(rows)})",
    ]
    for name, value in gate_pass_rates(rows).items():
        lines.append(f"  {name}: {value:.1%}")
    lines.append(
        "Tune zone keyframes, hazard timing, or steering "
        "(spawn.speed_min, spawn.speed_max, spawn.wander_accel, spawn.edge_accel) "
        "in configs/arena_default.yaml. Gates were not relaxed."
    )
    return "\n".join(lines)
