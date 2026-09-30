"""Fixed-timestep ball simulation, spawn throttle, and pacing auto-tune.

Image y grows downward, so gravity is +y. The integrator is semi-implicit Euler
at dt = 1/240 s. A video frame at 60 fps is four substeps. Frame 0 is the state
before any integration so the opening pose can be redrawn on the last frame.

Pacing is a ceiling, not a wish. N_target(t) is log-linear between keyframes and
then held at the cap from the last keyframe (default 0.985*Tg) through Tg.
That hold is what makes "reach the cap inside [0.97*Tg, Tg]" possible: a final
keyframe exactly at Tg would only allow the cap on the last instant.

Auto-tune retries a deterministic sequence of initial speed, gravity, and seed
offset. Attempt 0 also searches the spawn inset so the first wall hit lands in
[0.30, 0.60] s. Gaps are measured only between consecutive hits after that first
hit; the wait from t=0 is the first-hit window, not a gap.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fc_sat.config import Config, sim_fingerprint_payload

DT = 1.0 / 240.0
EVENT_DTYPE = np.dtype(
    [
        ("time", "f8"),
        ("ball_id", "i4"),
        ("impact_speed", "f8"),
        ("x", "f8"),
        ("y", "f8"),
        ("count_at_time", "i4"),
        ("spawned", "bool"),
    ]
)
MILESTONE_COUNTS = (10, 50, 100, 250, 500, 750, 1000)


@dataclass
class SimResult:
    positions: np.ndarray  # (n_growth_frames, cap, 2) float32
    counts: np.ndarray  # (n_growth_frames,) int32
    radii: np.ndarray  # (n_growth_frames,) float32
    spawn_index: np.ndarray  # (cap,) int32, -1 if unused
    hold_positions: np.ndarray  # (cap, 2) float32
    hold_count: int
    hold_radius: float
    events: np.ndarray
    milestones: np.ndarray  # (n, 2) time, count
    initial_pos: np.ndarray  # (2,) float64
    initial_vel: np.ndarray  # (2,) float64
    first_bounce: float
    count_at_014: int
    cap_time: float | None
    max_gap: float | None
    hit_times: np.ndarray
    gravity_used: float
    speed_used: float
    seed_offset_used: int
    outside_violations: int
    collisions_disabled: bool

    @property
    def final_count(self) -> int:
        return int(self.hold_count)


def ball_radius(count: int, cfg: Config) -> float:
    count = max(int(count), 1)
    radius = cfg.radius_base * (count ** cfg.radius_exp)
    return float(min(cfg.radius_max, max(cfg.radius_min, radius)))


def n_target(t: float, cfg: Config) -> float:
    """Log-linear throttle. At and after the last keyframe, hold that count."""
    tg = cfg.growth_seconds
    u = 0.0 if tg <= 0 else t / tg
    frames = cfg.throttle
    if u >= frames[-1][0]:
        return frames[-1][1]
    if u <= frames[0][0]:
        return frames[0][1]
    for (f0, n0), (f1, n1) in zip(frames, frames[1:]):
        if u <= f1:
            if f1 <= f0:
                return n1
            alpha = (u - f0) / (f1 - f0)
            # n0 is 1 at the origin, so log(n0) is 0 and the curve leaves 1 smoothly.
            log_n = math.log(n0) + alpha * (math.log(n1) - math.log(n0))
            return math.exp(log_n)
    return frames[-1][1]


def ceil_target(t: float, cfg: Config) -> int:
    return int(math.ceil(n_target(t, cfg) - 1e-12))


def pacing_failures(
    *,
    first_bounce: float | None,
    count_at_014: int,
    cap_time: float | None,
    hit_times: np.ndarray,
    growth_seconds: float,
    cap: int,
) -> list[str]:
    """Return human-readable pacing failures. An empty list means the run is valid.

    The gap check starts at the first hit. Time before that hit is judged only
    by the first-bounce window.
    """
    fails: list[str] = []
    if first_bounce is None or not (0.30 <= first_bounce <= 0.60):
        shown = "none" if first_bounce is None else f"{first_bounce:.4f}s"
        fails.append(f"first bounce at {shown} not in [0.30, 0.60] s")
    if not (6 <= int(count_at_014) <= 10):
        fails.append(f"count at 0.14*Tg is {int(count_at_014)}, want 8 +/- 2")
    lo = 0.97 * growth_seconds
    if cap_time is None or not (lo <= cap_time <= growth_seconds + 1e-9):
        shown = "never" if cap_time is None else f"{cap_time:.4f}s"
        fails.append(
            f"cap {cap} reached at {shown}, want inside [{lo:.4f}, {growth_seconds:.4f}] s"
        )
    if hit_times.size >= 2:
        gaps = np.diff(hit_times)
        worst = float(gaps.max()) if gaps.size else 0.0
        if worst > 0.5 + 1e-9:
            fails.append(f"wall-hit gap {worst:.4f}s exceeds 0.5s")
    return fails


def _attempt_params(cfg: Config, index: int) -> tuple[float, float, int]:
    s0, s1 = cfg.tune_speed_range
    g0, g1 = cfg.tune_gravity_range
    o0, o1 = cfg.tune_seed_offset_range
    if index == 0:
        gravity = min(max(cfg.gravity, g0), g1)
        speed = min(max(1500.0, s0), s1)
        return speed, gravity, int(o0)
    span = max(cfg.tune_max_attempts - 1, 1)
    speed = s0 + (s1 - s0) * ((index * 3) % span) / span
    gravity = g0 + (g1 - g0) * ((index * 5) % span) / span
    offset = o0 + ((index * 7) % (o1 - o0 + 1))
    return float(speed), float(gravity), int(offset)


def _make_start(cfg: Config, speed: float, inset: float) -> tuple[np.ndarray, np.ndarray]:
    """Place ball 0 on the left side of the ring, heading across a near-diameter.

    Path length is about 2*(R - r - inset). At ~1500 px/s that flight lasts
    about 0.45 s, which sits inside the required first-hit window, and the
    return chord stays under 0.5 s while speed remains high.
    """
    radius = ball_radius(1, cfg)
    reach = cfg.ring_radius - radius - inset
    reach = max(radius, min(reach, cfg.ring_radius - radius - 1.0))
    pos = np.array([cfg.ring_cx - reach, cfg.ring_cy], dtype=np.float64)
    # A few degrees downward so gravity bends the chord instead of cancelling it.
    angle = 0.05
    vel = np.array([math.cos(angle), math.sin(angle)], dtype=np.float64) * speed
    return pos, vel


def _integrate_ball(
    pos: np.ndarray,
    vel: np.ndarray,
    radius: float,
    cfg: Config,
    gravity: float,
    t_limit: float,
) -> float | None:
    """Integrate one ball until the first outward wall hit. Used by the inset search."""
    center = np.array([cfg.ring_cx, cfg.ring_cy], dtype=np.float64)
    t = 0.0
    p = pos.copy()
    v = vel.copy()
    while t < t_limit - 1e-12:
        t = min(t_limit, t + DT)
        v[1] += gravity * DT
        speed = float(np.hypot(v[0], v[1]))
        if speed > cfg.max_speed:
            v *= cfg.max_speed / speed
        p += v * DT
        rel = p - center
        dist = float(np.hypot(rel[0], rel[1]))
        if dist + radius > cfg.ring_radius and dist > 1e-8:
            normal = rel / dist
            if float(np.dot(v, normal)) > 0:
                return t
    return None


def solve_start(cfg: Config, speed: float, gravity: float) -> tuple[np.ndarray, np.ndarray]:
    """Search the wall inset so the first hit prefers the middle of [0.30, 0.60]."""
    best: tuple[float, np.ndarray, np.ndarray] | None = None
    fallback: tuple[np.ndarray, np.ndarray] | None = None
    for inset in (8.0, 16.0, 28.0, 40.0, 56.0, 72.0, 96.0, 120.0):
        pos, vel = _make_start(cfg, speed, inset)
        hit = _integrate_ball(pos, vel, ball_radius(1, cfg), cfg, gravity, 0.80)
        if hit is None:
            continue
        fallback = (pos, vel)
        if 0.30 <= hit <= 0.60:
            error = abs(hit - 0.45)
            if best is None or error < best[0]:
                best = (error, pos, vel)
    if best is not None:
        return best[1], best[2]
    if fallback is not None:
        return fallback
    return _make_start(cfg, speed, 16.0)


class _EventLog:
    def __init__(self) -> None:
        self._data = np.zeros(8192, dtype=EVENT_DTYPE)
        self._n = 0

    def append(
        self,
        t: float,
        ball_id: int,
        impact: float,
        x: float,
        y: float,
        count: int,
        spawned: bool,
    ) -> None:
        if self._n >= len(self._data):
            grown = np.zeros(len(self._data) * 2, dtype=EVENT_DTYPE)
            grown[: self._n] = self._data
            self._data = grown
        row = self._data[self._n]
        row["time"] = t
        row["ball_id"] = ball_id
        row["impact_speed"] = impact
        row["x"] = x
        row["y"] = y
        row["count_at_time"] = count
        row["spawned"] = spawned
        self._n += 1

    def freeze(self) -> np.ndarray:
        return self._data[: self._n].copy()


def _soft_collisions(
    pos: np.ndarray,
    count: int,
    radius: float,
    cfg: Config,
) -> float:
    """Uniform-grid positional push. Returns elapsed milliseconds."""
    started = time.perf_counter()
    if count < 2 or radius <= 0:
        return (time.perf_counter() - started) * 1000.0
    cell = max(radius * 2.0, 1.0)
    grid: dict[tuple[int, int], list[int]] = {}
    coords = pos[:count]
    keys = np.floor(coords / cell).astype(np.int32)
    for index in range(count):
        grid.setdefault((int(keys[index, 0]), int(keys[index, 1])), []).append(index)
    min_dist = radius * 2.0
    for (gx, gy), members in grid.items():
        neighbor_ids = list(members)
        for ox, oy in ((1, 0), (0, 1), (1, 1), (1, -1)):
            neighbor_ids.extend(grid.get((gx + ox, gy + oy), ()))
        for i_local, i in enumerate(members):
            pi = pos[i]
            for j in neighbor_ids:
                if j <= i:
                    continue
                delta = pi - pos[j]
                dist = float(math.hypot(delta[0], delta[1]))
                if dist >= min_dist or dist < 1e-8:
                    continue
                push = 0.5 * (min_dist - dist) / dist
                shift = delta * push
                pos[i, 0] += shift[0]
                pos[i, 1] += shift[1]
                pos[j, 0] -= shift[0]
                pos[j, 1] -= shift[1]
                pi = pos[i]
    center_x = cfg.ring_cx
    center_y = cfg.ring_cy
    limit = cfg.ring_radius - radius
    rel_x = pos[:count, 0] - center_x
    rel_y = pos[:count, 1] - center_y
    dist = np.hypot(rel_x, rel_y)
    outside = dist > limit
    if np.any(outside):
        scale = np.ones(count, dtype=np.float64)
        safe = np.maximum(dist[outside], 1e-8)
        scale[outside] = limit / safe
        pos[:count, 0] = center_x + rel_x * scale
        pos[:count, 1] = center_y + rel_y * scale
    return (time.perf_counter() - started) * 1000.0


def simulate_once(
    cfg: Config,
    *,
    speed: float,
    gravity: float,
    seed_offset: int,
    collisions: bool | None = None,
) -> SimResult:
    """Run one growth simulation. Does not auto-tune or raise on pacing."""
    use_collisions = cfg.collisions if collisions is None else collisions
    collisions_disabled = False
    warned = False
    radius0 = ball_radius(1, cfg)
    pos0, vel0 = solve_start(cfg, speed, gravity)
    cap = cfg.cap
    pos = np.zeros((cap, 2), dtype=np.float64)
    vel = np.zeros((cap, 2), dtype=np.float64)
    cooldown_until = np.zeros(cap, dtype=np.float64)
    spawn_index = np.full(cap, -1, dtype=np.int32)
    pos[0] = pos0
    vel[0] = vel0
    spawn_index[0] = 0
    count = 1
    rng = np.random.Generator(np.random.PCG64(cfg.seed + int(seed_offset)))

    full_fps, plan = cfg.frame_plan(preview=False)
    n_growth = dict(plan)["growth"]
    positions = np.zeros((n_growth, cap, 2), dtype=np.float32)
    counts = np.zeros(n_growth, dtype=np.int32)
    radii = np.zeros(n_growth, dtype=np.float32)

    center = np.array([cfg.ring_cx, cfg.ring_cy], dtype=np.float64)
    events = _EventLog()
    milestone_rows: list[tuple[float, int]] = []
    milestone_left = {m for m in MILESTONE_COUNTS if m <= cap}
    hit_times: list[float] = []
    first_bounce: float | None = None
    count_at_014: int | None = None
    cap_time: float | None = None
    check_t = 0.14 * cfg.growth_seconds
    outside_violations = 0
    t = 0.0
    substeps_done = 0

    def snapshot(frame_index: int) -> None:
        positions[frame_index, :count] = pos[:count]
        counts[frame_index] = count
        radii[frame_index] = ball_radius(count, cfg)

    def mark_milestones(now: float) -> None:
        reached = [m for m in milestone_left if count >= m]
        for milestone in reached:
            milestone_rows.append((now, milestone))
            milestone_left.discard(milestone)

    def substep(now: float) -> None:
        nonlocal count, first_bounce, cap_time, outside_violations, use_collisions, collisions_disabled, warned
        radius = ball_radius(count, cfg)
        n = count
        vel[:n, 1] += gravity * DT
        speed_now = np.hypot(vel[:n, 0], vel[:n, 1])
        hot = speed_now > cfg.max_speed
        if np.any(hot):
            scale = np.ones(n, dtype=np.float64)
            scale[hot] = cfg.max_speed / speed_now[hot]
            vel[:n] *= scale[:, None]
        pos[:n] += vel[:n] * DT
        if use_collisions:
            elapsed_ms = _soft_collisions(pos, n, radius, cfg)
            if elapsed_ms > 8.0:
                use_collisions = False
                collisions_disabled = True
                if not warned:
                    warned = True
                    print(
                        f"warning: ball-ball collisions took {elapsed_ms:.1f} ms "
                        "on a substep; disabling them for the rest of this run",
                        file=sys.stderr,
                    )
        rel = pos[:n] - center
        dist = np.hypot(rel[:, 0], rel[:, 1])
        limit = cfg.ring_radius - radius
        # Only balls that crossed the ring need a Python hit. The rest stay vectorized.
        hit_ids = np.flatnonzero(dist + radius > cfg.ring_radius)
        for index in hit_ids:
            index = int(index)
            if dist[index] <= 1e-8:
                continue
            normal = rel[index] / dist[index]
            vn = float(vel[index, 0] * normal[0] + vel[index, 1] * normal[1])
            pos[index] = center + normal * limit
            if vn > 0.0:
                impact = vn
                # Reflect, then keep the bounce from going quiet.
                vel[index] -= (1.0 + cfg.restitution) * vn * normal
                reflected = cfg.restitution * vn
                if reflected < cfg.min_bounce_speed:
                    current_n = float(
                        vel[index, 0] * normal[0] + vel[index, 1] * normal[1]
                    )
                    vel[index] -= current_n * normal
                    vel[index] -= cfg.min_bounce_speed * normal
                sp = float(np.hypot(vel[index, 0], vel[index, 1]))
                if sp > cfg.max_speed:
                    vel[index] *= cfg.max_speed / sp
                if first_bounce is None:
                    first_bounce = now
                hit_times.append(now)
                can_spawn = (
                    now + 1e-12 >= cooldown_until[index]
                    and count < cap
                    and count < ceil_target(now, cfg)
                )
                spawned = False
                if can_spawn:
                    child = count
                    jitter = math.radians(float(rng.uniform(-cfg.spawn_jitter_deg, cfg.spawn_jitter_deg)))
                    c = math.cos(jitter)
                    s = math.sin(jitter)
                    vx, vy = float(vel[index, 0]), float(vel[index, 1])
                    vel[child, 0] = c * vx - s * vy
                    vel[child, 1] = s * vx + c * vy
                    pos[child] = center + normal * (cfg.ring_radius - radius - 2.0)
                    spawn_index[child] = child
                    cooldown_until[child] = now + cfg.spawn_cooldown
                    cooldown_until[index] = now + cfg.spawn_cooldown
                    count += 1
                    spawned = True
                    mark_milestones(now)
                    if cap_time is None and count >= cap:
                        cap_time = now
                events.append(
                    now,
                    index,
                    impact,
                    float(pos[index, 0]),
                    float(pos[index, 1]),
                    count,
                    spawned,
                )
        # Numerical guard: every live ball is inside after the resolution.
        rel2 = pos[:count] - center
        dist2 = np.hypot(rel2[:, 0], rel2[:, 1])
        outside = dist2 + radius > cfg.ring_radius + 1e-4
        outside_violations += int(np.count_nonzero(outside))
        if np.any(outside):
            bad = np.flatnonzero(outside)
            for index in bad:
                d = float(dist2[index])
                if d <= 1e-8:
                    continue
                pos[index] = center + rel2[index] / d * (cfg.ring_radius - radius)

    snapshot(0)
    mark_milestones(0.0)
    target_substeps = (n_growth - 1) * 4
    for frame_index in range(1, n_growth):
        for _ in range(4):
            t += DT
            substeps_done += 1
            substep(t)
            if count_at_014 is None and t >= check_t:
                count_at_014 = count
        snapshot(frame_index)
    while t < cfg.growth_seconds - 1e-12:
        t = min(cfg.growth_seconds, t + DT)
        substep(t)
        if count_at_014 is None and t >= check_t:
            count_at_014 = count
    if count_at_014 is None:
        count_at_014 = count
    if cap_time is None and count >= cap:
        cap_time = t

    hold_radius = ball_radius(count, cfg)
    milestones = (
        np.array(milestone_rows, dtype=np.float64)
        if milestone_rows
        else np.zeros((0, 2), dtype=np.float64)
    )
    hits = np.array(hit_times, dtype=np.float64)
    max_gap = float(np.diff(hits).max()) if hits.size >= 2 else None
    return SimResult(
        positions=positions,
        counts=counts,
        radii=radii,
        spawn_index=spawn_index,
        hold_positions=pos.astype(np.float32),
        hold_count=count,
        hold_radius=hold_radius,
        events=events.freeze(),
        milestones=milestones,
        initial_pos=pos0.astype(np.float64),
        initial_vel=vel0.astype(np.float64),
        first_bounce=-1.0 if first_bounce is None else float(first_bounce),
        count_at_014=int(count_at_014),
        cap_time=None if cap_time is None else float(cap_time),
        max_gap=max_gap,
        hit_times=hits,
        gravity_used=float(gravity),
        speed_used=float(speed),
        seed_offset_used=int(seed_offset),
        outside_violations=outside_violations,
        collisions_disabled=collisions_disabled,
    )


def validate_result(cfg: Config, result: SimResult) -> list[str]:
    first = None if result.first_bounce < 0 else result.first_bounce
    return pacing_failures(
        first_bounce=first,
        count_at_014=result.count_at_014,
        cap_time=result.cap_time,
        hit_times=result.hit_times,
        growth_seconds=cfg.growth_seconds,
        cap=cfg.cap,
    )


def auto_tune(cfg: Config) -> SimResult:
    """Retry up to tune_max_attempts. Exit with the failing checks if none pass."""
    last: list[str] = []
    last_result: SimResult | None = None
    for index in range(cfg.tune_max_attempts):
        speed, gravity, offset = _attempt_params(cfg, index)
        result = simulate_once(cfg, speed=speed, gravity=gravity, seed_offset=offset)
        fails = validate_result(cfg, result)
        if not fails:
            print(
                f"sim: attempt {index + 1}/{cfg.tune_max_attempts} "
                f"speed={speed:.1f} gravity={gravity:.1f} seed_offset={offset} "
                f"first_bounce={result.first_bounce:.3f}s "
                f"count_at_0.14Tg={result.count_at_014} "
                f"cap_time={result.cap_time:.3f}s final={result.final_count}"
            )
            return result
        last = fails
        last_result = result
        print(
            f"sim: attempt {index + 1} failed ({'; '.join(fails)})",
            file=sys.stderr,
        )
    detail = "\n".join(f"  - {item}" for item in last)
    extra = ""
    if last_result is not None:
        extra = (
            f"\nLast run: first_bounce={last_result.first_bounce:.4f}s "
            f"count_at_0.14Tg={last_result.count_at_014} "
            f"cap_time={last_result.cap_time} max_gap={last_result.max_gap} "
            f"final={last_result.final_count}"
        )
    raise SystemExit(
        f"Pacing validation failed after {cfg.tune_max_attempts} attempts:\n{detail}{extra}"
    )


def cache_key(cfg: Config) -> str:
    payload = json.dumps(sim_fingerprint_payload(cfg), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def _result_to_npz(path: Path, result: SimResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "hold_count": result.hold_count,
        "hold_radius": result.hold_radius,
        "first_bounce": result.first_bounce,
        "count_at_014": result.count_at_014,
        "cap_time": -1.0 if result.cap_time is None else result.cap_time,
        "max_gap": -1.0 if result.max_gap is None else result.max_gap,
        "gravity_used": result.gravity_used,
        "speed_used": result.speed_used,
        "seed_offset_used": result.seed_offset_used,
        "outside_violations": result.outside_violations,
        "collisions_disabled": result.collisions_disabled,
    }
    np.savez_compressed(
        path,
        positions=result.positions,
        counts=result.counts,
        radii=result.radii,
        spawn_index=result.spawn_index,
        hold_positions=result.hold_positions,
        events=result.events,
        milestones=result.milestones,
        initial_pos=result.initial_pos,
        initial_vel=result.initial_vel,
        hit_times=result.hit_times,
        meta=np.array(json.dumps(meta)),
    )


def read_sim_npz(path: Path) -> SimResult:
    return _result_from_npz(path)


def _result_from_npz(path: Path) -> SimResult:
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
        cap_time = meta["cap_time"]
        max_gap = meta["max_gap"]
        return SimResult(
            positions=data["positions"],
            counts=data["counts"],
            radii=data["radii"],
            spawn_index=data["spawn_index"],
            hold_positions=data["hold_positions"],
            hold_count=int(meta["hold_count"]),
            hold_radius=float(meta["hold_radius"]),
            events=data["events"],
            milestones=data["milestones"],
            initial_pos=data["initial_pos"],
            initial_vel=data["initial_vel"],
            first_bounce=float(meta["first_bounce"]),
            count_at_014=int(meta["count_at_014"]),
            cap_time=None if cap_time < 0 else float(cap_time),
            max_gap=None if max_gap < 0 else float(max_gap),
            hit_times=data["hit_times"],
            gravity_used=float(meta["gravity_used"]),
            speed_used=float(meta["speed_used"]),
            seed_offset_used=int(meta["seed_offset_used"]),
            outside_violations=int(meta["outside_violations"]),
            collisions_disabled=bool(meta["collisions_disabled"]),
        )


def load_or_simulate(cfg: Config, *, use_cache: bool = True, cache_dir: Path | None = None) -> SimResult:
    directory = cache_dir or Path(".cache")
    path = directory / f"sim_{cache_key(cfg)}.npz"
    if use_cache and path.exists():
        print(f"sim: cache hit {path}")
        return _result_from_npz(path)
    started = time.perf_counter()
    result = auto_tune(cfg)
    _result_to_npz(path, result)
    print(f"sim: wrote {path} in {time.perf_counter() - started:.2f}s")
    return result
