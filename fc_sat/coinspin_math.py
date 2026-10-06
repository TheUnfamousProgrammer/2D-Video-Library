"""Where every body sits on each frame of the coinspin short, and how far the gold coin has turned.

Screen y grows downward. Angles run clockwise from straight up: d(a) = (sin a, -cos a).
``theta`` is the gold coin's total turn since frame 0, clockwise. 2 pi is one spin, and the
face is upright exactly when theta is a multiple of 2 pi. The claims count spins from
``gold_pose``, the same function the renderer draws.

Acts:
  1. frames 0-875: the gold coin rolls once around a same-size grey coin. 2 spins.
  2. frames 876-1343: the grey edge is laid flat and the coin rolls along it (1 spin),
     then the coin is bolted to a rod and carried around without rolling (1 spin).
  3. frames 1344-1799: the 1982 SAT case, grey coin 3x wider. 4 spins.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from fc_sat.easing import ease_in_out_cubic, ease_out_cubic

TAU = 2.0 * math.pi
CX = 540.0
CY = 915.0
R1 = 120.0
R_BIG = 210.0
R_SMALL = 70.0
ROAD_Y = CY + R1
ROAD_LENGTH = TAU * R1
ROAD_X0 = CX - ROAD_LENGTH / 2.0
ROAD_X1 = CX + ROAD_LENGTH / 2.0
N_FRAMES = 1824

# Act 1. The first half-lap is a smoothstep from frame 12 to 288; the face's quarter turns
# land on 102, 150, 198 and the upright on 288. The second half runs at 0.5 deg a frame
# after a 12-frame ramp, so its quarter turns land on 480, 570, 660.
LAP_START = 12
HALF = 288
RESUME = 384
RAMP_END = 396
SECOND_RATE = math.radians(0.5)
FREEZE = 744
DROP = 768
# Act 2.
UNROLL = (880, 932)
GLIDE_OUT = (880, 936)
ROAD = (936, 1032)
RETURN = (1056, 1096)
RECOLOR = (1096, 1104)
ROD_GROW = (1104, 1120)
BOLTS = (1116, 1124)
CARRY = (1128, 1224)
# Act 3.
RESIZE = (1344, 1368)
SAT_RAMP = (1384, 1400)
SAT_ZERO = 1392
SAT_RATE = math.radians(1.5)
SAT_END = 1632
SNAP = (1800, 1806)

PUNCHES = (DROP, SAT_END)


def direction(angle: float) -> np.ndarray:
    return np.array([math.sin(angle), -math.cos(angle)], dtype=np.float64)


def smoothstep(u: float) -> float:
    u = min(1.0, max(0.0, u))
    return u * u * (3.0 - 2.0 * u)


def _bezier(p0, p1, p2, p3, u: float) -> np.ndarray:
    a = np.asarray(p0, dtype=np.float64)
    b = np.asarray(p1, dtype=np.float64)
    c = np.asarray(p2, dtype=np.float64)
    d = np.asarray(p3, dtype=np.float64)
    v = 1.0 - u
    return v * v * v * a + 3 * v * v * u * b + 3 * v * u * u * c + u * u * u * d


def _ramped(frame: float, start: int, ramp_end: int, rate: float) -> float:
    """Angle swept from ``start``: speed ramps linearly to ``rate`` by ``ramp_end``, then holds."""
    span = float(ramp_end - start)
    t = float(frame) - start
    if t <= 0.0:
        return 0.0
    if t <= span:
        return rate * t * t / (2.0 * span)
    return rate * span / 2.0 + rate * (t - span)


def lap_phi(frame: float) -> float:
    """Orbit angle of the gold coin's center in act 1."""
    frame = float(frame)
    if frame < LAP_START:
        return 0.0
    if frame <= HALF:
        return math.pi * smoothstep((frame - LAP_START) / float(HALF - LAP_START))
    if frame < RESUME:
        return math.pi
    if frame < FREEZE:
        return math.pi + _ramped(frame, RESUME, RAMP_END, SECOND_RATE)
    if frame < DROP:
        return math.pi + _ramped(FREEZE, RESUME, RAMP_END, SECOND_RATE)
    return TAU


def sat_phi(frame: float) -> float:
    """Orbit angle in the SAT act. Exactly 1.5 deg a frame from frame 1392 once the ramp ends."""
    frame = float(frame)
    if frame < SAT_RAMP[0]:
        return 0.0
    if frame >= SAT_END:
        return TAU
    return _ramped(frame, SAT_RAMP[0], SAT_RAMP[1], SAT_RATE)


def rolling_pose(ratio: float, phi: float, radius: float = 1.0) -> dict:
    """A coin of ``radius`` rolled without slipping around a fixed coin ``ratio`` times wider."""
    center = np.array([CX, CY], dtype=np.float64)
    fixed = ratio * radius
    rolling = center + (fixed + radius) * direction(phi)
    return {
        "fixed_center": center,
        "fixed_radius": fixed,
        "center": rolling,
        "radius": radius,
        "theta": (ratio + 1.0) * phi,
    }


def inside_pose(ratio: float, phi: float, radius: float = 1.0) -> dict:
    """A coin rolled around the inside of a ring ``ratio`` times wider. It turns the other way."""
    center = np.array([CX, CY], dtype=np.float64)
    fixed = ratio * radius
    return {
        "fixed_center": center,
        "fixed_radius": fixed,
        "center": center + (fixed - radius) * direction(phi),
        "radius": radius,
        "theta": -(ratio - 1.0) * phi,
    }


@dataclass(frozen=True)
class Pose:
    """The gold coin on one frame. ``rolling_on`` is "rim", "road", "rod", or "" when lifted or still."""

    center: tuple[float, float]
    radius: float
    theta: float
    grey_radius: float
    rolling_on: str


def grey_radius(frame: int) -> float:
    if frame >= SNAP[0] or frame < RESIZE[0]:
        return R1
    if frame >= RESIZE[1]:
        return R_BIG
    u = ease_in_out_cubic((frame - RESIZE[0]) / float(RESIZE[1] - RESIZE[0]))
    return R1 + (R_BIG - R1) * u


def _top(grey: float, gold: float) -> tuple[float, float]:
    return CX, CY - grey - gold


def gold_pose(frame: float) -> Pose:
    frame = float(frame)
    if frame >= SNAP[0]:
        return Pose(_top(R1, R1), R1, 0.0, R1, "")
    if frame < DROP:
        phi = lap_phi(frame)
        center = np.array([CX, CY]) + 2.0 * R1 * direction(phi)
        moving = LAP_START <= frame <= HALF or RESUME <= frame < FREEZE
        return Pose((float(center[0]), float(center[1])), R1, 2.0 * phi, R1, "rim" if moving else "")
    turned = 2.0 * TAU
    if frame < GLIDE_OUT[0]:
        return Pose(_top(R1, R1), R1, turned, R1, "")
    if frame < GLIDE_OUT[1]:
        u = ease_in_out_cubic((frame - GLIDE_OUT[0]) / float(GLIDE_OUT[1] - GLIDE_OUT[0]))
        point = _bezier(_top(R1, R1), (CX - 200.0, CY - 330.0), (ROAD_X0, CY - 200.0), (ROAD_X0, CY), u)
        return Pose((float(point[0]), float(point[1])), R1, turned, R1, "")
    if frame < ROAD[1]:
        x = ROAD_X0 + ROAD_LENGTH * (frame - ROAD[0]) / float(ROAD[1] - ROAD[0])
        return Pose((x, CY), R1, turned + (x - ROAD_X0) / R1, R1, "road")
    turned += TAU
    if frame < RETURN[0]:
        return Pose((ROAD_X1, CY), R1, turned, R1, "")
    if frame < RETURN[1]:
        u = ease_in_out_cubic((frame - RETURN[0]) / float(RETURN[1] - RETURN[0]))
        point = _bezier((ROAD_X1, CY), (ROAD_X1, CY - 200.0), (CX + 200.0, CY - 330.0), _top(R1, R1), u)
        return Pose((float(point[0]), float(point[1])), R1, turned, R1, "")
    if frame < CARRY[0]:
        return Pose(_top(R1, R1), R1, turned, R1, "")
    if frame < CARRY[1]:
        beta = rod_angle(frame)
        center = np.array([CX, CY]) + 2.0 * R1 * direction(beta)
        return Pose((float(center[0]), float(center[1])), R1, turned + beta, R1, "rod")
    turned += TAU
    if frame < RESIZE[0]:
        return Pose(_top(R1, R1), R1, turned, R1, "")
    if frame < RESIZE[1]:
        u = ease_in_out_cubic((frame - RESIZE[0]) / float(RESIZE[1] - RESIZE[0]))
        grey = R1 + (R_BIG - R1) * u
        gold = R1 + (R_SMALL - R1) * u
        return Pose(_top(grey, gold), gold, turned, grey, "")
    if frame < SAT_RAMP[0]:
        return Pose(_top(R_BIG, R_SMALL), R_SMALL, turned, R_BIG, "")
    phi = sat_phi(frame)
    center = np.array([CX, CY]) + (R_BIG + R_SMALL) * direction(phi)
    moving = frame < SAT_END
    theta = turned + (R_BIG + R_SMALL) / R_SMALL * phi
    return Pose((float(center[0]), float(center[1])), R_SMALL, theta, R_BIG, "rim" if moving else "")


def rod_angle(frame: float) -> float:
    if frame < CARRY[0]:
        return 0.0
    if frame >= CARRY[1]:
        return TAU
    return TAU * (float(frame) - CARRY[0]) / float(CARRY[1] - CARRY[0])


def unroll_progress(frame: int) -> float:
    """0 is the cyan ring on the grey coin, 1 is the flat road. It rewinds on the way back."""
    if frame < UNROLL[0] or frame >= RETURN[1]:
        return 0.0
    if frame < UNROLL[1]:
        return ease_in_out_cubic((frame - UNROLL[0]) / float(UNROLL[1] - UNROLL[0]))
    if frame < RETURN[0]:
        return 1.0
    return 1.0 - ease_in_out_cubic((frame - RETURN[0]) / float(RETURN[1] - RETURN[0]))


def unroll_points(progress: float, samples: int = 90) -> tuple[np.ndarray, np.ndarray]:
    """The cyan ring opened at the grey coin's bottom point and laid along y = ROAD_Y.

    Each half rolls outward like tape coming off a spool: the part near the cut lies flat,
    and the rest stays on a circle of radius R1 that touches the road where the flat part ends.
    """
    flat = math.pi * min(1.0, max(0.0, progress))
    halves = []
    for side in (1.0, -1.0):
        alphas = np.linspace(0.0, math.pi, samples)
        xs = np.empty_like(alphas)
        ys = np.empty_like(alphas)
        laid = alphas <= flat
        xs[laid] = CX + side * R1 * alphas[laid]
        ys[laid] = ROAD_Y
        rest = ~laid
        contact = CX + side * R1 * flat
        xs[rest] = contact + side * R1 * np.sin(alphas[rest] - flat)
        ys[rest] = CY + R1 * np.cos(alphas[rest] - flat)
        halves.append(np.stack([xs, ys], axis=1))
    return halves[0], halves[1]


def paint_arc(frame: int) -> float:
    """How much of the grey rim, from 12 o'clock clockwise, the rolling coin has touched."""
    if frame < DROP:
        return lap_phi(frame)
    if frame < UNROLL[0]:
        return TAU
    if RESIZE[0] <= frame < SNAP[0]:
        return sat_phi(frame)
    return 0.0


def stage_scale(frame: int) -> float:
    for punch in PUNCHES:
        if punch <= frame < punch + 12:
            return 1.0 + 0.06 * math.sin(math.pi * (frame - punch) / 12.0)
    if SNAP[0] <= frame < SNAP[1]:
        return 0.97 + 0.03 * ease_out_cubic((frame - SNAP[0]) / float(SNAP[1] - SNAP[0]))
    return 1.0


def segment_spins(start: int, end: int) -> float:
    """Spins between two frames, read from the drawn pose."""
    return (gold_pose(end).theta - gold_pose(start).theta) / TAU


def upright_frames() -> list[int]:
    return list(_upright_frames())


@lru_cache(maxsize=1)
def _upright_frames() -> tuple[int, ...]:
    """Frames where the face comes back upright mid-act: every tick the counter counts."""
    frames = []
    for frame in range(1, SNAP[0]):
        now = gold_pose(frame)
        before = gold_pose(frame - 1)
        if now.rolling_on == "" and before.rolling_on == "" and frame not in (DROP,):
            continue
        turns_now = now.theta / TAU
        turns_before = before.theta / TAU
        if math.floor(turns_now + 1e-9) > math.floor(turns_before + 1e-9):
            frames.append(frame)
    return tuple(frames)


def quarter_frames() -> list[int]:
    return list(_quarter_frames())


@lru_cache(maxsize=1)
def _quarter_frames() -> tuple[int, ...]:
    """Frames where the face passes sideways or upside down while it moves."""
    frames = []
    for frame in range(1, SNAP[0]):
        now = gold_pose(frame).theta / (TAU / 4.0)
        before = gold_pose(frame - 1).theta / (TAU / 4.0)
        mark = math.floor(now + 1e-9)
        if mark > math.floor(before + 1e-9) and mark % 4 != 0:
            # Sound the frame nearest the exact crossing.
            frames.append(frame if now - mark <= mark - before else frame - 1)
    return tuple(frames)


def _walk(pose_at, samples: int, inside: bool) -> float:
    """Walk one lap, check contact and no slipping at every sample, return the total turn."""
    total = 0.0
    previous = None
    for index in range(samples + 1):
        phi = TAU * index / samples
        pose = pose_at(phi)
        gap = float(np.hypot(*(pose["center"] - pose["fixed_center"])))
        want = pose["fixed_radius"] - pose["radius"] if inside else pose["fixed_radius"] + pose["radius"]
        if abs(gap - want) > 1e-9 * max(1.0, want):
            raise ValueError(f"the coin left the rim at phi {phi:.4f}")
        if previous is not None:
            prev_phi, prev_theta = previous
            fixed_arc = pose["fixed_radius"] * (phi - prev_phi)
            contact = phi if inside else phi + math.pi
            prev_contact = prev_phi if inside else prev_phi + math.pi
            body = (contact - pose["theta"]) - (prev_contact - prev_theta)
            if abs(abs(body) * pose["radius"] - fixed_arc) > 1e-9 * max(1.0, fixed_arc):
                raise ValueError(f"the coin slipped at phi {phi:.4f}")
            total += pose["theta"] - prev_theta
        previous = (phi, pose["theta"])
    return total


def count_spins(ratio: float, samples: int = 4096) -> int:
    turn = _walk(lambda phi: rolling_pose(float(ratio), phi), samples, inside=False)
    return int(math.floor(abs(turn) / TAU + 1e-9))


def count_spins_inside(ratio: float, samples: int = 4096) -> int:
    turn = _walk(lambda phi: inside_pose(float(ratio), phi), samples, inside=True)
    return int(math.floor(abs(turn) / TAU + 1e-9))


def film_lap_spins() -> int:
    """Act 1 as drawn: the coin touches the grey rim and rolls without slipping."""
    return _film_rim_spins(LAP_START, DROP, R1, R1, lap_phi)


def sat_lap_spins() -> int:
    """Act 3 as drawn."""
    return _film_rim_spins(SAT_RAMP[0], SAT_END, R_BIG, R_SMALL, sat_phi)


def _film_rim_spins(start: int, end: int, grey: float, gold: float, phi_of) -> int:
    steps = 8
    previous = gold_pose(start)
    prev_phi = phi_of(start)
    for index in range(1, (end - start) * steps + 1):
        frame = start + index / steps
        if frame >= end:
            frame = end - 1e-6 if end == DROP else end
        pose = gold_pose(frame)
        phi = phi_of(frame)
        gap = math.hypot(pose.center[0] - CX, pose.center[1] - CY)
        if abs(gap - (grey + gold)) > 1e-6:
            raise ValueError(f"frame {frame:.3f}: centers {gap:.4f} apart, radii {grey + gold:.4f}")
        rolled = (pose.theta - previous.theta) - (phi - prev_phi)
        if abs(rolled * gold - grey * (phi - prev_phi)) > 1e-6:
            raise ValueError(f"frame {frame:.3f}: the coin slipped on the rim")
        previous, prev_phi = pose, phi
    if end == DROP:
        # The drop frame snaps the last 3 degrees home. Count the drawn turn up to it.
        return int(round(segment_spins(start, DROP)))
    return int(round(segment_spins(start, end)))


def road_spins() -> int:
    """Act 2a as drawn: rolled along the grey edge laid flat. Distance over radius, checked per step."""
    start, end = ROAD
    for index in range(1, (end - start) * 4 + 1):
        a = gold_pose(start + (index - 1) / 4.0)
        b = gold_pose(start + index / 4.0)
        if abs(a.center[1] - (ROAD_Y - R1)) > 1e-9 or abs(b.center[1] - (ROAD_Y - R1)) > 1e-9:
            raise ValueError("the coin left the road")
        if abs((b.center[0] - a.center[0]) - R1 * (b.theta - a.theta)) > 1e-9:
            raise ValueError("the coin slipped on the road")
    travelled = gold_pose(end).center[0] - gold_pose(start).center[0]
    if abs(travelled - ROAD_LENGTH) > 1e-9:
        raise ValueError("the road roll did not cover the whole grey edge")
    return int(round(segment_spins(start, end)))


def carry_spins() -> int:
    """Act 2b as drawn: bolted to the rod, the coin turns exactly as far as the rod."""
    start, end = CARRY
    for frame in range(start, end + 1):
        pose = gold_pose(frame)
        expected = gold_pose(start).theta + rod_angle(frame)
        if abs(pose.theta - expected) > 1e-9:
            raise ValueError(f"frame {frame}: the bolted coin turned against its rod")
    return int(round(segment_spins(start, end)))
