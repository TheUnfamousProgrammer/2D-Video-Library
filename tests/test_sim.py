import hashlib

import numpy as np
import pytest

from fc_sat.config import load_config
from fc_sat.sim import (
    ball_radius,
    ceil_target,
    n_target,
    pacing_failures,
    simulate_once,
)


@pytest.fixture(scope="module")
def cfg():
    return load_config("configs/default.yaml")


@pytest.fixture(scope="module")
def sim(cfg):
    return simulate_once(cfg, speed=1500, gravity=900, seed_offset=0)


def _digest(result) -> str:
    blob = hashlib.sha256()
    blob.update(np.ascontiguousarray(result.positions).tobytes())
    blob.update(np.ascontiguousarray(result.counts).tobytes())
    blob.update(np.ascontiguousarray(result.events).tobytes())
    blob.update(np.ascontiguousarray(result.hit_times).tobytes())
    return blob.hexdigest()


def test_determinism(cfg, sim):
    again = simulate_once(cfg, speed=1500, gravity=900, seed_offset=0)
    assert _digest(sim) == _digest(again)


def test_balls_stay_inside(cfg, sim):
    assert sim.outside_violations == 0
    center = np.array([cfg.ring_cx, cfg.ring_cy], dtype=np.float32)
    for index, count in enumerate(sim.counts):
        count = int(count)
        if count <= 0:
            continue
        dist = np.linalg.norm(sim.positions[index, :count] - center, axis=1)
        assert np.all(dist + sim.radii[index] <= cfg.ring_radius + 1e-3)


def test_throttle_never_exceeded(cfg, sim):
    for index, count in enumerate(sim.counts):
        t = index / cfg.fps
        assert int(count) <= ceil_target(t, cfg)
    assert n_target(0.985 * cfg.growth_seconds, cfg) == cfg.cap
    assert n_target(cfg.growth_seconds, cfg) == cfg.cap
    assert n_target(0.5 * cfg.growth_seconds, cfg) < cfg.cap


def test_cooldown_and_sorted_events(cfg, sim):
    events = sim.events
    assert events.size > 0
    assert np.all(np.diff(events["time"]) >= -1e-9)
    for ball_id in np.unique(events["ball_id"]):
        times = events["time"][(events["ball_id"] == ball_id) & events["spawned"]]
        if times.size >= 2:
            assert np.all(np.diff(np.sort(times)) >= cfg.spawn_cooldown - 1e-6)
    spawned = events[events["spawned"]]
    for row in spawned:
        child = int(row["count_at_time"]) - 1
        child_times = events["time"][(events["ball_id"] == child) & events["spawned"]]
        if child_times.size:
            assert float(child_times.min()) >= float(row["time"]) + cfg.spawn_cooldown - 1e-6


def test_radius_floor_is_configurable(cfg):
    assert ball_radius(1, cfg) == 28
    assert ball_radius(cfg.cap, cfg) == pytest.approx(cfg.radius_min)


def test_pacing_accepts_and_rejects():
    ok = pacing_failures(
        first_bounce=0.45,
        count_at_014=8,
        cap_time=18.8,
        hit_times=np.array([0.45, 0.8, 1.1]),
        growth_seconds=19.0,
        cap=1000,
    )
    assert ok == []
    # The wait before the first hit is not a gap.
    early = pacing_failures(
        first_bounce=0.45,
        count_at_014=8,
        cap_time=18.8,
        hit_times=np.array([0.45]),
        growth_seconds=19.0,
        cap=1000,
    )
    assert early == []
    bad = pacing_failures(
        first_bounce=0.1,
        count_at_014=2,
        cap_time=None,
        hit_times=np.array([0.40, 1.20]),
        growth_seconds=19.0,
        cap=1000,
    )
    text = " ".join(bad)
    assert "first bounce" in text
    assert "count at 0.14" in text
    assert "cap" in text
    assert "gap" in text
