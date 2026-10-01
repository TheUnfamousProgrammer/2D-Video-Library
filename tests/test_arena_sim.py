"""Analytic checks for the v2 top-down solver."""

import math

import numpy as np

from fc_sat.arena_config import load_arena_config
from fc_sat.arena_sim import (
    IMPULSE_SOURCES,
    attribute_cause,
    boss_allowed,
    combo_name,
    damp_factor,
    dash_travel,
    evaluate_gates,
    format_search_failure,
    resolve_collision,
    rounded_rect_sd,
    simulate,
    state_hash,
)


def test_head_on_equal_mass_keeps_momentum_and_restitution():
    va = np.array([400.0, 0.0])
    vb = np.array([-100.0, 0.0])
    normal = np.array([1.0, 0.0])
    na, nb, closing = resolve_collision(va, vb, 1.0, 1.0, 0.92, normal)
    assert closing == 500.0
    assert abs((na[0] + nb[0]) - (va[0] + vb[0])) / 500.0 < 0.02
    separation = float(nb[0] - na[0])
    assert abs(separation - 0.92 * closing) / closing < 0.02


def test_free_flight_matches_the_damping_exponential():
    speed = 800.0
    elapsed = 0.4
    expected = speed * damp_factor(2.2, elapsed)
    stepped = speed
    dt = 1.0 / 240.0
    for _ in range(int(round(elapsed / dt))):
        stepped *= math.exp(-2.2 * dt)
    assert abs(stepped - expected) / expected < 0.01


def test_no_gravity_center_of_mass_only_damps():
    rng = np.random.default_rng(4)
    pos = rng.normal(size=(6, 2)) * 40.0
    vel = rng.normal(size=(6, 2)) * 80.0
    com_v = vel.mean(axis=0)
    # Equal-mass contacts conserve momentum. Damping is the only speed change.
    for _ in range(80):
        vel *= math.exp(-2.2 / 240.0)
        pos += vel / 240.0
    expected = com_v * damp_factor(2.2, 80 / 240.0)
    assert float(np.linalg.norm(vel.mean(axis=0) - expected)) < 5.0


def test_speed_clamp_cannot_tunnel_a_diameter():
    cfg = load_arena_config("configs/arena_default.yaml")
    assert cfg.speed_clamp / 240.0 <= 7.5
    assert cfg.ball_radius * 2 == 88
    assert cfg.speed_clamp * cfg.dt <= cfg.max_step_px + 1e-9


def test_same_seed_has_the_same_hash():
    cfg = load_arena_config("configs/arena_default.yaml")
    first = state_hash(simulate(cfg, 3))
    second = state_hash(simulate(cfg, 3))
    assert first == second


def test_rounded_rect_corners_and_the_ring_out_sign():
    # Inside the flat side, on the corner circle, and clearly outside.
    assert rounded_rect_sd(540, 940, 540, 940, 410, 460, 0.35 * 410) < 0
    cr = 0.35 * 410
    corner_x = 540 + (410 - cr)
    corner_y = 940 + (460 - cr)
    assert abs(rounded_rect_sd(corner_x + cr, corner_y, 540, 940, 410, 460, cr)) < 1e-6
    assert rounded_rect_sd(540 + 410 + 5, 940, 540, 940, 410, 460, cr) > 0


def test_attribution_windows():
    assert attribute_cause(2.4, "ball", False, hit_window=2.5) == "ball"
    assert attribute_cause(2.6, "ball", True, hit_window=2.5) == "storm"
    assert attribute_cause(2.6, "ball", False, hit_window=2.5) == "self"
    assert attribute_cause(1.0, "boss", False, hit_window=2.5) == "boss"
    assert attribute_cause(3.0, "", False, hit_window=2.5) == "self"


def test_combo_names_and_boss_never_counts_as_alive():
    assert combo_name(1) == ""
    assert combo_name(2) == "double"
    assert combo_name(3) == "triple"
    assert combo_name(5) == "rampage"
    assert boss_allowed(12, 12)
    assert not boss_allowed(11, 12)


def test_dash_travel_shrinks_with_damping():
    assert dash_travel(1350, 2.2, 0.35) < 1350 * 0.35
    assert dash_travel(1350, 0.0, 0.35) == 1350 * 0.35


def test_logged_impulses_name_a_source():
    cfg = load_arena_config("configs/arena_default.yaml")
    result = simulate(cfg, 2)
    assert result.impulses
    assert {item.source for item in result.impulses} <= set(IMPULSE_SOURCES)
    assert any(item.source == "clash" for item in result.impulses)
    assert result.metrics["dash_windup_min"] >= 15.0 / 60.0 - 1e-6


def test_search_failure_names_the_knobs():
    rows = [{"passed": False, "gates": {"kinetic": False, "alive@4": True}}]
    text = format_search_failure(rows, 0.30)
    assert "under 30%" in text
    assert "aggression_scale" in text
    assert "Gates were not relaxed." in text


def test_gate_table_rejects_a_tie_and_a_storm_finish():
    cfg = load_arena_config("configs/arena_default.yaml")
    result = simulate(cfg, 1)
    gates = evaluate_gates(result, cfg)
    assert "no_tie" in gates
    assert "final_hit" in gates
    assert "kinetic" in gates
