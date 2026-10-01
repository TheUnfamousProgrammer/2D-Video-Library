import math

from fc_sat.arena_config import load_arena_config
from fc_sat.arena_sim import (
    Elim,
    SimResult,
    active_backend,
    alive_at,
    drama_components,
    evaluate_gates,
    format_search_failure,
    is_out,
    simulate,
    state_hash,
    sweeper_omega,
)


def _result(**overrides) -> SimResult:
    base = dict(
        seed=1,
        backend="test",
        elims=(),
        meteors=(),
        near_misses=(),
        t_win=None,
        winner=None,
        duel_start=None,
        cameo_exit=None,
        trace=None,
    )
    base.update(overrides)
    return SimResult(**base)


def _good_elims() -> tuple[Elim, ...]:
    times = (
        [1.8, 2.4, 3.4]
        + [5.0, 6.0, 7.0, 8.0, 8.5, 9.0, 9.5, 10.0]
        + [11.0, 12.0, 13.0, 14.0, 14.5, 15.0, 15.5, 16.0]
        + [17.0, 18.0, 19.0, 20.0, 21.0, 22.0]
        + [22.4, 22.8, 23.4, 24.0]
        + [28.0]
    )
    return tuple(Elim(time, index, "self") for index, time in enumerate(times))


def test_backend_hash_is_stable():
    cfg = load_arena_config("configs/arena_default.yaml")
    first = simulate(cfg, 1, horizon=4.0, record_trace=True)
    second = simulate(cfg, 1, horizon=4.0, record_trace=True)
    assert first.backend == active_backend()
    assert first.backend in {"pymunk", "numpy"}
    assert state_hash(first) == state_hash(second)


def test_no_elimination_in_the_opening():
    cfg = load_arena_config("configs/arena_default.yaml")
    result = simulate(cfg, 1, horizon=2.0)
    assert not result.elims or result.elims[0].time >= cfg.no_elim_before


def test_gate_evaluator_accepts_a_paced_log():
    cfg = load_arena_config("configs/arena_default.yaml")
    elims = _good_elims()
    result = _result(elims=elims, t_win=28.0, winner=31, duel_start=24.0)
    gates = evaluate_gates(result, cfg)
    assert all(gates.values()), gates
    assert alive_at(elims, 32, 4.0) == 29
    assert alive_at(elims, 32, 22.0) == 7


def test_gate_evaluator_rejects_an_early_exit_and_a_late_winner():
    cfg = load_arena_config("configs/arena_default.yaml")
    early = _result(elims=(Elim(0.4, 0, "self"),), t_win=28.0, winner=1, duel_start=24.0)
    assert evaluate_gates(early, cfg)["no_early_elim"] is False
    late = _result(elims=_good_elims(), t_win=34.0, winner=31, duel_start=30.0)
    gates = evaluate_gates(late, cfg)
    assert gates["winner_window"] is False
    assert gates["final_duel"] is False


def test_is_out_uses_half_a_radius():
    assert is_out(400.0, 380.0, 34.0, 0.5) is True
    assert is_out(390.0, 380.0, 34.0, 0.5) is False


def test_drama_score_and_repeat_penalty():
    cfg = load_arena_config("configs/arena_default.yaml")
    elims = (
        Elim(20.0, 0, "ball"),
        Elim(20.2, 1, "meteor"),
        Elim(26.0, 2, "self"),
        Elim(27.0, 3, "sweeper"),
        Elim(28.0, 4, "meteor"),
    )
    result = _result(
        elims=elims,
        near_misses=((0, 20.0), (1, 22.0), (2, 27.5)),
        t_win=29.0,
        winner=0,
        duel_start=24.0,
    )
    parts = drama_components(result, cfg, previous_winner="BR")
    assert parts["near_misses"] == 3
    assert parts["near_score"] == 3.0
    assert parts["duel"] == 5.0
    assert parts["duel_score"] == 5.0
    assert parts["late_elims"] == 5
    assert parts["late_score"] == 0.5
    assert parts["doubles"] == 1
    assert parts["double_score"] == 0.5
    assert parts["meteor_elims"] == 2
    assert parts["meteor_score"] == 0.3
    assert parts["repeat_penalty"] == 2.0
    assert math.isclose(parts["score"], 3 + 5 + 0.5 + 0.5 + 0.3 - 2)
    clear = drama_components(result, cfg, previous_winner="JP")
    assert clear["repeat_penalty"] == 0.0


def test_placements_are_elimination_order_with_the_winner_last():
    result = _result(elims=(Elim(2.0, 4, "self"), Elim(3.0, 1, "ball")), t_win=5.0, winner=7)
    assert result.placements == (4, 1, 7)


def test_cameo_is_not_a_country_and_cannot_win():
    cfg = load_arena_config("configs/arena_default.yaml")
    result = simulate(cfg, 3, horizon=20.0)
    assert result.cameo_exit is not None
    assert result.cameo_exit <= cfg.cameo_exit + 1e-6
    for item in result.elims:
        assert 0 <= item.index < 32
    if result.winner is not None:
        assert 0 <= result.winner < 32
        assert result.winner not in {item.index for item in result.elims}


def test_search_failure_names_the_gates_and_refuses_to_relax_them():
    rows = [{"passed": False, "gates": {"no_early_elim": False, "winner_window": True}}]
    text = format_search_failure(rows, 0.15)
    assert "under 15%" in text
    assert "no_early_elim" in text
    assert "Gates were not relaxed." in text


def test_sweeper_holds_then_ramps():
    cfg = load_arena_config("configs/arena_default.yaml")
    assert sweeper_omega(cfg, cfg.sweeper_start) == cfg.sweeper_omega_start
    assert sweeper_omega(cfg, cfg.sweeper_omega_end_time - 6.0) == cfg.sweeper_omega_start
    assert sweeper_omega(cfg, cfg.sweeper_omega_end_time) == cfg.sweeper_omega_end
