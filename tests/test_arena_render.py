import numpy as np

from fc_sat.arena_config import load_arena_config
from fc_sat.arena_render import (
    active_kill_lines,
    build_timeline,
    camera_view,
    event_video_times,
    video_time,
)
from fc_sat.arena_sim import Elim, SimResult, simulate


def _result(cfg, elims, t_win, winner=1, duel_start=None) -> SimResult:
    steps = 4
    count = len(cfg.countries)
    return SimResult(
        seed=1,
        backend="test",
        elims=tuple(elims),
        meteors=(),
        near_misses=(),
        t_win=t_win,
        winner=winner,
        duel_start=duel_start,
        cameo_exit=None,
        trace={
            "xy": np.zeros((steps, count, 2)),
            "angle": np.zeros((steps, count)),
            "alive": np.ones((steps, count), dtype=bool),
            "cameo_xy": np.zeros((steps, 2)),
            "cameo_alive": np.zeros(steps, dtype=bool),
        },
    )


def test_remap_is_monotone_and_slows_the_final_eliminations():
    cfg = load_arena_config("configs/arena_default.yaml")
    elims = [Elim(1.6 + index * 0.4, index, "self") for index in range(28)]
    elims.append(Elim(27.0, 28, "ball"))
    elims.append(Elim(27.4, 29, "ball"))
    elims.append(Elim(30.5, 30, "sweeper"))
    result = _result(cfg, elims, 30.5, winner=31, duel_start=27.4)
    timeline = build_timeline(result, cfg)
    samples = [0.0, 5.0, 10.0, 20.0, 27.0, 30.5]
    mapped = [video_time(timeline, moment) for moment in samples]
    assert mapped == sorted(mapped)
    assert video_time(timeline, 27.4) - video_time(timeline, 27.0) > (27.4 - 27.0) / cfg.slowmo_rate - 0.05
    assert timeline.winner_video == timeline.replay_video < timeline.celebration_video < timeline.duration
    assert abs((timeline.duration - timeline.celebration_video) - cfg.celebration) < 1e-9


def test_audio_events_follow_the_same_remap():
    cfg = load_arena_config("configs/arena_default.yaml")
    elims = [Elim(1.6 + index * 0.4, index, "self") for index in range(29)]
    elims.append(Elim(26.2, 29, "ball"))
    elims.append(Elim(29.4, 30, "ball"))
    result = _result(cfg, elims, 29.4, winner=31, duel_start=26.2)
    timeline = build_timeline(result, cfg)
    when = 29.0
    heard = event_video_times(timeline, when)
    assert heard[0] == video_time(timeline, when)
    assert len(heard) == 2
    assert heard[1] > timeline.replay_video


def test_camera_holds_the_zone_at_400_without_shake():
    cfg = load_arena_config("configs/arena_default.yaml")
    for moment in (2.0, 10.0):
        view = camera_view(cfg, moment, shake=0.0)
        assert abs(view.screen_radius - cfg.screen_radius) <= 1.0
        left, top, right, bottom = view.circle_bounds()
        assert cfg.zone_x[0] <= left and right <= cfg.zone_x[1]
        assert cfg.zone_y[0] <= top and bottom <= cfg.zone_y[1]
        assert view.shake_x == 0.0 and view.shake_y == 0.0


def test_kill_feed_never_exceeds_three_lines():
    cfg = load_arena_config("configs/arena_default.yaml")
    elims = [Elim(10.0 + index * 0.05, index, "ball") for index in range(8)]
    result = _result(cfg, elims, 20.0)
    timeline = build_timeline(result, cfg)
    shown = active_kill_lines(result.elims, timeline, video_time(timeline, 10.4), cfg.kill_fade, cfg.kill_lines)
    assert len(shown) <= 3
    assert len(shown) == 3


def test_frame_zero_text_stays_in_the_safe_zone():
    cfg = load_arena_config("configs/arena_default.yaml")
    result = simulate(cfg, 1, horizon=2.0, record_trace=True)
    from fc_sat.arena_render import ArenaRenderer

    renderer = ArenaRenderer(cfg, result)
    frame = renderer.render(0)
    assert frame.shape == (cfg.height, cfg.width, 3)
    assert renderer.boxes
    for left, top, right, bottom in renderer.boxes:
        assert cfg.safe_x[0] <= left and right <= cfg.safe_x[1]
        assert cfg.safe_y[0] <= top and bottom <= cfg.safe_y[1]
    hook_band = frame[int(cfg.hook_y) : int(cfg.hook_y) + 140, 130:950]
    assert int(hook_band.max()) > 180
