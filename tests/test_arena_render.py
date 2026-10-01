"""Timeline, camera box, and layout for arena v2."""

from fc_sat.arena_config import load_arena_config
from fc_sat.arena_render import (
    PLATFORM_BOX,
    build_timeline,
    camera_view,
    hit_stops,
    layout_boxes,
    layout_clear,
    sample_timeline,
    video_time,
)
from fc_sat.arena_sim import Elim, SimResult


def _result(cfg) -> SimResult:
    times = [2.0, 4.0, 8.0, 12.0, 16.0, 20.0, 24.0, 27.0]
    elims = tuple(
        Elim(when, index, "ball", killer=0, closing=900.0) for index, when in enumerate(times)
    )
    return SimResult(
        seed=1,
        backend="test",
        elims=elims,
        t_win=28.0,
        winner=31,
        duel_start=24.0,
        cameo_exit=None,
    )


def test_remap_is_monotone_and_caps_hit_stops():
    cfg = load_arena_config("configs/arena_default.yaml")
    result = _result(cfg)
    timeline = build_timeline(result, cfg)
    assert timeline.hitstop <= cfg.hitstop_cap + 1e-6
    previous = -1.0
    for step in range(0, 40):
        moment = timeline.t_win * step / 39.0
        current = video_time(timeline, moment)
        assert current + 1e-6 >= previous
        previous = current
    sim_a, _phase = sample_timeline(timeline, 0.2)
    sim_b, phase = sample_timeline(timeline, timeline.duration - 0.05)
    assert phase == "celebration"
    assert sim_b + 1e-6 >= sim_a
    holds = hit_stops(result, cfg, ())
    assert sum(item[2] for item in holds) <= cfg.hitstop_cap + 1e-6


def test_platform_box_stays_inside_the_screen_slot():
    cfg = load_arena_config("configs/arena_default.yaml")
    for moment, hw, hh in ((2.0, 1020.0, 1150.0), (10.0, 900.0, 1010.0)):
        view = camera_view(cfg, moment, hw, hh)
        assert PLATFORM_BOX[0] - 1 <= view.left <= view.right <= PLATFORM_BOX[1] + 1
        assert PLATFORM_BOX[2] - 1 <= view.top <= view.bottom <= PLATFORM_BOX[3] + 1
        assert hw * view.zoom <= cfg.screen_hw + 1.0
        assert hh * view.zoom <= cfg.screen_hh + 1.0


def test_layout_boxes_do_not_overlap_the_platform():
    cfg = load_arena_config("configs/arena_default.yaml")
    for video_t in (0.5, 3.2, 4.0, 8.0):
        boxes = layout_boxes(cfg, video_t, alive=20, phase="main")
        assert layout_clear(boxes)


def test_rendered_glyphs_stay_clear_of_the_platform():
    from fc_sat.arena_render import ArenaRenderer
    from fc_sat.arena_sim import simulate

    cfg = load_arena_config("configs/arena_default.yaml")
    result = simulate(cfg, 5, record_trace=True)
    renderer = ArenaRenderer(cfg, result, preview=True, reel=True)
    for video_t in (0.4, 3.2, 6.0):
        renderer.render(renderer.frame_at(video_t))
        assert layout_clear(renderer.hud_boxes)
