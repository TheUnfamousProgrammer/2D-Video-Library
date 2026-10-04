"""Fast checks for the line fit, the frame map, and the refused full render."""

from __future__ import annotations

import numpy as np
import pytest

from fc_sat.linedraw.choreo import (
    build_plan,
    caption_lines,
    drop_caption,
    flyoff_progress,
    hook_progress,
    loop_progress,
    throw_endpoints,
    zoom_at,
)
from fc_sat.linedraw.optimize import INK, find_l_aha, l_max_at, likeness, optimize, replay
from fc_sat.linedraw.picture import center_crop, importance_map, largest_face, select_clahe, spectral_peak


def _dark_target(h=12, w=10):
    target = np.full((h, w), 0.15, dtype=np.float64)
    target[:, : w // 2] = 0.85
    return target


def test_gain_prefers_the_ink_that_closes_the_error():
    from fc_sat.linedraw.optimize import _line_gain

    canvas = np.full((6, 8), 0.5, dtype=np.float64)
    target = np.zeros((6, 8), dtype=np.float64)
    weight = np.ones((6, 8), dtype=np.float64)
    dark, light, rr, cc, coverage = _line_gain(canvas, target, weight, 1, 3, 6, 3, INK)
    assert coverage.size > 0
    assert dark > 0
    assert light < 0
    assert dark > light
    manual = 0.0
    for y, x, cover in zip(rr, cc, coverage):
        delta = -INK * cover
        residual = target[y, x] - canvas[y, x]
        manual += weight[y, x] * (2 * delta * residual - delta * delta)
    assert dark == pytest.approx(manual)


def test_importance_weight_changes_the_gain():
    from fc_sat.linedraw.optimize import _line_gain

    canvas = np.full((6, 8), 0.5, dtype=np.float64)
    target = np.zeros((6, 8), dtype=np.float64)
    plain = np.ones((6, 8), dtype=np.float64)
    heavy = plain.copy()
    heavy[3, :] = 4.0
    dark_plain, _, *_ = _line_gain(canvas, target, plain, 1, 3, 6, 3, INK)
    dark_heavy, _, *_ = _line_gain(canvas, target, heavy, 1, 3, 6, 3, INK)
    assert dark_heavy > dark_plain


def test_length_schedule_shrinks():
    assert l_max_at(0) == pytest.approx(300.0)
    assert l_max_at(36_000) == pytest.approx(40.0)
    assert l_max_at(18_000) == pytest.approx(170.0)
    assert l_max_at(0) > l_max_at(1000) > l_max_at(36_000)


def test_optimizer_is_deterministic_and_logs_likeness():
    target = _dark_target()
    weight = np.ones_like(target)
    first = optimize(target, weight, seed=11, max_kept=40, candidates=12, planned=40, store_rejects=4)
    second = optimize(target, weight, seed=11, max_kept=40, candidates=12, planned=40, store_rejects=4)
    assert np.array_equal(first.x0, second.x0)
    assert np.array_equal(first.ink, second.ink)
    assert np.array_equal(first.thrown, second.thrown)
    assert len(first.likeness_counts) == len(first.x0) // 100 or len(first.x0) < 100
    if len(first.x0) >= 100:
        assert list(first.likeness_counts) == list(range(100, len(first.x0) - len(first.x0) % 100 + 1, 100))
    canvas = np.full(target.shape, 0.5, dtype=np.float64)
    from fc_sat.linedraw.optimize import _line_gain

    for index in range(len(first.x0)):
        dark, light, rr, cc, coverage = _line_gain(
            canvas, target, weight, float(first.x0[index]), float(first.y0[index]), float(first.x1[index]), float(first.y1[index]), INK
        )
        gain = dark if int(first.ink[index]) < 0 else light
        assert gain > 0
        assert int(first.ink[index]) == (-1 if dark >= light else 1)
        canvas[rr, cc] += int(first.ink[index]) * INK * coverage
    rebuilt = replay(first.x0, first.y0, first.x1, first.y1, first.ink, len(first.x0), target.shape)
    assert np.allclose(rebuilt, canvas)


def test_l_aha_is_the_first_count_at_sixty_percent_of_final():
    counts = np.array([100, 200, 300, 400])
    scores = np.array([0.10, 0.20, 0.31, 0.50])
    assert find_l_aha(counts, scores) == 300
    assert find_l_aha(np.array([100]), np.array([0.2])) == 100


def test_blank_image_has_no_face_and_center_crop_keeps_aspect():
    gray = np.full((80, 90), 40, dtype=np.uint8)
    assert largest_face(gray) is None
    x0, y0, x1, y1 = center_crop(90, 80)
    assert abs((x1 - x0) / (y1 - y0) - 0.69) < 0.02
    impulse = np.zeros((64, 80), dtype=np.float64)
    impulse[20, 15] = 1
    peak = spectral_peak(impulse)
    assert peak is not None
    assert abs(peak[0] - 15) <= 2 and abs(peak[1] - 20) <= 2
    assert spectral_peak(np.full((32, 32), 0.4)) is None


def test_clahe_trial_keeps_the_higher_likeness(tmp_path):
    rng = np.random.default_rng(0)
    grid = np.clip(rng.random((28, 20)), 0, 1)
    weight = importance_map(28, 20, 10, 8, 2.5)
    _target, clip, scores = select_clahe(grid, weight, seed=3, trial_lines=15, candidates=8)
    assert clip in {0.012, 0.02}
    assert clip == float(max(scores, key=scores.get))
    assert scores[f"{clip:.3f}"] == max(scores.values())


def test_hold_moves_back_when_the_spec_count_already_looks_like_the_picture():
    from fc_sat.linedraw.choreo import choose_hold

    counts = np.array([100, 200, 400, 1080])
    scores = np.array([0.27, 0.296, 0.40, 0.55])
    assert choose_hold(counts, scores, 1800) == 200
    plan = build_plan(36000, 1800, pre_drop=200)
    assert plan.pre_drop == 200
    assert plan.burst == 1600
    assert plan.counts[743] == 200
    assert plan.counts[768] == 1800
    assert np.all(np.diff(plan.counts[:1800].astype(np.int32)) >= 0)


def test_frame_counts_are_monotone_and_anchored():
    plan = build_plan(5000, 1000)
    assert plan.burst == 400
    assert plan.pre_drop == 600
    assert plan.counts[0] == 0
    assert plan.counts[192] == 16
    assert plan.counts[743] == 600
    assert np.all(plan.counts[744:768] == 600)
    assert plan.counts[768] == 1000
    assert plan.counts[768] - plan.counts[767] == plan.burst
    assert plan.counts[863] == plan.mid
    assert plan.counts[1320] == 5000
    assert np.all(np.diff(plan.counts[:1800]) >= 0)
    assert plan.count_at(1823) == 0
    assert plan.count_at(1799) == 5000


def test_throw_lands_on_the_stored_segment_and_starts_long():
    landed = throw_endpoints(10, 20, 40, 50, 1.0)
    assert landed == pytest.approx((10, 20, 40, 50))
    start = throw_endpoints(10, 20, 40, 50, 0.0)
    start_len = ((start[2] - start[0]) ** 2 + (start[3] - start[1]) ** 2) ** 0.5
    final_len = ((40 - 10) ** 2 + (50 - 20) ** 2) ** 0.5
    assert start_len == pytest.approx(final_len * 1.3, rel=0.02)
    assert hook_progress(0, 0) == pytest.approx(0.4)
    assert hook_progress(6, 0) == pytest.approx(1.0)
    assert loop_progress(1823) == pytest.approx(0.4)
    assert flyoff_progress(1799, 0, 100) == 0
    assert 0 < flyoff_progress(1800, 0, 100) < 0.2
    assert flyoff_progress(1811, 0, 100) == 1.0


def test_zoom_is_log_spaced_between_the_anchors():
    assert zoom_at(875) == pytest.approx(1.0)
    assert zoom_at(876) == pytest.approx(1.0)
    assert zoom_at(940) == pytest.approx(12.0)
    assert zoom_at(959) == pytest.approx(12.0)
    assert zoom_at(1056) == pytest.approx(1.0)
    mid = zoom_at(876 + (940 - 876) // 2)
    assert mid == pytest.approx(12.0 ** 0.5, rel=0.08)


def test_long_subject_stays_within_eighteen_characters():
    lines = drop_caption("supercalifragilistic expialidocious")
    assert lines[0] == "IT'S"
    assert all(len(line) <= 18 for line in lines)
    assert caption_lines(1440, "A", "", False, 100, 10) == ["THEN"]
    assert caption_lines(1440, "A", "", True, 100, 10) == ["THE ORIGINAL"]
    assert caption_lines(1536, "A", "", False, 100, 10) == ["NOW"]
    assert caption_lines(1536, "A", "", True, 100, 10) == ["THE LINES"]


def test_full_render_is_refused_without_approval():
    from make_linedraw import main

    with pytest.raises(SystemExit, match="approved"):
        main(["full"])


def test_likeness_of_identical_mid_gray_is_high():
    image = np.full((16, 12), 0.5)
    assert likeness(image, image) > 0.99


def test_persistent_buffer_matches_a_full_redraw_and_the_loop_frame():
    from fc_sat.linedraw.render import LineRenderer
    from fc_sat.verify import ssim_u8

    target = _dark_target(24, 18)
    weight = np.ones_like(target)
    fit = optimize(target, weight, seed=5, max_kept=24, candidates=8, planned=24)
    picture = {
        "target": target,
        "color": np.full((30, 24, 3), 120, dtype=np.uint8),
        "mouth": (9.0, 16.0),
        "weight": weight,
    }
    plan = build_plan(len(fit.x0), max(16, len(fit.x0) // 2))
    job = {"subject": "A VERY LONG SUBJECT NAME", "show_original": False, "credit": ""}
    renderer = LineRenderer(fit, picture, plan, job, width=270, height=480)
    renderer.commit(len(fit.x0))
    assert np.allclose(renderer.canvas, renderer.redraw(len(fit.x0)))
    opening = renderer.render(0)
    ending = renderer.render(1823)
    assert ssim_u8(opening, ending) >= 0.99
    renderer.render(768)
    assert renderer.last_caption_px >= 48
    assert all(len(line) <= 18 for line in drop_caption(job["subject"]))


def test_encode_command_locks_60fps_and_256k_aac():
    from fc_sat.linedraw.encode import encode_command

    command = encode_command("ffmpeg", width=1080, height=1920, fps=60, output=__import__("pathlib").Path("out.mp4"), audio_path=__import__("pathlib").Path("a.wav"), preset="slow")
    assert command[command.index("-framerate") + 1] == "60"
    assert command[command.index("-fps_mode") + 1] == "cfr"
    assert command[command.index("-b:a") + 1] == "256k"
    assert "scale=out_color_matrix=bt709:out_range=tv" in command


def test_audio_length_silence_tail_and_sidechain():
    from fc_sat.linedraw.audio import _cut_silence, _tape_stop, silence
    from fc_sat.linedraw.verify import sidechain_drops

    audio = silence()
    assert len(audio) == 1824 * 800
    audio[:] = 0.25
    _cut_silence(audio)
    assert float(np.max(np.abs(audio[744 * 800 : 768 * 800]))) == 0.0
    _tape_stop(audio)
    assert float(np.max(np.abs(audio[1806 * 800 :]))) == 0.0
    assert float(np.max(np.abs(audio[1800 * 800 : 1800 * 800 + 100]))) > 0.0
    assert sidechain_drops()


def test_delivery_modes_and_post_kit():
    from fc_sat.beatkit.delivery import require_delivery
    from fc_sat.linedraw.post import lint_post, render_post
    from fc_sat.linedraw.verify import sidechain_drops

    require_delivery("full", 1080, 1920, 60, 1824)
    require_delivery("preview", 540, 960, 30, 912)
    require_delivery("hooks", 540, 960, 30, 105)
    meta = {"kept": 12480, "thrown": 3140000, "rights": "unknown", "credit": ""}
    text = render_post(meta)
    assert lint_post(text, meta) == []
    meta["rights"] = "own"
    meta["credit"] = "Photo by Ada"
    assert lint_post(render_post(meta), meta) == []
    assert sidechain_drops()
