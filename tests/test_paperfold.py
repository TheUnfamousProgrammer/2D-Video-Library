"""Fast checks for the paperfold short."""

from __future__ import annotations

import pytest

from make_paperfold import main


def test_full_render_is_refused_without_approval():
    with pytest.raises(SystemExit, match="approved"):
        main(["full"])


def test_heights_lengths_and_pass_folds():
    from fc_sat.paperfold_math import (
        EARTH_CIRC_M,
        first_fold_past,
        height_km,
        height_m,
        length_km,
        length_m,
        light_years,
        milestone_fold,
        au_multiple,
    )

    assert height_m(14) == pytest.approx(1.638, abs=0.0005)
    assert height_m(15) == pytest.approx(3.277, abs=0.0005)
    assert height_m(22) == pytest.approx(419.4, abs=0.05)
    assert height_m(23) == pytest.approx(838.9, abs=0.05)
    assert height_km(26) == pytest.approx(6.711, abs=0.0005)
    assert height_km(27) == pytest.approx(13.42, abs=0.005)
    assert height_km(29) == pytest.approx(53.69, abs=0.005)
    assert height_km(30) == pytest.approx(107.37, abs=0.005)
    assert height_km(41) == pytest.approx(219_902, abs=0.5)
    assert height_km(42) == pytest.approx(439_805, abs=0.5)
    assert height_m(12) == pytest.approx(0.4096, abs=1e-6)
    assert milestone_fold("person") == 15
    assert milestone_fold("burj") == 23
    assert milestone_fold("everest") == 27
    assert milestone_fold("karman") == 30
    assert milestone_fold("moon") == 42
    assert length_m(7) == pytest.approx(0.878, abs=0.0005)
    assert length_m(12) == pytest.approx(879, abs=0.5)
    assert length_km(19) == pytest.approx(14_390, rel=0.001)
    assert length_km(20) == pytest.approx(57_570, abs=1)
    assert min(n for n in range(1, 30) if length_m(n) > EARTH_CIRC_M) == 20
    assert length_km(30) == pytest.approx(6.04e10, rel=0.01)
    assert au_multiple(30) == pytest.approx(403.5, abs=0.05)
    assert length_m(42) == pytest.approx(1.013e21, rel=0.001)
    assert light_years(42) == pytest.approx(107_100, rel=0.001)
    assert first_fold_past(149_597_870_700.0) == 51


def test_schedule_counts_and_frames():
    from fc_sat.paperfold_schedule import build_folds, fold_frame

    folds = build_folds()
    assert len(folds) == 42
    assert [fold_frame(n) for n in range(1, 15)] == [12 * k for k in range(1, 15)]
    assert [fold_frame(n) for n in range(15, 31)] == [216 + 24 * (n - 15) for n in range(15, 31)]
    assert [fold_frame(n) for n in range(31, 43)] == [864 + 24 * (n - 31) for n in range(31, 43)]
    assert fold_frame(30) == 576
    assert fold_frame(42) == 1128
    assert folds[0].frame == 12
    assert folds[-1].n == 42


def test_tower_scale_culling_and_log_axis():
    from fc_sat.paperfold_math import (
        height_m,
        length_m,
        log_axis_x,
        milestone_visible,
        tower_px,
    )

    for n in (7, 15, 30, 42):
        assert tower_px(height_m(n)) == pytest.approx(700.0)
    assert milestone_visible(1.8, height_m(15))
    assert not milestone_visible(1.8, height_m(30))
    assert not milestone_visible(384_400e3, height_m(7))
    assert milestone_visible(384_400e3, height_m(42))
    x12 = log_axis_x(length_m(12))
    x20 = log_axis_x(length_m(20))
    x30 = log_axis_x(length_m(30))
    assert 130 < x12 < x20 < x30 < 130 + 820


def test_claims_and_text_lint():
    from fc_sat.paperfold_claims import evaluate, lint_text, load_claims
    from fc_sat.paperfold_format import milky_way_label, stamp_lines
    from fc_sat.paperfold_text import lint_script, load_script

    book = load_claims()
    failed = [claim_id for claim_id, ok, _detail in evaluate(book) if not ok]
    assert failed == []
    script = load_script(book=book)
    assert lint_script(script, book) == []
    assert lint_text("it takes 99 folds", [], book, "caption")
    assert "400" in stamp_lines(30)[1]
    assert "EARTH TO SUN" in stamp_lines(30)[2]
    assert milky_way_label() == "107,000 LIGHT-YEARS"
    for lines in script.hooks.values():
        assert len(lines) <= 3
        assert all(len(line) <= 14 for line in lines)


def test_hook_variants_share_the_fold_grid():
    from fc_sat.paperfold_text import load_script, top_lines
    from fc_sat.paperfold_timeline import build_timeline

    timeline = build_timeline()
    script = load_script()
    assert timeline.folds[0].frame == 12
    assert timeline.kicks[0] == 0
    assert 552 not in timeline.kicks
    assert 576 in timeline.impacts
    for hook in ("A", "B", "C", "D"):
        assert top_lines(script, 0, hook)
        assert top_lines(script, 0, hook) == top_lines(script, 1823, hook)
