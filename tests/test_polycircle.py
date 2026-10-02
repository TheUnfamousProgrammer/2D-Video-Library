"""Fast checks for the polycircle short."""

from __future__ import annotations

import pytest

from fc_sat.polycircle_config import lightness_spread, load_config
from fc_sat.polycircle_format import format_count, format_gap
from make_polycircle import main


def test_full_render_is_refused_without_approval():
    with pytest.raises(SystemExit, match="approved"):
        main(["full"])


def test_bar_colors_share_a_lightness():
    cfg = load_config()
    assert len(cfg.bar_colors) == 8
    assert lightness_spread(cfg.bar_colors) <= 0.03
    assert cfg.bpm == 150
    assert cfg.frames == 1824
    assert cfg.radius == 370


def test_big_numbers_and_gaps_have_no_float_noise():
    assert format_count(6_291_456) == "6,291,456"
    assert format_count(24576) == "24,576"
    assert format_count(4) == "4"
    assert format_gap(0.4905867426127042) == "0.49"
    assert format_gap(7.131694814809082) == "7.1"
    assert format_gap(1.783043044636785) == "1.8"
    assert format_gap(0.4457682202174773) == "0.45"
    assert format_gap(0.02786065944888616) == "0.028"
    assert format_gap(0.006965166683015056) == "0.0070"
    assert format_gap(4.613087689619988e-11) == "0.000000000046"
    assert "e" not in format_gap(4.613087689619988e-11)
