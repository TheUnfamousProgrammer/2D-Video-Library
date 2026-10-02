"""Collatz math, formatting, and the iteration cap."""

from __future__ import annotations

import pytest

from fc_sat.collatz_format import display_hundredths, format_int
from fc_sat.collatz_math import (
    CollatzError,
    at_least_steps_below,
    first_step_above,
    first_step_below_after_peak,
    highest_peak_below,
    longest_below,
    next_n,
    peak,
    peak_index,
    steps,
    trajectory,
)


@pytest.mark.parametrize(
    ("n", "count"),
    [(1, 0), (2, 1), (3, 7), (6, 8), (7, 16), (9, 19), (12, 9), (19, 20), (27, 111), (97, 118)],
)
def test_known_steps(n, count):
    assert steps(n) == count


def test_peak_27():
    assert peak(27) == 9232
    assert peak_index(27) == 77
    assert trajectory(27)[77] == 9232


def test_records():
    assert longest_below(1000) == (871, 178)
    assert highest_peak_below(1000) == (250504, 703)
    assert at_least_steps_below(100, 111) == [(27, 111), (54, 112), (55, 112), (73, 115), (97, 118)]
    assert first_step_above(27, 1000) == 36
    assert first_step_below_after_peak(27, 100) == 92


def test_power_of_two():
    assert 2**71 == 2361183241434822606848


def test_rejects_bad_starts():
    with pytest.raises(CollatzError):
        next_n(0)
    with pytest.raises(CollatzError):
        steps(-3)
    with pytest.raises(CollatzError):
        steps(True)  # type: ignore[arg-type]
    with pytest.raises(CollatzError):
        steps(4.0)  # type: ignore[arg-type]


def test_iteration_cap(monkeypatch):
    monkeypatch.setattr("fc_sat.collatz_math.MAX_ITERS", 3)
    with pytest.raises(CollatzError, match="refusing"):
        steps(27)


def test_format_small_and_sextillion():
    assert format_int(9232) == "9,232"
    assert format_int(250504) == "250,504"
    assert format_int(2**71) == "2.36 sextillion"
    assert format_int(2075 * 2**60) == "2.39 sextillion"
    assert display_hundredths(2**71) == 236
    assert format_int(2**71).startswith("2.36 ")
    assert "e+" not in format_int(2**71)
