"""Toolchain check before any coinspin frame is drawn."""

from __future__ import annotations

from fc_sat.circlesquare_doctor import backend_name
from fc_sat.coinspin_config import load_config
from fc_sat.coinspin_math import carry_spins, film_lap_spins, road_spins, sat_lap_spins
from fc_sat.encode import assert_encoders, find_ffmpeg
from fc_sat.fonts import font_file


def doctor() -> int:
    failed = False
    try:
        ffmpeg = find_ffmpeg()
        assert_encoders(ffmpeg)
        print(f"ffmpeg: {ffmpeg}")
    except SystemExit as exc:
        print(f"ffmpeg: FAIL {exc}")
        failed = True
    for kind in ("mono", "word"):
        try:
            print(f"font {kind}: {font_file(kind)}")
        except SystemExit as exc:
            print(f"font {kind}: FAIL {exc}")
            failed = True
    print(f"renderer: {backend_name()}")
    try:
        cfg = load_config()
        print(f"config: {cfg.bpm:g} bpm, {cfg.frames} frames, hook {cfg.hook}")
        print(f"coins: radius {cfg.coin_radius:g}, SAT act {cfg.sat_grey_radius:g} and {cfg.sat_gold_radius:g}")
        print(f"spins as drawn: lap {film_lap_spins()}, road {road_spins()}, carry {carry_spins()}, SAT lap {sat_lap_spins()}")
    except SystemExit as exc:
        print(f"config: FAIL {exc}")
        failed = True
    return 1 if failed else 0
