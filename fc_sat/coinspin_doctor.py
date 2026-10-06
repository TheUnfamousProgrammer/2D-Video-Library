"""Toolchain check before any coinspin frame is drawn."""

from __future__ import annotations

from fc_sat.circlesquare_doctor import backend_name
from fc_sat.coinspin_config import lightness_spread, load_config
from fc_sat.coinspin_math import count_spins, scale_for
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
        print(f"trace lightness spread: {lightness_spread(cfg.trace_colors):.4f} OKLab L")
        print(f"same-size coin radius {scale_for(1):.1f} px; spins for 1x, 2x, 3x: {count_spins(1)}, {count_spins(2)}, {count_spins(3)}")
    except SystemExit as exc:
        print(f"config: FAIL {exc}")
        failed = True
    return 1 if failed else 0
