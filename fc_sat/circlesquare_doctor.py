"""Toolchain check before any circlesquare frame is drawn."""

from __future__ import annotations

from fc_sat.circlesquare_config import lightness_spread, load_config
from fc_sat.circlesquare_math import R1, first_k_within, gap
from fc_sat.encode import assert_encoders, find_ffmpeg
from fc_sat.fonts import font_file, glyph_present, load_font


def backend_name() -> str:
    try:
        import skia  # noqa: F401

        return "skia"
    except ImportError:
        pass
    try:
        import cairo  # noqa: F401

        return "cairo"
    except ImportError:
        return "pillow"


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
    try:
        present = glyph_present(load_font("mono", 64), "\u221e")
        print(f"infinity glyph: {'font' if present else 'vector lemniscate'}")
    except SystemExit as exc:
        print(f"infinity glyph: FAIL {exc}")
        failed = True
    print(f"renderer: {backend_name()}")
    try:
        cfg = load_config()
        spread = lightness_spread(cfg.bar_colors)
        print(f"config: {cfg.bpm:g} bpm, {cfg.frames} frames, hook {cfg.hook}")
        print(f"bar lightness spread: {spread:.4f} OKLab L")
        print(f"r1 {R1:.3f} px, gap(1) {gap(1):.2f} px, fool K {first_k_within(0.5)}")
    except SystemExit as exc:
        print(f"config: FAIL {exc}")
        failed = True
    return 1 if failed else 0
