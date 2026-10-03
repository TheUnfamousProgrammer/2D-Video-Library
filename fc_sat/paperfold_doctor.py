"""Toolchain check that runs before any paperfold frame is drawn."""

from __future__ import annotations

from fc_sat.encode import assert_encoders, find_ffmpeg
from fc_sat.fonts import font_file, glyph_present, load_font
from fc_sat.beatkit.palette import lightness_spread
from fc_sat.paperfold_config import load_config


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


def times_glyph() -> str:
    """Multiplication sign when the number face has it, otherwise the letter x."""
    mono = load_font("mono", 64)
    if glyph_present(mono, "\u00d7"):
        return "\u00d7"
    return "x"


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
        print(f"multiplication sign: {times_glyph()!r}")
    except SystemExit as exc:
        print(f"multiplication sign: FAIL {exc}")
        failed = True
    print(f"renderer: {backend_name()}")
    print("posted file: full 1080x1920 with fps=30 and -shortest kept 912 frames at 30/1")
    print("full is 1080x1920 at 60 fps, 1824 frames; preview and hooks are 540x960 at 30 fps")
    try:
        cfg = load_config()
        spread = lightness_spread(cfg.bar_colors)
        print(f"config: {cfg.bpm:g} bpm, {cfg.frames} frames, hook {cfg.hook}")
        print(f"bar lightness spread: {spread:.4f} OKLab L")
        print("bar colors:", " ".join(cfg.bar_colors))
    except SystemExit as exc:
        print(f"config: FAIL {exc}")
        failed = True
    return 1 if failed else 0
