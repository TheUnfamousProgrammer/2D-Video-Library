"""Toolchain check for the Collatz generator."""

from __future__ import annotations

import os

from fc_sat.collatz_config import load_yaml, validate_palette
from fc_sat.collatz_draw import backend_name
from fc_sat.collatz_fonts import font_file, operation_glyphs
from fc_sat.collatz_voice import key_present
from fc_sat.encode import assert_encoders, find_ffmpeg


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
    multiply, divide = operation_glyphs("word")
    print(f"glyphs: multiply {multiply!r} divide {divide!r}")
    print(f"renderer: {backend_name()}")
    try:
        validate_palette(load_yaml("collatz_timeline.yaml"))
        print("contrast: ok")
    except SystemExit as exc:
        print(f"contrast: FAIL {exc}")
        failed = True
    print(f"elevenlabs key: {'present' if os.environ.get('ELEVENLABS_API_KEY') else 'absent'}")
    print(f"elevenlabs voice: {'present' if os.environ.get('ELEVENLABS_VOICE_ID') else 'absent'}")
    print(f"voice ready: {'yes' if key_present() else 'no, write the manual script'}")
    return 1 if failed else 0
