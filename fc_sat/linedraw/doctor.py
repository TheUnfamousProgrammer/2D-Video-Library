"""Environment check for the linedraw short."""

from __future__ import annotations

import importlib
import shutil
import sys


def doctor() -> int:
    problems: list[str] = []
    print(f"python {sys.version.split()[0]}")
    if sys.version_info < (3, 11):
        problems.append("Python 3.11+ is required")
    ffmpeg = shutil.which("ffmpeg")
    print(f"ffmpeg {ffmpeg or 'missing'}")
    if not ffmpeg:
        problems.append("ffmpeg is not on PATH")
    else:
        from fc_sat.encode import assert_encoders

        try:
            assert_encoders(ffmpeg)
        except SystemExit as exc:
            problems.append(str(exc))
    for name in ("numpy", "scipy", "skimage", "PIL", "cv2", "skia", "soundfile", "pyloudnorm"):
        try:
            importlib.import_module(name)
            print(f"import {name} ok")
        except Exception as exc:
            problems.append(f"{name} import failed: {exc}")
    try:
        importlib.import_module("numba")
        print("import numba ok")
    except Exception as exc:
        print(f"numba unavailable, numpy fallback will be used ({exc})")
    from fc_sat.fonts import font_file

    for kind in ("word", "mono"):
        print(f"font {font_file(kind)}")
    if problems:
        for item in problems:
            print(f"problem: {item}")
        return 1
    print("doctor ok")
    return 0
