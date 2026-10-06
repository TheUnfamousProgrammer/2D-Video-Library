"""Toolchain and asset check before any scale frame is drawn."""

from __future__ import annotations

from fc_sat.circlesquare_doctor import backend_name
from fc_sat.encode import assert_encoders, find_ffmpeg
from fc_sat.fonts import font_file
from fc_sat.scale_scene import build_scene


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
        scene = build_scene()
    except SystemExit as exc:
        print(f"scene: FAIL {exc}")
        return 1
    ready = sum(scene.has_art)
    print(f"objects: {len(scene.placed)}, cutouts ready {ready}, placeholders {len(scene.placed) - ready}")
    for placed, has in zip(scene.placed, scene.has_art):
        if not has:
            print(f"  waiting on art: {placed.item.art}")
    return 1 if failed else 0
