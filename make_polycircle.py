#!/usr/bin/env python3
"""Sides to circle.

    python make_polycircle.py doctor
    python make_polycircle.py facts
    python make_polycircle.py full --approved --out out/polycircle.mp4

`full` is refused unless `--approved` is also passed. The default hook is A.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

from fc_sat.encode import pipe_raw_bgr
from fc_sat.polycircle_claims import evaluate, load_claims, render_report
from fc_sat.polycircle_doctor import doctor
from fc_sat.polycircle_render import PolyRenderer
from fc_sat.polycircle_text import lint_script, load_script
from fc_sat.polycircle_timeline import (
    build_timeline,
    facts_markdown,
    retention_markdown,
    write_timeline,
)
from fc_sat.stage_log import Heartbeat, StageLog

ROOT = Path(__file__).resolve().parent

ROOT_MODES = (
    "doctor",
    "facts",
    "timeline",
    "animatic",
    "hooks",
    "audio",
    "preview",
    "full",
    "postkit",
    "verify",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sides to circle")
    parser.add_argument("mode", choices=ROOT_MODES)
    parser.add_argument("--hook", default="A", choices=["A", "B", "C"])
    parser.add_argument("--out", default="")
    parser.add_argument("--approved", action="store_true")
    args = parser.parse_args(argv)
    if args.mode == "full" and not args.approved:
        raise SystemExit("full render refused without --approved")
    log = StageLog()
    if args.mode == "doctor":
        code = doctor()
        log.mark("doctor")
        return code
    if args.mode == "facts":
        code = _facts()
        log.mark("facts")
        return code
    if args.mode == "timeline":
        code = _timeline()
        log.mark("timeline")
        return code
    if args.mode == "animatic":
        code = _animatic(args.hook, Path(args.out or "out/polycircle_animatic.mp4"))
        log.mark("animatic")
        return code
    if args.mode == "hooks":
        code = _hooks()
        log.mark("hooks")
        return code
    print(f"{args.mode} is not built yet", file=sys.stderr)
    return 2


def _facts() -> int:
    book = load_claims()
    results = evaluate(book)
    script = load_script(book=book)
    errors = lint_script(script, book)
    timeline = build_timeline()
    out = ROOT / "out"
    out.mkdir(parents=True, exist_ok=True)
    (out / "facts.md").write_text(facts_markdown(timeline, book))
    (out / "claims_report.md").write_text(render_report(book, results))
    failed = [claim_id for claim_id, ok, detail in results if not ok]
    for claim_id, ok, detail in results:
        if not ok:
            print(f"FAIL {claim_id}: {detail}")
    for error in errors:
        print(error)
    print(f"claims {sum(1 for _, ok, _ in results if ok)}/{len(results)} passed")
    print(f"snap error {timeline.schedule.max_snap_error:.3e} frames")
    print(f"rotation scale {timeline.rotation_scale:.6f}, {timeline.rotation_turns} quarter-turns")
    return 1 if failed or errors else 0


def _timeline() -> int:
    timeline = build_timeline()
    path = write_timeline(timeline)
    report = ROOT / "out" / "retention_map.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(retention_markdown(timeline))
    payload = timeline.to_json()
    print(f"wrote {path}")
    print(
        f"adds {len(payload['adds'])} doublings {len(payload['doublings'])} "
        f"kicks {len(payload['kicks'])} impacts {len(payload['impacts'])}"
    )
    print(f"retention {report}")
    return 0


def _sheet(renderer: PolyRenderer, frames: list[int], path: Path) -> None:
    tiles = [Image.fromarray(renderer.render(frame)[:, :, ::-1]) for frame in frames]
    width, height = tiles[0].size
    sheet = Image.new("RGB", (width * len(tiles), height))
    for index, tile in enumerate(tiles):
        sheet.paste(tile, (index * width, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    print(f"wrote {path}")


def _animatic(hook: str, output: Path) -> int:
    timeline = build_timeline()
    renderer = PolyRenderer(timeline, width=540, height=960, hook=hook)
    ms = renderer.profile_ms()
    full = PolyRenderer(timeline, width=1080, height=1920, hook=hook)
    heavy = full.profile_ms()
    print(
        f"profile: {heavy:.0f} ms heavy frame at 1080x1920; "
        f"sequential full-render eta {heavy * timeline.n_frames / 60000:.1f} min",
        flush=True,
    )
    count = 912
    indices = [min(timeline.n_frames - 1, index * 2) for index in range(count)]

    def frames():
        for index, frame in enumerate(indices):
            if index % 60 == 0:
                print(f"animatic {index}/{count}", flush=True)
            yield renderer.render(frame)

    with Heartbeat("animatic still rendering"):
        pipe_raw_bgr(frames(), output, width=540, height=960, fps=30, audio_path=None, preset="veryfast", n_frames=count)
    _sheet(renderer, [0, 192, 768, 954, 1128, 1488, 1632, 1823], ROOT / "out" / "polycircle_contact.png")
    _sheet(renderer, [876, 912, 948, 960, 1128], ROOT / "out" / "polycircle_zoom_strip.png")
    print(f"draft profile {ms:.0f} ms/frame")
    return 0


def _hooks() -> int:
    timeline = build_timeline()
    folder = ROOT / "out" / "hooks"
    folder.mkdir(parents=True, exist_ok=True)
    count = 105
    indices = [min(timeline.n_frames - 1, index * 2) for index in range(count)]
    for hook in ("A", "B", "C"):
        renderer = PolyRenderer(timeline, width=540, height=960, hook=hook)
        output = folder / f"hook_{hook}.mp4"

        def frames(renderer=renderer):
            for index, frame in enumerate(indices):
                if index % 30 == 0:
                    print(f"hook {hook} {index}/{count}", flush=True)
                yield renderer.render(frame)

        pipe_raw_bgr(frames(), output, width=540, height=960, fps=30, audio_path=None, preset="veryfast", n_frames=count)
        print(f"wrote {output} (silent; audio is muxed after the mix exists)")
    strip = PolyRenderer(timeline, width=360, height=640, hook="A")
    _sheet(strip, [0, 30, 60, 120], folder / "thumb_strip.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
