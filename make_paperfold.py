#!/usr/bin/env python3
"""How many folds to the Moon.

    python make_paperfold.py doctor
    python make_paperfold.py facts
    python make_paperfold.py full --approved --out out/paperfold.mp4

`full` is refused unless `--approved` is also passed. The default hook is A.
Preview and hooks are 540x960 at 30 fps. The master is 1080x1920 at 60 fps.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from fc_sat.beatkit.delivery import frame_indices, require_delivery
from fc_sat.encode import pipe_raw_bgr
from fc_sat.paperfold_art import run_ingest
from fc_sat.paperfold_stills import render_heroes
from fc_sat.paperfold_claims import evaluate, load_claims, render_report
from fc_sat.paperfold_copy import lint_copy
from fc_sat.paperfold_doctor import doctor
from fc_sat.paperfold_render import PaperRenderer, paper_fraction
from fc_sat.paperfold_text import lint_script, load_script
from fc_sat.paperfold_timeline import build_timeline, facts_markdown, retention_markdown, write_timeline
from fc_sat.stage_log import Heartbeat, StageLog

ROOT = Path(__file__).resolve().parent

ROOT_MODES = (
    "doctor",
    "ingest",
    "stills",
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
    parser = argparse.ArgumentParser(description="How many folds to the Moon")
    parser.add_argument("mode", choices=ROOT_MODES)
    parser.add_argument("--hook", default="A", choices=["A", "B", "C", "D"])
    parser.add_argument("--out", default="")
    parser.add_argument("--approved", action="store_true")
    args = parser.parse_args(argv)
    if args.mode == "full" and not args.approved:
        raise SystemExit("full render refused without --approved")
    report = ROOT / "out" / "art_report.md"
    if args.mode == "full" and report.exists() and "UNAPPROVED" in report.read_text():
        raise SystemExit("full render refused: art is UNAPPROVED")
    log = StageLog()
    if args.mode == "ingest":
        code = run_ingest()
        log.mark("ingest")
        return code
    if args.mode == "stills":
        render_heroes(args.hook)
        log.mark("stills")
        return 0
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
        code = _animatic(args.hook, Path(args.out) if args.out else ROOT / "out" / "paperfold_animatic.mp4")
        log.mark("animatic")
        return code
    if args.mode == "hooks":
        code = _hooks(Path(args.out) if args.out else ROOT / "out")
        log.mark("hooks")
        return code
    print(f"{args.mode} is not built yet", file=sys.stderr)
    log.mark(args.mode)
    return 2


def _facts() -> int:
    book = load_claims()
    results = evaluate(book)
    script = load_script(book=book)
    errors = lint_script(script, book)
    errors.extend(lint_copy(book))
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
    print(f"snap error {timeline.max_snap_error:.3e} frames")
    print(f"folds {len(timeline.folds)} moon fold frame {timeline.folds[-1].frame}")
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
        f"folds {len(payload['folds'])} kicks {len(payload['kicks'])} "
        f"impacts {len(payload['impacts'])}"
    )
    print(f"retention {report}")
    return 0


def _animatic(hook: str, output: Path) -> int:
    spec = require_delivery("animatic", 540, 960, 30, 912)
    renderer = PaperRenderer(spec.width, spec.height, hook)
    full = PaperRenderer(1080, 1920, hook)
    heavy = full.profile_ms()
    print(f"profile: {heavy:.0f} ms heavy frame at 1080x1920; full-render eta {heavy * 1824 / 60000:.1f} min", flush=True)
    indices = frame_indices("animatic", 1824)

    def frames():
        for index, frame in enumerate(indices):
            if index % 60 == 0:
                print(f"animatic {index}/{spec.frames}", flush=True)
            yield renderer.render(frame)

    output.parent.mkdir(parents=True, exist_ok=True)
    with Heartbeat("animatic still rendering"):
        pipe_raw_bgr(
            frames(),
            output,
            width=spec.width,
            height=spec.height,
            fps=spec.fps,
            audio_path=None,
            preset="veryfast",
            n_frames=spec.frames,
        )
    _sheet(renderer, [0, 12, 72, 84, 96, 216, 408, 576, 792, 1128, 1152, 1344, 1800], ROOT / "out" / "paperfold_contact.png")
    _sheet(renderer, [12, 72, 84, 96, 216, 408, 576, 792, 1128, 1152, 1344, 1800], ROOT / "out" / "paperfold_tower_strip.png")
    _print_style(renderer, hook)
    print(f"wrote {output}")
    return 0


def _hooks(out_dir: Path) -> int:
    spec = require_delivery("hooks", 540, 960, 30, 105)
    out_dir.mkdir(parents=True, exist_ok=True)
    indices = frame_indices("hooks", 1824)
    thumbs = []
    for hook in ("A", "B", "C", "D"):
        renderer = PaperRenderer(spec.width, spec.height, hook)
        path = out_dir / f"paperfold_hook_{hook}.mp4"

        def frames(renderer=renderer):
            for frame in indices:
                yield renderer.render(frame)

        pipe_raw_bgr(
            frames(),
            path,
            width=spec.width,
            height=spec.height,
            fps=spec.fps,
            audio_path=None,
            preset="veryfast",
            n_frames=spec.frames,
        )
        _print_style(renderer, hook)
        row = []
        for seconds in (0.0, 0.2, 0.5, 1.0, 2.0):
            image = renderer.render(int(round(seconds * 60)))
            tile = Image.fromarray(image[:, :, ::-1])
            tile.thumbnail((360, 640))
            row.append(tile)
        thumbs.append(row)
        print(f"wrote {path} (silent; music is the next stage)")
    _thumb_strip(thumbs, out_dir / "paperfold_hook_thumbs.png")
    return 0


def _print_style(renderer: PaperRenderer, hook: str) -> None:
    images = [renderer.render(frame) for frame in range(25)]
    fracs = [paper_fraction(image) for image in images]
    energy = [
        float(np.mean(np.abs(images[index].astype(np.int16) - images[index + 1].astype(np.int16))))
        for index in range(24)
    ]
    changes = sum(1 for value in energy if value > 0.4)
    hook_fracs = [paper_fraction(renderer.render(frame)) for frame in (0, 12, 84, 168)]
    print(
        f"style {hook}: frame0 area {fracs[0]:.1%}, hook min {min(hook_fracs):.1%}, "
        f"top text {renderer.top_size:.0f}px, motion mean {float(np.mean(energy)):.2f}, "
        f"changes in 24 frames {changes}, min shape stroke 16px"
    )


def _sheet(renderer: PaperRenderer, frames: list[int], path: Path) -> None:
    tiles = [Image.fromarray(renderer.render(frame)[:, :, ::-1]) for frame in frames]
    width, height = tiles[0].size
    sheet = Image.new("RGB", (width * len(tiles), height))
    for index, tile in enumerate(tiles):
        sheet.paste(tile, (index * width, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    print(f"wrote {path}")


def _thumb_strip(rows: list[list[Image.Image]], path: Path) -> None:
    tile_w = max(tile.width for row in rows for tile in row)
    tile_h = max(tile.height for row in rows for tile in row)
    sheet = Image.new("RGB", (tile_w * 5, tile_h * len(rows)), (14, 17, 23))
    for row_index, row in enumerate(rows):
        for col, tile in enumerate(row):
            sheet.paste(tile, (col * tile_w, row_index * tile_h))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    print(f"wrote {path}")


if __name__ == "__main__":
    raise SystemExit(main())
