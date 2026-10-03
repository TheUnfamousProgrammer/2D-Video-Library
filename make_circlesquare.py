#!/usr/bin/env python3
"""How many circles does it take to draw a square?

    python make_circlesquare.py doctor
    python make_circlesquare.py full --approved --out out/circlesquare.mp4

`full` is refused unless `--approved` is also passed. The default hook is A.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from fc_sat.audio import write_wav
from fc_sat.beatkit.delivery import frame_indices, require_delivery
from fc_sat.encode import pipe_raw_bgr
from fc_sat.circlesquare_audio import master, mix, onset_sample, sync_rows
from fc_sat.circlesquare_claims import evaluate, load_claims, render_report
from fc_sat.circlesquare_doctor import doctor
from fc_sat.circlesquare_post import lint_post, render_postkit
from fc_sat.circlesquare_render import CircleRenderer
from fc_sat.circlesquare_text import lint_script, load_script
from fc_sat.circlesquare_timeline import build_timeline, facts_markdown, retention_markdown, write_timeline
from fc_sat.circlesquare_verify import verify
from fc_sat.stage_log import Heartbeat, StageLog

ROOT = Path(__file__).resolve().parent

ROOT_MODES = (
    "doctor",
    "facts",
    "timeline",
    "stills",
    "clips",
    "hooks",
    "audio",
    "preview",
    "full",
    "postkit",
    "verify",
)

HERO_FRAMES = (0, 120, 380, 700, 767, 954, 1128, 1320, 1488, 1700)
CLIP_RANGES = ((0, 180), (740, 830), (940, 1010))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Circles that try to draw a square")
    parser.add_argument("mode", choices=ROOT_MODES)
    parser.add_argument("--hook", default="A", choices=["A", "B", "D"])
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
    if args.mode == "stills":
        code = _stills(args.hook)
        log.mark("stills")
        return code
    if args.mode == "clips":
        code = _clips(args.hook)
        log.mark("clips")
        return code
    if args.mode == "hooks":
        code = _hooks()
        log.mark("hooks")
        return code
    if args.mode == "audio":
        code = _audio()
        log.mark("audio")
        return code
    if args.mode == "preview":
        code = _preview(args.hook, Path(args.out or "out/circlesquare_preview.mp4"))
        log.mark("preview")
        return code
    if args.mode == "postkit":
        code = _postkit()
        log.mark("postkit")
        return code
    if args.mode == "verify":
        target = Path(args.out) if args.out else None
        code = verify(target)
        log.mark("verify")
        return code
    if args.mode == "full":
        code = _full(args.hook, Path(args.out or "out/circlesquare.mp4"))
        log.mark("full")
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
    failed = [claim_id for claim_id, ok, _detail in results if not ok]
    for claim_id, ok, detail in results:
        if not ok:
            print(f"FAIL {claim_id}: {detail}")
    for error in errors:
        print(error)
    print(f"claims {sum(1 for _, ok, _ in results if ok)}/{len(results)} passed")
    print(f"snap error {timeline.schedule.max_snap_error:.3e} frames")
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


def _sheet(tiles: list[Image.Image], columns: int, path: Path) -> None:
    if not tiles:
        return
    width, height = tiles[0].size
    rows = (len(tiles) + columns - 1) // columns
    sheet = Image.new("RGB", (width * columns, height * rows), (14, 17, 23))
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % columns) * width, (index // columns) * height))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    print(f"wrote {path}")


def _stills(hook: str) -> int:
    timeline = build_timeline()
    renderer = CircleRenderer(timeline, width=1080, height=1920, hook=hook)
    folder = ROOT / "out" / "stills"
    folder.mkdir(parents=True, exist_ok=True)
    thumbs = []
    for frame in range(0, timeline.n_frames, 30):
        image = Image.fromarray(renderer.render(frame)[:, :, ::-1])
        if frame in HERO_FRAMES:
            image.save(folder / f"f{frame:04d}.png")
            print(f"wrote {folder / f'f{frame:04d}.png'}")
        thumbs.append(image.resize((135, 240), Image.Resampling.BOX))
        if frame and frame % 300 == 0:
            print(f"stills {frame}", flush=True)
    for frame in HERO_FRAMES:
        path = folder / f"f{frame:04d}.png"
        if not path.exists():
            image = Image.fromarray(renderer.render(frame)[:, :, ::-1])
            image.save(path)
            print(f"wrote {path}")
    _sheet(thumbs, 8, ROOT / "out" / "contact_sheet.png")
    return 0


def _clips(hook: str) -> int:
    timeline = build_timeline()
    spec = require_delivery("full", 1080, 1920, 60, timeline.n_frames)
    renderer = CircleRenderer(timeline, width=spec.width, height=spec.height, hook=hook)
    folder = ROOT / "out" / "clips"
    folder.mkdir(parents=True, exist_ok=True)
    for start, end in CLIP_RANGES:
        frames = list(range(start, end + 1))
        output = folder / f"f{start:04d}_{end:04d}.mp4"

        def frames_iter(frames=frames):
            for index, frame in enumerate(frames):
                if index % 30 == 0:
                    print(f"clip {frames[0]} {index}/{len(frames)}", flush=True)
                yield renderer.render(frame)

        with Heartbeat(f"clip {start}"):
            pipe_raw_bgr(
                frames_iter(),
                output,
                width=spec.width,
                height=spec.height,
                fps=spec.fps,
                audio_path=None,
                preset="veryfast",
                n_frames=len(frames),
            )
        print(f"wrote {output}")
    return 0


def _mix_files() -> tuple[Path, Path, float, float]:
    timeline = build_timeline()
    full, sfx, lufs, peak = mix(timeline.to_json(), seed=timeline.config.seed, n_frames=timeline.n_frames)
    folder = ROOT / "out"
    folder.mkdir(parents=True, exist_ok=True)
    music = folder / "circlesquare.wav"
    off_audio, off_lufs, off_peak = master(sfx)
    music_off = folder / "circlesquare.music_off.wav"
    write_wav(str(music), full)
    write_wav(str(music_off), off_audio)
    rows = sync_rows(timeline.to_json())
    (folder / "sync.md").write_text("# Sync\n\n" + "\n".join(rows) + "\n")
    onset = onset_sample(full)
    print(f"loudness {lufs:.2f} LUFS, true peak {peak:.2f} dBTP, onset sample {onset}")
    print(f"music-off {off_lufs:.2f} LUFS, true peak {off_peak:.2f} dBTP")
    print(f"wrote {music}")
    print(f"wrote {music_off}")
    return music, music_off, lufs, peak


def _audio() -> int:
    _mix_files()
    return 0


def _encode(renderer: CircleRenderer, indices: list[int], output: Path, audio: Path | None, fps: int, label: str, preset: str) -> None:
    def frames():
        for index, frame in enumerate(indices):
            if index % 60 == 0:
                print(f"{label} {index}/{len(indices)}", flush=True)
            yield renderer.render(frame)

    with Heartbeat(f"{label} still rendering"):
        pipe_raw_bgr(
            frames(),
            output,
            width=renderer.width,
            height=renderer.height,
            fps=fps,
            audio_path=audio,
            preset=preset,
            n_frames=len(indices),
        )
    print(f"wrote {output}")


def _hook_metrics(hook: str) -> None:
    timeline = build_timeline()
    renderer = CircleRenderer(timeline, width=540, height=960, hook=hook)
    full = CircleRenderer(timeline, width=1080, height=1920, hook=hook)
    area = full.bbox_fraction(0)
    stroke = min(full.design_stroke(frame) for frame in (0, 12, 24))
    text = full.design_text_size(0)
    small = CircleRenderer(timeline, width=270, height=480, hook=hook)
    previous = small.render(0).astype(np.float64)
    changes = 0
    energy = 0.0
    for frame in range(1, 25):
        image = small.render(frame).astype(np.float64)
        delta = float(np.mean(np.abs(image - previous)))
        energy += delta
        if delta > 0.4:
            changes += 1
        previous = image
    print(
        f"hook {hook}: bbox {area:.3f}, stroke {stroke:.0f} px, text {text:.0f} px, "
        f"motion {energy:.1f}, changes {changes}"
    )


def _hooks() -> int:
    timeline = build_timeline()
    music, _off, _lufs, _peak = _mix_files()
    folder = ROOT / "out" / "hooks"
    folder.mkdir(parents=True, exist_ok=True)
    spec = require_delivery("hooks", 540, 960, 30, 105)
    indices = frame_indices("hooks", timeline.n_frames)
    for hook in ("A", "B", "D"):
        _hook_metrics(hook)
        renderer = CircleRenderer(timeline, width=spec.width, height=spec.height, hook=hook)
        _encode(renderer, indices, folder / f"hook_{hook}.mp4", music, spec.fps, f"hook {hook}", "veryfast")
    strip = CircleRenderer(timeline, width=360, height=640, hook="A")
    tiles = [Image.fromarray(strip.render(frame)[:, :, ::-1]) for frame in (0, 12, 30, 60, 120)]
    _sheet(tiles, len(tiles), folder / "thumb_strip.png")
    return 0


def _preview(hook: str, output: Path) -> int:
    timeline = build_timeline()
    music, music_off, _lufs, _peak = _mix_files()
    spec = require_delivery("preview", 540, 960, 30, 912)
    renderer = CircleRenderer(timeline, width=spec.width, height=spec.height, hook=hook)
    indices = frame_indices("preview", timeline.n_frames)
    _encode(renderer, indices, output, music, spec.fps, "preview", "veryfast")
    off = output.with_name(output.stem + ".music_off.mp4")
    _encode(renderer, indices, off, music_off, spec.fps, "preview music-off", "veryfast")
    return 0


def _full(hook: str, output: Path) -> int:
    timeline = build_timeline()
    music, music_off, _lufs, _peak = _mix_files()
    spec = require_delivery("full", 1080, 1920, 60, timeline.n_frames)
    renderer = CircleRenderer(timeline, width=spec.width, height=spec.height, hook=hook)
    heavy = renderer.profile_ms()
    print(f"profile: {heavy:.0f} ms; eta {heavy * timeline.n_frames / 60000:.1f} min", flush=True)
    indices = frame_indices("full", timeline.n_frames)
    _encode(renderer, indices, output, music, spec.fps, "full", "slow")
    _encode(
        renderer,
        indices,
        output.with_name(output.stem + ".music_off.mp4"),
        music_off,
        spec.fps,
        "full music-off",
        "slow",
    )
    return 0


def _postkit() -> int:
    book = load_claims()
    errors = lint_post(book)
    for error in errors:
        print(error)
    folder = ROOT / "out"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "postkit.md"
    path.write_text(render_postkit(book))
    print(f"wrote {path}")
    timeline = build_timeline()
    renderer = CircleRenderer(timeline, width=1080, height=1920)
    thumbs = folder / "thumbs"
    thumbs.mkdir(parents=True, exist_ok=True)
    for name, frame in (("drop", 768), ("zoom", 954), ("answer", 1488)):
        image = Image.fromarray(renderer.render(frame)[:, :, ::-1])
        image.save(thumbs / f"{name}.png")
        print(f"wrote {thumbs / f'{name}.png'}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
