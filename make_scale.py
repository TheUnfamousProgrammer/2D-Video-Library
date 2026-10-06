#!/usr/bin/env python3
"""The smallest thing to the biggest thing in the universe, side by side, to scale.

    python make_scale.py doctor
    python make_scale.py ingest          # key and trim the art in assets/art/objects
    python make_scale.py stills
    python make_scale.py full --approved --out out/scale.mp4

`full` is refused unless `--approved` is also passed, and refused while any object is
still a placeholder unless `--allow-placeholders` is also passed. The default hook is A.
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
from fc_sat.scale_art import ingest
from fc_sat.scale_audio import master, mix, onset_sample, sync_rows, trim_for_aac
from fc_sat.scale_claims import display_errors, evaluate, load_claims, render_report
from fc_sat.scale_doctor import doctor
from fc_sat.scale_post import lint_post, render_postkit
from fc_sat.scale_render import ScaleRenderer
from fc_sat.scale_scene import build_scene, load_sizes
from fc_sat.scale_timeline import build_timeline, facts_markdown, retention_markdown, write_timeline
from fc_sat.scale_verify import art_errors, verify
from fc_sat.scale_world import load_catalog
from fc_sat.stage_log import Heartbeat, StageLog

ROOT = Path(__file__).resolve().parent

ROOT_MODES = (
    "doctor",
    "ingest",
    "facts",
    "timeline",
    "stills",
    "hooks",
    "audio",
    "preview",
    "full",
    "postkit",
    "verify",
)

HERO_FRAMES = (0, 60, 96, 288, 432, 576, 672, 720, 790, 912, 1008, 1104, 1152, 1200, 1260, 1344, 1392, 1440, 1520, 1640, 1720, 1823)
THUMBS = (("hook", 0), ("you", 790), ("earth", 1220), ("universe", 1650))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Smallest to biggest, to scale")
    parser.add_argument("mode", choices=ROOT_MODES)
    parser.add_argument("--hook", default="A", choices=["A", "B", "D"])
    parser.add_argument("--out", default="")
    parser.add_argument("--approved", action="store_true")
    parser.add_argument("--allow-placeholders", action="store_true")
    args = parser.parse_args(argv)
    if args.mode == "full" and not args.approved:
        raise SystemExit("full render refused without --approved")
    if args.mode == "full" and not args.allow_placeholders and art_errors():
        raise SystemExit("full render refused: " + "; ".join(art_errors()) + " (pass --allow-placeholders to render anyway)")
    log = StageLog()
    if args.mode == "doctor":
        code = doctor()
    elif args.mode == "ingest":
        code = _ingest()
    elif args.mode == "facts":
        code = _facts()
    elif args.mode == "timeline":
        code = _timeline()
    elif args.mode == "stills":
        code = _stills(args.hook)
    elif args.mode == "hooks":
        code = _hooks()
    elif args.mode == "audio":
        code = _audio()
    elif args.mode == "preview":
        code = _preview(args.hook, Path(args.out or "out/scale_preview.mp4"))
    elif args.mode == "postkit":
        code = _postkit()
    elif args.mode == "verify":
        code = verify(Path(args.out) if args.out else None, allow_placeholders=args.allow_placeholders)
    elif args.mode == "full":
        code = _full(args.hook, Path(args.out or "out/scale.mp4"))
    else:
        print(f"{args.mode} is not built yet", file=sys.stderr)
        return 2
    log.mark(args.mode)
    return code


def _ingest() -> int:
    arts = [str(entry["art"]) for entry in load_catalog()]
    lines = ingest(arts)
    for line in lines:
        print(line)
    build_scene.cache_clear()
    missing = [line for line in lines if line.endswith(": missing")]
    print(f"cutouts ready {len(arts) - len(missing)}/{len(arts)}")
    return 0


def _facts() -> int:
    book = load_claims()
    results = evaluate(book)
    sizes = load_sizes()
    errors = display_errors(sizes)
    timeline = build_timeline()
    out = ROOT / "out"
    out.mkdir(parents=True, exist_ok=True)
    (out / "scale_facts.md").write_text(facts_markdown(timeline, sizes))
    (out / "scale_claims_report.md").write_text(render_report(book, results))
    failed = [claim_id for claim_id, ok, _detail in results if not ok]
    for claim_id, ok, detail in results:
        if not ok:
            print(f"FAIL {claim_id}: {detail}")
    for error in errors:
        print(error)
    print(f"claims {sum(1 for _, ok, _ in results if ok)}/{len(results)} passed")
    print(f"grid error {timeline.max_snap_error:.3e} frames")
    print(f"wrote {out / 'scale_facts.md'}")
    return 1 if failed or errors else 0


def _timeline() -> int:
    timeline = build_timeline()
    path = write_timeline(timeline)
    report = ROOT / "out" / "scale_retention_map.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(retention_markdown(timeline))
    payload = timeline.to_json()
    print(f"wrote {path}")
    print(f"lands {len(payload['lands'])} kicks {len(payload['kicks'])} bells {len(payload['bells'])} impacts {len(payload['impacts'])}")
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
    renderer = ScaleRenderer(width=1080, height=1920, hook=hook)
    folder = ROOT / "out" / "scale_stills"
    folder.mkdir(parents=True, exist_ok=True)
    thumbs = []
    for frame in range(0, 1824, 24):
        image = Image.fromarray(renderer.render(frame)[:, :, ::-1])
        thumbs.append(image.resize((135, 240), Image.Resampling.BOX))
        if frame and frame % 480 == 0:
            print(f"stills {frame}", flush=True)
    for frame in HERO_FRAMES:
        path = folder / f"f{frame:04d}.png"
        Image.fromarray(renderer.render(frame)[:, :, ::-1]).save(path)
    print(f"wrote {len(HERO_FRAMES)} stills in {folder}")
    _sheet(thumbs, 10, ROOT / "out" / "scale_contact_sheet.png")
    return 0


def _mix_files() -> tuple[Path, Path, float, float]:
    timeline = build_timeline()
    full, sfx, lufs, peak = mix(timeline.to_json(), seed=timeline.seed, n_frames=timeline.n_frames)
    folder = ROOT / "out"
    folder.mkdir(parents=True, exist_ok=True)
    music = folder / "scale.wav"
    off_audio, off_lufs, off_peak = trim_for_aac(*master(sfx))
    music_off = folder / "scale.music_off.wav"
    write_wav(str(music), full)
    write_wav(str(music_off), off_audio)
    rows = sync_rows(timeline.to_json())
    (folder / "scale_sync.md").write_text("# Sync\n\n" + "\n".join(rows) + "\n")
    onset = onset_sample(full)
    print(f"loudness {lufs:.2f} LUFS, true peak {peak:.2f} dBTP, onset sample {onset}")
    print(f"music-off {off_lufs:.2f} LUFS, true peak {off_peak:.2f} dBTP")
    print(f"wrote {music}")
    return music, music_off, lufs, peak


def _audio() -> int:
    _mix_files()
    return 0


def _encode(renderer: ScaleRenderer, indices: list[int], output: Path, audio: Path | None, fps: int, label: str, preset: str) -> None:
    def frames():
        for index, frame in enumerate(indices):
            if index % 120 == 0:
                print(f"{label} {index}/{len(indices)}", flush=True)
            yield renderer.render(frame)

    output.parent.mkdir(parents=True, exist_ok=True)
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
    small = ScaleRenderer(width=270, height=480, hook=hook)
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
    print(f"hook {hook}: motion {energy:.1f}, changes {changes}")


def _hooks() -> int:
    music, _off, _lufs, _peak = _mix_files()
    folder = ROOT / "out" / "scale_hooks"
    folder.mkdir(parents=True, exist_ok=True)
    spec = require_delivery("hooks", 540, 960, 30, 105)
    indices = frame_indices("hooks", 1824)
    for hook in ("A", "B", "D"):
        _hook_metrics(hook)
        renderer = ScaleRenderer(width=spec.width, height=spec.height, hook=hook)
        _encode(renderer, indices, folder / f"hook_{hook}.mp4", music, spec.fps, f"hook {hook}", "veryfast")
    strip = ScaleRenderer(width=360, height=640, hook="A")
    tiles = [Image.fromarray(strip.render(frame)[:, :, ::-1]) for frame in (0, 12, 30, 60, 120)]
    _sheet(tiles, len(tiles), folder / "thumb_strip.png")
    return 0


def _preview(hook: str, output: Path) -> int:
    music, _music_off, _lufs, _peak = _mix_files()
    spec = require_delivery("preview", 540, 960, 30, 912)
    renderer = ScaleRenderer(width=spec.width, height=spec.height, hook=hook)
    indices = frame_indices("preview", 1824)
    _encode(renderer, indices, output, music, spec.fps, "preview", "veryfast")
    return 0


def _full(hook: str, output: Path) -> int:
    music, music_off, _lufs, _peak = _mix_files()
    spec = require_delivery("full", 1080, 1920, 60, 1824)
    renderer = ScaleRenderer(width=spec.width, height=spec.height, hook=hook)
    heavy = renderer.profile_ms()
    print(f"profile: {heavy:.0f} ms; eta {heavy * 1824 / 60000:.1f} min", flush=True)
    indices = frame_indices("full", 1824)
    _encode(renderer, indices, output, music, spec.fps, "full", "slow")
    _encode(renderer, indices, output.with_name(output.stem + ".music_off.mp4"), music_off, spec.fps, "full music-off", "slow")
    return 0


def _postkit() -> int:
    book = load_claims()
    errors = lint_post(book)
    for error in errors:
        print(error)
    folder = ROOT / "out"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "scale_postkit.md"
    path.write_text(render_postkit(book))
    print(f"wrote {path}")
    renderer = ScaleRenderer(width=1080, height=1920)
    thumbs = folder / "scale_thumbs"
    thumbs.mkdir(parents=True, exist_ok=True)
    for name, frame in THUMBS:
        Image.fromarray(renderer.render(frame)[:, :, ::-1]).save(thumbs / f"{name}.png")
        print(f"wrote {thumbs / f'{name}.png'}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
