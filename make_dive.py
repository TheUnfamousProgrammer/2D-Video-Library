#!/usr/bin/env python3
"""The deepest and highest we've ever reached: from the sea surface down to the bottom of the
deepest hole, then from sea level up to Voyager 1.

    python make_dive.py doctor
    python make_dive.py ingest
    python make_dive.py full --approved --out out/dive.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from fc_sat.audio import write_wav
from fc_sat.beatkit.delivery import frame_indices, require_delivery
from fc_sat.dive_render import DiveRenderer
from fc_sat.dive_scene import build_scene, load_values
from fc_sat.dive_timeline import build_timeline, facts_markdown, write_timeline
from fc_sat.dive_verify import art_errors, fact_errors, verify
from fc_sat.dive_world import load_stops
from fc_sat.encode import pipe_raw_bgr
from fc_sat.scale_art import ingest
from fc_sat.scale_audio import master, mix, onset_sample, trim_for_aac
from fc_sat.stage_log import Heartbeat, StageLog

ROOT = Path(__file__).resolve().parent
MODES = ("doctor", "ingest", "facts", "stills", "audio", "preview", "full", "verify")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Deepest to highest")
    parser.add_argument("mode", choices=MODES)
    parser.add_argument("--hook", default="A", choices=["A", "B", "D"])
    parser.add_argument("--out", default="")
    parser.add_argument("--approved", action="store_true")
    parser.add_argument("--allow-placeholders", action="store_true")
    args = parser.parse_args(argv)
    if args.mode == "full" and not args.approved:
        raise SystemExit("full render refused without --approved")
    if args.mode == "full" and not args.allow_placeholders and art_errors():
        raise SystemExit("full render refused: " + "; ".join(art_errors()))
    log = StageLog()
    code = {
        "doctor": _doctor,
        "ingest": _ingest,
        "facts": _facts,
        "stills": lambda: _stills(args.hook),
        "audio": lambda: (_mix(), 0)[1],
        "preview": lambda: _preview(args.hook, Path(args.out or "out/dive_preview.mp4")),
        "full": lambda: _full(args.hook, Path(args.out or "out/dive.mp4")),
        "verify": lambda: verify(Path(args.out) if args.out else None, allow_placeholders=args.allow_placeholders),
    }[args.mode]()
    log.mark(args.mode)
    return code


def _doctor() -> int:
    scene = build_scene()
    ready = sum(scene.has_art)
    print(f"stops {len(scene.stops)}, cutouts ready {ready}")
    for line in art_errors():
        print(line)
    return 0


def _ingest() -> int:
    arts = [str(entry["art"]) for entry in load_stops()]
    for line in ingest(arts):
        print(line)
    build_scene.cache_clear()
    return 0


def _facts() -> int:
    errors = fact_errors()
    (ROOT / "out").mkdir(exist_ok=True)
    (ROOT / "out" / "dive_facts.md").write_text(facts_markdown(build_timeline(), load_values()))
    write_timeline(build_timeline())
    for error in errors:
        print(error)
    print("facts ok" if not errors else f"{len(errors)} problems")
    return 1 if errors else 0


def _stills(hook: str) -> int:
    renderer = DiveRenderer(hook=hook)
    tiles = [Image.fromarray(renderer.render(f)[:, :, ::-1]).resize((135, 240), Image.Resampling.BOX) for f in range(0, 1824, 24)]
    sheet = Image.new("RGB", (135 * 10, 240 * ((len(tiles) + 9) // 10)), (14, 17, 23))
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % 10) * 135, (i // 10) * 240))
    sheet.save(ROOT / "out" / "dive_contact_sheet.png")
    print("wrote out/dive_contact_sheet.png")
    return 0


def _mix() -> tuple[Path, Path]:
    timeline = build_timeline()
    full, sfx, lufs, peak = mix(timeline.to_json(), seed=timeline.seed, n_frames=timeline.n_frames)
    music, music_off = ROOT / "out" / "dive.wav", ROOT / "out" / "dive.music_off.wav"
    write_wav(str(music), full)
    write_wav(str(music_off), trim_for_aac(*master(sfx))[0])
    print(f"loudness {lufs:.2f} LUFS, true peak {peak:.2f} dBTP, onset sample {onset_sample(full)}")
    return music, music_off


def _encode(renderer: DiveRenderer, indices: list[int], output: Path, audio: Path, fps: int, label: str, preset: str) -> None:
    def frames():
        for index, frame in enumerate(indices):
            if index % 240 == 0:
                print(f"{label} {index}/{len(indices)}", flush=True)
            yield renderer.render(frame)

    with Heartbeat(f"{label} still rendering"):
        pipe_raw_bgr(frames(), output, width=renderer.width, height=renderer.height, fps=fps, audio_path=audio, preset=preset, n_frames=len(indices))
    print(f"wrote {output}")


def _preview(hook: str, output: Path) -> int:
    music, _ = _mix()
    spec = require_delivery("preview", 540, 960, 30, 912)
    _encode(DiveRenderer(width=spec.width, height=spec.height, hook=hook), frame_indices("preview", 1824), output, music, spec.fps, "preview", "veryfast")
    return 0


def _full(hook: str, output: Path) -> int:
    music, music_off = _mix()
    spec = require_delivery("full", 1080, 1920, 60, 1824)
    renderer = DiveRenderer(width=spec.width, height=spec.height, hook=hook)
    indices = frame_indices("full", 1824)
    _encode(renderer, indices, output, music, spec.fps, "full", "slow")
    _encode(renderer, indices, output.with_name(output.stem + ".music_off.mp4"), music_off, spec.fps, "full music-off", "slow")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
