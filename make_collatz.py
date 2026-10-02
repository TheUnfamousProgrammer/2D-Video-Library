#!/usr/bin/env python3
"""Collatz conjecture short.

    python make_collatz.py doctor
    python make_collatz.py preview --hook A --out out/collatz_preview.mp4
    python make_collatz.py full --approved --out out/collatz.mp4

`full` is refused unless `--approved` is also passed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml
from PIL import Image

from fc_sat.audio import write_wav
from fc_sat.collatz_audio import mix
from fc_sat.collatz_bound import fetch_page
from fc_sat.collatz_claims import evaluate, load_claims, render_report
from fc_sat.collatz_config import load_yaml
from fc_sat.collatz_doctor import doctor
from fc_sat.collatz_encode import write_mp4
from fc_sat.collatz_layout import Film
from fc_sat.collatz_post import render_postkit
from fc_sat.collatz_render import CollatzRenderer
from fc_sat.collatz_script import estimate_characters, lint_script, load_script, script_markdown
from fc_sat.collatz_timeline import (
    build_timeline,
    facts_markdown,
    retention_markdown,
    retention_rows,
)
from fc_sat.collatz_verify import style_law_errors, verify_movie
from fc_sat.collatz_voice import (
    apply_pronunciation,
    key_present,
    load_pronunciation,
    parse_selection,
    save_picks,
    stem_from_dir,
    write_audition,
    write_manual_script,
)
from fc_sat.stage_log import Heartbeat, StageLog

ROOT = Path(__file__).resolve().parent


def _film(hook: str = "A"):
    return Film(build_timeline(hook=hook), load_script(), load_claims())


def _hash_inputs(path: Path) -> None:
    names = [
        "configs/claims.yaml",
        "configs/collatz.yaml",
        "configs/collatz_script.yaml",
        "configs/collatz_timeline.yaml",
        "configs/collatz_hooks.yaml",
        "configs/collatz_pronunciation.yaml",
        "configs/vo_picks.yaml",
    ]
    payload = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names if (ROOT / name).exists()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _write_facts(fetch: bool) -> int:
    book = load_claims()
    html = None
    if fetch:
        try:
            html = fetch_page()
        except Exception as exc:
            print(f"bound recheck failed ({exc}); keeping the stored bound")
    results, note = evaluate(book, html=html, fetch=False)
    print(note)
    out = ROOT / "out"
    out.mkdir(parents=True, exist_ok=True)
    (out / "facts.md").write_text(facts_markdown(book))
    (out / "claims_report.md").write_text(render_report(book, results))
    lines = load_script()
    errors = lint_script(lines, book)
    (out / "script.md").write_text(script_markdown(lines))
    timeline = build_timeline()
    (out / "retention_map.md").write_text(retention_markdown(retention_rows(timeline, lines)))
    _hash_inputs(out / "collatz_inputs.json")
    if errors or any(not ok for _, ok, _ in results):
        for error in errors:
            print(error)
        return 1
    print(f"claims {sum(1 for _, ok, _ in results if ok)}/{len(results)} passed")
    return 0


def _frames(film: Film, width: int, height: int, fps: int, limit: int | None = None):
    renderer = CollatzRenderer(film, width=width, height=height, fps=fps)
    count = renderer.n_frames if limit is None else min(renderer.n_frames, limit)
    ms = renderer.profile_ms()
    frames = film.timeline.n_frames
    sequential = (renderer.paint_ms * frames + renderer.heavy_ms) / 1000.0
    print(
        f"profile: {ms:.0f} ms heavy frame, {renderer.paint_ms:.0f} ms paint at {width}x{height}; "
        f"sequential full-render eta {sequential / 60:.1f} min",
        flush=True,
    )

    def generate():
        for index in range(count):
            if index % 60 == 0:
                print(f"frame {index}/{count}", flush=True)
            yield renderer.render(index)

    return generate(), count


def _contact(film: Film, path: Path) -> None:
    renderer = CollatzRenderer(film, width=270, height=480, fps=30)
    tiles = []
    for when in (0, 2.6, 6.5, 10.8, 13.3, 19.72, 25.5, 29.5):
        index = min(renderer.n_frames - 1, int(round(when * 30)))
        tiles.append(Image.fromarray(renderer.render(index)[:, :, ::-1]))
    sheet = Image.new("RGB", (270 * len(tiles), 480))
    for index, tile in enumerate(tiles):
        sheet.paste(tile, (index * 270, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def _thumbs(film: Film, folder: Path) -> None:
    renderer = CollatzRenderer(film, width=1080, height=1920, fps=60)
    folder.mkdir(parents=True, exist_ok=True)
    for name, when in (("twist", 13.35), ("peak", 19.72), ("funnel", 26.2)):
        index = min(film.timeline.n_frames - 1, int(round(when * 60)))
        Image.fromarray(renderer.render(index)[:, :, ::-1]).save(folder / f"{name}.png")
        print(f"thumb {name}", flush=True)


def _thumb_strip(film: Film, path: Path) -> None:
    renderer = CollatzRenderer(film, width=360, height=640, fps=30)
    tiles = []
    for index in range(0, min(renderer.n_frames, 60), 15):
        tiles.append(Image.fromarray(renderer.render(index)[:, :, ::-1]))
    sheet = Image.new("RGB", (360 * len(tiles), 640))
    for index, tile in enumerate(tiles):
        sheet.paste(tile, (index * 360, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collatz conjecture short")
    parser.add_argument(
        "mode",
        choices=["doctor", "facts", "script", "vo", "vo-select", "timeline", "animatic", "hooks", "audio", "preview", "full", "postkit", "verify"],
    )
    parser.add_argument("--hook", default="A", choices=["A", "B", "C"])
    parser.add_argument("--out", default="")
    parser.add_argument("--approved", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--vo-dir", default="")
    parser.add_argument("selection", nargs="*")
    args = parser.parse_args(argv)
    if not args.out:
        args.out = {
            "animatic": "out/collatz_animatic.mp4",
            "preview": "out/collatz_preview.mp4",
        }.get(args.mode, "out/collatz.mp4")
    log = StageLog()
    if args.mode == "full" and not args.approved:
        raise SystemExit("full render refused without --approved")
    if args.mode == "doctor":
        code = doctor()
        log.mark("doctor")
        return code
    if args.mode == "facts":
        code = _write_facts(fetch=True)
        log.mark("facts")
        return code
    if args.mode == "script":
        book = load_claims()
        lines = load_script()
        errors = lint_script(lines, book)
        path = ROOT / "out" / "script.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(script_markdown(lines))
        print(f"estimated characters for 3 takes: {estimate_characters(lines)}")
        for error in errors:
            print(error)
        log.mark("script")
        return 1 if errors else 0
    if args.mode == "vo":
        lines = load_script()
        pron = load_pronunciation()
        print(f"estimated characters for 3 takes: {estimate_characters(lines)}")
        print(f"pronunciation sample: {apply_pronunciation(lines[-1].tts_text, pron['picks'], pron['variants'])}")
        write_manual_script(
            lines,
            "No model id is hardcoded. Call GET /v1/models and pick the single v4 text-to-speech model.",
            ROOT / "out" / "script_for_manual_tts.md",
        )
        write_audition(lines, ROOT / "out" / "vo" / "audition.html")
        if args.dry_run or not key_present():
            print("wrote out/script_for_manual_tts.md and out/vo/audition.html")
        log.mark("vo")
        return 0
    if args.mode == "vo-select":
        save_picks(parse_selection(args.selection))
        print("updated configs/vo_picks.yaml")
        log.mark("vo-select")
        return 0
    if args.mode == "timeline":
        timeline = build_timeline(hook=args.hook)
        text = retention_markdown(retention_rows(timeline, load_script()))
        path = ROOT / "out" / "retention_map.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        print(text)
        log.mark("timeline")
        return 0
    film = _film(args.hook)
    if args.mode == "postkit":
        text, errors = render_postkit(film.book, reply=bool(load_yaml("collatz.yaml").get("reply_commitment", True)))
        path = ROOT / "out" / "postkit.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        _thumbs(film, ROOT / "out" / "thumbs")
        for error in errors:
            print(error)
        log.mark("postkit")
        return 1 if errors else 0
    if args.mode == "animatic":
        with Heartbeat("animatic working"):
            frames, count = _frames(film, 540, 960, 30)
            write_mp4(frames, Path(args.out), width=540, height=960, fps=30, audio_path=None, preset="veryfast", n_frames=count)
            _contact(film, Path(args.out).with_suffix(".contact.png"))
        log.mark("animatic")
        return 0
    if args.mode == "hooks":
        hooks = yaml.safe_load((ROOT / "configs" / "collatz_hooks.yaml").read_text())
        folder = ROOT / "out" / "hooks"
        folder.mkdir(parents=True, exist_ok=True)
        with Heartbeat("hooks working"):
            for key, body in hooks["variants"].items():
                variant = _film(key)
                variant.hook_captions = [(float(item["t"]), item["text"]) for item in body["captions"]]
                frames, count = _frames(variant, 540, 960, 30, limit=int(round(3.5 * 30)))
                write_mp4(frames, folder / f"hook_{key}.mp4", width=540, height=960, fps=30, audio_path=None, preset="veryfast", n_frames=count)
                _thumb_strip(variant, folder / f"hook_{key}_thumb.png")
                print(f"hook {key} {body['name']}")
        log.mark("hooks")
        return 0
    if args.mode in {"audio", "preview", "full"}:
        with Heartbeat(f"{args.mode} working"):
            voice = None
            if args.vo_dir:
                voice = stem_from_dir(
                    Path(args.vo_dir),
                    load_script(),
                    film.timeline.n_frames * 800,
                    shift_s=film.timeline.shifted_by,
                )
            audio, lufs, true_peak, sync = mix(film.timeline, voice)
            wav = Path(args.out).with_suffix(".wav")
            wav.parent.mkdir(parents=True, exist_ok=True)
            write_wav(str(wav), audio)
            print(f"loudness {lufs:.2f} LUFS  true peak {true_peak:.2f} dBTP  samples {len(audio)}")
            wav.with_suffix(".sync.json").write_text(json.dumps(sync, indent=2))
            log.mark("audio")
            if args.mode == "audio":
                return 0
            width, height, fps = (1080, 1920, 60) if args.mode == "full" else (540, 960, 30)
            preset = "slow" if args.mode == "full" else "veryfast"
            frames, count = _frames(film, width, height, fps)
            write_mp4(frames, Path(args.out), width=width, height=height, fps=fps, audio_path=wav, preset=preset, n_frames=count)
        log.mark(args.mode)
        return 0
    if args.mode == "verify":
        errors = list(style_law_errors())
        movie = Path(args.out)
        if movie.exists():
            errors.extend(verify_movie(movie))
        else:
            print(f"no movie at {movie}; checked the renderer source only")
        for error in errors:
            print(f"FAIL {error}")
        log.mark("verify")
        return 1 if errors else 0
    return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        raise
