#!/usr/bin/env python3
"""Render a Flag Arena short.

python make_arena.py --cast configs/casts/world.yaml --out out/arena.mp4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path

import numpy as np

from fc_sat.arena_audio import render_arena_mix
from fc_sat.arena_config import load_arena_config, load_cast
from fc_sat.arena_render import (
    ArenaRenderer,
    build_timeline,
    contact_sheet,
    init_arena_worker,
    render_arena_chunk,
    write_sim_cache,
)
from fc_sat.arena_sim import (
    format_search_failure,
    gate_pass_rates,
    read_previous_winner,
    search_seeds,
    simulate,
)
from fc_sat.arena_voice import load_voice, write_announcer
from fc_sat.audio import write_wav
from fc_sat.encode import find_ffmpeg, iter_ordered_frames, pipe_raw_bgr, resolve_workers


def output_name(base: Path, cast_path: str, seed: int) -> Path:
    stem = Path(cast_path).stem
    return base.with_name(f"{base.stem}_{stem}_s{seed}{base.suffix}")


def _with_cast(cfg, cast_path: str, allow_sensitive: bool):
    countries, warnings = load_cast(cast_path, cfg.guards, allow_sensitive=allow_sensitive)
    return replace(
        cfg,
        countries=countries,
        cast_path=str(cast_path),
        cast_size=len(countries),
        sensitive_warnings=warnings,
    )


def _pick_hook(cfg):
    if cfg.hook_index is not None:
        index = int(cfg.hook_index)
    else:
        index = int(np.random.Generator(np.random.PCG64(cfg.seed)).integers(0, len(cfg.hooks)))
    return replace(cfg, hook=cfg.hooks[index], hook_index=index)


def _write_seeds(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    gates = list(rows[0]["gates"])
    parts = list(rows[0]["components"])
    lines = [",".join(["seed", "passed", *gates, *parts, "winner", "t_win"])]
    for row in rows:
        values = [
            str(row["seed"]),
            "1" if row["passed"] else "0",
            *[("1" if row["gates"][name] else "0") for name in gates],
            *[f"{row['components'][name]:.4f}" for name in parts],
            "" if row["winner"] is None else str(row["winner"]),
            "" if row["t_win"] is None else f"{row['t_win']:.4f}",
        ]
        lines.append(",".join(values))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _post_text(cfg, winner: str) -> str:
    return (
        f"{cfg.hook} #shorts\n\n"
        "32 countries. Same size. Same weight. Pure physics. Comment your country.\n\n"
        "#shorts #flags #countries #countryballs #satisfying #physics\n\n"
        f"Pinned comment: {winner} survived. Did yours?\n"
    )


def _sample_boxes(renderer: ArenaRenderer) -> list[list[int]]:
    moments = [0.0, 4.0, 10.0, 16.0, 22.0, renderer.timeline.winner_video, renderer.timeline.celebration_video + 0.3]
    boxes: list[list[int]] = []
    for moment in moments:
        renderer.render(renderer.frame_at(moment))
        boxes.extend([list(box) for box in renderer.boxes])
    return boxes


def _mux_audio_copy(video: Path, wav: Path, dest: Path) -> None:
    """Same picture, different audio. The video stream is copied."""
    import os
    import subprocess

    ffmpeg = find_ffmpeg()
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_name(f".{dest.stem}.partial.mp4")
    if temporary.exists():
        temporary.unlink()
    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video),
        "-i",
        str(wav),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-ar",
        "48000",
        "-ac",
        "2",
        "-shortest",
        "-movflags",
        "+faststart",
        str(temporary),
    ]
    proc = subprocess.run(command, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        if temporary.exists():
            temporary.unlink()
        raise SystemExit(f"sfx mux failed:\n{proc.stderr.strip()}")
    os.replace(temporary, dest)


def render_job(
    cfg,
    output: Path,
    *,
    preview: bool,
    verify: bool,
    keep_temp: bool,
    safe_overlay: bool,
    contact: bool,
    music: bool,
    voice: bool,
    rows: list[dict] | None,
) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    result = simulate(cfg, cfg.seed, record_trace=True)
    timeline = build_timeline(result, cfg)
    renderer = ArenaRenderer(cfg, result, preview=preview, safe_overlay=False)
    ms_frame = renderer.profile_ms()
    workers = 1 if renderer.n_frames < 4 else resolve_workers(cfg.workers)
    eta = (ms_frame / 1000.0) * renderer.n_frames / workers
    print(f"profile: {ms_frame:.1f} ms/frame; eta {eta / 60.0:.1f} min with {workers} workers", flush=True)
    boxes = _sample_boxes(renderer)
    cache = output.with_suffix(".sim.npz")
    write_sim_cache(cache, result)
    wav_path = output.with_suffix(".wav")
    sfx_wav = output.with_suffix(".sfx.wav")
    audio_path = None
    lufs = None
    true_peak = None
    if not preview:
        spoken = None
        if voice:
            spoken = load_voice(cfg, result, timeline, renderer.n_frames * 800)
        mix = render_arena_mix(
            cfg,
            result,
            renderer.n_frames,
            fps=renderer.fps,
            timeline=timeline,
            voice=spoken,
            music=music,
        )
        write_wav(str(wav_path), mix.full)
        write_wav(str(sfx_wav), mix.sfx_only)
        audio_path = wav_path
        lufs = mix.lufs
        true_peak = mix.true_peak
        print(f"audio: {lufs:.2f} LUFS, true peak {true_peak:.2f} dBTP", flush=True)
    write_announcer(output.with_suffix(".announcer.txt"), cfg, result, timeline)
    if rows:
        _write_seeds(output.with_suffix(".seeds.csv"), rows)

    def render_index(index: int) -> np.ndarray:
        return renderer.render(index)

    started = time.perf_counter()
    pipe_raw_bgr(
        iter_ordered_frames(
            renderer.n_frames,
            workers=workers,
            render_one=render_index,
            initializer=init_arena_worker,
            initargs=(cfg, str(cache), preview, False),
            render_chunk=render_arena_chunk,
        ),
        output,
        width=renderer.width,
        height=renderer.height,
        fps=renderer.fps,
        audio_path=audio_path,
        n_frames=renderer.n_frames if workers <= 1 else None,
    )
    if not preview and sfx_wav.exists():
        _mux_audio_copy(output, sfx_wav, output.with_name(output.stem + ".sfx_only.mp4"))
    if safe_overlay:
        pipe_raw_bgr(
            iter_ordered_frames(
                renderer.n_frames,
                workers=workers,
                render_one=render_index,
                initializer=init_arena_worker,
                initargs=(cfg, str(cache), preview, True),
                render_chunk=render_arena_chunk,
            ),
            output.with_name(output.stem + ".safe.mp4"),
            width=renderer.width,
            height=renderer.height,
            fps=renderer.fps,
            audio_path=None,
            n_frames=renderer.n_frames if workers <= 1 else None,
        )
    elapsed = time.perf_counter() - started
    if contact:
        contact_sheet(renderer, output.with_suffix(".contact.png"))
        print(f"contact sheet: {output.with_suffix('.contact.png')}", flush=True)
    winner = None if result.winner is None else cfg.countries[result.winner].name
    winner_code = None if result.winner is None else cfg.countries[result.winner].iso2
    slow = [[span.video0, span.video1] for span in timeline.spans if span.phase in {"slow", "replay"}]
    payload = {
        "seed": cfg.seed,
        "winner": winner_code,
        "winner_name": winner,
        "placements": [cfg.countries[index].iso2 for index in result.placements],
        "gates": {} if not rows else next((row["gates"] for row in rows if row["seed"] == cfg.seed), {}),
        "drama": {} if not rows else next((row["components"] for row in rows if row["seed"] == cfg.seed), {}),
        "config": cfg.to_public_dict(),
        "loudness": None if lufs is None else {"lufs": lufs, "true_peak": true_peak},
        "render_seconds": elapsed,
        "ms_per_frame": ms_frame,
        "credits": "flag-icons v7.5.0 MIT; Ohio cameo is a stylized burgee",
        "elims": [{"time": item.time, "iso2": cfg.countries[item.index].iso2, "cause": item.cause} for item in result.elims],
        "t_win": result.t_win,
        "first_elim": None if not result.elims else result.elims[0].time,
        "winner_video": timeline.winner_video,
        "celebration_video": timeline.celebration_video,
        "duration": timeline.duration,
        "text_boxes": boxes,
        "cast": [country.iso2 for country in cfg.countries],
        "slow_spans": slow,
        "pre_drop": [timeline.winner_video - cfg.pre_drop, timeline.winner_video],
        "preview": preview,
    }
    if not payload["gates"]:
        from fc_sat.arena_sim import evaluate_gates

        payload["gates"] = evaluate_gates(result, cfg)
    output.with_suffix(".json").write_text(json.dumps(payload, default=list), encoding="utf-8")
    output.with_suffix(".post.txt").write_text(_post_text(cfg, winner or "Nobody"), encoding="utf-8")
    if not preview and winner_code:
        history = Path(".cache/arena_history.json")
        history.parent.mkdir(parents=True, exist_ok=True)
        history.write_text(json.dumps({"winner": winner_code}), encoding="utf-8")
    if not keep_temp:
        for temp in (cache, wav_path, sfx_wav):
            if temp.exists():
                temp.unlink()
    print(f"wrote {output} in {elapsed:.1f}s ({ms_frame:.1f} ms/frame)", flush=True)
    if verify:
        from fc_sat.verify import format_table, verify_arena_file

        checks = verify_arena_file(output, cfg, payload)
        print(format_table(checks))
        if not all(item.ok for item in checks):
            raise SystemExit(f"verification failed for {output}")
    return payload


def _choose_seed(cfg, count: int, output: Path) -> tuple[int, list[dict]]:
    previous = read_previous_winner()
    workers = resolve_workers(cfg.workers)
    started = time.perf_counter()
    rows = search_seeds(cfg, count, workers=workers, previous_winner=previous)
    elapsed = time.perf_counter() - started
    passed = [row for row in rows if row["passed"]]
    print(f"search: {len(passed)}/{len(rows)} passed in {elapsed:.1f}s", flush=True)
    print(gate_pass_rates(rows), flush=True)
    ranked = sorted(passed, key=lambda row: (-row["components"]["score"], row["seed"]))
    print("top seeds:", flush=True)
    for row in ranked[:5]:
        twin = None if row["t_win"] is None else round(row["t_win"], 2)
        print(f"  {row['seed']} score {row['components']['score']:.2f} {row['winner']} t={twin}", flush=True)
    rate = len(passed) / len(rows) if rows else 0.0
    if rate < cfg.min_pass_rate:
        print(format_search_failure(rows, cfg.min_pass_rate), file=sys.stderr)
        _write_seeds(output.with_suffix(".seeds.csv"), rows)
        raise SystemExit(1)
    return ranked[0]["seed"], rows


def _parse(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a Flag Arena short")
    parser.add_argument("--config", default="configs/arena_default.yaml")
    parser.add_argument("--cast", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--search", type=int, default=None)
    parser.add_argument("--hook-index", type=int, default=None)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--no-music", action="store_true")
    parser.add_argument("--no-voice", action="store_true")
    parser.add_argument("--safe-overlay", action="store_true")
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--cast-list", default=None)
    parser.add_argument("--keep-temp", action="store_true")
    parser.add_argument("--contact-sheet", action="store_true")
    parser.add_argument("--allow-sensitive", action="store_true")
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args(argv)


def _one(cfg, base: Path, args, cast_path: str) -> None:
    cfg = _with_cast(cfg, cast_path, args.allow_sensitive)
    rows = None
    if args.seed is None:
        count = cfg.search if args.search is None else args.search
        chosen, rows = _choose_seed(cfg, count, output_name(base, cast_path, cfg.seed))
        cfg = replace(cfg, seed=chosen)
    else:
        cfg = replace(cfg, seed=args.seed)
    cfg = _pick_hook(cfg)
    output = output_name(base, cast_path, cfg.seed)
    verify = (not args.preview) or bool(args.verify)
    render_job(
        cfg,
        output,
        preview=args.preview,
        verify=verify and not args.preview or bool(args.verify),
        keep_temp=args.keep_temp,
        safe_overlay=args.safe_overlay,
        contact=args.contact_sheet,
        music=not args.no_music,
        voice=not args.no_voice and not args.preview,
        rows=rows,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    cfg = load_arena_config(args.config, allow_sensitive=args.allow_sensitive)
    if args.workers is not None:
        cfg = replace(cfg, workers=args.workers)
    if args.hook_index is not None:
        cfg = replace(cfg, hook_index=args.hook_index)
    casts = [args.cast or cfg.cast_path]
    if args.cast_list:
        casts = [item.strip() for item in args.cast_list.split(",") if item.strip()]
    if args.batch is not None:
        casts = casts[: args.batch]
    base = Path(args.out)
    failed = 0
    for cast_path in casts:
        try:
            _one(cfg, base, args, cast_path)
        except SystemExit as exc:
            failed += 1
            print(f"batch item failed: {cast_path}: {exc}", file=sys.stderr)
            if args.batch is None:
                return int(exc.code) if isinstance(exc.code, int) else 1
        except Exception:
            failed += 1
            print(f"batch item failed: {cast_path}", file=sys.stderr)
            traceback.print_exc()
            if args.batch is None:
                return 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
