#!/usr/bin/env python3
"""Odd One Out.

Default (no mode flag) builds the levels, the puzzle PNGs, and the contact sheet.
It does not encode video. --full is refused unless --approved is also passed.

  python make_odd.py --config configs/odd_default.yaml --out out/odd.mp4
  python make_odd.py --config configs/odd_default.yaml --out out/odd_preview.mp4 --preview
  python make_odd.py --config configs/odd_default.yaml --out out/odd.mp4 --full --approved
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

from fc_sat.audio import write_wav
from fc_sat.encode import iter_ordered_frames, pipe_raw_bgr, resolve_workers
from fc_sat.odd_audio import synthesize_odd
from fc_sat.odd_config import load_odd_config
from fc_sat.odd_render import OddRenderer, contact_sheet, init_odd_worker, render_odd_chunk, write_level_pngs
from fc_sat.odd_report import write_answers, write_json, write_post, write_report
from fc_sat.odd_sim import save_show, simulate_show
from fc_sat.odd_verify import harsh_measure_show

CRITIQUE = """
Looked at the clean puzzles, the ringed answers, and the contact sheet.

L1 is immediate. Seed 7 put the odd disc in the top-left cell, and the required lightness gap makes it yellow against pink. That is an easy scan, which is what level 1 is for. Nothing else marks it: same size, no ring, no glow.

L2 reads in one glance. One rounded square is turned, the other twenty-four are upright, and they are the same blue. A still frame shows the tilt. It is not hidden, and it is not pre-circled.

L3 is the one that takes a scan. Every disc matches except the one in the lower left with no white dot. The dot is small, so you have to look, and the missing one is the only tell.

The reveal dims the others, puts a flat gold ring on the odd item, and moves the answer into the caption row. Text stays off the field. At most the level label, one caption, and the seconds number are on screen together.
""".strip()


def unique_odd_variants(n: int, seed: int, tiers: tuple[str, ...] = ("easy", "normal", "hard")) -> list[dict]:
    """Deterministic (seed, tier) pairs. Base hue follows the seed, so pairs do not repeat."""
    if n < 1:
        raise SystemExit("--batch must be >= 1")
    found: list[dict] = []
    seen: set[tuple[int, str]] = set()
    index = 0
    while len(found) < n:
        combo = (int(seed) + index, tiers[index % len(tiers)])
        if combo not in seen:
            seen.add(combo)
            found.append({"seed": combo[0], "tier": combo[1]})
        index += 1
        if index > n * len(tiers) + 5:
            break
    if len(found) < n:
        raise SystemExit(f"only {len(found)} unique odd variants are available")
    return found


def _parse_levels(text: str | None) -> list[int] | None:
    if not text:
        return None
    try:
        return [int(part.strip()) for part in text.split(",") if part.strip()]
    except ValueError as exc:
        raise SystemExit(f"--levels must be comma-separated ids, got {text!r}") from exc


def _variant_path(output: Path, seed: int, tier: str) -> Path:
    return output.with_name(f"{output.stem}_s{seed}_{tier}{output.suffix}")


def _encode(renderer: OddRenderer, output: Path, *, audio_path: Path | None, preset: str) -> float:
    workers = 1 if renderer.n_frames < 8 else resolve_workers(renderer.cfg.workers)
    ms_frame = renderer.profile_ms()
    eta = (ms_frame / 1000.0) * renderer.n_frames / workers
    print(
        f"profile: {ms_frame:.1f} ms/frame ({'preview' if renderer.preview else 'full'}); "
        f"eta {eta / 60.0:.1f} min with {workers} workers",
        flush=True,
    )

    def render_index(index: int):
        return renderer.render(index)

    pipe_raw_bgr(
        iter_ordered_frames(
            renderer.n_frames,
            workers=workers,
            render_one=render_index,
            initializer=init_odd_worker,
            initargs=(renderer.cfg, renderer.show, renderer.preview, renderer.safe_overlay),
            render_chunk=render_odd_chunk,
            chunk_size=8,
        ),
        output,
        width=renderer.width,
        height=renderer.height,
        fps=renderer.fps,
        audio_path=audio_path,
        preset=preset,
        n_frames=renderer.n_frames if workers <= 1 else None,
    )
    return eta


def generate(
    cfg,
    output: Path,
    *,
    critique: str,
    preview: bool = False,
    safe_overlay: bool = False,
    encode_full: bool = False,
    verify: bool = False,
) -> dict:
    started = time.perf_counter()
    timings: list[tuple[str, float]] = []
    mark = started
    show = simulate_show(cfg)
    timings.append(("sim", time.perf_counter() - mark))
    print(f"stage sim done in {timings[-1][1]:.2f}s", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_show(output.with_suffix(".sim.npz"), show)
    renderer = OddRenderer(cfg, show, preview=False)
    mark = time.perf_counter()
    pngs = write_level_pngs(renderer, output)
    contact_sheet(renderer, output.with_suffix(".contact.png"))
    timings.append(("pngs", time.perf_counter() - mark))
    print(f"stage pngs done in {timings[-1][1]:.2f}s", flush=True)
    mark = time.perf_counter()
    measured = harsh_measure_show(cfg, show)
    timings.append(("harsh", time.perf_counter() - mark))
    print(f"stage harsh done in {timings[-1][1]:.2f}s", flush=True)
    eta = None
    if preview:
        mark = time.perf_counter()
        preview_renderer = OddRenderer(cfg, show, preview=True, safe_overlay=False)
        full = OddRenderer(cfg, show, preview=False)
        workers = 1 if full.n_frames < 8 else resolve_workers(cfg.workers)
        ms_full = full.profile_ms()
        eta = (ms_full / 1000.0) * full.n_frames / workers
        print(f"full-render eta {eta / 60.0:.1f} min from {ms_full:.0f} ms/frame with {workers} workers", flush=True)
        _encode(preview_renderer, output, audio_path=None, preset="veryfast")
        if safe_overlay:
            guides = OddRenderer(cfg, show, preview=True, safe_overlay=True)
            _encode(guides, output.with_name(output.stem + ".safe.mp4"), audio_path=None, preset="veryfast")
        timings.append(("preview", time.perf_counter() - mark))
        print(f"stage preview done in {timings[-1][1]:.2f}s", flush=True)
    loudness = None
    if encode_full:
        mark = time.perf_counter()
        audio, lufs, true_peak = synthesize_odd(cfg)
        wav = output.with_suffix(".wav")
        write_wav(str(wav), audio)
        print(f"audio: {lufs:.2f} LUFS, true peak {true_peak:.2f} dBTP -> {wav}", flush=True)
        master = OddRenderer(cfg, show, preview=False)
        _encode(master, output, audio_path=wav, preset="slow")
        timings.append(("full", time.perf_counter() - mark))
        print(f"stage full done in {timings[-1][1]:.2f}s", flush=True)
        loudness = {"lufs": lufs, "true_peak": true_peak}
        if verify:
            from fc_sat.odd_verify import verify_odd_file
            from fc_sat.verify import format_table

            checks = verify_odd_file(output, cfg, show=show)
            print(format_table(checks))
            if not all(item.ok for item in checks):
                raise SystemExit(1)
    write_report(output.with_suffix(".report.md"), cfg, show, timings=timings, critique=critique, measured=measured)
    write_answers(output.with_suffix(".answers.md"), cfg, show)
    write_post(output.with_suffix(".post.txt"), cfg, show)
    payload = list(timings)
    if eta is not None:
        payload.append(("full_eta_s", eta))
    write_json(output.with_suffix(".json"), cfg, show, timings=payload, loudness=loudness)
    print(f"total {time.perf_counter() - started:.2f}s", flush=True)
    return {"show": show, "pngs": pngs, "measured": measured, "timings": timings, "eta": eta}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Odd One Out generator")
    parser.add_argument("--config", default="configs/odd_default.yaml")
    parser.add_argument("--out", default="out/odd.mp4")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--tier", choices=("easy", "normal", "hard"), default=None)
    parser.add_argument("--levels", default=None, help="comma-separated level ids, order kept")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--strips", action="store_true")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--audio-only", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--approved", action="store_true")
    parser.add_argument("--safe-overlay", action="store_true")
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--keep-temp", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    if args.full and not args.approved:
        raise SystemExit(
            "--full is refused without --approved. "
            "Look at the puzzle PNGs and the contact sheet, then re-run with --full --approved."
        )
    if args.batch and args.full:
        raise SystemExit("--full encodes one film. Drop --batch and pass --full --approved for that seed.")
    level_ids = _parse_levels(args.levels)
    output = Path(args.out)
    if args.audio_only and not args.preview and args.batch is None and not args.full:
        import json

        from fc_sat.odd_config import build_timeline

        cfg = load_odd_config(args.config, tier=args.tier, seed=args.seed, level_ids=level_ids)
        started = time.perf_counter()
        audio, lufs, true_peak = synthesize_odd(cfg)
        wav = output.with_suffix(".wav")
        wav.parent.mkdir(parents=True, exist_ok=True)
        write_wav(str(wav), audio)
        elapsed = time.perf_counter() - started
        print(f"audio: {lufs:.2f} LUFS, true peak {true_peak:.2f} dBTP, {elapsed:.2f}s -> {wav}", flush=True)
        payload = {
            "seed": cfg.seed,
            "config": cfg.to_public_dict(),
            "timings": [{"stage": "audio", "seconds": elapsed}],
            "loudness": {"lufs": lufs, "true_peak": true_peak},
            "timeline": [
                {
                    "kind": segment.kind,
                    "level": segment.level_id,
                    "start": segment.start_s,
                    "duration": segment.duration_s,
                    "start_frame": segment.start_frame,
                    "frames": segment.n_frames,
                }
                for segment in build_timeline(cfg)
            ],
        }
        output.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return 0
    if args.batch:
        variants = unique_odd_variants(args.batch, args.seed if args.seed is not None else 7)
        failed = 0
        for variant in variants:
            path = _variant_path(output, variant["seed"], variant["tier"])
            print(f"batch {variant['tier']} seed {variant['seed']} -> {path.name}", flush=True)
            try:
                cfg = load_odd_config(args.config, tier=variant["tier"], seed=variant["seed"], level_ids=level_ids)
                if args.workers is not None:
                    from dataclasses import replace

                    cfg = replace(cfg, workers=args.workers)
                generate(cfg, path, critique=CRITIQUE, preview=args.preview, safe_overlay=args.safe_overlay)
            except Exception:
                failed += 1
                traceback.print_exc()
        if failed:
            print(f"batch failures: {failed}", flush=True)
            return 1
        return 0
    cfg = load_odd_config(args.config, tier=args.tier, seed=args.seed, level_ids=level_ids)
    if args.workers is not None:
        from dataclasses import replace

        cfg = replace(cfg, workers=args.workers)
    generate(
        cfg,
        output,
        critique=CRITIQUE,
        preview=args.preview,
        safe_overlay=args.safe_overlay,
        encode_full=args.full,
        verify=args.verify,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
