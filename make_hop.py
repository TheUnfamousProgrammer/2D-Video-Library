#!/usr/bin/env python3
"""Render a Melody Hop loop.

Examples:
  python make_hop.py --config configs/hop_default.yaml --song songs/ode_to_joy.yaml --out out/hop_ode.mp4
  python make_hop.py --config configs/hop_default.yaml --song songs/ode_to_joy.yaml --out out/hop_preview.mp4 --preview
  python make_hop.py --config configs/hop_default.yaml --song songs/ode_to_joy.yaml --out out/hop_batch.mp4 --batch 3 --vary palette,hook
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

from fc_sat.audio import write_wav
from fc_sat.config import _load_palettes
from fc_sat.encode import iter_ordered_frames, pipe_raw_bgr
from fc_sat.hop_audio import synthesize_hop
from fc_sat.hop_config import HopConfig, load_hop_config, with_hop_overrides
from fc_sat.hop_render import HopRenderer, init_hop_worker, render_hop_chunk
from fc_sat.song import load_song


def unique_hop_variants(
    n: int,
    *,
    palette: str,
    hook_index: int,
    octave: int,
    bpm: float,
    vary: str,
    palette_names: list[str],
    n_hooks: int,
) -> list[tuple[str, int, int, float]]:
    """Deterministic (palette, hook index, octave, bpm) tuples. No repeats."""
    if n < 1:
        raise SystemExit("--batch must be >= 1")
    vary_set = {part.strip() for part in vary.split(",") if part.strip()}
    unknown = vary_set - {"palette", "hook", "octave", "bpm"}
    if unknown:
        raise SystemExit(f"unknown --vary fields: {', '.join(sorted(unknown))}")
    if not vary_set:
        raise SystemExit("--vary must include at least one of palette, hook, octave, bpm")
    palettes = list(palette_names) if "palette" in vary_set else [palette]
    hooks = list(range(n_hooks)) if "hook" in vary_set else [hook_index]
    if "octave" in vary_set:
        octaves: list[int] = []
        for delta in (0, 1, -1, 2, -2):
            value = octave + delta
            if -4 <= value <= 4 and value not in octaves:
                octaves.append(value)
    else:
        octaves = [octave]
    if "bpm" in vary_set:
        bpms = [float(bpm), float(bpm) + 10.0, float(bpm) - 10.0]
    else:
        bpms = [float(bpm)]
    if not palettes or not hooks or not octaves or not bpms:
        raise SystemExit("no variants available for this batch")
    found: list[tuple[str, int, int, float]] = []
    seen: set[tuple[str, int, int, float]] = set()
    span = len(palettes) * len(hooks) * len(octaves) * len(bpms)
    for index in range(span):
        palette_i = index % len(palettes)
        hook_i = (index // len(palettes)) % len(hooks)
        octave_i = (index // (len(palettes) * len(hooks))) % len(octaves)
        bpm_i = index // (len(palettes) * len(hooks) * len(octaves))
        key = (palettes[palette_i], hooks[hook_i], octaves[octave_i], bpms[bpm_i])
        if key not in seen:
            seen.add(key)
            found.append(key)
        if len(found) == n:
            return found
    raise SystemExit(f"only {len(found)} unique hop variants are available for --vary {vary}")


def variant_name(
    output: Path,
    song_stem: str,
    palette: str,
    hook_index: int,
    *,
    octave: int | None = None,
    bpm: float | None = None,
) -> Path:
    parts = [output.stem, song_stem, palette, str(hook_index)]
    extra = []
    if octave is not None:
        extra.append(f"oct{octave}")
    if bpm is not None:
        extra.append(f"bpm{int(round(bpm))}")
    if extra:
        parts.append("_".join(extra))
    return output.with_name("_".join(parts) + output.suffix)


def _hook_index(cfg: HopConfig) -> int:
    try:
        return cfg.hooks.index(cfg.hook)
    except ValueError:
        return 0


def _palette_names(cfg_path: Path) -> list[str]:
    return list(_load_palettes(cfg_path.parent / "palettes.yaml").keys())


def _apply_variant(cfg: HopConfig, palette: str, hook_index: int, octave: int, bpm: float, base_bpm: float) -> HopConfig:
    stops = _load_palettes(Path("configs/palettes.yaml"))[palette]
    hook = cfg.hooks[hook_index]
    changes: dict = {
        "palette": palette,
        "palette_stops": stops,
        "hook": hook,
        "octave_shift": octave,
    }
    if abs(bpm - base_bpm) > 1e-6:
        changes["bpm"] = bpm
    else:
        changes["bpm"] = cfg.bpm
    return with_hop_overrides(cfg, **changes)


def _write_sidecar(output: Path, cfg: HopConfig, song_title: str, composer: str, stats: dict) -> None:
    payload = {"config": cfg.to_public_dict(), "stats": stats}
    output.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    output.with_suffix(".post.txt").write_text(
        f"{cfg.hook} #shorts\n\n"
        "#shorts #guessthesong #satisfying #music #asmr\n\n"
        f"Answer: {song_title} ({composer})\n",
        encoding="utf-8",
    )


def render_one(cfg: HopConfig, output: Path, *, preview: bool, verify: bool, keep_temp: bool) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    renderer = HopRenderer(cfg, preview=preview)
    song = renderer.dance.song
    wav_path = output.with_suffix(".wav")
    lufs = None
    true_peak = None
    audio_path = None
    if not preview:
        audio, lufs, true_peak = synthesize_hop(
            renderer.dance,
            seed=cfg.seed,
            reverb_wet=cfg.reverb_wet,
            rt60=cfg.rt60,
            audio_offset_ms=cfg.audio_offset_ms,
            repeats=cfg.repeats,
        )
        write_wav(str(wav_path), audio)
        audio_path = wav_path
        print(f"audio: {lufs:.2f} LUFS, true peak {true_peak:.2f} dBTP")
    started = time.perf_counter()
    profile_frame = renderer.render(0)
    ms_frame = (time.perf_counter() - started) * 1000.0
    workers = 1 if renderer.n_frames < 4 else max(1, cfg.workers)
    eta = (ms_frame / 1000.0) * renderer.n_frames / workers
    print(f"profile: {ms_frame:.1f} ms/frame; eta {eta / 60.0:.1f} min with {workers} workers")

    def render_index(index: int):
        if index == 0:
            return profile_frame
        return renderer.render(index)

    encode_started = time.perf_counter()
    try:
        pipe_raw_bgr(
            iter_ordered_frames(
                renderer.n_frames,
                workers=workers,
                render_one=render_index,
                initializer=init_hop_worker,
                initargs=(cfg, preview),
                render_chunk=render_hop_chunk,
            ),
            output,
            width=renderer.width,
            height=renderer.height,
            fps=renderer.fps,
            audio_path=audio_path,
            n_frames=renderer.n_frames if workers <= 1 else None,
        )
    finally:
        if audio_path is not None and audio_path.exists() and not keep_temp:
            audio_path.unlink()
    elapsed = time.perf_counter() - encode_started
    stats = {
        "n_frames": int(renderer.dance.grid.n_frames),
        "repeats": int(cfg.repeats),
        "note_count": len(song.notes),
        "distinct_pads": len(renderer.dance.layout.pads),
        "loop_seconds": float(renderer.dance.grid.duration),
        "lufs": None if lufs is None else float(lufs),
        "true_peak_dbtp": None if true_peak is None else float(true_peak),
        "first_onset_sample": int(renderer.dance.grid.onset_samples[0]),
        "render_seconds": float(elapsed),
        "ms_per_frame": float(ms_frame),
        "preview": preview,
    }
    _write_sidecar(output, cfg, song.title, song.composer, stats)
    print(f"wrote {output} in {elapsed:.1f}s ({ms_frame:.1f} ms/frame)")
    if verify:
        from fc_sat.verify import format_table, verify_hop_file

        checks = verify_hop_file(output, cfg)
        print(format_table(checks))
        if not all(item.ok for item in checks):
            raise SystemExit(f"verification failed for {output}")
    return stats


def _parse(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a Melody Hop loop")
    parser.add_argument("--config", required=True)
    parser.add_argument("--song", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--preview", action="store_true", help="540x960, 30 fps, no audio")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--vary", default="palette,hook")
    parser.add_argument("--verify", action="store_true", help="verify after render (default for full renders)")
    parser.add_argument("--keep-temp", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    cfg = load_hop_config(args.config)
    if args.song:
        cfg = with_hop_overrides(cfg, song=args.song)
    if args.seed is not None:
        cfg = with_hop_overrides(cfg, seed=args.seed)
    if args.workers is not None:
        cfg = with_hop_overrides(cfg, workers=args.workers)
    if not Path(cfg.song).exists():
        raise SystemExit(f"config field 'song' does not exist: {cfg.song}")
    verify = True if not args.preview else bool(args.verify)
    output = Path(args.out)
    if args.batch:
        song = load_song(cfg.song, octave_shift=cfg.octave_shift, bpm=cfg.bpm)
        base_bpm = float(song.bpm)
        names = _palette_names(Path(args.config))
        combos = unique_hop_variants(
            args.batch,
            palette=cfg.palette,
            hook_index=_hook_index(cfg),
            octave=cfg.octave_shift,
            bpm=base_bpm,
            vary=args.vary,
            palette_names=names,
            n_hooks=len(cfg.hooks),
        )
        vary_set = {part.strip() for part in args.vary.split(",") if part.strip()}
        failed = 0
        song_stem = Path(cfg.song).stem
        for palette, hook_index, octave, bpm in combos:
            extra_octave = octave if "octave" in vary_set else None
            extra_bpm = bpm if "bpm" in vary_set else None
            target = variant_name(
                output,
                song_stem,
                palette,
                hook_index,
                octave=extra_octave,
                bpm=extra_bpm,
            )
            try:
                variant = _apply_variant(cfg, palette, hook_index, octave, bpm, base_bpm)
                print(f"batch: {target.name}")
                render_one(variant, target, preview=args.preview, verify=verify, keep_temp=args.keep_temp)
            except KeyboardInterrupt:
                raise
            except (Exception, SystemExit) as exc:
                failed += 1
                print(f"variant failed {target.name}: {exc}", file=sys.stderr)
                traceback.print_exc()
        if failed:
            print(f"{failed} of {len(combos)} variants failed", file=sys.stderr)
            return 1
        return 0
    render_one(cfg, output, preview=args.preview, verify=verify, keep_temp=args.keep_temp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
