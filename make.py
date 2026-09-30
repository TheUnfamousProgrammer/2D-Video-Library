#!/usr/bin/env python3
"""Render a vertical oddly satisfying Short.

Examples:
  python make.py --config configs/default.yaml --out out/a.mp4
  python make.py --config configs/default.yaml --out out/preview.mp4 --preview
  python make.py --config configs/default.yaml --out out/batch.mp4 --batch 4 --vary seed,palette
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

from fc_sat.audio import synthesize, write_wav
from fc_sat.config import Config, load_config, with_overrides
from fc_sat.encode import encode_frames
from fc_sat.render import Renderer, geometric_fill, pixel_coverage
from fc_sat.sim import cache_key, load_or_simulate, validate_result
from fc_sat.verify import format_table, verify_file


def unique_variants(
    n: int,
    seed: int,
    palette: str,
    hook_index: int,
    vary: str,
    palette_names: list[str],
    n_hooks: int,
) -> list[tuple[int, str, int]]:
    """Deterministic (seed, palette, hook index) tuples with no repeated combination."""
    if n < 1:
        raise SystemExit("--batch must be >= 1")
    vary_set = {part.strip() for part in vary.split(",") if part.strip()}
    unknown = vary_set - {"seed", "palette", "hook"}
    if unknown:
        raise SystemExit(f"unknown --vary fields: {', '.join(sorted(unknown))}")
    if not vary_set:
        raise SystemExit("--vary must include at least one of seed, palette, hook")
    palettes = list(palette_names) if "palette" in vary_set else [palette]
    hooks = list(range(n_hooks)) if "hook" in vary_set else [hook_index]
    if not palettes or not hooks:
        raise SystemExit("no palettes or hooks available for this batch")
    found: list[tuple[int, str, int]] = []
    seen: set[tuple[int, str, int]] = set()
    index = 0
    while len(found) < n and index < 100000:
        palette_i = index % len(palettes)
        hook_i = (index // len(palettes)) % len(hooks)
        seed_i = index // (len(palettes) * len(hooks))
        if "seed" not in vary_set and seed_i > 0:
            break
        this_seed = seed + seed_i if "seed" in vary_set else seed
        key = (this_seed, palettes[palette_i], hooks[hook_i])
        if key not in seen:
            seen.add(key)
            found.append(key)
        index += 1
    if len(found) < n:
        raise SystemExit(
            f"only {len(found)} unique (seed, palette, hook) combinations "
            f"are available for --vary {vary}"
        )
    return found


def variant_path(output: Path, seed: int, palette: str, hook_index: int) -> Path:
    return output.with_name(f"{output.stem}_{seed}_{palette}_{hook_index}{output.suffix}")


def _hook_index(cfg: Config) -> int:
    try:
        return cfg.hooks.index(cfg.hook)
    except ValueError:
        return 0


def _palettes_path() -> Path:
    return Path(__file__).resolve().parent / "configs" / "palettes.yaml"


def _palette_names(cfg: Config) -> list[str]:
    import yaml

    data = yaml.safe_load(_palettes_path().read_text(encoding="utf-8"))
    return list(data.keys())


def _apply_variant(cfg: Config, seed: int, palette: str, hook_index: int) -> Config:
    import yaml

    hook = cfg.hooks[hook_index] if cfg.hooks else cfg.hook
    stops = tuple(yaml.safe_load(_palettes_path().read_text(encoding="utf-8"))[palette])
    return with_overrides(cfg, seed=seed, hook=hook, palette=palette, palette_stops=stops)


def _write_sidecar(output: Path, cfg: Config, stats: dict) -> None:
    payload = {"config": cfg.to_public_dict(), "stats": stats}
    output.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    post = output.with_suffix(".post.txt")
    post.write_text(
        f"{cfg.hook}\n\n#shorts #oddlysatisfying #satisfying #asmr\n",
        encoding="utf-8",
    )


def render_one(
    cfg: Config,
    output: Path,
    *,
    preview: bool,
    verify: bool,
    use_cache: bool,
    keep_temp: bool,
) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    sim = load_or_simulate(cfg, use_cache=use_cache)
    failures = validate_result(cfg, sim)
    if failures:
        raise SystemExit("sim failed pacing:\n" + "\n".join(failures))
    npz_path = Path(".cache") / f"sim_{cache_key(cfg)}.npz"
    wav_path = output.with_suffix(".wav")
    lufs = None
    true_peak = None
    if preview:
        audio_path = None
    else:
        audio, lufs, true_peak = synthesize(cfg, sim)
        write_wav(str(wav_path), audio)
        audio_path = wav_path
        print(f"audio: {lufs:.2f} LUFS, true peak {true_peak:.2f} dBTP")
    started = time.perf_counter()
    try:
        profile = encode_frames(
            cfg,
            sim,
            npz_path,
            output,
            preview=preview,
            audio_path=audio_path,
            workers=cfg.workers,
        )
    finally:
        if audio_path is not None and audio_path.exists() and not keep_temp:
            audio_path.unlink()
    elapsed = time.perf_counter() - started
    renderer = Renderer(cfg, sim, preview=preview)
    growth_n = dict(renderer.plan)["growth"]
    coverage_frame = renderer.render(growth_n - 1)
    coverage = pixel_coverage(cfg, sim, coverage_frame) if not preview else None
    fill = geometric_fill(cfg, sim)
    stats = {
        "final_count": int(sim.final_count),
        "first_bounce_time": float(sim.first_bounce),
        "count_at_0.14_tg": int(sim.count_at_014),
        "lufs": None if lufs is None else float(lufs),
        "true_peak_dbtp": None if true_peak is None else float(true_peak),
        "render_seconds": float(elapsed),
        "ms_per_frame": float(profile["ms_per_frame"]),
        "geometric_fill": float(fill),
        "pixel_coverage": None if coverage is None else float(coverage),
        "cap_time": None if sim.cap_time is None else float(sim.cap_time),
        "max_hit_gap": None if sim.max_gap is None else float(sim.max_gap),
        "preview": preview,
    }
    _write_sidecar(output, cfg, stats)
    print(
        f"fill density at cap: geometric {fill:.4f} "
        f"(count * r^2 / R^2)"
        + (f", pixel coverage {coverage:.4f}" if coverage is not None else "")
    )
    print(f"wrote {output} in {elapsed:.1f}s ({profile['ms_per_frame']:.1f} ms/frame)")
    if verify:
        checks = verify_file(output, cfg, use_cache=True)
        print(format_table(checks))
        if not all(item.ok for item in checks):
            raise SystemExit(f"verification failed for {output}")
    return stats


def _parse(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render an oddly satisfying bouncing-ball Short")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--preview", action="store_true", help="540x960, 30 fps, no audio")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--vary", default="seed,palette,hook")
    parser.add_argument("--verify", action="store_true", help="verify after render (default for full renders)")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--keep-temp", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    cfg = load_config(args.config)
    if args.seed is not None:
        cfg = with_overrides(cfg, seed=args.seed)
    if args.workers is not None:
        cfg = with_overrides(cfg, workers=args.workers)
    verify = True if not args.preview else bool(args.verify)
    output = Path(args.out)
    if args.batch:
        names = _palette_names(cfg)
        combos = unique_variants(
            args.batch,
            cfg.seed,
            cfg.palette,
            _hook_index(cfg),
            args.vary,
            names,
            len(cfg.hooks),
        )
        failed = 0
        for seed, palette, hook_index in combos:
            target = variant_path(output, seed, palette, hook_index)
            variant = _apply_variant(cfg, seed, palette, hook_index)
            print(f"batch: {target.name}")
            try:
                render_one(
                    variant,
                    target,
                    preview=args.preview,
                    verify=verify,
                    use_cache=not args.no_cache,
                    keep_temp=args.keep_temp,
                )
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
    render_one(
        cfg,
        output,
        preview=args.preview,
        verify=verify,
        use_cache=not args.no_cache,
        keep_temp=args.keep_temp,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
