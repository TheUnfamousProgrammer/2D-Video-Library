#!/usr/bin/env python3
"""Odd One Out.

Default (no mode flag) builds the levels and the strips. It does not encode video.
--full is refused unless --approved is also passed.

  python make_odd.py --config configs/odd_default.yaml --out out/odd.mp4
  python make_odd.py --config configs/odd_default.yaml --out out/odd.mp4 --preview
  python make_odd.py --config configs/odd_default.yaml --out out/odd.mp4 --full --approved
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

from fc_sat.odd_config import load_odd_config
from fc_sat.odd_render import OddRenderer, write_level_strips, write_pause_test
from fc_sat.odd_report import write_answers, write_json, write_post, write_report
from fc_sat.odd_sim import simulate_show

CRITIQUE = """
L1 hue is findable in the first second: one lighter disc among 36, which is what the 0.15 lightness gap is for. Nothing else marks it. It is an easy scan, not a pre-lit answer.

L2 size is the one a still frame can hide. The 18% radius gap does not pop among 64 discs, and the color is the same. That is the medium round. The reveal is what makes it obvious.

L3 spin does not survive a pause. Every disc has a tick, the pause frame draws no circle, and the odd disc is only the one going the other way once the clip is moving. "Pausing won't help" sits clear of the timer.

L4 pulse is the same idea. Brightness already differs from disc to disc because each one has its own phase, so a brighter disc is not the tell. The different rate shows up only over time. "Last chance" stays left of the seconds.

HUD text stays outside the field. The caption band ends before the seconds. The reveal pill is inside the field, clamped, and offset from the odd disc. No sampled frame had two text boxes touching.
""".strip()


def unique_odd_variants(n: int, seed: int, tiers: tuple[str, ...] = ("normal", "brutal", "quick")) -> list[dict]:
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


def generate(cfg, output: Path, *, critique: str) -> dict:
    started = time.perf_counter()
    timings: list[tuple[str, float]] = []
    mark = started
    show = simulate_show(cfg)
    timings.append(("sim", time.perf_counter() - mark))
    print(f"stage sim done in {timings[-1][1]:.2f}s", flush=True)
    mark = time.perf_counter()
    renderer = OddRenderer(cfg, show, preview=False)
    strips = write_level_strips(renderer, output)
    pause = write_pause_test(renderer, output)
    timings.append(("strips", time.perf_counter() - mark))
    print(f"stage strips done in {timings[-1][1]:.2f}s", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_report(output.with_suffix(".report.md"), cfg, show, timings=timings, critique=critique)
    write_answers(output.with_suffix(".answers.md"), cfg, show)
    write_post(output.with_suffix(".post.txt"), cfg, show)
    write_json(output.with_suffix(".json"), cfg, show, timings=timings, loudness=None)
    elapsed = time.perf_counter() - started
    print(f"total {elapsed:.2f}s", flush=True)
    return {"show": show, "renderer": renderer, "strips": strips, "pause": pause, "timings": timings}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Odd One Out generator")
    parser.add_argument("--config", default="configs/odd_default.yaml")
    parser.add_argument("--out", default="out/odd.mp4")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--tier", choices=("normal", "brutal", "quick"), default=None)
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
            "Look at the strips, the pause test, and the preview, then re-run with --full --approved."
        )
    if args.preview or args.audio_only or (args.full and args.approved):
        raise SystemExit("preview, audio-only, and full encode are later stages; this run only builds levels and strips")
    level_ids = _parse_levels(args.levels)
    output = Path(args.out)
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
                generate(cfg, path, critique=CRITIQUE)
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
    generate(cfg, output, critique=CRITIQUE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
