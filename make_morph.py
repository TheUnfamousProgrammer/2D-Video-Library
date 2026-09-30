#!/usr/bin/env python3
"""Render a Pixel Morph loop.

Examples:
  python make_morph.py --config configs/morph_default.yaml --a a.png --b b.png --out out/morph_x.mp4
  python make_morph.py --config configs/morph_default.yaml --a a.png --b b.png --out out/morph_preview.mp4 --preview --contact-sheet
  python make_morph.py --config configs/morph_default.yaml --pairs configs/pairs.yaml --out out/morph_x.mp4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import yaml

from fc_sat.audio import write_wav
from fc_sat.encode import iter_ordered_frames, pipe_raw_bgr, resolve_workers
from fc_sat.morph_audio import synthesize_morph
from fc_sat.morph_config import MorphConfig, load_morph_config, with_morph_overrides
from fc_sat.morph_images import file_sha256
from fc_sat.morph_render import MorphRenderer, contact_sheet, init_morph_worker, render_morph_chunk


def load_pairs(path: Path) -> list[dict]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise SystemExit(f"{path} must be a list of {{a, b}} mappings")
    entries = []
    for index, item in enumerate(data):
        if not isinstance(item, dict) or "a" not in item or "b" not in item:
            raise SystemExit(f"pair {index} must include a and b")
        entries.append(item)
    return entries


def pair_tuple(entry: dict, hooks: tuple[str, ...], default_hook: str) -> tuple[str, str, int]:
    """``(stem A, stem B, hook index)``. Repeats are rejected by the caller."""
    hook = str(entry.get("hook") or default_hook)
    try:
        index = tuple(hooks).index(hook)
    except ValueError as exc:
        raise SystemExit(f"pair hook {hook!r} is not in the morph hook list") from exc
    return (Path(str(entry["a"])).stem, Path(str(entry["b"])).stem, index)


def unique_pairs(entries: list[dict], hooks: tuple[str, ...], default_hook: str) -> list[tuple[dict, tuple[str, str, int]]]:
    seen: set[tuple[str, str, int]] = set()
    chosen = []
    for entry in entries:
        identity = pair_tuple(entry, hooks, default_hook)
        if identity in seen:
            raise SystemExit(f"repeated pair {identity[0]}, {identity[1]}, hook {identity[2]}")
        seen.add(identity)
        chosen.append((entry, identity))
    return chosen


def _credit_line(path: Path) -> str:
    credits_path = Path("assets/images/credits.json")
    if not credits_path.is_file():
        return f"{path.name} — not listed in assets/images/credits.json"
    try:
        payload = json.loads(credits_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return f"{path.name} — credits.json could not be read"
    images = payload.get("images", []) if isinstance(payload, dict) else payload
    digest = file_sha256(path)
    for item in images:
        if not isinstance(item, dict):
            continue
        if item.get("sha256") == digest or item.get("file") == path.name:
            title = item.get("title") or path.name
            artist = item.get("artist") or "unknown"
            license_name = item.get("license") or "unspecified"
            return f"{title} — {artist} — {license_name}"
    return f"{path.name} — not listed in assets/images/credits.json"


def _write_sidecar(
    output: Path,
    cfg: MorphConfig,
    path_a: Path,
    path_b: Path,
    renderer: MorphRenderer,
    stats: dict,
) -> None:
    payload = {
        "config": cfg.to_public_dict(),
        "image_a": str(path_a),
        "image_b": str(path_b),
        "image_a_sha256": file_sha256(path_a),
        "image_b_sha256": file_sha256(path_b),
        "assignment": {
            "mean_oklab": renderer.assignment.mean_error,
            "p95_oklab": renderer.assignment.p95_error,
            "solve_seconds": renderer.assignment.solve_seconds,
            "cache_key": renderer.assignment.cache_key,
            "from_cache": renderer.assignment.from_cache,
            "particles": int(renderer.perm.size),
        },
        "stats": stats,
    }
    output.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    output.with_suffix(".post.txt").write_text(
        f"{cfg.hook} #shorts\n\n"
        "#shorts #satisfying #pixels #art #morph\n\n"
        "Pinned comment: What did the picture turn into?\n\n"
        "Image credits:\n"
        f"A: {_credit_line(path_a)}\n"
        f"B: {_credit_line(path_b)}\n"
        "You are responsible for having rights to pictures you add.\n",
        encoding="utf-8",
    )


def render_one(
    cfg: MorphConfig,
    path_a: Path,
    path_b: Path,
    output: Path,
    *,
    preview: bool,
    verify: bool,
    keep_temp: bool,
    contact: bool,
) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    renderer = MorphRenderer(cfg, path_a, path_b, preview=preview)
    assignment = renderer.assignment
    print(
        f"assignment: mean OKLab {assignment.mean_error:.4f}, p95 {assignment.p95_error:.4f}, "
        f"solve {assignment.solve_seconds:.2f}s"
        + (" (cached)" if assignment.from_cache else "")
    )
    wav_path = output.with_suffix(".wav")
    audio_path = None
    lufs = None
    true_peak = None
    if not preview:
        audio, lufs, true_peak = synthesize_morph(
            cfg,
            delay_ab=renderer.delay_ab,
            delay_ba=renderer.delay_ba,
            x_b=renderer.centers_b[:, 0],
            x_a=renderer.centers_a[:, 0],
            n_frames=renderer.n_frames,
            fps=renderer.fps,
        )
        write_wav(str(wav_path), audio)
        audio_path = wav_path
        print(f"audio: {lufs:.2f} LUFS, true peak {true_peak:.2f} dBTP")
    heavy = renderer.heaviest_index()
    started = time.perf_counter()
    profile_frame = renderer.render(heavy)
    ms_frame = (time.perf_counter() - started) * 1000.0
    workers = 1 if renderer.n_frames < 4 else resolve_workers(cfg.workers)
    eta = (ms_frame / 1000.0) * renderer.n_frames / workers
    print(f"profile: {ms_frame:.1f} ms/frame; eta {eta / 60.0:.1f} min with {workers} workers")
    if contact:
        sheet = output.with_suffix(".contact.png")
        contact_sheet(renderer, sheet)
        print(f"contact sheet: {sheet}")

    def render_index(index: int):
        if index == heavy:
            return profile_frame
        return renderer.render(index)

    encode_started = time.perf_counter()
    try:
        pipe_raw_bgr(
            iter_ordered_frames(
                renderer.n_frames,
                workers=workers,
                render_one=render_index,
                initializer=init_morph_worker,
                initargs=(cfg, str(path_a), str(path_b), preview),
                render_chunk=render_morph_chunk,
            ),
            output,
            width=renderer.width,
            height=renderer.height,
            fps=renderer.fps,
            audio_path=audio_path,
            preset="veryfast" if preview else "slow",
            n_frames=renderer.n_frames if workers <= 1 else None,
        )
    finally:
        if audio_path is not None and audio_path.exists() and not keep_temp:
            audio_path.unlink()
    elapsed = time.perf_counter() - encode_started
    stats = {
        "n_frames": int(renderer.n_frames),
        "fps": int(renderer.fps),
        "preview": bool(preview),
        "ms_per_frame": float(ms_frame),
        "render_seconds": float(elapsed),
        "workers": int(workers),
        "lufs": None if lufs is None else float(lufs),
        "true_peak_db": None if true_peak is None else float(true_peak),
        "heaviest_frame": int(heavy),
    }
    _write_sidecar(output, cfg, path_a, path_b, renderer, stats)
    if verify and not preview:
        from fc_sat.verify import format_table, verify_morph_file

        checks = verify_morph_file(output, cfg, image_a=path_a, image_b=path_b)
        print(format_table(checks))
        if not all(item.ok for item in checks):
            raise SystemExit(f"verification failed for {output}")
    elif verify and preview:
        print("preview skips the full-frame verifier; render without --preview to verify")
    return stats


def _apply_pair(cfg: MorphConfig, entry: dict) -> tuple[MorphConfig, Path, Path]:
    changes = {}
    if entry.get("hook"):
        changes["hook"] = entry["hook"]
    if entry.get("focus_a") is not None:
        changes["focus_a"] = entry["focus_a"]
    if entry.get("focus_b") is not None:
        changes["focus_b"] = entry["focus_b"]
    updated = with_morph_overrides(cfg, **changes) if changes else cfg
    return updated, Path(str(entry["a"])), Path(str(entry["b"]))


def _parse(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a Pixel Morph loop")
    parser.add_argument("--config", required=True)
    parser.add_argument("--a", default=None)
    parser.add_argument("--b", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--preview", action="store_true", help="540x960, 30 fps, no audio")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--pairs", default=None, help="YAML list of {a, b, hook?, focus_a?, focus_b?}")
    parser.add_argument("--verify", action="store_true", help="verify after render (default for full renders)")
    parser.add_argument("--keep-temp", action="store_true")
    parser.add_argument("--contact-sheet", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    cfg = load_morph_config(args.config)
    if args.seed is not None:
        cfg = with_morph_overrides(cfg, seed=args.seed)
    if args.workers is not None:
        cfg = with_morph_overrides(cfg, workers=args.workers)
    verify = True if not args.preview else bool(args.verify)
    output = Path(args.out)
    if args.pairs:
        entries = load_pairs(Path(args.pairs))
        chosen = unique_pairs(entries, cfg.hooks, cfg.hook)
        failed = 0
        for entry, identity in chosen:
            target = output.parent / f"{identity[0]}_{identity[1]}_{identity[2]}.mp4"
            try:
                variant, path_a, path_b = _apply_pair(cfg, entry)
                print(f"batch: {target.name}")
                render_one(
                    variant,
                    path_a,
                    path_b,
                    target,
                    preview=args.preview,
                    verify=verify,
                    keep_temp=args.keep_temp,
                    contact=args.contact_sheet,
                )
            except KeyboardInterrupt:
                raise
            except (Exception, SystemExit) as exc:
                failed += 1
                print(f"pair failed {target.name}: {exc}", file=sys.stderr)
                traceback.print_exc()
        if failed:
            print(f"{failed} of {len(chosen)} pairs failed", file=sys.stderr)
            return 1
        return 0
    if not args.a or not args.b:
        raise SystemExit("--a and --b are required unless --pairs is set")
    render_one(
        cfg,
        Path(args.a),
        Path(args.b),
        output,
        preview=args.preview,
        verify=verify,
        keep_temp=args.keep_temp,
        contact=args.contact_sheet,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
