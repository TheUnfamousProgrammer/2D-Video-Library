#!/usr/bin/env python3
"""Random lines become a picture.

    python make_linedraw.py doctor
    python make_linedraw.py optimize --target assets/target/target.png
    python make_linedraw.py full --approved --target assets/target/target.png

`full` is refused unless `--approved` is also passed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from fc_sat.linedraw.choreo import build_plan, choose_hold
from fc_sat.linedraw.doctor import doctor
from fc_sat.linedraw.optimize import find_l_aha, load_fit, optimize, replay, save_fit
from fc_sat.linedraw.picture import load_picture, parse_crop, prepare, save_picture
from fc_sat.linedraw.tone import paper_rgb
from fc_sat.stage_log import StageLog

ROOT = Path(__file__).resolve().parent
MODES = (
    "doctor",
    "prepare",
    "optimize",
    "stills",
    "motion",
    "audio",
    "hooks",
    "preview",
    "postkit",
    "verify",
    "full",
)


def _show_original(rights: str, flag: str) -> bool:
    if flag == "yes":
        return True
    if flag == "no":
        return False
    return rights in {"public-domain", "own", "licensed"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Turn a picture into a random-line short")
    parser.add_argument("mode", choices=MODES)
    parser.add_argument("--target", default="assets/target/target.png")
    parser.add_argument("--subject", default="")
    parser.add_argument("--credit", default="")
    parser.add_argument("--rights", default="unknown", choices=("public-domain", "own", "licensed", "unknown"))
    parser.add_argument("--show-original", default="auto", choices=("auto", "yes", "no"))
    parser.add_argument("--crop", default="")
    parser.add_argument("--weight", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default="")
    parser.add_argument("--approved", action="store_true")
    parser.add_argument("--hook", default="A", choices=("A", "B", "C"))
    return parser


def job_from_args(args: argparse.Namespace) -> dict:
    if not 0.0 <= args.weight <= 4.0:
        raise SystemExit("--weight must be between 0 and 4")
    target = Path(args.target)
    if not target.is_absolute():
        target = ROOT / target
    return {
        "target": target,
        "subject": " ".join(args.subject.split()),
        "credit": args.credit.strip(),
        "rights": args.rights,
        "show_original": _show_original(args.rights, args.show_original),
        "crop": parse_crop(args.crop) if args.crop else None,
        "weight": float(args.weight),
        "seed": int(args.seed),
        "hook": args.hook,
        "out": Path(args.out) if args.out else ROOT / "out" / "linedraw.mp4",
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "full" and not args.approved:
        raise SystemExit("full render refused without --approved")
    log = StageLog()
    job = job_from_args(args) if args.mode != "doctor" else None
    if args.mode == "doctor":
        code = doctor()
        log.mark("doctor")
        return code
    assert job is not None
    out = ROOT / "out"
    out.mkdir(parents=True, exist_ok=True)
    if args.mode == "prepare":
        code = _prepare(job, out)
        log.mark("prepare")
        return code
    if args.mode == "optimize":
        code = _optimize(job, out)
        log.mark("optimize")
        return code
    from fc_sat.linedraw.pipeline import load_state, render_full, write_audio, write_delivery, write_hooks, write_motion
    from fc_sat.linedraw.post import lint_post, render_post, retention_map
    from fc_sat.linedraw.qa import contact_sheet, write_stills
    from fc_sat.linedraw.render import LineRenderer
    from fc_sat.linedraw.verify import verify

    if args.mode == "full":
        code = render_full(job, log)
        log.mark("full")
        return code
    if not (out / "lines.npz").exists():
        _optimize(job, out)
        log.mark("optimize")
    meta, picture, fit, plan = load_state(out, job)
    if args.mode == "audio":
        write_audio(out, fit, plan, job["seed"])
        log.mark("audio")
        return 0
    renderer = LineRenderer(fit, picture, plan, job, width=1080, height=1920, hook=job["hook"])
    if args.mode == "stills":
        write_stills(renderer, out / "stills")
        contact_sheet(renderer, out / "contact_sheet.png")
        log.mark("stills")
        return 0
    if args.mode == "motion":
        write_motion(out, renderer)
        log.mark("motion")
        return 0
    if args.mode == "hooks":
        if not (out / "linedraw.wav").exists():
            write_audio(out, fit, plan, job["seed"])
        write_hooks(out, picture, fit, plan, job)
        log.mark("hooks")
        return 0
    if args.mode == "preview":
        if not (out / "linedraw.wav").exists():
            write_audio(out, fit, plan, job["seed"])
        small = LineRenderer(fit, picture, plan, job, width=540, height=960, hook=job["hook"])
        write_delivery(small, "preview", Path(args.out) if args.out else out / "linedraw_preview.mp4", out / "linedraw.wav", "veryfast")
        log.mark("preview")
        return 0
    if args.mode == "postkit":
        post = render_post(meta)
        errors = lint_post(post, meta)
        (out / "postkit.md").write_text(post)
        (out / "retention_map.md").write_text(retention_map(meta))
        log.mark("postkit")
        return 1 if errors else 0
    if args.mode == "verify":
        code, _report = verify(out, Path(args.out) if args.out else out / "linedraw.mp4")
        log.mark("verify")
        return code
    raise SystemExit(f"{args.mode} is not built yet")


def _prepare(job: dict, out: Path) -> int:
    picture = prepare(job["target"], weight=job["weight"], crop=job["crop"], seed=job["seed"])
    save_picture(picture, out / "picture.npz")
    stills = out / "stills"
    stills.mkdir(parents=True, exist_ok=True)
    Image.fromarray(picture.color, mode="RGB").save(stills / "crop.png")
    meta = {
        "method": picture.method,
        "clip": picture.clip,
        "trials": picture.trial_scores,
        "upscaled": picture.upscaled,
        "upscale_from": picture.upscale_from,
        "crop_box": picture.crop_box,
        "face_box": picture.face_box,
        "focus": picture.focus,
        "mouth": picture.mouth,
        "source_size": picture.source_size,
        "working_size": picture.working_size,
        "show_original": job["show_original"],
        "rights": job["rights"],
        "subject": job["subject"],
        "credit": job["credit"],
        "seed": job["seed"],
        "weight": job["weight"],
    }
    (out / "picture.json").write_text(json.dumps(meta, indent=2))
    print(f"crop via {picture.method}, clahe {picture.clip}, face {picture.face_box}")
    return 0


def _optimize(job: dict, out: Path) -> int:
    if not (out / "picture.npz").exists():
        _prepare(job, out)
    meta = json.loads((out / "picture.json").read_text())
    picture = load_picture(out / "picture.npz")
    fit = optimize(
        picture["target"],
        picture["weight"],
        seed=job["seed"],
        store_rejects=900,
    )
    save_fit(fit, out / "lines.npz")
    kept = int(len(fit.x0))
    thrown = int(fit.thrown[-1]) if kept else 0
    l_aha = find_l_aha(fit.likeness_counts, fit.likeness) if kept >= 100 else kept
    hold = choose_hold(fit.likeness_counts, fit.likeness, l_aha) if kept >= 100 else kept
    plan = build_plan(kept, l_aha, pre_drop=hold)
    _write_aha(picture["target"], fit, l_aha, out / "stills")
    dark = int(np.sum(fit.ink < 0))
    light = int(np.sum(fit.ink > 0))
    meta.update(
        {
            "kept": kept,
            "thrown": thrown,
            "l_aha": l_aha,
            "burst": plan.burst,
            "pre_drop": plan.pre_drop,
            "mid": plan.mid,
            "reveal_fraction": plan.reveal_fraction,
            "hold_count": hold,
            "hold_note": (
                "The 0.55 and 0.45 fractions of L_aha already read as a face, so the silent-beat "
                "hold is the last logged count under 30% likeness."
                if hold < l_aha - int(round(0.4 * l_aha))
                else "The spec hold at 60% of L_aha stayed under 30% likeness."
            ),
            "optimize_seconds": fit.seconds,
            "dark_ink": dark,
            "light_ink": light,
            "final_likeness": float(fit.likeness[-1]) if len(fit.likeness) else 0.0,
        }
    )
    (out / "picture.json").write_text(json.dumps(meta, indent=2))
    (out / "lines_report.md").write_text(_lines_report(fit, meta))
    print(f"L_aha {l_aha} kept {kept} thrown {thrown} dark {dark} light {light}")
    return 0


def _write_aha(target: np.ndarray, fit, l_aha: int, stills: Path) -> None:
    stills.mkdir(parents=True, exist_ok=True)
    fraction = 0.55
    marks = {
        "aha_early.png": max(1, int(round(fraction * l_aha))),
        "aha_drop.png": max(1, l_aha),
        "aha_final.png": len(fit.x0),
    }
    for name, count in marks.items():
        canvas = replay(fit.x0, fit.y0, fit.x1, fit.y1, fit.ink, count, target.shape)
        rgb = paper_rgb(canvas, 762, 1104)
        Image.fromarray(rgb, mode="RGB").save(stills / name)
        print(f"wrote {stills / name} at {count} lines", flush=True)


def _lines_report(fit, meta: dict) -> str:
    rows = ["| kept | likeness |", "| --- | --- |"]
    for count, score in zip(fit.likeness_counts[::10], fit.likeness[::10]):
        rows.append(f"| {int(count):,} | {float(score) * 100:.1f}% |")
    if len(fit.likeness):
        rows.append(f"| {int(fit.likeness_counts[-1]):,} | {float(fit.likeness[-1]) * 100:.1f}% |")
    curve = "\n".join(rows)
    return f"""# Lines

Kept {meta['kept']:,} lines out of {meta['thrown']:,} thrown in {meta['optimize_seconds']:.1f}s.
Dark ink {meta['dark_ink']:,}, light ink {meta['light_ink']:,}.
CLAHE clip {meta['clip']} (trial scores {meta['trials']}). Crop method `{meta['method']}`.
L_aha is {meta['l_aha']:,}, the first logged count at 60% of the final likeness ({meta['final_likeness'] * 100:.1f}%).
The pre-drop hold shows {meta['pre_drop']:,} lines, then the drop adds the burst of {meta['burst']:,}.
The inspection still `out/stills/aha_early.png` is {meta['reveal_fraction']:.2f} of L_aha. On this picture that still already reads as a face, and so does 0.45 of L_aha. {meta.get('hold_note', '')}
The silent beat therefore holds at {meta['pre_drop']:,} lines instead of 60% of L_aha. The drop still lands on L_aha.

{curve}
"""


if __name__ == "__main__":
    raise SystemExit(main())
