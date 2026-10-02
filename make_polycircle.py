#!/usr/bin/env python3
"""Sides to circle.

    python make_polycircle.py doctor
    python make_polycircle.py facts
    python make_polycircle.py full --approved --out out/polycircle.mp4

`full` is refused unless `--approved` is also passed. The default hook is A.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fc_sat.polycircle_claims import evaluate, load_claims, render_report
from fc_sat.polycircle_doctor import doctor
from fc_sat.polycircle_text import lint_script, load_script
from fc_sat.polycircle_timeline import (
    build_timeline,
    facts_markdown,
    retention_markdown,
    write_timeline,
)
from fc_sat.stage_log import StageLog

ROOT = Path(__file__).resolve().parent

ROOT_MODES = (
    "doctor",
    "facts",
    "timeline",
    "animatic",
    "hooks",
    "audio",
    "preview",
    "full",
    "postkit",
    "verify",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sides to circle")
    parser.add_argument("mode", choices=ROOT_MODES)
    parser.add_argument("--hook", default="A", choices=["A", "B", "C"])
    parser.add_argument("--out", default="")
    parser.add_argument("--approved", action="store_true")
    args = parser.parse_args(argv)
    if args.mode == "full" and not args.approved:
        raise SystemExit("full render refused without --approved")
    log = StageLog()
    if args.mode == "doctor":
        code = doctor()
        log.mark("doctor")
        return code
    if args.mode == "facts":
        code = _facts()
        log.mark("facts")
        return code
    if args.mode == "timeline":
        code = _timeline()
        log.mark("timeline")
        return code
    print(f"{args.mode} is not built yet", file=sys.stderr)
    return 2


def _facts() -> int:
    book = load_claims()
    results = evaluate(book)
    script = load_script(book=book)
    errors = lint_script(script, book)
    timeline = build_timeline()
    out = ROOT / "out"
    out.mkdir(parents=True, exist_ok=True)
    (out / "facts.md").write_text(facts_markdown(timeline, book))
    (out / "claims_report.md").write_text(render_report(book, results))
    failed = [claim_id for claim_id, ok, detail in results if not ok]
    for claim_id, ok, detail in results:
        if not ok:
            print(f"FAIL {claim_id}: {detail}")
    for error in errors:
        print(error)
    print(f"claims {sum(1 for _, ok, _ in results if ok)}/{len(results)} passed")
    print(f"snap error {timeline.schedule.max_snap_error:.3e} frames")
    print(f"rotation scale {timeline.rotation_scale:.6f}, {timeline.rotation_turns} quarter-turns")
    return 1 if failed or errors else 0


def _timeline() -> int:
    timeline = build_timeline()
    path = write_timeline(timeline)
    report = ROOT / "out" / "retention_map.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(retention_markdown(timeline))
    payload = timeline.to_json()
    print(f"wrote {path}")
    print(
        f"adds {len(payload['adds'])} doublings {len(payload['doublings'])} "
        f"kicks {len(payload['kicks'])} impacts {len(payload['impacts'])}"
    )
    print(f"retention {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
