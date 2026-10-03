#!/usr/bin/env python3
"""How many folds to the Moon.

    python make_paperfold.py doctor
    python make_paperfold.py facts
    python make_paperfold.py full --approved --out out/paperfold.mp4

`full` is refused unless `--approved` is also passed. The default hook is A.
Preview and hooks are 540x960 at 30 fps. The master is 1080x1920 at 60 fps.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fc_sat.paperfold_doctor import doctor
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
    parser = argparse.ArgumentParser(description="How many folds to the Moon")
    parser.add_argument("mode", choices=ROOT_MODES)
    parser.add_argument("--hook", default="A", choices=["A", "B", "C", "D"])
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
    print(f"{args.mode} is not built yet", file=sys.stderr)
    log.mark(args.mode)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
