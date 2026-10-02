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

from fc_sat.polycircle_doctor import doctor
from fc_sat.stage_log import StageLog

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
    print(f"{args.mode} is not built yet", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
