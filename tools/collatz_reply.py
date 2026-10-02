#!/usr/bin/env python3
"""Print a Collatz reply for one starting number.

    python tools/collatz_reply.py 27
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fc_sat.collatz_format import format_int
from fc_sat.collatz_math import CollatzError, peak, steps, trajectory


def reply_text(n: int) -> str:
    count = steps(n)
    top = peak(n)
    return f"{format_int(n)} takes {format_int(count)} steps and peaks at {format_int(top)} before dropping to 1."


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1:
        print("usage: python tools/collatz_reply.py N", file=sys.stderr)
        return 2
    try:
        n = int(argv[0])
    except ValueError:
        print(f"{argv[0]!r} is not an integer", file=sys.stderr)
        return 2
    try:
        values = trajectory(n)
    except CollatzError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    shown = ", ".join(format_int(value) for value in values[:12])
    print(f"steps {steps(n)}")
    print(f"peak {format_int(peak(n))}")
    print(f"first 12: {shown}")
    print(reply_text(n))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
