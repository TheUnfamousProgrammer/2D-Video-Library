#!/usr/bin/env python3
"""How many folds to pass a place or a distance, and how tall the stack is.

    python tools/fold_reply.py moon
    python tools/fold_reply.py 100 km
    python tools/fold_reply.py 1 ly
"""

from __future__ import annotations

import sys

from fc_sat.paperfold_math import (
    AU_M,
    BURJ_M,
    EVEREST_M,
    ISS_M,
    KARMAN_M,
    LIGHT_YEAR_M,
    MOON_M,
    PERSON_M,
    first_fold_past,
    height_m,
)

PLACES = {
    "moon": MOON_M,
    "everest": EVEREST_M,
    "burj": BURJ_M,
    "burj khalifa": BURJ_M,
    "iss": ISS_M,
    "space": KARMAN_M,
    "karman": KARMAN_M,
    "person": PERSON_M,
    "sun": AU_M,
}


def parse_distance(text: str) -> float:
    cleaned = " ".join(text.strip().lower().replace("earth to sun", "sun").split())
    if not cleaned:
        raise ValueError("empty")
    if cleaned in PLACES:
        return PLACES[cleaned]
    parts = cleaned.split()
    if len(parts) == 1:
        raise ValueError(f"unknown name {text!r}")
    if len(parts) != 2:
        raise ValueError(f"could not read {text!r}")
    try:
        value = float(parts[0])
    except ValueError as exc:
        raise ValueError(f"unknown name {text!r}") from exc
    if value <= 0:
        raise ValueError(f"distance must be positive, got {value}")
    unit = parts[1]
    scale = {"m": 1.0, "km": 1000.0, "au": AU_M, "ly": LIGHT_YEAR_M}.get(unit)
    if scale is None:
        raise ValueError(f"unknown unit {unit!r}")
    return value * scale


def reply(text: str) -> str:
    meters = parse_distance(text)
    fold = first_fold_past(meters)
    height = height_m(fold)
    if height < meters:
        return f"beyond {fold} folds; the stack is still shorter than that distance"
    if height >= LIGHT_YEAR_M:
        shown = f"{height / LIGHT_YEAR_M:.3g} ly"
    elif height >= AU_M:
        shown = f"{height / AU_M:.3g} AU"
    elif height >= 1000:
        shown = f"{height / 1000:.3g} km"
    else:
        shown = f"{height:.3g} m"
    return f"{fold} folds, tower {shown}"


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("usage: python tools/fold_reply.py <place or distance>", file=sys.stderr)
        return 2
    try:
        print(reply(" ".join(argv)))
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
