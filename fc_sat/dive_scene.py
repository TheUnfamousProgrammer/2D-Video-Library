"""The deepest-to-highest short's stops: catalog order and timing, cited values, drawn proportions."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from fc_sat.dive_world import CLAIMS_PATH, Stop, check_order, load_stops
from fc_sat.scale_art import cutout_aspect
from fc_sat.scale_format import metres


@dataclass(frozen=True)
class DiveScene:
    stops: tuple[Stop, ...]
    counters: tuple[str, ...]
    displays: tuple[str, ...]
    has_art: tuple[bool, ...]


def load_values(path: Path | None = None) -> dict[str, dict]:
    return dict(yaml.safe_load((path or CLAIMS_PATH).read_text())["claims"])


def counter_text(value_m: float, digits: int, act: str) -> str:
    """Plain metres. Depths carry a minus sign; sea level is 0 m."""
    if value_m <= 0:
        return "0 m"
    text = metres(value_m, digits)
    return "-" + text if act == "down" else text


@lru_cache(maxsize=1)
def build_scene() -> DiveScene:
    values = load_values()
    stops, has_art = [], []
    for entry in load_stops():
        claim = values.get(entry["id"])
        if claim is None:
            raise SystemExit(f"no value claim for {entry['id']}")
        aspect = cutout_aspect(entry["art"])
        has_art.append(aspect is not None)
        stops.append(
            Stop(
                id=entry["id"],
                name=str(entry["name"]),
                art=str(entry["art"]),
                act=str(entry["act"]),
                kind=str(entry["kind"]),
                land=int(entry["land"]),
                realm=str(entry["realm"]),
                label=str(entry["label"]),
                value_m=float(claim["value"]),
                aspect=float(aspect if aspect is not None else 1.0),
            )
        )
    errors = check_order(stops)
    if errors:
        raise SystemExit("; ".join(errors))
    counters = tuple(counter_text(s.value_m, int(values[s.id].get("digits", 2)), s.act) for s in stops)
    displays = tuple(str(values[s.id]["display"]) for s in stops)
    return DiveScene(tuple(stops), counters, displays, tuple(has_art))
