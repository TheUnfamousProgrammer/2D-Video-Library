"""The scale short's objects: catalog order and timing, cited sizes, and drawn proportions."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from fc_sat.scale_art import cutout_aspect
from fc_sat.scale_format import metres
from fc_sat.scale_world import Item, Placed, check_order, load_catalog, place

ROOT = Path(__file__).resolve().parents[1]
CLAIMS_PATH = ROOT / "configs" / "scale_claims.yaml"
# Width over height for each placeholder shape, used only until the cutout exists.
PLACEHOLDER_ASPECT = {"tall": 0.35, "wide": 3.0, "circle": 1.0, "square": 0.9, "peak": 2.0, "galaxy": 1.0}


@dataclass(frozen=True)
class Scene:
    placed: tuple[Placed, ...]
    counters: tuple[str, ...]
    displays: tuple[str, ...]
    has_art: tuple[bool, ...]

    @property
    def items(self) -> list[Item]:
        return [p.item for p in self.placed]


def load_sizes(path: Path | None = None) -> dict[str, dict]:
    raw = yaml.safe_load((path or CLAIMS_PATH).read_text())
    return dict(raw["claims"])


@lru_cache(maxsize=1)
def build_scene() -> Scene:
    sizes = load_sizes()
    items = []
    has_art = []
    for entry in load_catalog():
        claim = sizes.get(entry["id"])
        if claim is None:
            raise SystemExit(f"no size claim for {entry['id']}")
        aspect = cutout_aspect(entry["art"])
        has_art.append(aspect is not None)
        if aspect is None:
            aspect = PLACEHOLDER_ASPECT[entry["shape"]]
        items.append(
            Item(
                id=entry["id"],
                name=str(entry["name"]),
                art=str(entry["art"]),
                measure=str(entry["measure"]),
                land=int(entry["land"]),
                realm=str(entry["realm"]),
                shape=str(entry["shape"]),
                color=str(entry["color"]),
                size_m=float(claim["value"]),
                aspect=float(aspect),
            )
        )
    errors = check_order(items)
    if errors:
        raise SystemExit("; ".join(errors))
    counters = tuple(metres(item.size_m, int(sizes[item.id].get("digits", 2))) for item in items)
    displays = tuple(str(sizes[item.id]["display"]) for item in items)
    return Scene(tuple(place(items)), counters, displays, tuple(has_art))
