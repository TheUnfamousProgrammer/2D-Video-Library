"""Seven plates, one ground line, and the pull-back that joins them.

World height is the metres from the aligned ground line up to the top of the
frame. px_per_m stays constant inside a scene. A hand-off starts on the first
frame of the fold that would push the stack top above y = 0.30, and log-blends
the scale for 24 frames. Deep space has no next plate, so it never hands off.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from fc_sat.easing import ease_in_out_cubic
from fc_sat.paperfold_math import MOON_M, height_m
from fc_sat.paperfold_schedule import fold_frame

FRAME_W = 1080
FRAME_H = 1920
GROUND = 0.68
LIMIT = 0.30
HANDOFF_FRAMES = 24
FOLD_ANIM_FRAMES = 10


@dataclass(frozen=True)
class Scene:
    name: str
    plate: str
    ground_y: float
    world_m: float
    moon_cx: float | None = None
    moon_cy: float | None = None
    moon_radius: float | None = None


def scenes_from_manifest(manifest: dict | None = None, deep_space_m: float | None = None) -> tuple[Scene, ...]:
    if manifest is None:
        from fc_sat.paperfold_art import load_manifest

        manifest = load_manifest()
    rows = []
    for item in manifest["plates"]:
        world = item.get("world_m")
        if world is None:
            world = deep_space_m if deep_space_m is not None else deep_space_world_m(
                float(item["ground_y"]),
                float(item["moon_cy"]),
            )
        rows.append(
            Scene(
                name=str(item["scene"]),
                plate=str(item["id"]),
                ground_y=float(item["ground_y"]),
                world_m=float(world),
                moon_cx=_optional_float(item.get("moon_cx")),
                moon_cy=_optional_float(item.get("moon_cy")),
                moon_radius=_optional_float(item.get("moon_radius")),
            )
        )
    return tuple(rows)


def _optional_float(value) -> float | None:
    if value is None:
        return None
    return float(value)


def align_scale(ground_y: float, src_w: int, src_h: int, frame_w: int = FRAME_W, frame_h: int = FRAME_H) -> float:
    """Smallest scale of at least 1 that covers the frame above the ground panel and the full width."""
    cover_above = (GROUND * frame_h) / (ground_y * src_h)
    cover_width = frame_w / src_w
    return max(1.0, cover_above, cover_width)


def align_translate(
    ground_y: float,
    src_w: int,
    src_h: int,
    frame_w: int = FRAME_W,
    frame_h: int = FRAME_H,
) -> tuple[float, float, float]:
    """Return (scale, tx, ty) so the plate's ground line sits at y = 0.68 of the frame."""
    scale = align_scale(ground_y, src_w, src_h, frame_w, frame_h)
    tx = (frame_w - src_w * scale) / 2.0
    ty = GROUND * frame_h - ground_y * src_h * scale
    return scale, tx, ty


def aligned_ground_y(ground_y: float, src_w: int, src_h: int, frame_h: int = FRAME_H) -> float:
    scale, _tx, ty = align_translate(ground_y, src_w, src_h, FRAME_W, frame_h)
    return ground_y * src_h * scale + ty


def px_per_m(world_m: float, frame_h: int = FRAME_H) -> float:
    if world_m <= 0:
        raise ValueError(f"world height must be positive, got {world_m}")
    return (GROUND * frame_h) / float(world_m)


def stack_top_y(height: float, world_m: float, frame_h: int = FRAME_H) -> float:
    return GROUND * frame_h - height * px_per_m(world_m, frame_h)


def crosses_limit(height: float, world_m: float, frame_h: int = FRAME_H) -> bool:
    return stack_top_y(height, world_m, frame_h) < LIMIT * frame_h


def break_fold(world_m: float, limit: int = 60) -> int:
    """Smallest n whose landed stack would rise above y = 0.30. Past `limit` means it does not."""
    for n in range(1, limit + 1):
        if crosses_limit(height_m(n), world_m):
            return n
    return limit + 1


def handoff_folds(scene_rows: tuple[Scene, ...] | None = None) -> list[tuple[str, str, int]]:
    """(from scene, to scene, fold). The last plate has no successor, so it is not listed."""
    rows = scene_rows or scenes_from_manifest()
    pairs = []
    for current, nxt in zip(rows, rows[1:]):
        pairs.append((current.name, nxt.name, break_fold(current.world_m)))
    return pairs


def handoff_start_frame(fold: int) -> int:
    """The pull-back begins with the fold animation, 10 frames before that fold lands."""
    return fold_frame(fold) - FOLD_ANIM_FRAMES


def log_blend(old: float, new: float, u: float) -> float:
    u = ease_in_out_cubic(u)
    return math.exp(math.log(old) + (math.log(new) - math.log(old)) * u)


def px_per_m_at(frame: int, scene_rows: tuple[Scene, ...] | None = None) -> float:
    """Scale at a frame. Constant inside a scene, log-blended across each 24-frame hand-off."""
    rows = scene_rows or scenes_from_manifest()
    switches = []
    for current, nxt in zip(rows, rows[1:]):
        fold = break_fold(current.world_m)
        switches.append((handoff_start_frame(fold), current, nxt))
    scale = px_per_m(rows[0].world_m)
    scene = rows[0]
    for start, current, nxt in switches:
        if frame < start:
            return px_per_m(current.world_m)
        end = start + HANDOFF_FRAMES
        if frame < end:
            u = (frame - start) / (HANDOFF_FRAMES - 1)
            return log_blend(px_per_m(current.world_m), px_per_m(nxt.world_m), u)
        scale = px_per_m(nxt.world_m)
        scene = nxt
    return px_per_m(scene.world_m) if frame >= 0 else scale


def deep_space_world_m(
    ground_y: float,
    moon_cy: float,
    src_w: int = 768,
    src_h: int = 1376,
    moon_distance_m: float = MOON_M,
) -> float:
    """Metres from the aligned ground line to the top of the frame, given the Moon's plate position.

    The Moon center is `moon_distance_m` above the ground. The same px_per_m
    reaches the top of the frame.
    """
    scale, _tx, ty = align_translate(ground_y, src_w, src_h)
    moon_y = moon_cy * src_h * scale + ty
    pixels = GROUND * FRAME_H - moon_y
    if pixels <= 0:
        raise ValueError("the Moon center is not above the ground line")
    return (GROUND * FRAME_H) * moon_distance_m / pixels
