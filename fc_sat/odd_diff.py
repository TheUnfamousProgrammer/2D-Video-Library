"""The three Odd One Out differences. Each one can be measured back from pixels.

Hue, tilt, and a moved dot. A new type needs a measurer before it can ship.
Fills are flat: no antialiasing, no glow.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from fc_sat.color import cvd_distance, lch_to_oklab, oklab_delta, oklab_to_rgb_u8, rgb_u8_to_oklab
from fc_sat.odd_config import DETAIL_OFFSET, DOT_FRACTION, TILT_DEGREES, TierParams

REVEAL_LABELS = {
    "hue": "It was a different color",
    "tilt": "It was tilted",
    "detail": "The dot was off center",
}
DIR_NAMES = (
    "east",
    "southeast",
    "south",
    "southwest",
    "west",
    "northwest",
    "north",
    "northeast",
)
CVD_KINDS = ("protan", "deutan", "tritan")


def reveal_label(kind: str, mode: str = "moved") -> str:
    if kind == "detail" and mode == "missing":
        return "It had no dot"
    if kind not in REVEAL_LABELS:
        raise ValueError(f"no reveal label for {kind!r}")
    return REVEAL_LABELS[kind]


def lab_to_bgr(lab: np.ndarray) -> tuple[int, int, int]:
    rgb = oklab_to_rgb_u8(np.asarray(lab, dtype=np.float64))
    return int(rgb[2]), int(rgb[1]), int(rgb[0])


def choose_odd_lab(
    base: np.ndarray,
    *,
    min_distance: float,
    min_lightness: float,
    min_cvd: float,
    chroma: float,
) -> dict:
    """First grid color that clears distance, lightness, and the CVD floor.

    The floors are not relaxed. If nothing clears them the search raises.
    """
    base = np.asarray(base, dtype=np.float64)
    base_h = math.atan2(float(base[2]), float(base[1]))
    rejected = 0
    light_steps = (0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.18, 0.22, 0.28, 0.34)
    for mag in light_steps:
        for sign in (1.0, -1.0):
            if mag + 1e-9 < min_lightness:
                continue
            lightness = float(np.clip(base[0] + sign * mag, 0.12, 0.92))
            actual_dl = lightness - float(base[0])
            if abs(actual_dl) + 1e-9 < min_lightness:
                continue
            for step in range(0, 36):
                hue = base_h + math.radians(8 + step * 5)
                odd = lch_to_oklab(lightness, chroma, hue)
                distance = oklab_delta(base, odd)
                if distance + 1e-9 < min_distance:
                    rejected += 1
                    continue
                cvd = {kind: cvd_distance(base, odd, kind) for kind in CVD_KINDS}
                if min(cvd.values()) + 1e-9 < min_cvd:
                    rejected += 1
                    continue
                return {
                    "lab": odd,
                    "distance": distance,
                    "lightness_delta": actual_dl,
                    "hue_offset_rad": hue - base_h,
                    "cvd": cvd,
                    "rejected_offsets": rejected,
                    "requested_distance": min_distance,
                    "requested_cvd": min_cvd,
                }
    raise RuntimeError(
        "no hue offset met the OKLab distance, the lightness gap, and the CVD floor of "
        f"{min_cvd:.2f}. Not loosening those rules. Rejected {rejected} candidates. "
        f"Requested distance was {min_distance:.2f}."
    )


def apply_look(
    kind: str,
    tier: TierParams,
    *,
    is_odd: bool,
    base_lab: np.ndarray,
    odd_lab: np.ndarray,
    size: float,
    params: dict | None = None,
) -> dict:
    """Drawable state of one static item. Nothing here depends on time."""
    if kind not in REVEAL_LABELS:
        raise ValueError(f"no measurer for difference {kind!r}")
    params = params or {}
    lab = np.array(odd_lab if is_odd and kind == "hue" else base_lab, dtype=np.float64)
    angle = 0.0
    if is_odd and kind == "tilt":
        angle = float(params.get("tilt_degrees", TILT_DEGREES[tier.tilt_rung]))
    dot = None
    if kind == "detail":
        item_radius = size / 2.0
        radius = float(params.get("dot_radius", float(params.get("dot_fraction", DOT_FRACTION)) * item_radius))
        mode = str(params.get("detail_mode", "moved"))
        if is_odd and mode == "missing":
            dot = None
        elif is_odd and mode == "moved":
            offset = float(params.get("offset", DETAIL_OFFSET[tier.detail_rung]))
            direction = int(params.get("direction", 0)) % 8
            dist = offset * item_radius
            theta = direction * math.pi / 4.0
            dot = {"ox": math.cos(theta) * dist, "oy": math.sin(theta) * dist, "radius": radius}
        else:
            dot = {"ox": 0.0, "oy": 0.0, "radius": radius}
    shape = "square" if kind == "tilt" else "disc"
    return {
        "kind": kind,
        "shape": shape,
        "lab": lab,
        "bgr": lab_to_bgr(lab),
        "size": float(size),
        "angle": angle,
        "dot": dot,
        "is_odd": is_odd,
    }


def _rounded_square_mask(side: int, corner_frac: float, angle_deg: float) -> np.ndarray:
    side = max(2, int(side))
    canvas = int(math.ceil(side * (abs(math.cos(math.radians(angle_deg))) + abs(math.sin(math.radians(angle_deg)))) + 4))
    canvas = max(canvas, side + 2)
    if canvas % 2:
        canvas += 1
    mask = np.zeros((canvas, canvas), dtype=np.uint8)
    radius = max(1, int(round(corner_frac * side)))
    left = (canvas - side) // 2
    top = left
    right = left + side - 1
    bottom = top + side - 1
    cv2.rectangle(mask, (left + radius, top), (right - radius, bottom), 255, -1)
    cv2.rectangle(mask, (left, top + radius), (right, bottom - radius), 255, -1)
    cv2.circle(mask, (left + radius, top + radius), radius, 255, -1, lineType=cv2.LINE_8)
    cv2.circle(mask, (right - radius, top + radius), radius, 255, -1, lineType=cv2.LINE_8)
    cv2.circle(mask, (left + radius, bottom - radius), radius, 255, -1, lineType=cv2.LINE_8)
    cv2.circle(mask, (right - radius, bottom - radius), radius, 255, -1, lineType=cv2.LINE_8)
    if abs(angle_deg) > 1e-3:
        center = (canvas / 2.0, canvas / 2.0)
        matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
        mask = cv2.warpAffine(mask, matrix, (canvas, canvas), flags=cv2.INTER_NEAREST)
    return mask


def _disc_mask(diameter: int) -> np.ndarray:
    diameter = max(2, int(diameter))
    if diameter % 2 == 0:
        diameter += 1
    mask = np.zeros((diameter, diameter), dtype=np.uint8)
    radius = diameter // 2
    cv2.circle(mask, (radius, radius), radius, 255, -1, lineType=cv2.LINE_8)
    return mask


def blit_mask(
    frame: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    cx: float,
    cy: float,
    alpha: float,
) -> None:
    """Stamp a flat color through ``mask``. ``alpha`` blends over the frame."""
    if alpha <= 0:
        return
    height, width = mask.shape
    x0 = int(round(cx)) - width // 2
    y0 = int(round(cy)) - height // 2
    fx0 = max(0, x0)
    fy0 = max(0, y0)
    fx1 = min(frame.shape[1], x0 + width)
    fy1 = min(frame.shape[0], y0 + height)
    if fx0 >= fx1 or fy0 >= fy1:
        return
    crop = mask[fy0 - y0 : fy1 - y0, fx0 - x0 : fx1 - x0]
    on = crop > 0
    if not np.any(on):
        return
    region = frame[fy0:fy1, fx0:fx1]
    if alpha >= 0.999:
        region[on] = color
        return
    painted = region.copy()
    painted[on] = color
    mixed = region.astype(np.float32) * (1.0 - alpha) + painted.astype(np.float32) * alpha
    frame[fy0:fy1, fx0:fx1] = mixed.astype(np.uint8)


def paint_item(
    frame: np.ndarray,
    look: dict,
    cx: float,
    cy: float,
    *,
    corner_frac: float,
    alpha: float = 1.0,
    scale: float = 1.0,
) -> None:
    size = look["size"] * scale
    if look["shape"] == "square":
        mask = _rounded_square_mask(int(round(size)), corner_frac, look["angle"])
    else:
        mask = _disc_mask(int(round(size)))
    blit_mask(frame, mask, look["bgr"], cx * scale, cy * scale, alpha)
    dot = look.get("dot")
    if dot is not None and alpha > 0:
        radius = max(1, int(round(dot["radius"] * scale)))
        dot_mask = _disc_mask(radius * 2 + 1)
        blit_mask(
            frame,
            dot_mask,
            (255, 255, 255),
            cx * scale + dot["ox"] * scale,
            cy * scale + dot["oy"] * scale,
            alpha,
        )


def _patch_lab(frame: np.ndarray, x: float, y: float) -> np.ndarray:
    ix = int(round(x))
    iy = int(round(y))
    patch = frame[iy - 2 : iy + 3, ix - 2 : ix + 3]
    if patch.shape[0] != 5 or patch.shape[1] != 5:
        raise RuntimeError(f"5x5 patch at ({ix},{iy}) left the frame")
    return rgb_u8_to_oklab(patch[..., ::-1]).mean(axis=(0, 1))


def measure_hue(
    frame: np.ndarray,
    odd_xy: tuple[float, float],
    other_xy: list[tuple[float, float]],
) -> float:
    """OKLab distance from the odd center to the median color of the other centers."""
    odd = _patch_lab(frame, odd_xy[0], odd_xy[1])
    others = np.stack([_patch_lab(frame, x, y) for x, y in other_xy], axis=0)
    median = np.median(others, axis=0)
    return oklab_delta(odd, median)


def _axis_delta(angle: float) -> float:
    folded = abs(float(angle)) % 90.0
    if folded > 45.0:
        folded = 90.0 - folded
    return folded


def _cell_angle(frame: np.ndarray, box: tuple[float, float, float, float], bg_bgr: tuple[int, int, int]) -> float:
    x0 = max(0, int(math.floor(box[0])))
    y0 = max(0, int(math.floor(box[1])))
    x1 = min(frame.shape[1], int(math.ceil(box[2])))
    y1 = min(frame.shape[0], int(math.ceil(box[3])))
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        return 0.0
    bg = np.array(bg_bgr, dtype=np.int16)
    dist = np.max(np.abs(crop.astype(np.int16) - bg), axis=-1)
    # The compressed halo rounds a small tilt back to 0. Keep the solid core.
    mask = (dist > 48).astype(np.uint8) * 255
    mask = cv2.erode(mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 8:
        return 0.0
    rect = cv2.minAreaRect(contour)
    return _axis_delta(rect[2])


def measure_tilt(
    frame: np.ndarray,
    cells: list[tuple[float, float, float, float]],
    odd_index: int,
    bg_bgr: tuple[int, int, int],
) -> dict:
    """Odd angle minus the median of the others, in degrees, folded into 0..45."""
    angles = [_cell_angle(frame, box, bg_bgr) for box in cells]
    odd = angles[odd_index]
    others = [angle for index, angle in enumerate(angles) if index != odd_index]
    median = float(np.median(others)) if others else 0.0
    return {"odd": odd, "median": median, "delta": abs(odd - median), "angles": angles}


def _centroid_offset(frame: np.ndarray, x: float, y: float, item_radius: float) -> float:
    """Distance of the white-dot centroid from the item center, in radii.

    The dot is the region clearly brighter than the disc. Comparing against the
    fractional center keeps a centered dot near zero after the 720p re-encode.
    """
    item_radius = max(1.0, float(item_radius))
    ix = int(round(x))
    iy = int(round(y))
    r = int(math.ceil(item_radius)) + 2
    if iy - r < 0 or ix - r < 0 or iy + r + 1 > frame.shape[0] or ix + r + 1 > frame.shape[1]:
        return 0.0
    crop = frame[iy - r : iy + r + 1, ix - r : ix + r + 1]
    yy, xx = np.ogrid[: crop.shape[0], : crop.shape[1]]
    circle = (xx - r) ** 2 + (yy - r) ** 2 <= (item_radius * 0.92) ** 2
    luma = 0.114 * crop[..., 0] + 0.587 * crop[..., 1] + 0.299 * crop[..., 2]
    if not np.any(circle):
        return 0.0
    base = float(np.median(luma[circle]))
    white = circle & (luma >= base + 60.0) & (crop.min(axis=-1) >= 160)
    if int(np.count_nonzero(white)) < 3:
        return 0.0
    ys, xs = np.nonzero(white)
    true_x = float(x) - (ix - r)
    true_y = float(y) - (iy - r)
    return float(math.hypot(float(xs.mean()) - true_x, float(ys.mean()) - true_y) / item_radius)


def measure_detail(
    frame: np.ndarray,
    odd_xy: tuple[float, float],
    other_xy: list[tuple[float, float]],
    item_radius: float,
) -> dict:
    """White-dot centroid offset from the item center, divided by the item radius."""
    odd = _centroid_offset(frame, odd_xy[0], odd_xy[1], item_radius)
    others = [_centroid_offset(frame, x, y, item_radius) for x, y in other_xy]
    peak = float(max(others)) if others else 0.0
    return {"odd": odd, "others": others, "max_other": peak}


def paint_synthetic(
    width: int,
    height: int,
    looks: list[dict],
    centers: list[tuple[float, float]],
    background: tuple[int, int, int],
    *,
    corner_frac: float = 0.18,
) -> np.ndarray:
    frame = np.empty((height, width, 3), dtype=np.uint8)
    frame[:] = background
    for look, (x, y) in zip(looks, centers):
        paint_item(frame, look, x, y, corner_frac=corner_frac)
    return frame
