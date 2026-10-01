"""The four Odd One Out differences.

Each type can draw itself (``apply_look``) and measure itself from pixels.
A new type needs both. v1 is hue, size, spin, and pulse.

Pulse and spin are pause-proof: a single frame does not mark the odd item.
Normals and the odd item share the same amplitude or the same tick length.
Each item draws its own phase. Normals share a frequency, not one brightness.
A shared brightness would show the odd item in a still, which the pause test rejects.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from fc_sat.color import cvd_distance, lch_to_oklab, oklab_delta, oklab_to_rgb_u8, rgb_u8_to_oklab
from fc_sat.odd_config import TierParams

REVEAL_LABELS = {
    "hue": "It was a different color",
    "size": "It was slightly bigger",
    "spin": "It spun the other way",
    "pulse": "It pulsed at a different speed",
}
CVD_KINDS = ("protan", "deutan", "tritan")


def lab_to_bgr(lab: np.ndarray) -> tuple[int, int, int]:
    rgb = oklab_to_rgb_u8(np.asarray(lab, dtype=np.float64))
    return int(rgb[2]), int(rgb[1]), int(rgb[0])


def choose_odd_lab(base: np.ndarray, tier: TierParams) -> dict:
    """Smallest grid match that clears distance, lightness, and every CVD floor.

    Brutal asks for a hue distance of 0.08. The CVD floor stays 0.10, so the
    color that comes back can be farther than 0.08. That is logged, not loosened.
    """
    base = np.asarray(base, dtype=np.float64)
    base_h = math.atan2(float(base[2]), float(base[1]))
    base_c = tier.base_c
    rejected = 0
    light_steps = (0.08, 0.10, 0.12, 0.15, 0.16, 0.18, 0.22, 0.28)
    for mag in light_steps:
        for sign in (1.0, -1.0):
            d_l = sign * mag
            if abs(d_l) + 1e-9 < tier.hue_min_lightness:
                continue
            lightness = float(np.clip(base[0] + d_l, 0.12, 0.92))
            actual_dl = lightness - float(base[0])
            if abs(actual_dl) + 1e-9 < tier.hue_min_lightness:
                continue
            for step in range(0, 36):
                hue = base_h + math.radians(8 + step * 5)
                odd = lch_to_oklab(lightness, base_c, hue)
                distance = oklab_delta(base, odd)
                if distance + 1e-9 < tier.hue_min_distance:
                    rejected += 1
                    continue
                cvd = {kind: cvd_distance(base, odd, kind) for kind in CVD_KINDS}
                if min(cvd.values()) + 1e-9 < tier.cvd_min_distance:
                    rejected += 1
                    continue
                return {
                    "lab": odd,
                    "distance": distance,
                    "lightness_delta": actual_dl,
                    "hue": hue,
                    "hue_offset_rad": hue - base_h,
                    "cvd": cvd,
                    "rejected_offsets": rejected,
                    "requested_distance": tier.hue_min_distance,
                }
    raise RuntimeError(
        "no hue offset met the OKLab distance, the lightness gap, and the CVD floor of "
        f"{tier.cvd_min_distance:.2f}. Not loosening those rules. "
        f"Rejected {rejected} candidates. Requested distance was {tier.hue_min_distance:.2f}."
    )


def apply_look(
    kind: str,
    tier: TierParams,
    *,
    is_odd: bool,
    base_lab: np.ndarray,
    odd_lab: np.ndarray,
    radius: float,
    t: float,
    spin_phase: float,
    pulse_phase: float,
    exaggerate: float,
) -> dict:
    """One item's drawable state at sim time ``t``.

    ``exaggerate`` is 0 or 1. For one second of the reveal the odd item's
    difference is doubled: hue offset, size excess (1.18 becomes 1.36), pulse
    amplitude. The spin tick gets longer and brighter; its speed stays 1.2 rev/s
    so the direction, not a new speed, is what the reveal shows.
    """
    if kind not in REVEAL_LABELS:
        raise ValueError(f"no measurer for difference {kind!r}")
    gain = 1.0 + float(exaggerate) if is_odd else 1.0
    lab = np.array(odd_lab if is_odd else base_lab, dtype=np.float64, copy=True)
    draw_radius = float(radius)
    tick = None
    if kind == "hue" and is_odd and exaggerate:
        lab = np.asarray(base_lab, dtype=np.float64) + gain * (np.asarray(odd_lab, dtype=np.float64) - base_lab)
        lab[0] = float(np.clip(lab[0], 0.08, 0.95))
    elif kind == "size" and is_odd:
        ratio = 1.0 + (tier.size_ratio - 1.0) * gain
        draw_radius = float(radius) * ratio
    elif kind == "spin":
        direction = -1.0 if is_odd else 1.0
        angle = float(spin_phase + direction * tier.spin_rev_s * 2.0 * math.pi * t)
        longer = 1.0 + (0.55 if is_odd and exaggerate else 0.0)
        tick = {
            "angle": angle,
            "inner": 0.25 / longer,
            "outer": min(0.98, 0.85 * longer),
            "width": 3.0 + (3.0 if is_odd and exaggerate else 0.0),
            "alpha": 1.0 if not (is_odd and exaggerate) else 1.0,
            "bright": bool(is_odd and exaggerate),
        }
    elif kind == "pulse":
        amp = tier.pulse_amplitude * (gain if is_odd else 1.0)
        freq = tier.pulse_odd_hz if is_odd else tier.pulse_normal_hz
        wave = math.sin(2.0 * math.pi * freq * t + pulse_phase)
        lab = np.array(lab, dtype=np.float64, copy=True)
        lab[0] = float(np.clip(lab[0] * (1.0 + amp * wave), 0.05, 0.98))
    return {
        "lab": lab,
        "radius": draw_radius,
        "tick": tick,
        "kind": kind,
        "is_odd": is_odd,
    }


def _patch_lab(frame: np.ndarray, x: float, y: float) -> np.ndarray:
    ix = int(round(x))
    iy = int(round(y))
    patch = frame[iy - 2 : iy + 3, ix - 2 : ix + 3]
    if patch.shape[0] != 5 or patch.shape[1] != 5:
        raise RuntimeError(f"5x5 patch at ({ix},{iy}) left the frame")
    return rgb_u8_to_oklab(patch[..., ::-1]).mean(axis=(0, 1))


def measure_hue(frames: list[np.ndarray], odd_xy: list[tuple[float, float]], other_xy: list[list[tuple[float, float]]]) -> float:
    """Mean OKLab distance from the odd center to each comparison item, averaged over frames."""
    odd = np.mean([_patch_lab(frame, x, y) for frame, (x, y) in zip(frames, odd_xy)], axis=0)
    distances = []
    for index in range(len(other_xy[0])):
        other = np.mean(
            [_patch_lab(frame, points[index][0], points[index][1]) for frame, points in zip(frames, other_xy)],
            axis=0,
        )
        distances.append(oklab_delta(odd, other))
    return float(np.mean(distances))


def _area(frame: np.ndarray, x: float, y: float, window: float, bg_lab: np.ndarray) -> int:
    radius = int(math.ceil(window))
    ix = int(round(x))
    iy = int(round(y))
    if iy - radius < 0 or ix - radius < 0 or iy + radius + 1 > frame.shape[0] or ix + radius + 1 > frame.shape[1]:
        return 0
    crop = frame[iy - radius : iy + radius + 1, ix - radius : ix + radius + 1]
    lab = rgb_u8_to_oklab(crop[..., ::-1])
    dist = np.linalg.norm(lab - bg_lab, axis=-1)
    yy, xx = np.ogrid[: crop.shape[0], : crop.shape[1]]
    circle = (xx - radius) ** 2 + (yy - radius) ** 2 <= window * window
    return int(np.count_nonzero((dist > 0.08) & circle))


def measure_size(
    frame: np.ndarray,
    odd_xy: tuple[float, float],
    other_xy: list[tuple[float, float]],
    window: float,
    bg_lab: np.ndarray,
) -> dict:
    odd_area = _area(frame, odd_xy[0], odd_xy[1], window, bg_lab)
    others = [_area(frame, x, y, window, bg_lab) for x, y in other_xy]
    median = float(np.median(others)) if others else 0.0
    ratio = float(odd_area / median) if median > 0 else 0.0
    return {"odd_area": odd_area, "median_area": median, "ratio": ratio}


def _brightest_angle(frame: np.ndarray, x: float, y: float, radius: float) -> float:
    count = 24
    angles = np.linspace(0.0, 2.0 * math.pi, count, endpoint=False)
    ring = 0.6 * radius
    values = np.empty(count, dtype=np.float64)
    for index, angle in enumerate(angles):
        px = int(round(x + math.cos(angle) * ring))
        py = int(round(y + math.sin(angle) * ring))
        patch = frame[py - 1 : py + 2, px - 1 : px + 2]
        values[index] = float(patch.mean()) if patch.size else 0.0
    peak = int(np.argmax(values))
    left = values[(peak - 1) % count]
    mid = values[peak]
    right = values[(peak + 1) % count]
    denom = left - 2.0 * mid + right
    delta = 0.0 if abs(denom) < 1e-6 else float(np.clip(0.5 * (left - right) / denom, -1.0, 1.0))
    return float(angles[peak] + delta * (2.0 * math.pi / count))


def spin_from_angles(
    angles,
    fps: float,
    *,
    expect_sign: float,
    nominal_rev_s: float,
) -> dict:
    """Unwrap angles of the brightest ring sample. ``expect_sign`` is +1 clockwise."""
    series = np.unwrap(np.asarray(angles, dtype=np.float64))
    win = max(3, int(round(0.5 * fps)))
    nominal = nominal_rev_s * 2.0 * math.pi
    if len(series) <= win:
        return {"sign_fraction": 0.0, "omega": 0.0, "nominal": nominal, "windows": 0}
    signs = []
    magnitudes = []
    step = max(1, win // 2)
    for start in range(0, len(series) - win, step):
        vel = np.diff(series[start : start + win]) * fps
        med = float(np.median(vel))
        signs.append(math.copysign(1.0, med) == math.copysign(1.0, expect_sign) if abs(med) > 1e-6 else False)
        magnitudes.append(abs(med))
    return {
        "sign_fraction": float(np.mean(signs)) if signs else 0.0,
        "omega": float(np.median(magnitudes)) if magnitudes else 0.0,
        "nominal": nominal,
        "windows": len(signs),
    }


def measure_spin(
    frames: list[np.ndarray],
    points: list[tuple[float, float]],
    radius: float,
    fps: float,
    *,
    expect_sign: float,
    nominal_rev_s: float,
) -> dict:
    """Unwrap the brightest point on a 0.6 r ring. ``expect_sign`` is +1 clockwise."""
    angles = [_brightest_angle(frame, x, y, radius) for frame, (x, y) in zip(frames, points)]
    return spin_from_angles(angles, fps, expect_sign=expect_sign, nominal_rev_s=nominal_rev_s)


def center_luma(frame: np.ndarray, x: float, y: float) -> float:
    ix = int(round(x))
    iy = int(round(y))
    patch = frame[iy - 2 : iy + 3, ix - 2 : ix + 3].astype(np.float64)
    if patch.shape[0] != 5 or patch.shape[1] != 5:
        return 0.0
    luma = 0.0722 * patch[..., 0] + 0.7152 * patch[..., 1] + 0.2126 * patch[..., 2]
    return float(luma.mean() / 255.0)


def pulse_peak_hz(series, fps: float) -> float:
    """Peak frequency of a brightness series, in Hz. The DC bin is ignored."""
    values = np.asarray(series, dtype=np.float64)
    if len(values) < 8:
        return 0.0
    values = values - float(values.mean())
    window = np.hanning(len(values))
    spec = np.abs(np.fft.rfft(values * window))
    freqs = np.fft.rfftfreq(len(values), d=1.0 / fps)
    spec[0] = 0.0
    return float(freqs[int(np.argmax(spec))])


def measure_pulse(frames: list[np.ndarray], points: list[tuple[float, float]], fps: float) -> float:
    """Peak frequency of the item-center brightness, in Hz."""
    series = [center_luma(frame, x, y) for frame, (x, y) in zip(frames, points)]
    return pulse_peak_hz(series, fps)


def paint_synthetic(width: int, height: int, looks: list[dict], background: tuple[int, int, int]) -> np.ndarray:
    """Solid discs and ticks, no bloom. Used by the round-trip tests."""
    frame = np.empty((height, width, 3), dtype=np.uint8)
    frame[:] = background
    for look in looks:
        color = lab_to_bgr(look["lab"])
        center = (int(round(look["x"])), int(round(look["y"])))
        cv2.circle(frame, center, max(1, int(round(look["radius"]))), color, -1, lineType=cv2.LINE_AA)
        tick = look.get("tick")
        if tick is not None:
            radius = float(look["radius"])
            x0 = int(round(look["x"] + math.cos(tick["angle"]) * radius * tick["inner"]))
            y0 = int(round(look["y"] + math.sin(tick["angle"]) * radius * tick["inner"]))
            x1 = int(round(look["x"] + math.cos(tick["angle"]) * radius * tick["outer"]))
            y1 = int(round(look["y"] + math.sin(tick["angle"]) * radius * tick["outer"]))
            cv2.line(frame, (x0, y0), (x1, y1), (255, 255, 255), int(round(tick["width"])), lineType=cv2.LINE_AA)
    return frame
