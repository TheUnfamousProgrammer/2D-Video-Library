"""Locate supplied art, fit plates to the frame, and key the magenta cutouts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from fc_sat.paperfold_math import BURJ_M, PERSON_M
from fc_sat.paperfold_scenes import FRAME_H, FRAME_W, GROUND, LIMIT, align_translate

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "assets" / "art" / "MANIFEST.yaml"
ART_ROOT = ROOT / "assets" / "art"

# Written after looking at the aligned plates. The stack stands on the center line.
BUSY_REGIONS = (
    (
        "p01_hook_night_desk_moon",
        "The moon fills the center lane. Crater texture sits behind the stack, and the cream disk is under 3:1 against white paper.",
    ),
    (
        "p02_street_day",
        "A flat cream sky fills the gap between the buildings. That is the whole center lane, so a white stack has almost no separation (1.04:1).",
    ),
    (
        "p03_city_day",
        "A cream tower stands in the center lane, with windowed towers on both sides. The stack and that tower are nearly the same value (1.10:1).",
    ),
    (
        "p04_mountains_clouds",
        "The upper sky clears 3:1. Pale clouds and the cream valley still cross the center lane lower down, where the stack base will sit.",
    ),
    (
        "p05_upper_atmosphere",
        "Open dark sky. Nothing busy behind the stack.",
    ),
    (
        "p06_low_orbit_limb",
        "The grey cap is now #1B2030. Stars are small and sit off the stack edges.",
    ),
    (
        "p07_deep_space_moon",
        "The teal cap is now near-black sky. The Moon stays on the right. Earth's apex is at y = 0.58, so the cap shows above the bottom band.",
    ),
)


def art_id(filename: str) -> str:
    """Id is the prefix before a `.png_` or `_png_` upload suffix."""
    name = Path(filename).name
    for token in (".png_", "_png_"):
        if token in name:
            return name.split(token, 1)[0]
    return Path(name).stem


def load_manifest(path: Path | None = None) -> dict:
    path = path or MANIFEST_PATH
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise SystemExit(f"{path} is not a mapping")
    return raw


def find_source(folder: Path, asset_id: str) -> Path | None:
    hits = sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"} and art_id(path.name) == asset_id
    )
    return hits[0] if hits else None


def missing_assets(root: Path | None = None, manifest: dict | None = None) -> list[str]:
    root = root or ART_ROOT
    manifest = manifest or load_manifest()
    missing = []
    for kind in ("plates", "objects"):
        folder = root / kind
        if not folder.is_dir():
            missing.extend(item["id"] for item in manifest.get(kind) or [])
            continue
        for item in manifest.get(kind) or []:
            if find_source(folder, item["id"]) is None:
                missing.append(item["id"])
    return missing


TARGET_ASPECT = FRAME_W / FRAME_H
ASPECT_TOLERANCE = 0.015
PANEL_BGR = (0x30, 0x20, 0x1B)  # #1B2030


@dataclass(frozen=True)
class PlateResult:
    asset_id: str
    notes: tuple[str, ...]
    contrast: float
    image: np.ndarray


@dataclass(frozen=True)
class KeyResult:
    asset_id: str
    border_magenta: float
    bbox: tuple[int, int, int, int]
    image: np.ndarray


def aspect_note(width: int, height: int) -> str | None:
    aspect = width / height
    if abs(aspect / TARGET_ASPECT - 1.0) > ASPECT_TOLERANCE:
        return f"WRONG ASPECT {aspect:.4f} against {TARGET_ASPECT:.4f}"
    return None


def denoise_chroma(bgr: np.ndarray) -> np.ndarray:
    """Bilateral filter on Cb and Cr only. Luma, and therefore edges, stay put."""
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    cr = cv2.bilateralFilter(cr, 7, 16, 8)
    cb = cv2.bilateralFilter(cb, 7, 16, 8)
    return cv2.cvtColor(cv2.merge([y, cr, cb]), cv2.COLOR_YCrCb2BGR)


def _runs(indexes: np.ndarray) -> list[tuple[int, int]]:
    if len(indexes) == 0:
        return []
    groups = []
    start = prev = int(indexes[0])
    for value in indexes[1:]:
        value = int(value)
        if value == prev + 1:
            prev = value
            continue
        groups.append((start, prev))
        start = prev = value
    groups.append((start, prev))
    return groups


FILL_NOISE = 1.5
FILL_BLEND = 48


def _column_color(band: np.ndarray) -> np.ndarray:
    """One colour per column: the median of the band, skipping black specks and white clouds."""
    values = band.astype(np.float32)
    luma = values.max(axis=2)
    keep = (luma > 45.0) & (luma < 170.0)
    width = values.shape[1]
    color = np.empty((width, 3), np.float32)
    kept = values[keep]
    fallback = np.median(kept, axis=0) if kept.size else np.median(values.reshape(-1, 3), axis=0)
    for x in range(width):
        chosen = values[keep[:, x], x]
        color[x] = np.median(chosen, axis=0) if chosen.shape[0] >= 3 else fallback
    return cv2.GaussianBlur(color.reshape(1, width, 3), (0, 0), 40).reshape(width, 3)


def _paint_rows(image: np.ndarray, y0: int, y1: int, color: np.ndarray, rng: np.random.Generator) -> None:
    """Per-column colour plus Gaussian noise. Float math, then round. No clamp on the noise."""
    rows = y1 - y0 + 1
    noise = rng.normal(0.0, FILL_NOISE, size=(rows, color.shape[0], 3)).astype(np.float32)
    band = color[None, :, :] + noise
    fade = min(FILL_BLEND, rows)
    if y0 > 0 and fade > 1:
        edge = _column_color(image[max(0, y0 - 8) : y0])
        # Start on the real boundary row when it already matches that colour,
        # so a dark speck in the last row cannot paint a dark ramp.
        boundary = image[y0 - 1].astype(np.float32)
        close = np.max(np.abs(boundary - edge), axis=1) <= 12.0
        edge = np.where(close[:, None], boundary, edge)
        edge = cv2.GaussianBlur(edge.reshape(1, -1, 3), (0, 0), 3).reshape(-1, 3)
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)[:, None, None]
        band[:fade] = edge[None, :, :] * (1.0 - ramp) + band[:fade] * ramp
    image[y0 : y1 + 1] = np.rint(band)


def _row_color(band: np.ndarray) -> np.ndarray:
    """One colour per row, skipping black specks and white clouds."""
    values = band.astype(np.float32)
    luma = values.max(axis=2)
    keep = (luma > 45.0) & (luma < 170.0)
    height = values.shape[0]
    color = np.empty((height, 3), np.float32)
    fallback = np.median(values.reshape(-1, 3), axis=0)
    for y in range(height):
        chosen = values[y][keep[y]]
        color[y] = np.median(chosen, axis=0) if chosen.shape[0] >= 3 else fallback
    return cv2.GaussianBlur(color.reshape(height, 1, 3), (0, 0), 6).reshape(height, 3)


def _paint_cols(image: np.ndarray, x0: int, x1: int, color: np.ndarray, rng: np.random.Generator) -> None:
    cols = x1 - x0 + 1
    noise = rng.normal(0.0, FILL_NOISE, size=(color.shape[0], cols, 3)).astype(np.float32)
    band = color[:, None, :] + noise
    fade = min(FILL_BLEND, cols)
    if x0 > 0 and fade > 1:
        edge = _row_color(image[:, max(0, x0 - 8) : x0])
        boundary = image[:, x0 - 1].astype(np.float32)
        close = np.max(np.abs(boundary - edge), axis=1) <= 12.0
        edge = np.where(close[:, None], boundary, edge)
        edge = cv2.GaussianBlur(edge.reshape(-1, 1, 3), (0, 0), 3).reshape(-1, 3)
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)[None, :, None]
        band[:, :fade] = edge[:, None, :] * (1.0 - ramp) + band[:, :fade] * ramp
    image[:, x0 : x1 + 1] = np.rint(band)


def max_fill_neighbor_step(image: np.ndarray, mask: np.ndarray) -> float:
    """Largest mean absolute step between neighbouring rows inside a fill."""
    steps = []
    for y in range(1, image.shape[0]):
        both = mask[y] & mask[y - 1]
        if int(both.sum()) < image.shape[1] // 2:
            continue
        delta = np.abs(image[y].astype(np.float32) - image[y - 1].astype(np.float32)).mean(axis=1)
        steps.append(float(delta[both].mean()))
    return max(steps) if steps else 0.0


def _fill_uncovered(placed: np.ndarray, cover: np.ndarray) -> tuple[int, np.ndarray, float]:
    """Fill warp gaps from each column's own bottom-band colour, plus sigma-1.5 noise."""
    covered = cover >= 0.5
    mask = np.zeros(covered.shape, dtype=bool)
    if covered.all():
        return 0, mask, 0.0
    rng = np.random.default_rng(7)
    filled = 0
    row_gap = covered.mean(axis=1) < 0.05
    row_ok = ~row_gap
    for y0, y1 in _runs(np.nonzero(row_gap)[0]):
        hits = np.nonzero(row_ok)[0]
        if len(hits) == 0:
            continue
        if y0 == 0:
            sample = placed[int(hits[0]) : int(hits[0]) + 24]
        else:
            sample = placed[max(0, y0 - 24) : y0]
        if sample.size == 0:
            continue
        _paint_rows(placed, y0, y1, _column_color(sample), rng)
        mask[y0 : y1 + 1] = True
        filled += int(mask[y0 : y1 + 1].sum())
    # Side gaps use the same per-row colour plus sigma-1.5 noise.
    col_gap = covered.mean(axis=0) < 0.05
    col_ok = ~col_gap
    for x0, x1 in _runs(np.nonzero(col_gap)[0]):
        hits = np.nonzero(col_ok)[0]
        if len(hits) == 0:
            continue
        if x0 == 0:
            sample = placed[:, int(hits[0]) : int(hits[0]) + 24]
        else:
            sample = placed[:, max(0, x0 - 24) : x0]
        if sample.size == 0:
            continue
        _paint_cols(placed, x0, x1, _row_color(sample), rng)
        fresh = ~mask[:, x0 : x1 + 1]
        mask[:, x0 : x1 + 1] = True
        filled += int(fresh.sum())
    return filled, mask, max_fill_neighbor_step(placed, mask)


def place_plate(
    bgr: np.ndarray, ground_y: float, screen_ground: float = GROUND
) -> tuple[np.ndarray, list[str]]:
    """Scale to cover, align the ground line, and crop to 1080x1920.

    Rows the shift leaves empty are a sampled flat colour plus matched grain.
    p06 and p07 pass screen_ground 0.58 so the Earth's apex sits higher than 0.68.
    """
    src_h, src_w = bgr.shape[:2]
    notes = []
    wrong = aspect_note(src_w, src_h)
    if wrong:
        notes.append(wrong)
    if src_w < FRAME_W or src_h < FRAME_H:
        notes.append("UPSCALED")
    scale, tx, ty = align_translate(ground_y, src_w, src_h, screen_ground=screen_ground)
    notes.append(f"scale {scale:.4f} tx {tx:.1f} ty {ty:.1f} ground {screen_ground:.2f}")
    matrix = np.array([[scale, 0.0, tx], [0.0, scale, ty]], dtype=np.float32)
    placed = cv2.warpAffine(
        bgr.astype(np.float32),
        matrix,
        (FRAME_W, FRAME_H),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    cover = cv2.warpAffine(
        np.ones((src_h, src_w), np.float32),
        matrix,
        (FRAME_W, FRAME_H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    if "UPSCALED" in notes:
        soft = cv2.GaussianBlur(placed, (0, 0), 1.0)
        sharp = np.clip(placed * 1.15 - soft * 0.15, 0, 255)
        mask = cover >= 0.5
        placed[mask] = sharp[mask]
    filled, mask, step = _fill_uncovered(placed, cover)
    if filled:
        notes.append(f"grain fill {filled} px")
        notes.append(f"fill step {step:.2f}")
    return np.clip(np.rint(placed), 0, 255).astype(np.uint8), notes


def recolor_grey_cap(bgr: np.ndarray) -> int:
    """Paint a neutral grey cap at the top of the frame with the panel colour. Returns the row count."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    rows = 0
    for y in range(bgr.shape[0]):
        saturation = float(np.median(hsv[y, :, 1]))
        value = float(np.median(hsv[y, :, 2]))
        if saturation < 30 and 40 <= value <= 110:
            rows += 1
            continue
        break
    if rows == 0:
        return 0
    band = hsv[:rows]
    neutral = (band[:, :, 1] < 40) & (band[:, :, 2] > 30)
    region = bgr[:rows]
    region[neutral] = PANEL_BGR
    return rows


def _relative_luma(rgb: np.ndarray) -> float:
    def channel(value: float) -> float:
        value = value / 255.0
        if value <= 0.04045:
            return value / 12.92
        return ((value + 0.055) / 1.055) ** 2.4

    red, green, blue = (channel(float(rgb[0])), channel(float(rgb[1])), channel(float(rgb[2])))
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def center_lane_contrast(bgr: np.ndarray) -> float:
    """White stack (#F2F4F8) against the median colour of the centre lane."""
    height, width = bgr.shape[:2]
    y0, y1 = int(0.32 * height), int(0.62 * height)
    x0, x1 = width // 2 - 24, width // 2 + 24
    patch = bgr[y0:y1, x0:x1, ::-1].reshape(-1, 3)
    median = np.median(patch, axis=0)
    paper = _relative_luma(np.array([242, 244, 248], dtype=np.float32))
    background = _relative_luma(median)
    lighter = max(paper, background)
    darker = min(paper, background)
    return (lighter + 0.05) / (darker + 0.05)


def key_magenta(bgr: np.ndarray) -> tuple[np.ndarray, float, tuple[int, int, int, int]]:
    """Chroma-key #FF00FF. Returns BGRA, the pure-magenta fraction of the border, and the alpha box."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)
    magenta = (hue >= 140) & (hue <= 165) & (sat >= 70) & (val >= 80)
    border = np.zeros(magenta.shape, dtype=bool)
    border[:8, :] = True
    border[-8:, :] = True
    border[:, :8] = True
    border[:, -8:] = True
    fraction = float(magenta[border].mean()) if border.any() else 0.0
    rgb = bgr[:, :, ::-1].astype(np.int16)
    red, green, blue = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    spill = np.clip(np.minimum(red, blue) - green, 0, None)
    red = np.clip(red - spill, 0, 255)
    blue = np.clip(blue - spill, 0, 255)
    alpha = np.where(magenta, 0, 255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    alpha = cv2.erode(alpha, kernel)
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0.5)
    bgra = np.dstack([blue.astype(np.uint8), green.astype(np.uint8), red.astype(np.uint8), alpha])
    ys, xs = np.nonzero(alpha > 16)
    if len(xs) == 0:
        box = (0, 0, 0, 0)
    else:
        box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    return bgra, fraction, box


def detect_circle(
    bgr: np.ndarray, hint: tuple[float, float, float] | None = None
) -> tuple[float, float, float] | None:
    grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    grey = cv2.GaussianBlur(grey, (9, 9), 1.5)
    height, width = grey.shape
    circles = cv2.HoughCircles(
        grey,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=height // 3,
        param1=120,
        param2=40,
        minRadius=int(width * 0.18),
        maxRadius=int(width * 0.48),
    )
    if circles is None:
        return None
    rows = circles[0]
    if hint is None:
        chosen = max(rows, key=lambda row: row[2])
    else:
        # The plate also contains Earth. Keep the circle nearest the manifest moon.
        def distance(row: np.ndarray) -> float:
            return (
                abs(float(row[0]) / width - hint[0])
                + abs(float(row[1]) / height - hint[1])
                + abs(float(row[2]) / width - hint[2])
            )

        chosen = min(rows, key=distance)
    x, y, radius = chosen
    return float(x) / width, float(y) / height, float(radius) / width


def restyle_deep_space_sky(
    bgr: np.ndarray, moon_cx: float, moon_cy: float, moon_radius: float
) -> int:
    """Replace the flat teal cap with near-black sky and a few stars.

    The mask feathers over 32 px so the boundary is not a horizontal seam.
    The Moon disc is left untouched.
    """
    height, width = bgr.shape[:2]
    band = np.median(bgr[:8, :48], axis=(0, 1))
    distance = np.linalg.norm(bgr.astype(np.float32) - band, axis=2)
    yy, xx = np.mgrid[0:height, 0:width]
    moon = (xx - moon_cx * width) ** 2 + (yy - moon_cy * height) ** 2 <= (moon_radius * width) ** 2
    hard = ((distance < 26) & ~moon).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    hard = cv2.morphologyEx(hard, cv2.MORPH_CLOSE, kernel)
    # Cover the bright rim, then feather into the dark sky past it.
    hard = cv2.dilate(hard, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    guard = cv2.dilate(moon.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) > 0
    hard[guard] = 0
    if int(hard.sum()) == 0:
        return 0
    inside = cv2.distanceTransform(hard, cv2.DIST_L2, 5)
    outside = cv2.distanceTransform(1 - hard, cv2.DIST_L2, 5)
    weight = np.clip(0.5 + (inside - outside) / 32.0, 0.0, 1.0)
    weight[moon] = 0.0
    sky = np.empty_like(bgr)
    sky[:] = (8, 10, 12)
    ys, xs = np.nonzero(weight > 0.92)
    rng = np.random.default_rng(7)
    if len(ys):
        picks = rng.choice(len(ys), size=min(12, len(ys)), replace=False)
        for index in picks:
            radius = int(rng.integers(1, 3))
            cv2.circle(sky, (int(xs[index]), int(ys[index])), radius, (242, 244, 248), -1, cv2.LINE_AA)
    soft = weight[:, :, None]
    blended = bgr.astype(np.float32) * (1.0 - soft) + sky.astype(np.float32) * soft
    bgr[:] = np.clip(blended, 0, 255).astype(np.uint8)
    _smooth_sky_horizon(bgr, moon)
    return int((weight > 0.5).sum())


def clean_earth_cap(bgr: np.ndarray, apex_y: int) -> int:
    """Drop isolated mask-edge specks just outside the Earth cap. The cap itself stays."""
    height = bgr.shape[0]
    apex_y = int(np.clip(apex_y, 1, height - 1))
    blue = (bgr[:, :, 0] > 110) & (bgr[:, :, 0] > bgr[:, :, 2] + 20) & (bgr[:, :, 0] > bgr[:, :, 1])
    blue[: max(0, apex_y - 36)] = False
    count, labels, stats, _ = cv2.connectedComponentsWithStats(blue.astype(np.uint8), 8)
    if count <= 1:
        return 0
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    earth = labels == biggest
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    ring = (cv2.dilate(earth.astype(np.uint8), kernel) > 0) & ~earth
    sky_band = bgr[max(0, apex_y - 56) : max(1, apex_y - 12)]
    if sky_band.size == 0:
        return 0
    sky = np.median(sky_band.reshape(-1, 3), axis=0)
    dist = np.linalg.norm(bgr.astype(np.float32) - sky, axis=2)
    odd = (ring & (dist > 22)).astype(np.uint8)
    pieces, lab, piece_stats, _ = cv2.connectedComponentsWithStats(odd, 8)
    speckle = np.zeros(earth.shape, dtype=bool)
    for index in range(1, pieces):
        if piece_stats[index, cv2.CC_STAT_AREA] <= 12:
            speckle |= lab == index
    replaced = int(speckle.sum())
    if replaced:
        bgr[speckle] = np.clip(sky, 0, 255).astype(np.uint8)
    return replaced


def clear_earth_silhouettes(bgr: np.ndarray) -> int:
    """Inpaint the jagged limb shapes into the surrounding sky. The disc itself stays."""
    rgb = bgr[:, :, ::-1]
    cloud = (rgb[:, :, 0] > 195) & (rgb[:, :, 1] > 180) & (rgb[:, :, 2] > 160) & (bgr.max(axis=2) > 180)
    cloud[:1060] = False
    cloud[1310:] = False
    earth = (bgr[:, :, 0] > 90) & (bgr[:, :, 0] > bgr[:, :, 2] + 8) & (bgr.max(axis=2) > 85) & ~cloud
    earth[1320:] = False
    earth[:1040] = False
    if int(earth.sum()) < 10:
        return 0
    side = np.zeros(bgr.shape[:2], dtype=bool)
    side[1220:1410, :250] = True
    side[1220:1410, 830:] = True
    target = side & (bgr.max(axis=2) < 85) & ~earth & ~cloud
    removed = int(target.sum())
    if removed == 0:
        return 0
    painted = cv2.inpaint(bgr, target.astype(np.uint8) * 255, 15, cv2.INPAINT_TELEA)
    bgr[target] = painted[target]
    return removed


def _smooth_sky_horizon(bgr: np.ndarray, moon: np.ndarray, y_center: int = 250) -> None:
    """Blur the sky vertically across the old teal edge so that row is not a line."""
    y0 = max(0, y_center - 48)
    y1 = min(bgr.shape[0], y_center + 48)
    strip = bgr[y0:y1].astype(np.float32)
    # Stars are punched out before the blur so a bright pixel cannot smear into a line.
    sky_color = np.array([8.0, 10.0, 12.0], np.float32)
    stars = strip.max(axis=2) > 40.0
    clean = strip.copy()
    clean[stars] = sky_color
    blurred = cv2.GaussianBlur(clean, (1, 51), 0.01, sigmaY=16)
    ramp = np.minimum(np.arange(strip.shape[0]), np.arange(strip.shape[0])[::-1]).astype(np.float32)
    ramp = np.clip(ramp / 20.0, 0.0, 1.0)
    mix = ramp[:, None] * (~moon[y0:y1]).astype(np.float32)
    mix[stars] = 0.0
    bgr[y0:y1] = np.rint(strip * (1.0 - mix[:, :, None]) + blurred * mix[:, :, None]).astype(np.uint8)


def soften_cloud_shadows(bgr: np.ndarray) -> int:
    """Replace the painted cloud halo with a 4 px offset, 6 px blur, alpha 0.20."""
    rgb = bgr[:, :, ::-1]
    cloud = (rgb[:, :, 0] > 195) & (rgb[:, :, 1] > 180) & (rgb[:, :, 2] > 160) & (bgr.max(axis=2) > 180)
    cloud[:1060] = False
    cloud[1310:] = False
    if int(cloud.sum()) < 20:
        return 0
    earth = (bgr[:, :, 0] > 90) & (bgr[:, :, 0] > bgr[:, :, 2] + 8) & (bgr.max(axis=2) > 85) & ~cloud
    earth[1320:] = False
    earth[:1040] = False
    if int(earth.sum()) < 10:
        return 0
    earth_color = np.median(bgr[earth], axis=0)
    near = cv2.dilate(cloud.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))) > 0
    disc = cv2.dilate(earth.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))) > 0
    painted = (near | disc) & ~cloud & ~earth & (bgr.max(axis=2) < 100)
    replaced = int(painted.sum())
    bgr[painted] = np.rint(earth_color).astype(np.uint8)
    shifted = np.zeros(cloud.shape, np.float32)
    shifted[4:, 4:] = cloud[:-4, :-4].astype(np.float32)
    blur = cv2.GaussianBlur(shifted, (0, 0), 6)
    alpha = np.clip(blur, 0.0, 1.0) * 0.20
    alpha[cloud] = 0.0
    alpha[~disc] = 0.0
    base = bgr.astype(np.float32)
    bgr[:] = np.rint(base * (1.0 - alpha[:, :, None])).astype(np.uint8)
    return replaced


def rebuild_below(image: np.ndarray, y0: int = 1306) -> tuple[int, float]:
    """Replace every row from y0 down with per-column bottom-band colour plus sigma-1.5 noise."""
    if image.shape[0] <= y0:
        return 0, 0.0
    work = image.astype(np.float32)
    sample = work[max(0, y0 - 24) : y0]
    _paint_rows(work, y0, work.shape[0] - 1, _column_color(sample), np.random.default_rng(11))
    rounded = np.rint(work)
    if float(rounded[y0:].min()) < 0.0 or float(rounded[y0:].max()) > 255.0:
        raise RuntimeError("bottom fill left the 0-255 range without a clamp")
    image[:] = rounded.astype(np.uint8)
    mask = np.zeros(image.shape[:2], dtype=bool)
    mask[y0:] = True
    return int(mask.sum()), max_fill_neighbor_step(image, mask)


def finish_iss_key(bgra: np.ndarray) -> np.ndarray:
    """Erode the ISS matte by 2 px and pull the pink fringe back toward the station colour."""
    alpha = bgra[:, :, 3]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    eroded = cv2.erode(alpha, kernel)
    edge = cv2.dilate((eroded < 16).astype(np.uint8), kernel) > 0
    color = bgra[:, :, :3].astype(np.int16)
    blue, green, red = color[:, :, 0], color[:, :, 1], color[:, :, 2]
    pink = edge & (eroded > 0) & (red > green + 18) & (blue + 8 > green) & (red > 90)
    red = np.where(pink, green + (red - green) // 5, red)
    blue = np.where(pink, green + np.maximum(blue - green, 0) // 5, blue)
    out = bgra.copy()
    out[:, :, 0] = np.clip(blue, 0, 255).astype(np.uint8)
    out[:, :, 1] = np.clip(green, 0, 255).astype(np.uint8)
    out[:, :, 2] = np.clip(red, 0, 255).astype(np.uint8)
    out[:, :, 3] = eroded
    return out


def mask_moon_circle(bgra: np.ndarray) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Keep a perfect circle. Everything outside it, including the teal field, goes clear."""
    color = bgra[:, :, :3]
    alpha = bgra[:, :, 3]
    opaque = alpha > 16
    border = np.zeros(opaque.shape, dtype=bool)
    border[:12, :] = True
    border[-12:, :] = True
    border[:, :12] = True
    border[:, -12:] = True
    samples = color[border & opaque]
    if len(samples) == 0:
        samples = color[opaque]
    background = np.median(samples, axis=0)
    distance = np.linalg.norm(color.astype(np.float32) - background, axis=2)
    content = opaque & (distance > 28)
    ys, xs = np.nonzero(content)
    if len(xs) < 30:
        raise SystemExit("o06_moon has no disc to mask")
    points = np.ascontiguousarray(np.stack([xs, ys], axis=1).astype(np.float32))
    (cx, cy), radius = cv2.minEnclosingCircle(points)
    yy, xx = np.mgrid[0:bgra.shape[0], 0:bgra.shape[1]]
    inside = (xx - cx) ** 2 + (yy - cy) ** 2 <= (radius + 0.5) ** 2
    out = bgra.copy()
    out[:, :, 3] = np.where(inside, alpha, 0).astype(np.uint8)
    leftover = (out[:, :, 3] > 16) & ~inside
    if leftover.any():
        raise SystemExit(f"non-moon pixels remain outside the circle: {int(leftover.sum())}")
    return out, (float(cx), float(cy), float(radius))


# Tallest on-plate reference the decorative towers are checked against.
STRUCTURE_REFERENCES = {
    "street": ("person", PERSON_M),
    "city": ("Burj Khalifa", BURJ_M),
}


def tallest_structure_m(bgr: np.ndarray, screen_ground: float, world_m: float) -> float:
    """Height of the tallest ground-rooted shape, in metres at this plate's scale."""
    height, width = bgr.shape[:2]
    ground = int(round(screen_ground * height))
    sky_y = max(0, int(0.16 * height))
    sky = np.median(bgr[sky_y : sky_y + 16, width // 2 - 24 : width // 2 + 24], axis=(0, 1))
    diff = np.abs(bgr.astype(np.int16) - sky.astype(np.int16)).sum(axis=2)
    floor = int(0.08 * height)
    tallest = 0
    for x in range(16, width - 16, 3):
        y = ground - 1
        while y > floor and diff[y, x] > 48:
            y -= 1
        tallest = max(tallest, ground - y)
    ppm = (screen_ground * height) / world_m
    if ppm <= 0:
        return 0.0
    return tallest / ppm


def placeholder(label: str, width: int, height: int) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:] = (23, 17, 14)
    cv2.putText(image, label, (24, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (242, 244, 248), 2, cv2.LINE_AA)
    return image


def _checker(width: int, height: int, cell: int = 24) -> np.ndarray:
    board = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(0, height, cell):
        for x in range(0, width, cell):
            if ((x // cell) + (y // cell)) % 2 == 0:
                board[y : y + cell, x : x + cell] = (40, 40, 40)
            else:
                board[y : y + cell, x : x + cell] = (70, 70, 70)
    return board


def _paste(board: np.ndarray, sprite: np.ndarray, x: int, y: int) -> None:
    sh, sw = sprite.shape[:2]
    x1, y1 = min(board.shape[1], x + sw), min(board.shape[0], y + sh)
    crop = sprite[: y1 - y, : x1 - x]
    if crop.shape[2] == 4:
        alpha = crop[:, :, 3:4].astype(np.float32) / 255.0
        board[y:y1, x:x1] = (crop[:, :, :3].astype(np.float32) * alpha + board[y:y1, x:x1].astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
    else:
        board[y:y1, x:x1] = crop


def contact_sheet(plates: list[tuple[str, np.ndarray]], objects: list[tuple[str, np.ndarray]]) -> np.ndarray:
    tile_h = 420
    tiles = []
    for label, image in plates:
        scale = tile_h / image.shape[0]
        tile = cv2.resize(image, (int(image.shape[1] * scale), tile_h), interpolation=cv2.INTER_AREA)
        ground = int(GROUND * tile.shape[0])
        limit = int(LIMIT * tile.shape[0])
        cv2.line(tile, (0, ground), (tile.shape[1], ground), (80, 200, 255), 3)
        cv2.line(tile, (0, limit), (tile.shape[1], limit), (80, 220, 80), 2)
        cv2.line(tile, (tile.shape[1] // 2, 0), (tile.shape[1] // 2, tile.shape[0]), (180, 180, 180), 1)
        cv2.putText(tile, label, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(tile)
    width = sum(tile.shape[1] for tile in tiles) + 8 * (len(tiles) + 1)
    height = tile_h + 520
    sheet = np.zeros((height, width, 3), dtype=np.uint8)
    sheet[:] = (14, 17, 23)
    x = 8
    for tile in tiles:
        sheet[8 : 8 + tile.shape[0], x : x + tile.shape[1]] = tile
        x += tile.shape[1] + 8
    thumb = 220
    x = 8
    for label, image in objects:
        board = _checker(thumb, thumb)
        sprite = image
        scale = min(thumb / sprite.shape[1], thumb / sprite.shape[0]) * 0.9
        sized = cv2.resize(sprite, (max(1, int(sprite.shape[1] * scale)), max(1, int(sprite.shape[0] * scale))), interpolation=cv2.INTER_AREA)
        _paste(board, sized, (thumb - sized.shape[1]) // 2, (thumb - sized.shape[0]) // 2)
        y = tile_h + 24
        sheet[y : y + thumb, x : x + thumb] = board
        cv2.putText(sheet, label, (x, y + thumb + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (242, 244, 248), 1, cv2.LINE_AA)
        x += thumb + 16
    return sheet


def chroma_speck_count(bgr: np.ndarray) -> int:
    """Isolated cyan or red pixels: a channel spike the four-neighbors do not share."""
    if bgr is None or bgr.size == 0:
        return 0
    blue, green, red = cv2.split(bgr.astype(np.int16))
    cyan = (blue > green + 40) & (blue > red + 40)
    hot = (red > green + 40) & (red > blue + 40)
    mask = cyan | hot
    kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], np.uint8)
    neighbors = cv2.filter2D(mask.astype(np.uint8), -1, kernel)
    return int(np.count_nonzero(mask & (neighbors == 0)))


def paint_space_gradient(image: np.ndarray, cx: float, cy: float, radius: float) -> int:
    """Replace the flat teal cap with a vertical gradient. The Moon disc is left as it is."""
    height, width = image.shape[:2]
    band = min(height, 250 + 48)
    sample_y = min(height - 1, 280)
    xs = np.arange(width)
    moon_row = (xs - cx * width) ** 2 + (sample_y - cy * height) ** 2 > (radius * width + 8) ** 2
    if moon_row.any():
        sky = np.median(image[sample_y, moon_row].astype(np.float32), axis=0)
    else:
        sky = np.array([12.0, 10.0, 8.0], np.float32)
    top = np.array([8.0, 10.0, 12.0], np.float32)
    y = np.arange(band, dtype=np.float32)[:, None]
    blend = np.clip(y / 250.0, 0.0, 1.0)
    color = top[None, :] * (1.0 - blend) + sky[None, :] * blend
    fade = np.ones((band, width), np.float32)
    feather = y.ravel() >= (250 - 48)
    fade[feather, :] = np.clip((250 + 48 - y.ravel()[feather]) / 48.0, 0.0, 1.0)[:, None]
    yy, xx = np.mgrid[0:band, 0:width]
    moon = (xx - cx * width) ** 2 + (yy - cy * height) ** 2 <= (radius * width) ** 2
    fade[moon] = 0.0
    region = image[:band].astype(np.float32)
    mixed = region * (1.0 - fade[..., None]) + color[:, None, :] * fade[..., None]
    image[:band] = np.clip(np.rint(mixed), 0, 255).astype(np.uint8)
    return int(np.count_nonzero(fade > 0.05))


def procedural_city(width: int = FRAME_W, height: int = FRAME_H) -> np.ndarray:
    """Paper-cut low-rise skyline. The tallest roof is at most 30% of the frame above the ground."""
    image = np.zeros((height, width, 3), np.float32)
    ground = int(round(0.68 * height))
    for y in range(ground):
        t = y / max(1, ground - 1)
        # Warm, dark enough that white paper clears 3:1 in the open center lane.
        image[y] = (1.0 - t) * np.array([48.0, 42.0, 62.0]) + t * np.array([78.0, 86.0, 118.0])
    rng = np.random.default_rng(3)
    layers = (
        ((127, 107, 79), 0.22, 9),
        ((100, 90, 31), 0.28, 7),
        ((207, 230, 239), 0.16, 6),
    )
    limit = int(round(0.30 * height))
    center0 = int(round(width / 3))
    center1 = int(round(2 * width / 3))
    for color, max_frac, count in layers:
        shadow = image.copy()
        for index in range(count):
            bw = int(rng.integers(70, 140))
            bh = int(rng.integers(int(0.08 * height), int(max_frac * height)))
            bh = min(bh, limit)
            if index % 2 == 0:
                x0 = int(rng.integers(16, max(17, center0 - bw - 8)))
            else:
                x0 = int(rng.integers(center1 + 8, max(center1 + 9, width - bw - 16)))
            y0 = ground - bh
            image[y0 + 6 : ground + 6, x0 + 8 : x0 + bw + 8] = np.array(color, np.float32) * 0.55
            image[y0:ground, x0 : x0 + bw] = color
            win = (max(0, color[0] - 40), max(0, color[1] - 36), max(0, color[2] - 28))
            step_y = 28
            step_x = 22
            for wy in range(y0 + 16, ground - 18, step_y):
                for wx in range(x0 + 10, x0 + bw - 16, step_x):
                    image[wy : wy + 12, wx : wx + 10] = win
        _ = shadow
    awning = (92, 113, 232)
    image[ground - 36 : ground - 24, 48:160] = awning
    band = np.array([168.0, 176.0, 186.0], np.float32)
    rng2 = np.random.default_rng(11)
    grain = rng2.normal(0.0, 1.5, (height - ground, width, 1)).astype(np.float32)
    image[ground:] = np.clip(band + grain, 0, 255)
    return np.clip(np.rint(image), 0, 255).astype(np.uint8)


def _note_city(which: str) -> None:
    path = ROOT / "out" / "qa_log.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    line = f"\n- city plate: {which}\n"
    existing = path.read_text() if path.exists() else "# QA log\n"
    if line.strip() in existing:
        return
    path.write_text(existing.rstrip() + "\n" + line)


def run_ingest(root: Path | None = None, out_dir: Path | None = None, manifest_path: Path | None = None) -> int:
    root = root or ART_ROOT
    out_dir = out_dir or (ROOT / "out")
    manifest = load_manifest(manifest_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    normalized = root / "normalized"
    keyed = root / "keyed"
    normalized.mkdir(parents=True, exist_ok=True)
    keyed.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Art ingest",
        "",
        "Center-lane contrast is #F2F4F8 against the median of the center 48 px, from 32% to 62% of the aligned frame.",
        "",
    ]
    unapproved = False
    plate_tiles = []
    object_tiles = []
    for item in manifest["plates"]:
        asset_id = item["id"]
        source = find_source(root / "plates", asset_id)
        if source is None:
            unapproved = True
            image = placeholder(asset_id, FRAME_W, FRAME_H)
            lines.append(f"- {asset_id}: MISSING, placeholder, UNAPPROVED")
            plate_tiles.append((asset_id, image))
            continue
        raw = cv2.imread(str(source), cv2.IMREAD_COLOR)
        screen = float(item.get("screen_ground", GROUND))
        extra: list[str] = []
        if asset_id in ("p02_street_day", "p03_city_day") and chroma_speck_count(raw) > 0:
            raw = denoise_chroma(raw)
            extra.append("chroma denoised")
        image, notes = place_plate(raw, float(item["ground_y"]), screen)
        if asset_id == "p07_deep_space_moon":
            scale, tx, ty = align_translate(float(item["ground_y"]), raw.shape[1], raw.shape[0], screen_ground=screen)
            cx = (float(item["moon_cx"]) * raw.shape[1] * scale + tx) / FRAME_W
            cy = (float(item["moon_cy"]) * raw.shape[0] * scale + ty) / FRAME_H
            radius = float(item["moon_radius"]) * raw.shape[1] * scale / FRAME_W
            painted = paint_space_gradient(image, cx, cy, radius)
            extra.append(f"teal cap replaced with a vertical gradient ({painted} px), feather 48 px")
        notes = extra + notes
        reference = STRUCTURE_REFERENCES.get(str(item.get("scene")))
        if reference and item.get("world_m"):
            meters = tallest_structure_m(image, screen, float(item["world_m"]))
            name, limit_m = reference
            scale_note = f"tallest structure {meters:.0f} m"
            if meters > limit_m:
                shown = f"{limit_m:.0f}" if limit_m >= 10 else f"{limit_m:.1f}"
                scale_note += f" SCALE WARN taller than {name} {shown} m"
                if item.get("scene") == "city":
                    image = procedural_city()
                    scale_note += "; replaced with the procedural low-rise skyline"
                    _note_city("procedural low-rise skyline")
            elif item.get("scene") == "city":
                _note_city("source plate p03")
            notes.append(scale_note)
        if asset_id == "p06_low_orbit_limb":
            rows = recolor_grey_cap(image)
            notes.append(f"grey cap recolored {rows} rows")
        contrast = center_lane_contrast(image)
        flag = " FLAG under 3:1" if contrast < 3.0 else ""
        if asset_id == "p07_deep_space_moon":
            expect = (float(item["moon_cx"]), float(item["moon_cy"]), float(item["moon_radius"]))
            found = detect_circle(raw, expect)
            if found is None:
                notes.append("moon circle NOT FOUND")
            else:
                deltas = [abs(found[i] - expect[i]) / expect[i] for i in range(3)]
                notes.append(
                    f"moon detected cx {found[0]:.3f} cy {found[1]:.3f} r {found[2]:.3f}"
                )
                if max(deltas) > 0.03:
                    notes.append(f"moon WARN off by {max(deltas):.1%}")
        cv2.imwrite(str(normalized / f"{asset_id}.png"), image)
        lines.append(f"- {asset_id}: {', '.join(notes)}; center-lane contrast {contrast:.2f}:1{flag}")
        plate_tiles.append((asset_id, image))
    for item in manifest["objects"]:
        asset_id = item["id"]
        source = find_source(root / "objects", asset_id)
        if source is None:
            unapproved = True
            image = np.dstack([placeholder(asset_id, 768, 1376), np.full((1376, 768), 255, np.uint8)])
            lines.append(f"- {asset_id}: MISSING, placeholder, UNAPPROVED")
            object_tiles.append((asset_id, image))
            continue
        raw = cv2.imread(str(source), cv2.IMREAD_COLOR)
        keyed_image, fraction, box = key_magenta(raw)
        mark = "" if fraction >= 0.90 else " FLAG border under 90%"
        if asset_id == "o05_iss":
            keyed_image = finish_iss_key(keyed_image)
            ys, xs = np.nonzero(keyed_image[:, :, 3] > 16)
            if len(xs):
                box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
            mark += ", eroded 2 px, pink fringe despilled"
        if asset_id == "o06_moon":
            keyed_image, circle = mask_moon_circle(keyed_image)
            ys, xs = np.nonzero(keyed_image[:, :, 3] > 16)
            box = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
            mark += f", circle r {circle[2]:.0f} px, teal field removed"
        cv2.imwrite(str(keyed / f"{asset_id}.png"), keyed_image)
        lines.append(f"- {asset_id}: border magenta {fraction:.1%}, alpha box {box}{mark}")
        object_tiles.append((asset_id, keyed_image))
    lines += ["", "## Busy regions behind the stack", ""]
    lines += [f"- {plate_id}: {note}" for plate_id, note in BUSY_REGIONS]
    lines.append("")
    lines.append(
        "o04_burj_khalifa keeps a small red patch on the right face. It is in the source plate, not a magenta leak. The tower touches the bottom edge, which is why the border fraction is 92.6% rather than 100%."
    )
    status = "UNAPPROVED" if unapproved else "APPROVED"
    lines += ["", f"Status: {status}", ""]
    report = "\n".join(lines)
    (out_dir / "art_report.md").write_text(report + "\n")
    sheet = contact_sheet(plate_tiles, object_tiles)
    cv2.imwrite(str(out_dir / "art_contact_sheet.png"), sheet)
    print(report)
    print(f"wrote {out_dir / 'art_report.md'}")
    print(f"wrote {out_dir / 'art_contact_sheet.png'}")
    return 1 if unapproved else 0
