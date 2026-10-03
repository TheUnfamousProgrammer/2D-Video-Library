"""Locate supplied art, fit plates to the frame, and key the magenta cutouts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

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
        "Stars in the center lane. The Moon is on the right and Earth is below the ground line, so neither sits behind the stack.",
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


def place_plate(bgr: np.ndarray, ground_y: float) -> tuple[np.ndarray, list[str]]:
    """Scale to cover, align the ground line to y = 0.68, and crop to 1080x1920."""
    src_h, src_w = bgr.shape[:2]
    notes = []
    wrong = aspect_note(src_w, src_h)
    if wrong:
        notes.append(wrong)
    if src_w < FRAME_W or src_h < FRAME_H:
        notes.append("UPSCALED")
    scale, tx, ty = align_translate(ground_y, src_w, src_h)
    notes.append(f"scale {scale:.4f} tx {tx:.1f} ty {ty:.1f}")
    matrix = np.array([[scale, 0.0, tx], [0.0, scale, ty]], dtype=np.float32)
    placed = cv2.warpAffine(
        bgr.astype(np.float32),
        matrix,
        (FRAME_W, FRAME_H),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REPLICATE,
    )
    if "UPSCALED" in notes:
        soft = cv2.GaussianBlur(placed, (0, 0), 1.0)
        placed = np.clip(placed * 1.15 - soft * 0.15, 0, 255)
    return placed.astype(np.uint8), notes


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
        image, notes = place_plate(raw, float(item["ground_y"]))
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
        cv2.imwrite(str(keyed / f"{asset_id}.png"), keyed_image)
        mark = "" if fraction >= 0.90 else " FLAG border under 90%"
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
