"""Load, orient, and crop the two pictures a Pixel Morph rearranges."""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

# Matches the vignette base so transparent pixels disappear into the background.
_BG_RGB = (7, 7, 11)


class ImageError(ValueError):
    """A source picture cannot be used."""


def file_sha256(path: Path) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rgb(path: Path) -> np.ndarray:
    """Return HxWx3 uint8 RGB after EXIF orientation.

    RGBA and LA are composited onto ``#07070B``. Every other mode (CMYK,
    palette, grayscale) is converted to RGB. Unreadable files raise
    ``ImageError``.
    """
    path = Path(path)
    try:
        with Image.open(path) as image:
            image.load()
            oriented = ImageOps.exif_transpose(image)
            rgb = _to_rgb(oriented)
            array = np.asarray(rgb, dtype=np.uint8)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageError(f"unreadable image: {path}") from exc
    if array.ndim != 3 or array.shape[2] != 3:
        raise ImageError(f"unreadable image: {path}")
    return np.ascontiguousarray(array)


def _to_rgb(image: Image.Image) -> Image.Image:
    if image.mode in ("RGBA", "LA"):
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, _BG_RGB + (255,))
        return Image.alpha_composite(background, rgba).convert("RGB")
    if image.mode != "RGB":
        return image.convert("RGB")
    return image


def prepare_grid(
    rgb: np.ndarray,
    cols: int,
    rows: int,
    focus: tuple[float, float] = (0.5, 0.5),
) -> np.ndarray:
    """Center-crop to ``cols:rows`` and area-resize to ``(rows, cols, 3)``.

    ``focus`` is ``(fx, fy)`` in 0..1. 0.5 is the center of the leftover
    margin; 0 pins the crop to the left or top, 1 to the right or bottom.
    Pictures smaller than the grid are rejected so the resize never upscales
    from below the cell count.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ImageError("prepared image must be RGB")
    height, width = rgb.shape[:2]
    if width < cols or height < rows:
        raise ImageError(f"image is {width}x{height}, smaller than the {cols}x{rows} grid")
    fx, fy = focus
    if not (0.0 <= fx <= 1.0 and 0.0 <= fy <= 1.0):
        raise ImageError(f"focus {focus} is outside 0..1")
    cropped = _crop(rgb, cols, rows, fx, fy)
    resized = cv2.resize(cropped, (cols, rows), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(resized)


def _crop(rgb: np.ndarray, cols: int, rows: int, fx: float, fy: float) -> np.ndarray:
    height, width = rgb.shape[:2]
    target = cols / rows
    current = width / height
    if current > target:
        crop_h = height
        crop_w = max(cols, int(round(height * target)))
        crop_w = min(crop_w, width)
        x0 = int(round(fx * (width - crop_w)))
        y0 = 0
    else:
        crop_w = width
        crop_h = max(rows, int(round(width / target)))
        crop_h = min(crop_h, height)
        x0 = 0
        y0 = int(round(fy * (height - crop_h)))
    return rgb[y0 : y0 + crop_h, x0 : x0 + crop_w]


def reject_identical(path_a: Path, path_b: Path) -> None:
    """Refuse a morph whose two files are the same bytes."""
    path_a, path_b = Path(path_a), Path(path_b)
    if file_sha256(path_a) == file_sha256(path_b):
        raise ImageError("image A and image B are identical (same sha256)")
