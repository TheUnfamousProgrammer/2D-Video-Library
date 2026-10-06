"""Cutouts for the scale short: key the supplied magenta art, trim it, and report what is missing.

Sources are matched by id the same way as the paperfold plates: ``s04_virus.png``,
``s04_virus.png_20261006.jpg`` and ``s04_virus_png_x.jpg`` all match ``s04_virus``.
New art is read from assets/art/objects; the reused paperfold cutouts (o01..o07) are
read from assets/art/keyed. Trimmed RGBA cutouts land in assets/art/scale.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from fc_sat.paperfold_art import find_source, key_magenta

ROOT = Path(__file__).resolve().parents[1]
ART_ROOT = ROOT / "assets" / "art"
SOURCES = ART_ROOT / "objects"
REUSED = ART_ROOT / "keyed"
OUT = ART_ROOT / "scale"
ALPHA_FLOOR = 16
MAX_SIDE = 1400


def trim(bgra: np.ndarray, pad: int = 4) -> np.ndarray:
    ys, xs = np.nonzero(bgra[:, :, 3] > ALPHA_FLOOR)
    if xs.size == 0:
        raise ValueError("cutout is empty after keying")
    x0 = max(0, int(xs.min()) - pad)
    y0 = max(0, int(ys.min()) - pad)
    x1 = min(bgra.shape[1], int(xs.max()) + 1 + pad)
    y1 = min(bgra.shape[0], int(ys.max()) + 1 + pad)
    return np.ascontiguousarray(bgra[y0:y1, x0:x1])


def _limit(bgra: np.ndarray) -> np.ndarray:
    longest = max(bgra.shape[:2])
    if longest <= MAX_SIDE:
        return bgra
    factor = MAX_SIDE / float(longest)
    size = (max(1, int(round(bgra.shape[1] * factor))), max(1, int(round(bgra.shape[0] * factor))))
    return cv2.resize(bgra, size, interpolation=cv2.INTER_AREA)


def ingest_one(art: str) -> tuple[Path | None, str]:
    """Key and trim one cutout. Returns (written path or None, a one-line note)."""
    OUT.mkdir(parents=True, exist_ok=True)
    reused = REUSED / f"{art}.png"
    if art.startswith("o") and reused.exists():
        bgra = cv2.imread(str(reused), cv2.IMREAD_UNCHANGED)
        note = "reused paperfold cutout"
    else:
        source = find_source(SOURCES, art) if SOURCES.exists() else None
        if source is None:
            return None, "missing"
        bgr = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if bgr is None:
            return None, f"unreadable {source.name}"
        bgra, fraction, _box = key_magenta(bgr)
        note = f"keyed {source.name}, border {fraction:.0%} magenta"
        if fraction < 0.9:
            note += " (WARNING: background is not flat magenta)"
    cut = _limit(trim(bgra))
    path = OUT / f"{art}.png"
    cv2.imwrite(str(path), cut)
    return path, note


def ingest(arts: list[str]) -> list[str]:
    lines = []
    for art in arts:
        path, note = ingest_one(art)
        lines.append(f"{art}: {note}" + (f" -> {path.relative_to(ROOT)}" if path else ""))
    return lines


def load_cutout(art: str) -> np.ndarray | None:
    """Trimmed BGRA cutout, or None when the art has not been supplied yet."""
    path = OUT / f"{art}.png"
    if not path.exists():
        return None
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim != 3 or image.shape[2] != 4:
        return None
    return image


def cutout_aspect(art: str) -> float | None:
    image = load_cutout(art)
    if image is None:
        return None
    ys, xs = np.nonzero(image[:, :, 3] > ALPHA_FLOOR)
    return (float(xs.max() - xs.min() + 1)) / float(ys.max() - ys.min() + 1)
