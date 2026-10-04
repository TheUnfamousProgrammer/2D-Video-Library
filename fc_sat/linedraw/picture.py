"""Load a picture, crop it to the short's portrait window, and build the target grid."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from skimage.exposure import equalize_adapthist

from fc_sat.linedraw.optimize import GRID_H, GRID_W

ASPECT = 0.69
MIN_SHORT_SIDE = 600
CLAHE_CLIPS = (0.012, 0.02)


@dataclass
class Picture:
    color: np.ndarray  # crop, RGB uint8, source resolution
    luma: np.ndarray  # crop luma, float 0..1
    target: np.ndarray  # GRID_H x GRID_W after the chosen CLAHE
    weight: np.ndarray
    clip: float
    trial_scores: dict[str, float]
    method: str
    upscaled: bool
    upscale_from: tuple[int, int]
    crop_box: tuple[int, int, int, int]  # x0, y0, x1, y1 on the working image
    face_box: tuple[int, int, int, int] | None  # x, y, w, h inside the crop
    focus: tuple[float, float]  # grid x, y of the importance center
    mouth: tuple[float, float]  # grid x, y of the zoom target
    source_size: tuple[int, int]
    working_size: tuple[int, int]


def _luma(rgb: np.ndarray) -> np.ndarray:
    image = rgb.astype(np.float64) / 255.0
    return 0.2126 * image[..., 0] + 0.7152 * image[..., 1] + 0.0722 * image[..., 2]


def _cascades() -> list[cv2.CascadeClassifier]:
    names = ("haarcascade_frontalface_default.xml", "haarcascade_frontalface_alt2.xml")
    found = []
    for name in names:
        path = cv2.data.haarcascades + name
        cascade = cv2.CascadeClassifier(path)
        if not cascade.empty():
            found.append(cascade)
    return found


def largest_face(gray_u8: np.ndarray) -> tuple[int, int, int, int] | None:
    """Largest frontal face from the default and alt2 cascades, on the raw frame and a blurred copy."""
    if gray_u8.ndim != 2:
        raise ValueError("face detection expects one grayscale plane")
    blurred = cv2.GaussianBlur(gray_u8, (0, 0), 1.4)
    best: tuple[int, int, int, int] | None = None
    best_area = 0
    for image in (gray_u8, blurred):
        for cascade in _cascades():
            boxes = cascade.detectMultiScale(image, scaleFactor=1.08, minNeighbors=3, minSize=(24, 24))
            for x, y, w, h in boxes:
                area = int(w) * int(h)
                if area > best_area:
                    best_area = area
                    best = (int(x), int(y), int(w), int(h))
    return best


def spectral_peak(luma: np.ndarray) -> tuple[float, float] | None:
    """Spectral-residual saliency peak as (x, y). None when the map is flat."""
    from scipy.ndimage import gaussian_filter, uniform_filter

    small = np.asarray(luma, dtype=np.float64)
    if float(np.std(small)) < 1e-4:
        return None
    if max(small.shape) > 480:
        scale = 480.0 / max(small.shape)
        small = cv2.resize(
            small.astype(np.float32),
            (max(8, int(round(small.shape[1] * scale))), max(8, int(round(small.shape[0] * scale)))),
        ).astype(np.float64)
    spectrum = np.fft.fft2(small)
    log_amplitude = np.log(np.abs(spectrum) + 1e-8)
    phase = np.angle(spectrum)
    residual = log_amplitude - uniform_filter(log_amplitude, size=3, mode="nearest")
    saliency = gaussian_filter(np.abs(np.fft.ifft2(np.exp(residual + 1j * phase))) ** 2, 1.2)
    if float(saliency.max()) <= float(saliency.min()) + 1e-12:
        return None
    y, x = np.unravel_index(int(np.argmax(saliency)), saliency.shape)
    return float(x) * luma.shape[1] / saliency.shape[1], float(y) * luma.shape[0] / saliency.shape[0]


def _fit_aspect(width: int, height: int, aspect: float = ASPECT) -> tuple[int, int]:
    if width / max(height, 1) >= aspect:
        crop_h = height
        crop_w = max(1, int(round(crop_h * aspect)))
    else:
        crop_w = width
        crop_h = max(1, int(round(crop_w / aspect)))
    return min(crop_w, width), min(crop_h, height)


def _clamp_window(left: int, top: int, crop_w: int, crop_h: int, width: int, height: int) -> tuple[int, int]:
    return max(0, min(left, width - crop_w)), max(0, min(top, height - crop_h))


def center_crop(width: int, height: int, aspect: float = ASPECT) -> tuple[int, int, int, int]:
    crop_w, crop_h = _fit_aspect(width, height, aspect)
    left, top = _clamp_window((width - crop_w) // 2, (height - crop_h) // 2, crop_w, crop_h, width, height)
    return left, top, left + crop_w, top + crop_h


def crop_around_point(
    width: int,
    height: int,
    focus_x: float,
    focus_y: float,
    *,
    y_frac: float = 0.30,
    aspect: float = ASPECT,
    must_contain: tuple[int, int, int, int] | None = None,
) -> tuple[int, int, int, int]:
    """Largest aspect window that puts ``focus`` near ``y_frac`` of the crop height."""
    chosen = center_crop(width, height, aspect)
    for scale in (1.0, 0.94, 0.86, 0.78, 0.70, 0.62):
        crop_h = max(1, int(round(height * scale)))
        crop_w = max(1, int(round(crop_h * aspect)))
        if crop_w > width or crop_h > height:
            crop_w, crop_h = _fit_aspect(width, height, aspect)
            crop_h = min(crop_h, height)
        left = int(round(focus_x - 0.5 * crop_w))
        top = int(round(focus_y - y_frac * crop_h))
        left, top = _clamp_window(left, top, crop_w, crop_h, width, height)
        if must_contain is not None:
            fx, fy, fw, fh = must_contain
            if not (left <= fx and top <= fy and left + crop_w >= fx + fw and top + crop_h >= fy + fh):
                continue
        face_y = focus_y - top
        if face_y <= crop_h / 3.0 + (0 if must_contain is None else must_contain[3] * 0.2):
            return left, top, left + crop_w, top + crop_h
        chosen = (left, top, left + crop_w, top + crop_h)
    return chosen


def parse_crop(text: str) -> tuple[int, int, int, int]:
    parts = [int(float(item.strip())) for item in text.split(",")]
    if len(parts) != 4:
        raise SystemExit("--crop expects x0,y0,x1,y1")
    x0, y0, x1, y1 = parts
    if x1 <= x0 or y1 <= y0:
        raise SystemExit("--crop box is empty")
    return x0, y0, x1, y1


def importance_map(height: int, width: int, center_x: float, center_y: float, weight: float) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    dy = (yy - center_y) / (0.22 * height)
    dx = (xx - center_x) / (0.24 * width)
    return 1.0 + float(weight) * np.exp(-(dy * dy + dx * dx))


def _resize_luma(luma: np.ndarray, width: int, height: int) -> np.ndarray:
    image = Image.fromarray(np.clip(np.rint(luma * 255.0), 0, 255).astype(np.uint8), mode="L")
    resized = image.resize((width, height), Image.Resampling.LANCZOS)
    return np.asarray(resized, dtype=np.float64) / 255.0


def select_clahe(
    grid: np.ndarray,
    weight: np.ndarray,
    seed: int,
    clips: tuple[float, ...] = CLAHE_CLIPS,
    trial_lines: int = 2000,
    candidates: int = 150,
) -> tuple[np.ndarray, float, dict[str, float]]:
    """Keep the clip limit whose short line trial matches the target more closely."""
    from fc_sat.linedraw.optimize import likeness, optimize

    scores: dict[str, float] = {}
    best_clip = float(clips[0])
    best_score = -2.0
    best_target = np.asarray(grid, dtype=np.float64)
    for clip in clips:
        enhanced = np.asarray(equalize_adapthist(np.clip(grid, 0.0, 1.0), clip_limit=float(clip)), dtype=np.float64)
        trial = optimize(
            enhanced,
            weight,
            seed=seed,
            max_kept=trial_lines,
            candidates=candidates,
            planned=max(trial_lines, 1),
            record_likeness=False,
        )
        score = likeness(trial.canvas, enhanced)
        scores[f"{float(clip):.3f}"] = score
        print(f"clahe clip {float(clip):.3f} trial likeness {score:.3f}", flush=True)
        if score > best_score:
            best_score = score
            best_clip = float(clip)
            best_target = enhanced
    return best_target, best_clip, scores


def prepare(
    path: Path,
    *,
    weight: float = 2.5,
    crop: tuple[int, int, int, int] | None = None,
    seed: int = 7,
    trial_lines: int = 2000,
) -> Picture:
    if not path.exists():
        raise SystemExit(f"target image not found: {path}")
    source = Image.open(path).convert("RGB")
    source_size = source.size
    upscaled = False
    working = source
    if min(source.size) < MIN_SHORT_SIDE:
        scale = MIN_SHORT_SIDE / min(source.size)
        resized = (max(1, int(round(source.size[0] * scale))), max(1, int(round(source.size[1] * scale))))
        working = source.resize(resized, Image.Resampling.LANCZOS)
        upscaled = True
        print(f"upscaled {source.size[0]}x{source.size[1]} to {working.size[0]}x{working.size[1]} with Lanczos", flush=True)
    rgb = np.asarray(working, dtype=np.uint8)
    luma = _luma(rgb)
    height, width = luma.shape
    gray = np.clip(np.rint(luma * 255.0), 0, 255).astype(np.uint8)
    manual = crop
    if manual is not None and upscaled and manual[2] <= source_size[0] and manual[3] <= source_size[1]:
        scale_x = width / source_size[0]
        scale_y = height / source_size[1]
        manual = (
            int(round(manual[0] * scale_x)),
            int(round(manual[1] * scale_y)),
            int(round(manual[2] * scale_x)),
            int(round(manual[3] * scale_y)),
        )
    face = None if manual is not None else largest_face(gray)
    if manual is not None:
        x0, y0, x1, y1 = manual
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(width, x1), min(height, y1)
        box_w, box_h = _fit_aspect(max(1, x1 - x0), max(1, y1 - y0))
        left = x0 + max(0, (x1 - x0 - box_w) // 2)
        top = y0 + max(0, (y1 - y0 - box_h) // 2)
        left, top = _clamp_window(left, top, box_w, box_h, width, height)
        window = (left, top, left + box_w, top + box_h)
        method = "manual"
    elif face is not None:
        fx, fy, fw, fh = face
        window = crop_around_point(width, height, fx + fw / 2.0, fy + fh / 2.0, must_contain=face)
        method = "face"
    else:
        peak = spectral_peak(luma)
        if peak is None:
            window = center_crop(width, height)
            method = "center"
        else:
            window = crop_around_point(width, height, peak[0], peak[1])
            method = "saliency"
    x0, y0, x1, y1 = window
    color = rgb[y0:y1, x0:x1]
    crop_luma = luma[y0:y1, x0:x1]
    face_box = None
    if face is not None and method == "face":
        fx, fy, fw, fh = face
        face_box = (fx - x0, fy - y0, fw, fh)
        focus_x = (fx + fw / 2.0 - x0) * GRID_W / color.shape[1]
        focus_y = (fy + fh / 2.0 - y0) * GRID_H / color.shape[0]
        mouth = (focus_x, (fy - y0 + 0.78 * fh) * GRID_H / color.shape[0])
    elif method == "saliency":
        peak = spectral_peak(crop_luma)
        if peak is None:
            focus_x, focus_y = GRID_W / 2.0, GRID_H / 2.0
        else:
            focus_x = peak[0] * GRID_W / color.shape[1]
            focus_y = peak[1] * GRID_H / color.shape[0]
        mouth = (focus_x, focus_y)
    else:
        focus_x, focus_y = GRID_W / 2.0, GRID_H / 3.0
        mouth = (focus_x, focus_y)
    grid = _resize_luma(crop_luma, GRID_W, GRID_H)
    imp = importance_map(GRID_H, GRID_W, focus_x, focus_y, weight)
    best_target, best_clip, scores = select_clahe(grid, imp, seed, trial_lines=trial_lines)
    return Picture(
        color=np.ascontiguousarray(color),
        luma=crop_luma,
        target=best_target,
        weight=imp,
        clip=best_clip,
        trial_scores=scores,
        method=method,
        upscaled=upscaled,
        upscale_from=source_size,
        crop_box=(x0, y0, x1, y1),
        face_box=face_box,
        focus=(float(focus_x), float(focus_y)),
        mouth=(float(mouth[0]), float(mouth[1])),
        source_size=source_size,
        working_size=(width, height),
    )


def save_picture(picture: Picture, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    face = np.array(picture.face_box if picture.face_box is not None else (-1, -1, -1, -1), dtype=np.int32)
    np.savez_compressed(
        path,
        color=picture.color,
        target=picture.target.astype(np.float32),
        weight=picture.weight.astype(np.float32),
        clip=np.float64(picture.clip),
        face=face,
        focus=np.array(picture.focus, dtype=np.float64),
        mouth=np.array(picture.mouth, dtype=np.float64),
        crop_box=np.array(picture.crop_box, dtype=np.int32),
    )


def load_picture(path: Path) -> dict:
    data = np.load(path)
    face = tuple(int(v) for v in data["face"])
    return {
        "color": data["color"],
        "target": data["target"].astype(np.float64),
        "weight": data["weight"].astype(np.float64),
        "clip": float(data["clip"]),
        "face": None if face[2] <= 0 else face,
        "focus": tuple(float(v) for v in data["focus"]),
        "mouth": tuple(float(v) for v in data["mouth"]),
        "crop_box": tuple(int(v) for v in data["crop_box"]),
    }
