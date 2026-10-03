"""Hero stills. The stack is an oblique paper prism on the plate. No slate strip.

p06 and p07 put the Earth's apex at y = 0.58. Every other plate keeps 0.68.
The bottom band is the plate's own ground. The counter sits on a small plate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from fc_sat.paperfold_art import ART_ROOT, _relative_luma
from fc_sat.paperfold_draw import PaperCanvas, bgr, text_width
from fc_sat.polycircle_draw import measure
from fc_sat.paperfold_math import (
    BURJ_M,
    ISS_M,
    KARMAN_M,
    MILESTONES,
    MOON_M,
    MUG_M,
    PERSON_M,
    PHONE_M,
    height_m,
    milestone_fold,
)
from fc_sat.easing import ease_in_out_cubic
from fc_sat.paperfold_render import fold_motion, folds_done, height_label, length_sub
from fc_sat.paperfold_scenes import (
    FRAME_H,
    FRAME_W,
    GROUND,
    HANDOFF_FRAMES,
    Scene,
    align_translate,
    break_fold,
    handoff_start_frame,
    log_blend,
    px_per_m,
    px_per_m_at,
    scenes_from_manifest,
)
from fc_sat.paperfold_schedule import EDGE_END, EDGE_START, SEGMENTS, fold_frame, segment_at
from fc_sat.paperfold_text import Script, load_script, top_lines

ROOT = Path(__file__).resolve().parents[1]
OUTLINE = "#1B2030"
PAPER = "#F2F4F8"
PAPER_RGB = (242, 244, 248)
SEAM_RGB = (228, 232, 240)
SIDE = "#B8C1D4"
TOP_FACE = "#D5DCE8"
SHADOW = "#0E1117"
STACK_W = 150.0
SIDE_W = 28.0
TOP_H = 12.0
OUTLINE_PX = 10.0
STACK_OUTLINE = 6.0
CARD_W = 520.0
CARD_H = 700.0
CARD_BASE = 1306.0
HOOK_PROGRESS = 0.35
HOOK_SIZE = 84.0
HOOK_BAND_TOP = 200.0
HOOK_BAND_BOTTOM = 400.0
SAFE_L = 130.0
SAFE_R = 950.0
PLATE_ALPHA = 0.92
HOOK_PLATE_ALPHA = 0.96
COUNTER_ALPHA = 1.0

HEROES = (
    ("hook", 0),
    ("fold_15", 15),
    ("fold_23", 23),
    ("fold_30", 30),
    ("fold_32", 32),
    ("fold_42", 42),
)

SCENE_OBJECTS: dict[str, tuple[tuple[str, float, float], ...]] = {
    "desk": (("o02_mug", 0.24, MUG_M), ("o03_smartphone", 0.76, PHONE_M)),
    "street": (("o01_person", 0.24, PERSON_M),),
    "city": (("o04_burj_khalifa", 0.76, BURJ_M),),
}


@dataclass(frozen=True)
class PaperBox:
    x: float
    y: float
    w: float
    h: float


@dataclass(frozen=True)
class StillFrame:
    name: str
    frame: int
    fold: int
    plate: str
    scene: str
    image: np.ndarray
    paper: PaperBox
    ratios: tuple[float, float, float]
    notes: tuple[str, ...]


def scene_at(frame: int, rows: tuple[Scene, ...] | None = None) -> Scene:
    rows = rows or scenes_from_manifest()
    current = rows[0]
    for cur, nxt in zip(rows, rows[1:]):
        start = handoff_start_frame(break_fold(cur.world_m, screen_ground=cur.screen_ground))
        if frame < start:
            return cur
        if frame < start + HANDOFF_FRAMES:
            if frame >= start + int(0.6 * HANDOFF_FRAMES):
                return nxt
            return cur
        current = nxt
    return current


def _contrast(foreground: np.ndarray, background: np.ndarray) -> float:
    lighter = max(_relative_luma(foreground), _relative_luma(background))
    darker = min(_relative_luma(foreground), _relative_luma(background))
    return (lighter + 0.05) / (darker + 0.05)


def edge_contrast(image_bgr: np.ndarray, paper: PaperBox, outline: float = OUTLINE_PX) -> tuple[float, float, float]:
    """White front face against the plate just outside the outline, at three heights."""
    height, width = image_bgr.shape[:2]
    ratios = []
    for frac in (0.25, 0.50, 0.75):
        sy = int(round(paper.y + paper.h * frac))
        sy = min(height - 3, max(2, sy))
        inside_x = min(width - 3, max(2, int(round(paper.x + 8))))
        outside_x = min(width - 3, max(2, int(round(paper.x - outline - 8))))
        paper_rgb = np.median(image_bgr[sy - 2 : sy + 3, inside_x - 2 : inside_x + 3, ::-1], axis=(0, 1))
        back_rgb = np.median(image_bgr[sy - 2 : sy + 3, outside_x - 2 : outside_x + 3, ::-1], axis=(0, 1))
        ratios.append(_contrast(paper_rgb, back_rgb))
    return tuple(ratios)


def _paper_texture(width: int, height: int, folds: int, seed: int) -> np.ndarray:
    """Opaque paper. Low-frequency grain, sigma 1.5, no clip."""
    rng = np.random.default_rng(seed)
    bands = min(6, max(1, folds))
    height = max(1, height)
    width = max(1, width)
    base = np.empty((height, width, 3), np.float32)
    for index in range(bands):
        y0 = int(round(index * height / bands))
        y1 = int(round((index + 1) * height / bands))
        tone = PAPER_RGB if index % 2 == 0 else SEAM_RGB
        base[y0:y1, :, 0] = tone[0]
        base[y0:y1, :, 1] = tone[1]
        base[y0:y1, :, 2] = tone[2]
    noise = rng.standard_normal((height, width)).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), 0.8)
    std = float(noise.std()) or 1.0
    noise *= (0.03 * float(PAPER_RGB[0])) / std
    rounded = np.rint(base + noise[:, :, None])
    overflow = (rounded < 0.0) | (rounded > 255.0)
    if overflow.any():
        rounded[overflow] = base[overflow]
    face = np.empty((height, width, 4), np.uint8)
    face[:, :, :3] = rounded.astype(np.uint8)
    face[:, :, 3] = 255
    if int(face[:, :, 3].min()) != 255:
        raise RuntimeError("paper face alpha is not 1")
    paper = np.array(PAPER_RGB, np.float32)
    seam = np.array(SEAM_RGB, np.float32)
    rgb = face[:, :, :3].astype(np.float32)
    nearest = np.minimum(np.abs(rgb - paper).max(axis=2), np.abs(rgb - seam).max(axis=2))
    if float(nearest.mean()) > 12.0:
        raise RuntimeError(f"paper face is {nearest.mean():.2f} levels from the page tones")
    return face


def _plate_drift(frame: int) -> float:
    """Slow zoom from 1.00 to 1.06 across the current segment. The ground point stays put."""
    name = segment_at(frame)
    for seg_name, start, end in SEGMENTS:
        if seg_name == name:
            span = max(1, end - start - 1)
            return 1.0 + 0.06 * ((frame - start) / span)
    return 1.0


def _zoom_about(rgb: np.ndarray, scale: float, gx: float, gy: float) -> np.ndarray:
    if scale <= 1.001:
        return rgb
    height, width = rgb.shape[:2]
    matrix = np.array(
        [[scale, 0.0, gx - scale * gx], [0.0, scale, gy - scale * gy]],
        dtype=np.float32,
    )
    return cv2.warpAffine(rgb, matrix, (width, height), flags=cv2.INTER_LINEAR)


def _blit_plate(
    canvas: PaperCanvas,
    bgr_image: np.ndarray,
    scale: float,
    ground_x: float,
    ground_y: float,
    alpha: float,
) -> None:
    """Scale a plate about the ground point. Alpha 0 draws nothing."""
    if alpha <= 0.01 or scale < 0.004:
        return
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    sized_w = max(1, int(round(width * scale)))
    sized_h = max(1, int(round(height * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    sized = cv2.resize(rgb, (sized_w, sized_h), interpolation=interp)
    left = ground_x * (1.0 - scale)
    top = ground_y * (1.0 - scale)
    if alpha < 0.999:
        layer = np.dstack([sized, np.full((sized_h, sized_w), int(round(alpha * 255)), np.uint8)])
        canvas.image(layer, left, top)
        return
    canvas.image(sized, left, top)


def _load_plate(plate_id: str) -> np.ndarray:
    path = ART_ROOT / "normalized" / f"{plate_id}.png"
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"missing normalized plate {path}. Run ingest first.")
    return image


def _scaled_cutout(object_id: str, height_px: float | None = None, width_px: float | None = None) -> np.ndarray:
    path = ART_ROOT / "keyed" / f"{object_id}.png"
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim != 3 or image.shape[2] != 4:
        raise SystemExit(f"missing keyed cutout {path}")
    alpha = image[:, :, 3]
    ys, xs = np.nonzero(alpha > 16)
    if len(ys) == 0:
        raise SystemExit(f"{object_id} has no opaque pixels")
    crop = image[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    if width_px is not None:
        dest_w = max(1, int(round(width_px)))
        dest_h = max(1, int(round(crop.shape[0] * (dest_w / crop.shape[1]))))
    else:
        dest_h = max(1, int(round(height_px or 1)))
        dest_w = max(1, int(round(crop.shape[1] * (dest_h / crop.shape[0]))))
    scaled = cv2.resize(crop.astype(np.float32), (dest_w, dest_h), interpolation=cv2.INTER_LANCZOS4)
    rgb = np.clip(scaled[:, :, :3], 0, 255).astype(np.uint8)
    return np.dstack(
        [rgb[:, :, 2], rgb[:, :, 1], rgb[:, :, 0], np.clip(scaled[:, :, 3], 0, 255).astype(np.uint8)]
    )


def _stamp(fold: int) -> str:
    for key, _distance, label in MILESTONES:
        if milestone_fold(key) == fold:
            return f"= {label}"
    return ""


def context_sub(script: Script, frame: int, fold: int) -> str:
    """Context under the counter. The stack label owns the height, so this never repeats it."""
    for sub in script.subs:
        if sub.start <= frame < sub.end and sub.id not in {"space_km", "h6"}:
            return sub.text
    if frame >= 96 and fold > 0:
        text = length_sub(fold)
        if not text.startswith("PAPER NEEDED"):
            text = f"PAPER NEEDED {text}"
        return text
    return ""


def _moon_on_plate(scene: Scene) -> tuple[float, float, float] | None:
    if scene.moon_cx is None or scene.moon_cy is None or scene.moon_radius is None:
        return None
    scale, tx, ty = align_translate(scene.ground_y, 768, 1376, screen_ground=scene.screen_ground)
    return (
        scene.moon_cx * 768 * scale + tx,
        scene.moon_cy * 1376 * scale + ty,
        scene.moon_radius * 768 * scale,
    )


def _chip_plate(text: str, cx: float, top: float, kind: str, anchor: str) -> tuple[float, float, float, float]:
    """Plate rectangle (x, y, w, h) after the safe-zone clamp. Text sits 8 px below the plate top."""
    size = 36.0
    pad_x = 16.0
    max_w = SAFE_R - SAFE_L - 2 * pad_x
    while size > 28 and text_width(kind, size, text) > max_w:
        size -= 2
    plate_w = text_width(kind, size, text) + 2 * pad_x
    if anchor == "right":
        left = cx - plate_w
    elif anchor == "left":
        left = cx
    else:
        left = cx - plate_w / 2.0
    left = min(max(left, SAFE_L), SAFE_R - plate_w)
    top = max(top, 208.0)
    return left, top - 8.0, plate_w, size + 20.0


def _pad_box(box: tuple[float, float, float, float], gap: float) -> tuple[float, float, float, float]:
    x, y, w, h = box
    return (x - gap, y - gap, w + gap * 2, h + gap * 2)


def _rects_overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah


def _fit_sub(text: str) -> tuple[tuple[str, ...], float, str]:
    """One line at 34 px, else 28 px, else two lines. The words are never cut."""
    kind = "mono" if any(ch.isdigit() for ch in text) else "word"
    max_w = SAFE_R - SAFE_L - 48
    for size in (34.0, 28.0):
        if text_width(kind, size, text) <= max_w:
            return (text,), size, kind
    if text.endswith(" EARTH TO SUN"):
        parts = (text[: -len(" EARTH TO SUN")], "EARTH TO SUN")
    else:
        words = text.split()
        mid = max(1, len(words) // 2)
        parts = (" ".join(words[:mid]), " ".join(words[mid:]))
    for size in (34.0, 28.0):
        if max(text_width(kind, size, part) for part in parts) <= max_w:
            return parts, size, kind
    return parts, 28.0, kind


def _moon_vertical_span(plate_bgr: np.ndarray) -> tuple[float, float] | None:
    """Top of the cream Moon and its vertical size, from the plate itself."""
    rgb = plate_bgr[:, :, ::-1]
    bright = (rgb[:, :, 0] > 180) & (rgb[:, :, 1] > 170) & (rgb[:, :, 2] > 150)
    mid = plate_bgr.shape[1] // 2
    rows = np.nonzero(bright[:, mid - 180 : mid + 180].any(axis=1))[0]
    if len(rows) < 20:
        return None
    top = float(rows.min())
    return top, max(1.0, float(rows.max() - rows.min()))


class DioramaStill:
    def __init__(self, hook: str = "A", script: Script | None = None, scenes: tuple[Scene, ...] | None = None) -> None:
        self.hook = hook
        self.script = script or load_script()
        self.scenes = scenes or scenes_from_manifest()
        self._plates: dict[str, np.ndarray] = {}
        self._iss_box: tuple[float, float, float, float] | None = None
        self._boxes: dict[str, tuple[float, float, float, float]] = {}
        self._height_box: tuple[float, float, float, float] | None = None

    def plate(self, plate_id: str) -> np.ndarray:
        if plate_id not in self._plates:
            self._plates[plate_id] = _load_plate(plate_id)
        return self._plates[plate_id]

    def render(self, frame: int) -> StillFrame:
        self._boxes = {}
        self._iss_box = None
        scene = scene_at(frame, self.scenes)
        fold = folds_done(frame)
        canvas = PaperCanvas(FRAME_W, FRAME_H)
        rgb = cv2.cvtColor(self.plate(scene.plate), cv2.COLOR_BGR2RGB)
        canvas.image(rgb, 0, 0)
        notes: list[str] = []
        scale = px_per_m(scene.world_m, screen_ground=scene.screen_ground)
        self._objects(canvas, scene, scale, notes)
        if frame < 84:
            paper = self._card(canvas)
            notes.append(f"card {CARD_W:.0f}x{CARD_H:.0f} base {CARD_BASE:.0f}")
        else:
            paper = self._stack(canvas, fold, scale, scene)
            self._altitude(canvas, scene, scale, fold)
        self._stamp(canvas, frame, fold, scene, scale)
        self._top(canvas, frame, scene, notes)
        self._bottom(canvas, frame, fold)
        image = bgr(canvas)
        ratios = edge_contrast(image, paper, STACK_OUTLINE if frame >= 84 else OUTLINE_PX)
        name = "hook" if frame == 0 else f"fold_{fold}"
        return StillFrame(name, frame, fold, scene.plate, scene.name, image, paper, ratios, tuple(notes))

    def _objects(self, canvas: PaperCanvas, scene: Scene, scale: float, notes: list[str]) -> None:
        ground = scene.screen_ground * FRAME_H
        for object_id, x_frac, meters in SCENE_OBJECTS.get(scene.name, ()):
            height_px = meters * scale
            cutout = _scaled_cutout(object_id, height_px=height_px)
            left = FRAME_W * x_frac - cutout.shape[1] / 2.0
            top = ground - cutout.shape[0]
            self._boxes[object_id] = (left, top, float(cutout.shape[1]), float(cutout.shape[0]))
            canvas.image(cutout, left, top)
            error = abs(cutout.shape[0] - height_px) / height_px if height_px else 0.0
            notes.append(f"{object_id} {meters:g} m -> {cutout.shape[0]} px (scale error {error:.2%})")
        if scene.name == "orbit":
            cutout = _scaled_cutout("o05_iss", width_px=0.18 * FRAME_W)
            altitude = ground - ISS_M * scale
            left = 0.76 * FRAME_W - cutout.shape[1] / 2.0
            top = altitude - cutout.shape[0] / 2.0
            self._iss_box = (left, top, float(cutout.shape[1]), float(cutout.shape[0]))
            canvas.image(cutout, left, top)
            notes.append(f"o05_iss width {cutout.shape[1]} px (18% of the frame) at {ISS_M/1000:.0f} km")

    def _stack(self, canvas: PaperCanvas, fold: int, scale: float, scene: Scene, label: bool = True) -> PaperBox:
        base = scene.screen_ground * FRAME_H
        height_px = max(24.0, height_m(fold) * scale)
        top = base - height_px
        x = FRAME_W / 2.0 - STACK_W / 2.0
        silhouette = self._silhouette(x, top, base)
        canvas.soft_polygon(silhouette, SHADOW, 0.28, 16, dx=24, dy=10)
        canvas.stroke_polygon(silhouette, OUTLINE, STACK_OUTLINE * 2, alpha=0.80)
        front = [(x, top), (x + STACK_W, top), (x + STACK_W, base), (x, base)]
        cap = [
            (x, top),
            (x + STACK_W, top),
            (x + STACK_W + SIDE_W, top - TOP_H),
            (x + SIDE_W, top - TOP_H),
        ]
        side = [
            (x + STACK_W, top),
            (x + STACK_W + SIDE_W, top - TOP_H),
            (x + STACK_W + SIDE_W, base - TOP_H),
            (x + STACK_W, base),
        ]
        canvas.polygon(side, SIDE)
        canvas.polygon(cap, TOP_FACE)
        canvas.rect(x, top, STACK_W, height_px, PAPER)
        texture = _paper_texture(int(STACK_W), int(round(height_px)), fold, seed=7)
        canvas.image(texture, x, top)
        canvas.rect(x, top, STACK_W, 2, "#FFFFFF")
        if label:
            text = height_label(fold)
            label_top = top - TOP_H - 56
            # A chip whose plate would sit above y = 200 moves into the top band.
            # On the Moon card the title and the disc block the right side, so it sits in the open sky.
            if label_top < 208.0:
                self._chip_in_open_sky(canvas, text, scene)
            else:
                self._chip(canvas, text, FRAME_W / 2.0, label_top, "mono")
        return PaperBox(x, top, STACK_W, height_px)

    def _silhouette(self, x: float, top: float, base: float) -> list[tuple[float, float]]:
        return [
            (x, base),
            (x, top),
            (x + SIDE_W, top - TOP_H),
            (x + STACK_W + SIDE_W, top - TOP_H),
            (x + STACK_W + SIDE_W, base - TOP_H),
            (x + STACK_W, base),
        ]

    def _card(self, canvas: PaperCanvas) -> PaperBox:
        return self._card_at(canvas, HOOK_PROGRESS, 1.0)

    def _card_at(self, canvas: PaperCanvas, progress: float, squash: float) -> PaperBox:
        page_h = CARD_H * squash
        top = CARD_BASE - page_h
        half = CARD_W / 2.0
        left = FRAME_W / 2.0 - half
        right_w = max(1.0, half * abs(math.cos(progress * math.pi)))
        canvas.soft_polygon(
            [(left, CARD_BASE), (left, top), (left + half + right_w, top), (left + half + right_w, CARD_BASE)],
            SHADOW,
            0.30,
            8,
            dx=20,
        )
        canvas.rect(left - OUTLINE_PX, top - OUTLINE_PX, half + OUTLINE_PX, page_h + OUTLINE_PX, OUTLINE)
        canvas.rect(FRAME_W / 2.0, top - OUTLINE_PX, right_w + OUTLINE_PX, page_h + OUTLINE_PX, OUTLINE)
        canvas.rect(left, top, half, page_h, PAPER)
        canvas.image(_paper_texture(int(half), int(round(page_h)), 1, 7), left, top)
        right = _paper_texture(max(1, int(round(right_w))), max(1, int(round(page_h))), 1, 11).astype(np.float32)
        right[:, :, :3] *= 0.86
        right[:, :, 3] = 255
        canvas.image(np.clip(right, 0, 255).astype(np.uint8), FRAME_W / 2.0, top)
        return PaperBox(left, top, half, page_h)

    def _altitude(self, canvas: PaperCanvas, scene: Scene, scale: float, fold: int) -> None:
        ground = scene.screen_ground * FRAME_H
        if scene.name == "atmosphere":
            y = ground - KARMAN_M * scale
            self._dashed(canvas, y)
            self._chip(canvas, "SPACE", SAFE_L + 90, y - 52, "word")
        if scene.name == "deep_space":
            y = ground - MOON_M * scale
            moon = _moon_on_plate(scene)
            right = SAFE_R
            if moon is not None:
                right = min(right, self._chord_left(moon, y) - 8)
            self._dashed(canvas, y, right=right)

    def _dashed(self, canvas: PaperCanvas, y: float, right: float = SAFE_R) -> None:
        gap_l = FRAME_W / 2.0 - STACK_W / 2.0 - OUTLINE_PX - 10
        gap_r = FRAME_W / 2.0 + STACK_W / 2.0 + SIDE_W + OUTLINE_PX + 10
        runs = [(SAFE_L, gap_l)]
        if gap_r < right:
            runs.append((gap_r, right))
        for x0, x1 in runs:
            x = x0
            while x < x1:
                canvas.rect(x, y - 3, min(16.0, x1 - x), 6, PAPER)
                x += 28

    def _chord_left(self, moon: tuple[float, float, float], y: float) -> float:
        cx, cy, radius = moon
        dy = y - cy
        if abs(dy) >= radius:
            return cx - radius
        return cx - math.sqrt(radius * radius - dy * dy)

    def _stamp(self, canvas: PaperCanvas, frame: int, fold: int, scene: Scene, scale: float) -> None:
        if fold in (30, 42):
            return
        text = _stamp(fold)
        if not text or frame != fold_frame(fold):
            return
        ground = scene.screen_ground * FRAME_H
        if scene.name == "street":
            self._label_above(canvas, text, self._boxes.get("o01_person"), ground - PERSON_M * scale, 0.24 * FRAME_W)
        elif scene.name == "city":
            self._label_above(canvas, text, self._boxes.get("o04_burj_khalifa"), ground - BURJ_M * scale, 0.76 * FRAME_W)
        elif scene.name == "orbit":
            box = self._iss_box
            self._label_above(
                canvas,
                text,
                box,
                ground - ISS_M * scale,
                0.76 * FRAME_W,
                below_title=True,
            )

    def _chip_in_open_sky(self, canvas: PaperCanvas, text: str, scene: Scene) -> None:
        """Height chip at y 200–260, clear of the centered title and the Moon."""
        if scene.name != "deep_space":
            self._chip(canvas, text, FRAME_W / 2.0 + STACK_W / 2.0 + SIDE_W + 16.0, 208.0, "mono", anchor="left")
            return
        title_w = text_width("word", 72.0, "THE MOON")
        title_left = FRAME_W / 2.0 - title_w / 2.0 - 20.0
        room = title_left - 12.0 - SAFE_L
        size = 36.0
        while size > 22.0 and text_width("mono", size, text) + 32.0 > room:
            size -= 2.0
        plate_w = text_width("mono", size, text) + 32.0
        left = SAFE_L
        top = 360.0
        self._height_box = (left, top - 8.0, plate_w, size + 20.0)
        canvas.round_rect(left, top - 8.0, plate_w, size + 20.0, 16, OUTLINE, PLATE_ALPHA)
        canvas.text(text, left + 16.0, top, "mono", size, PAPER)

    def _label_above(
        self,
        canvas: PaperCanvas,
        text: str,
        box: tuple[float, float, float, float] | None,
        tip_y: float,
        tip_x: float,
        below_title: bool = False,
    ) -> None:
        """Above and to the right of the object. Never on top of the sprite."""
        size = 36.0
        if box is not None:
            left, top, width, height = box
            tip_x = left + width / 2.0
            tip_y = top
        else:
            left = top = width = height = 0.0
        text_top = tip_y - size - 32.0
        anchor_x = tip_x + 8.0

        def plate_of(y: float, x: float) -> tuple[float, float, float, float]:
            return _chip_plate(text, x, y, "word", "left")

        plate = plate_of(text_top, anchor_x)
        sprite = (left, top, width, height) if box is not None else None
        if sprite is not None and _rects_overlap(plate, sprite):
            text_top = top + height + 16.0
            plate = plate_of(text_top, anchor_x)
        if below_title and _rects_overlap(plate, (SAFE_L, 200.0, SAFE_R - SAFE_L, 240.0)):
            if box is not None:
                text_top = top + height + 16.0
                plate = plate_of(text_top, anchor_x)
        stack = (FRAME_W / 2.0 - STACK_W / 2.0 - 16.0, 200.0, STACK_W + SIDE_W + 32.0, FRAME_H)
        if box is not None and left + width / 2.0 < FRAME_W / 2.0 and _rects_overlap(plate, stack):
            anchor_x -= (plate[0] + plate[2]) - (stack[0] - 12.0)
            plate = plate_of(text_top, anchor_x)
        height_box = self._height_box
        if height_box is not None and _rects_overlap(plate, _pad_box(height_box, 24.0)):
            anchor_x = SAFE_L
            text_top = 480.0
        self._chip(canvas, text, anchor_x, text_top, "word", anchor="left")

    def _chip(
        self,
        canvas: PaperCanvas,
        text: str,
        cx: float,
        top: float,
        kind: str,
        anchor: str = "center",
    ) -> None:
        left, plate_y, plate_w, plate_h = _chip_plate(text, cx, top, kind, anchor)
        size = plate_h - 20.0
        if kind == "mono":
            self._height_box = (left, plate_y, plate_w, plate_h)
        canvas.round_rect(left, plate_y, plate_w, plate_h, 16, OUTLINE, PLATE_ALPHA)
        canvas.text(text, left + 16.0, plate_y + 8.0, kind, size, PAPER)

    def _top(self, canvas: PaperCanvas, frame: int, scene: Scene, notes: list[str]) -> None:
        lines = top_lines(self.script, frame, self.hook)
        if not lines:
            return
        if frame <= 180 or frame >= 1800:
            self._hook_title(canvas, lines, notes, self.plate(scene.plate))
            return
        if scene.name == "deep_space":
            self._moon_title(canvas, lines)
            return
        self._centered_title(canvas, lines, 216.0)

    def _hook_title(
        self, canvas: PaperCanvas, lines: tuple[str, ...], notes: list[str], plate_bgr: np.ndarray
    ) -> None:
        size = HOOK_SIZE
        ink_h = max(measure("word", size, line).height for line in lines)
        gap = 8.0
        pad = 16.0
        widths = [text_width("word", size, line) for line in lines]
        block_w = max(widths)
        block_h = len(lines) * ink_h + (len(lines) - 1) * gap
        x = FRAME_W / 2.0 - block_w / 2.0
        y = HOOK_BAND_TOP + pad
        plate_bottom = y + block_h + pad
        if plate_bottom > HOOK_BAND_BOTTOM:
            y -= plate_bottom - HOOK_BAND_BOTTOM
            plate_bottom = HOOK_BAND_BOTTOM
        canvas.round_rect(
            x - pad, HOOK_BAND_TOP, block_w + pad * 2, plate_bottom - HOOK_BAND_TOP, 16, OUTLINE, HOOK_PLATE_ALPHA
        )
        span = _moon_vertical_span(plate_bgr)
        if span is not None:
            moon_top, diameter = span
            covered = max(0.0, plate_bottom - moon_top)
            share = covered / diameter if diameter else 0.0
            notes.append(f"hook plate covers {share:.1%} of the Moon disc")
        cursor = y
        for line, width in zip(lines, widths):
            canvas.text(line, FRAME_W / 2.0 - width / 2.0, cursor, "word", size, PAPER)
            cursor += ink_h + gap

    def _centered_title(self, canvas: PaperCanvas, lines: tuple[str, ...], y: float) -> None:
        size = 100.0
        pad = 24.0
        max_w = SAFE_R - SAFE_L - 2 * pad
        while size > 72 and max(text_width("word", size, line) for line in lines) > max_w:
            size -= 2
        block_w = max(text_width("word", size, line) for line in lines)
        block_h = len(lines) * (size + 8)
        x = min(max(FRAME_W / 2.0 - block_w / 2.0, SAFE_L + pad), SAFE_R - pad - block_w)
        canvas.round_rect(x - pad, y - 16, block_w + pad * 2, block_h + 8, 16, OUTLINE, PLATE_ALPHA)
        cursor = y
        for line in lines:
            width = text_width("word", size, line)
            canvas.text(line, x + (block_w - width) / 2.0, cursor, "word", size, PAPER)
            cursor += size + 8

    def _moon_title(self, canvas: PaperCanvas, lines: tuple[str, ...]) -> None:
        """Center the title in the top slot. It does not sit in a corner."""
        size = 72.0
        pad = 20.0
        # Stay left of the stack so the title never covers the paper.
        max_w = FRAME_W / 2.0 - STACK_W / 2.0 - 24.0 - SAFE_L - 2 * pad
        while size > 36 and max(text_width("word", size, line) for line in lines) > max_w:
            size -= 2
        block_w = max(text_width("word", size, line) for line in lines)
        line_h = size + 6
        block_h = len(lines) * line_h
        x = SAFE_L
        y = 208.0
        if y + block_h > 300:
            y = max(208.0, 300.0 - block_h)
        canvas.round_rect(x, y - 12, block_w + pad * 2, block_h + 8, 16, OUTLINE, PLATE_ALPHA)
        cursor = y
        for line in lines:
            canvas.text(line, x + pad, cursor, "word", size, PAPER)
            cursor += line_h

    def _bottom(self, canvas: PaperCanvas, frame: int, fold: int) -> None:
        sub = context_sub(self.script, frame, fold)
        label_w = text_width("word", 36, "FOLDS")
        number = str(fold)
        num_w = text_width("mono", 130, number)
        sub_lines: tuple[str, ...] = ()
        sub_size = 34.0
        sub_kind = "word"
        sub_w = 0.0
        if sub:
            sub_lines, sub_size, sub_kind = _fit_sub(sub)
            sub_w = max(text_width(sub_kind, sub_size, line) for line in sub_lines)
        sub_h = 0.0
        if sub_lines:
            sub_h = len(sub_lines) * sub_size + (len(sub_lines) - 1) * 6
        block_w = max(label_w, num_w, sub_w)
        block_h = 36 + 8 + 130 + (8 + sub_h if sub_lines else 0)
        pad = 24.0
        x = FRAME_W / 2.0 - (block_w + pad * 2) / 2.0
        y = 1316.0
        canvas.round_rect(x, y, block_w + pad * 2, block_h + pad * 2, 16, OUTLINE, COUNTER_ALPHA)
        cursor = y + pad
        canvas.text("FOLDS", FRAME_W / 2.0 - label_w / 2.0, cursor, "word", 36, "#8A93A6")
        cursor += 36 + 8
        canvas.text(number, FRAME_W / 2.0 - num_w / 2.0, cursor, "mono", 130, PAPER)
        if sub_lines:
            cursor += 130 + 8
            for line in sub_lines:
                width = text_width(sub_kind, sub_size, line)
                canvas.text(line, FRAME_W / 2.0 - width / 2.0, cursor, sub_kind, sub_size, "#C5CDD8")
                cursor += sub_size + 6


    def render_animated(self, frame: int) -> np.ndarray:
        """One master frame with the fold, the edge turn, or the pull-back in motion."""
        self._boxes = {}
        self._iss_box = None
        handoff = self._handoff_at(frame)
        canvas = PaperCanvas(FRAME_W, FRAME_H)
        self._height_box = None
        notes: list[str] = []
        if handoff is None:
            scene = scene_at(frame, self.scenes)
            rgb = cv2.cvtColor(self.plate(scene.plate), cv2.COLOR_BGR2RGB)
            rgb = _zoom_about(rgb, _plate_drift(frame), FRAME_W / 2.0, scene.screen_ground * FRAME_H)
            canvas.image(rgb, 0, 0)
            scale = px_per_m(scene.world_m, screen_ground=scene.screen_ground)
            self._objects(canvas, scene, scale, notes)
        else:
            current, nxt, raw_u = handoff
            scale = px_per_m_at(frame, self.scenes)
            self._draw_pullback(canvas, current, nxt, raw_u, scale)
            scene = nxt if raw_u >= 0.6 else current
        self._animated_paper(canvas, frame, scale, scene)
        self._stamp(canvas, frame, folds_done(frame), scene, scale)
        self._top(canvas, frame, scene, notes)
        self._bottom(canvas, frame, folds_done(frame))
        return bgr(canvas)

    def _handoff_at(self, frame: int) -> tuple[Scene, Scene, float] | None:
        for current, nxt in zip(self.scenes, self.scenes[1:]):
            fold = break_fold(current.world_m, screen_ground=current.screen_ground)
            start = handoff_start_frame(fold)
            end = start + HANDOFF_FRAMES
            if start <= frame < end:
                raw_u = (frame - start) / (HANDOFF_FRAMES - 1)
                return current, nxt, raw_u
        return None

    def _draw_pullback(self, canvas: PaperCanvas, current: Scene, nxt: Scene, raw_u: float, scale: float) -> None:
        eased = ease_in_out_cubic(raw_u)
        old_ppm = px_per_m(current.world_m, screen_ground=current.screen_ground)
        new_ppm = px_per_m(nxt.world_m, screen_ground=nxt.screen_ground)
        ground_x = FRAME_W / 2.0
        ground_y = current.screen_ground * FRAME_H
        new_alpha = 0.2 + 0.8 * eased
        new_scale = 1.6 + (1.0 - 1.6) * eased
        _blit_plate(canvas, self.plate(nxt.plate), new_scale, ground_x, ground_y, new_alpha)
        shrink = log_blend(1.0, new_ppm / old_ppm, raw_u)
        old_alpha = 1.0 if raw_u < 0.6 else max(0.0, 1.0 - (raw_u - 0.6) / 0.4)
        _blit_plate(canvas, self.plate(current.plate), shrink, ground_x, ground_y, old_alpha)
        self._objects_about(canvas, current, old_ppm, shrink, ground_x, ground_y, old_alpha)
        self._objects_about(canvas, nxt, new_ppm, new_scale, ground_x, ground_y, new_alpha)

    def _objects_about(
        self,
        canvas: PaperCanvas,
        scene: Scene,
        ppm: float,
        shrink: float,
        ground_x: float,
        ground_y: float,
        alpha: float,
    ) -> None:
        if alpha <= 0.01:
            return
        for object_id, x_frac, meters in SCENE_OBJECTS.get(scene.name, ()):
            cutout = _scaled_cutout(object_id, height_px=max(1.0, meters * ppm * shrink))
            if alpha < 0.999:
                cutout = cutout.copy()
                cutout[:, :, 3] = np.clip(cutout[:, :, 3].astype(np.float32) * alpha, 0, 255).astype(np.uint8)
            feet_x = ground_x + (FRAME_W * x_frac - ground_x) * shrink
            left = feet_x - cutout.shape[1] / 2.0
            top = ground_y - cutout.shape[0]
            self._boxes[object_id] = (left, top, float(cutout.shape[1]), float(cutout.shape[0]))
            canvas.image(cutout, left, top)

    def _animated_paper(self, canvas: PaperCanvas, frame: int, scale: float, scene: Scene) -> None:
        moving, progress, phase = fold_motion(frame)
        if frame < EDGE_START:
            if phase == "flip":
                self._card_at(canvas, progress, 1.0)
            elif phase == "squash":
                self._card_at(canvas, 1.0, 1.0)
            else:
                self._card_at(canvas, 0.0, 1.0)
            return
        if frame < EDGE_END:
            self._edge_turn(canvas, frame, scale, scene)
            return
        if moving >= 8 and phase == "flip":
            self._doubling(canvas, moving, progress, scale, scene)
            return
        if moving >= 8 and phase == "squash":
            self._stack(canvas, moving, scale, scene)
            return
        self._stack(canvas, folds_done(frame), scale, scene)

    def _edge_turn(self, canvas: PaperCanvas, frame: int, scale: float, scene: Scene) -> None:
        """Twelve frames. The closed card turns onto its edge and becomes the short tower."""
        target = max(24.0, height_m(8) * scale)
        _ = scene.name
        span = max(1, EDGE_END - EDGE_START)
        u = ease_in_out_cubic((frame - EDGE_START) / span)
        width = STACK_W + (CARD_W - STACK_W) * abs(math.cos(u * math.pi / 2.0))
        height = CARD_H + (target - CARD_H) * u
        arrived = u
        left = FRAME_W / 2.0 - width / 2.0
        top = CARD_BASE - height
        side_w = SIDE_W * arrived
        canvas.soft_polygon(
            [
                (left, CARD_BASE),
                (left, top),
                (left + side_w, top - TOP_H * arrived),
                (left + width + side_w, top - TOP_H * arrived),
                (left + width + side_w, CARD_BASE - TOP_H * arrived),
                (left + width, CARD_BASE),
            ],
            SHADOW,
            0.28,
            16,
            dx=24,
            dy=10,
        )
        if side_w > 1:
            canvas.polygon(
                [
                    (left + width, top),
                    (left + width + side_w, top - TOP_H * arrived),
                    (left + width + side_w, CARD_BASE - TOP_H * arrived),
                    (left + width, CARD_BASE),
                ],
                SIDE,
            )
        canvas.rect(left, top, width, height, PAPER)
        canvas.image(
            _paper_texture(max(1, int(round(width))), max(1, int(round(height))), 8, 7),
            left,
            top,
        )
        # The doubling copy occupies the top half of the landing tower.
        if arrived > 0.45:
            copy_h = height * 0.5 * min(1.0, (arrived - 0.45) / 0.55)
            canvas.rect(left, top, width, max(copy_h, 1.0), PAPER)
            canvas.image(
                _paper_texture(max(1, int(round(width))), max(1, int(round(copy_h))), 4, 3),
                left,
                top,
            )

    def _doubling(self, canvas: PaperCanvas, fold: int, progress: float, scale: float, scene: Scene) -> None:
        """The copy flips onto the top. The label changes when the page has landed."""
        base = scene.screen_ground * FRAME_H
        height_px = max(24.0, height_m(fold - 1) * scale)
        extra = 0.0 if progress < 0.5 else height_px * math.sin((progress - 0.5) * math.pi)
        top = base - height_px - extra
        x = FRAME_W / 2.0 - STACK_W / 2.0
        squash = 1.0 - 0.12 * math.sin(progress * math.pi)
        self._prism(canvas, x, top, (height_px + extra) * squash, fold, label_fold=fold - 1)

    def _prism(
        self,
        canvas: PaperCanvas,
        x: float,
        top: float,
        height_px: float,
        fold: int,
        label_fold: int | None = None,
    ) -> None:
        base = top + height_px
        silhouette = self._silhouette(x, top, base)
        canvas.soft_polygon(silhouette, SHADOW, 0.28, 16, dx=24, dy=10)
        canvas.stroke_polygon(silhouette, OUTLINE, STACK_OUTLINE * 2, alpha=0.80)
        cap = [
            (x, top),
            (x + STACK_W, top),
            (x + STACK_W + SIDE_W, top - TOP_H),
            (x + SIDE_W, top - TOP_H),
        ]
        side = [
            (x + STACK_W, top),
            (x + STACK_W + SIDE_W, top - TOP_H),
            (x + STACK_W + SIDE_W, base - TOP_H),
            (x + STACK_W, base),
        ]
        canvas.polygon(side, SIDE)
        canvas.polygon(cap, TOP_FACE)
        canvas.rect(x, top, STACK_W, height_px, PAPER)
        canvas.image(_paper_texture(int(STACK_W), max(1, int(round(height_px))), max(1, fold), seed=7), x, top)
        canvas.rect(x, top, STACK_W, 2, "#FFFFFF")
        shown = fold if label_fold is None else label_fold
        if shown > 0:
            label_top = top - TOP_H - 56
            if label_top < 208.0:
                self._chip(canvas, height_label(shown), x + STACK_W + SIDE_W + 16.0, 208.0, "mono", anchor="left")
            else:
                self._chip(canvas, height_label(shown), FRAME_W / 2.0, label_top, "mono")


def render_heroes(hook: str = "A", out_dir: Path | None = None) -> list[StillFrame]:
    out_dir = out_dir or (ROOT / "out" / "stills")
    out_dir.mkdir(parents=True, exist_ok=True)
    drawer = DioramaStill(hook)
    frames = []
    lines = [
        "# Hero stills",
        "",
        "Stack-edge contrast is the white front face against the plate 8 px outside the 6 px outline, at 25%, 50% and 75% of the face.",
        "The slate strip is gone. p03 is the current plate; a replacement has not arrived.",
        "",
    ]
    for name, fold in HEROES:
        frame = 0 if fold == 0 else fold_frame(fold)
        still = drawer.render(frame)
        path = out_dir / f"{name}.png"
        cv2.imwrite(str(path), still.image)
        ratios = still.ratios
        flag = " PASS" if min(ratios) >= 3.0 else " under 3:1"
        print(
            f"{name} frame {still.frame} {still.scene} {still.plate} "
            f"contrast {ratios[0]:.2f} {ratios[1]:.2f} {ratios[2]:.2f}{flag}"
        )
        for note in still.notes:
            print(f"  {note}")
        lines.append(
            f"- {name}: frame {still.frame}, {still.scene}, {still.plate}, "
            f"{ratios[0]:.2f}:1, {ratios[1]:.2f}:1, {ratios[2]:.2f}:1{flag}"
        )
        for note in still.notes:
            lines.append(f"  - {note}")
        frames.append(still)
    lines += ["", CRITIQUE]
    report = ROOT / "out" / "stills_contrast.md"
    report.write_text("\n".join(lines) + "\n")
    print(f"wrote {report}")
    return frames


CRITIQUE = """## Critique

The stack is a 150 px oblique prism: white front with the plate's grain at 6%, a 2 px light top edge, #B8C1D4 side 28 px, top face 12 px, 6 px outline at 80% opacity, and a shadow 24 px right and 10 px down. There is no slate strip. Where the front face sits on cream, the edge contrast can fall under 3:1. Those numbers are the measured ones.

The hook card is 520 by 700, base at y = 1306, in front of the lower Moon. Both title lines sit in a solid #1B2030 plate at 96% opacity, Montserrat ExtraBold at 84 px, inside y 190-400. The plate may cover the top 15% of the Moon.

p06 and p07 put the Earth's apex at y = 0.58. Empty rows from the shift are a flat sample plus grain, not a copied edge. p07's teal cap is near-black sky, feathered over 24 px. The counter plate is opaque. Fold 30 and fold 32 say how many Earth-to-Sun distances of paper, from the claims engine. "THE MOON" is centered in the top slot.

p03 is still the current city plate. Its tallest decorative tower is taller than the 828 m Burj, which the ingest warns about. A replacement plate has not arrived.
"""
