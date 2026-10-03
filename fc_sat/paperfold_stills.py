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
from fc_sat.paperfold_render import folds_done, height_label, length_sub
from fc_sat.paperfold_scenes import (
    FRAME_H,
    FRAME_W,
    GROUND,
    HANDOFF_FRAMES,
    Scene,
    align_translate,
    break_fold,
    handoff_start_frame,
    px_per_m,
    scenes_from_manifest,
)
from fc_sat.paperfold_schedule import fold_frame
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
CARD_W = 520.0
CARD_H = 700.0
CARD_BASE = 1306.0
HOOK_PROGRESS = 0.35
HOOK_SIZE = 100.0
HOOK_Y = 190.0
HOOK_Y_MAX = 420.0
SAFE_L = 130.0
SAFE_R = 950.0
PLATE_ALPHA = 0.92

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
    rng = np.random.default_rng(seed)
    bands = min(6, max(1, folds))
    image = np.empty((max(1, height), max(1, width), 4), np.float32)
    image[:, :, 3] = 255
    for index in range(bands):
        y0 = int(round(index * image.shape[0] / bands))
        y1 = int(round((index + 1) * image.shape[0] / bands))
        tone = PAPER_RGB if index % 2 == 0 else SEAM_RGB
        image[y0:y1, :, 0] = tone[0]
        image[y0:y1, :, 1] = tone[1]
        image[y0:y1, :, 2] = tone[2]
    noise = rng.normal(0.0, 2.5, size=(image.shape[0], image.shape[1], 1))
    image[:, :, :3] = np.clip(image[:, :, :3] + noise, 0, 255)
    return image.astype(np.uint8)


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


def _moon_face_overlap(image_bgr: np.ndarray, rects: list[tuple[float, float, float, float]]) -> float:
    """Share of the text ink boxes that sit on the cream Moon face."""
    rgb = image_bgr[:, :, ::-1].astype(np.int16)
    bright = (rgb[:, :, 0] > 180) & (rgb[:, :, 1] > 170) & (rgb[:, :, 2] > 150)
    covered = 0
    total = 0
    height, width = bright.shape
    for x, y, w, h in rects:
        x0 = max(0, int(x))
        y0 = max(0, int(y))
        x1 = min(width, int(round(x + w)))
        y1 = min(height, int(round(y + h)))
        if x1 <= x0 or y1 <= y0:
            continue
        patch = bright[y0:y1, x0:x1]
        total += patch.size
        covered += int(patch.sum())
    if total == 0:
        return 0.0
    return covered / total


class DioramaStill:
    def __init__(self, hook: str = "A", script: Script | None = None, scenes: tuple[Scene, ...] | None = None) -> None:
        self.hook = hook
        self.script = script or load_script()
        self.scenes = scenes or scenes_from_manifest()
        self._plates: dict[str, np.ndarray] = {}

    def plate(self, plate_id: str) -> np.ndarray:
        if plate_id not in self._plates:
            self._plates[plate_id] = _load_plate(plate_id)
        return self._plates[plate_id]

    def render(self, frame: int) -> StillFrame:
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
        ratios = edge_contrast(image, paper)
        name = "hook" if frame == 0 else f"fold_{fold}"
        return StillFrame(name, frame, fold, scene.plate, scene.name, image, paper, ratios, tuple(notes))

    def _objects(self, canvas: PaperCanvas, scene: Scene, scale: float, notes: list[str]) -> None:
        ground = scene.screen_ground * FRAME_H
        for object_id, x_frac, meters in SCENE_OBJECTS.get(scene.name, ()):
            height_px = meters * scale
            cutout = _scaled_cutout(object_id, height_px=height_px)
            left = FRAME_W * x_frac - cutout.shape[1] / 2.0
            top = ground - cutout.shape[0]
            canvas.image(cutout, left, top)
            error = abs(cutout.shape[0] - height_px) / height_px if height_px else 0.0
            notes.append(f"{object_id} {meters:g} m -> {cutout.shape[0]} px (scale error {error:.2%})")
        if scene.name == "orbit":
            cutout = _scaled_cutout("o05_iss", width_px=0.18 * FRAME_W)
            altitude = ground - ISS_M * scale
            left = 0.76 * FRAME_W - cutout.shape[1] / 2.0
            top = altitude - cutout.shape[0] / 2.0
            canvas.image(cutout, left, top)
            notes.append(f"o05_iss width {cutout.shape[1]} px (18% of the frame) at {ISS_M/1000:.0f} km")

    def _stack(self, canvas: PaperCanvas, fold: int, scale: float, scene: Scene) -> PaperBox:
        base = scene.screen_ground * FRAME_H
        height_px = max(24.0, height_m(fold) * scale)
        top = base - height_px
        x = FRAME_W / 2.0 - STACK_W / 2.0
        silhouette = self._silhouette(x, top, base)
        canvas.soft_polygon(silhouette, SHADOW, 0.30, 8, dx=20)
        canvas.stroke_polygon(silhouette, OUTLINE, OUTLINE_PX * 2)
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
        texture = _paper_texture(int(STACK_W), int(round(height_px)), fold, seed=7)
        canvas.image(texture, x, top)
        label = height_label(fold)
        label_cx = FRAME_W / 2.0
        if scene.name == "deep_space":
            label_cx = self._left_of_moon(scene, label, "mono", 36.0, top - TOP_H - 56 + 18)
        self._chip(canvas, label, label_cx, top - TOP_H - 56, "mono")
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
        top = CARD_BASE - CARD_H
        half = CARD_W / 2.0
        left = FRAME_W / 2.0 - half
        right_w = half * abs(math.cos(HOOK_PROGRESS * math.pi))
        canvas.soft_polygon(
            [(left, CARD_BASE), (left, top), (left + half + right_w, top), (left + half + right_w, CARD_BASE)],
            SHADOW,
            0.30,
            8,
            dx=20,
        )
        canvas.rect(left - OUTLINE_PX, top - OUTLINE_PX, half + OUTLINE_PX, CARD_H + OUTLINE_PX, OUTLINE)
        canvas.rect(FRAME_W / 2.0, top - OUTLINE_PX, right_w + OUTLINE_PX, CARD_H + OUTLINE_PX, OUTLINE)
        canvas.image(_paper_texture(int(half), int(CARD_H), 1, 7), left, top)
        right = _paper_texture(max(1, int(round(right_w))), int(CARD_H), 1, 11).astype(np.float32)
        right[:, :, :3] *= 0.86
        canvas.image(np.clip(right, 0, 255).astype(np.uint8), FRAME_W / 2.0, top)
        return PaperBox(left, top, half, CARD_H)

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

    def _left_of_moon(self, scene: Scene, text: str, kind: str, size: float, mid_y: float) -> float:
        """Center a label so its plate stays left of the Moon disc."""
        moon = _moon_on_plate(scene)
        width = text_width(kind, size, text)
        center = FRAME_W / 2.0
        if moon is None:
            return center
        limit = self._chord_left(moon, mid_y) - 16 - 12 - width / 2.0
        return min(center, limit)

    def _stamp(self, canvas: PaperCanvas, frame: int, fold: int, scene: Scene, scale: float) -> None:
        if fold in (30, 42):
            return
        text = _stamp(fold)
        if not text or frame != fold_frame(fold):
            return
        ground = scene.screen_ground * FRAME_H
        if scene.name == "street":
            self._chip(canvas, text, 250, ground - PERSON_M * scale - 64, "word")
        elif scene.name == "city":
            self._chip(canvas, text, 900, ground - BURJ_M * scale + 160, "word")
        elif scene.name == "orbit":
            self._chip(canvas, text, 0.76 * FRAME_W, ground - ISS_M * scale - 80, "word")

    def _chip(self, canvas: PaperCanvas, text: str, cx: float, top: float, kind: str) -> None:
        size = 36.0
        while size > 22 and text_width(kind, size, text) > 460:
            size -= 2
        width = text_width(kind, size, text)
        x = min(SAFE_R - width - 16, max(SAFE_L, cx - width / 2.0))
        canvas.round_rect(x - 16, top - 8, width + 32, size + 20, 16, OUTLINE, PLATE_ALPHA)
        canvas.text(text, x, top, kind, size, PAPER)

    def _top(self, canvas: PaperCanvas, frame: int, scene: Scene, notes: list[str]) -> None:
        lines = top_lines(self.script, frame, self.hook)
        if not lines:
            return
        if frame <= 180 or frame >= 1800:
            self._hook_title(canvas, lines, notes)
            return
        if scene.name == "deep_space":
            self._clear_of_moon(canvas, lines, scene)
            return
        self._centered_title(canvas, lines, 210.0)

    def _hook_title(self, canvas: PaperCanvas, lines: tuple[str, ...], notes: list[str]) -> None:
        size = HOOK_SIZE
        ink_h = max(measure("word", size, line).height for line in lines)
        gap = 8.0
        widths = [text_width("word", size, line) for line in lines]
        block_w = max(widths)
        x = FRAME_W / 2.0 - block_w / 2.0
        y = HOOK_Y
        rects = []
        cursor = y
        for line, width in zip(lines, widths):
            rects.append((FRAME_W / 2.0 - width / 2.0, cursor, width, ink_h))
            cursor += ink_h + gap
        if cursor - gap > HOOK_Y_MAX:
            notes.append(f"hook text extends to {cursor - gap:.0f}, past y {HOOK_Y_MAX:.0f}")
        plate = bgr(canvas)
        worst = 99.0
        for rx, ry, rw, rh in rects:
            patch = plate[int(ry) : int(ry + rh), int(rx) : int(rx + rw)]
            if patch.size == 0:
                continue
            median = np.median(patch.reshape(-1, 3)[:, ::-1], axis=0)
            worst = min(worst, _contrast(np.array(PAPER_RGB, dtype=np.float32), median))
        if worst < 4.5:
            canvas.round_rect(x - 24, y - 16, block_w + 48, (cursor - y) + 8, 16, OUTLINE, PLATE_ALPHA)
            notes.append(f"hook plate on, worst contrast {worst:.2f}:1")
        else:
            notes.append(f"hook plate off, worst contrast {worst:.2f}:1")
        overlap = _moon_face_overlap(plate, rects)
        notes.append(f"moon overlap {overlap:.1%} of the text")
        cursor = y
        for line, width in zip(lines, widths):
            canvas.text(line, FRAME_W / 2.0 - width / 2.0, cursor, "word", size, PAPER)
            cursor += ink_h + gap

    def _centered_title(self, canvas: PaperCanvas, lines: tuple[str, ...], y: float) -> None:
        size = 100.0
        while size > 72 and max(text_width("word", size, line) for line in lines) > (SAFE_R - SAFE_L):
            size -= 2
        block_w = max(text_width("word", size, line) for line in lines)
        block_h = len(lines) * (size + 8)
        x = FRAME_W / 2.0 - block_w / 2.0
        patch = bgr(canvas)[int(y) : int(y + block_h), int(x) : int(x + block_w)]
        if patch.size:
            median = np.median(patch.reshape(-1, 3)[:, ::-1], axis=0)
            if _contrast(np.array(PAPER_RGB, dtype=np.float32), median) < 4.5:
                canvas.round_rect(x - 24, y - 16, block_w + 48, block_h + 8, 16, OUTLINE, PLATE_ALPHA)
        cursor = y
        for line in lines:
            width = text_width("word", size, line)
            canvas.text(line, FRAME_W / 2.0 - width / 2.0, cursor, "word", size, PAPER)
            cursor += size + 8

    def _clear_of_moon(self, canvas: PaperCanvas, lines: tuple[str, ...], scene: Scene) -> None:
        moon = _moon_on_plate(scene)
        size = 88.0
        text = lines[0] if len(lines) == 1 else " ".join(lines)
        # One line in the dark band, left of the Moon.
        if len(lines) > 1:
            # Keep the scripted breaks, stacked on the left.
            pass
        width = max(text_width("word", size, line) for line in lines)
        x = SAFE_L
        # Above the stack label. The Moon occupies the upper right, so the words stay on the left.
        y = 78.0
        if moon is not None:
            cx, _cy, radius = moon
            x = min(x, cx - radius - width - 24)
            x = max(40.0, x)
        block_h = len(lines) * (size + 8)
        canvas.round_rect(x - 20, y - 12, width + 40, block_h + 8, 16, OUTLINE, PLATE_ALPHA)
        cursor = y
        for line in lines:
            canvas.text(line, x, cursor, "word", size, PAPER)
            cursor += size + 8

    def _bottom(self, canvas: PaperCanvas, frame: int, fold: int) -> None:
        sub = context_sub(self.script, frame, fold)
        label_w = text_width("word", 36, "FOLDS")
        number = str(fold)
        num_w = text_width("mono", 130, number)
        sub_size = 34.0
        sub_kind = "word"
        sub_w = 0.0
        if sub:
            sub_kind = "mono" if any(ch.isdigit() for ch in sub) else "word"
            while sub_size > 22 and text_width(sub_kind, sub_size, sub) > (SAFE_R - SAFE_L):
                sub_size -= 2
            sub_w = text_width(sub_kind, sub_size, sub)
        block_w = max(label_w, num_w, sub_w)
        block_h = 36 + 8 + 130 + (8 + sub_size if sub else 0)
        pad = 24.0
        x = FRAME_W / 2.0 - (block_w + pad * 2) / 2.0
        y = 1316.0
        canvas.round_rect(x, y, block_w + pad * 2, block_h + pad * 2, 16, OUTLINE, PLATE_ALPHA)
        cursor = y + pad
        canvas.text("FOLDS", FRAME_W / 2.0 - label_w / 2.0, cursor, "word", 36, "#8A93A6")
        cursor += 36 + 8
        canvas.text(number, FRAME_W / 2.0 - num_w / 2.0, cursor, "mono", 130, PAPER)
        if sub:
            cursor += 130 + 8
            canvas.text(sub, FRAME_W / 2.0 - sub_w / 2.0, cursor, sub_kind, sub_size, "#C5CDD8")


def render_heroes(hook: str = "A", out_dir: Path | None = None) -> list[StillFrame]:
    out_dir = out_dir or (ROOT / "out" / "stills")
    out_dir.mkdir(parents=True, exist_ok=True)
    drawer = DioramaStill(hook)
    frames = []
    lines = [
        "# Hero stills",
        "",
        "Stack-edge contrast is the white front face against the plate 8 px outside the 10 px outline, at 25%, 50% and 75% of the face.",
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

The stack is a 150 px oblique prism: white front, #B8C1D4 side 28 px, top face 12 px, 10 px outline, shadow 20 px to the right. There is no slate strip. Where the front face sits on cream (the street sky, the city, the atmosphere horizon), the edge contrast can fall under 3:1. Those numbers are the measured ones.

The hook card is 520 by 700, base at y = 1306, in front of the lower Moon. The title is two lines of Montserrat ExtraBold at 100 px in y 190-420. Variant B is still three lines because "REACH THE MOON?" does not fit in 14 characters.

p06 and p07 put the Earth's apex at y = 0.58. The bottom of the frame is the plate's own ground, with the last row repeated where the shift runs out. p05 is unchanged. p07's flat teal cap is near-black sky with a few stars; the Moon circle is masked so the disc is untouched.

The counter is a #1B2030 plate at 92% opacity, sized to the words. The stack label is the height. The sub-line is context (0.1 MM PAPER, or PAPER NEEDED). Fold 15 and fold 23 have no top caption. Fold 42's title sits in the dark band to the left of the Moon.

p03 is still the current city plate. Its tallest decorative tower is taller than the 828 m Burj, which the ingest warns about. A replacement plate has not arrived.
"""
