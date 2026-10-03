"""Hero stills for the paper diorama.

Plates are the ingested PNGs. This module never rewrites them. A 240 px slate
strip sits behind the stack on the low-contrast plates, and on the atmosphere
plate because its cream horizon fails the same test. The opening card is about
700 px wide, so that still widens the same slate past the card. Tower stills
keep the 240 px strip.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from fc_sat.paperfold_art import ART_ROOT, _relative_luma
from fc_sat.paperfold_draw import PaperCanvas, bgr, text_width
from fc_sat.paperfold_math import (
    BURJ_M,
    KARMAN_M,
    MILESTONES,
    MOON_M,
    MUG_M,
    PERSON_M,
    PHONE_M,
    height_m,
    milestone_fold,
)
from fc_sat.paperfold_render import folds_done, height_label, sub_line
from fc_sat.paperfold_scenes import (
    FRAME_H,
    FRAME_W,
    GROUND,
    HANDOFF_FRAMES,
    LIMIT,
    Scene,
    break_fold,
    handoff_start_frame,
    px_per_m,
    scenes_from_manifest,
    stack_top_y,
)
from fc_sat.paperfold_schedule import fold_frame
from fc_sat.paperfold_text import Script, load_script, top_lines

ROOT = Path(__file__).resolve().parents[1]
PANEL_Y = int(round(GROUND * FRAME_H))
BACKING = "#2B3A4D"
OUTLINE = "#1B2030"
PAPER = "#F2F4F8"
PAPER_RGB = (242, 244, 248)
SEAM_RGB = (228, 232, 240)
SHADOW = "#0E1117"
STACK_W = 120.0
OUTLINE_PX = 12.0
BACKING_W = 240.0
# 4% of the frame above the y = 0.30 limit, so slate shows above a maxed stack.
BACKING_TOP = LIMIT - 0.04
CARD_W = 700.0
CARD_H = 640.0
HOOK_PROGRESS = 0.35
SAFE_L = 130.0
SAFE_R = 950.0

# p01-p04 are the brief's low-contrast plates. p05's center lane clears 3:1,
# but the cream horizon beside the lower stack does not, so it gets the same strip.
BACKING_PLATES = frozenset(
    {
        "p01_hook_night_desk_moon",
        "p02_street_day",
        "p03_city_day",
        "p04_mountains_clouds",
        "p05_upper_atmosphere",
    }
)

HEROES = (
    ("hook", 0),
    ("fold_15", 15),
    ("fold_23", 23),
    ("fold_30", 30),
    ("fold_42", 42),
)

# (object id, x as a fraction of the frame, height in metres)
SCENE_OBJECTS: dict[str, tuple[tuple[str, float, float], ...]] = {
    "desk": (("o02_mug", 0.24, MUG_M), ("o03_smartphone", 0.76, PHONE_M)),
    "street": (("o01_person", 0.24, PERSON_M),),
    "city": (("o04_burj_khalifa", 0.76, BURJ_M),),
}


@dataclass(frozen=True)
class PaperBox:
    """White fill, not including the outline."""

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


def backing_box(frame_w: int = FRAME_W, frame_h: int = FRAME_H) -> tuple[float, float, float, float]:
    """Slate strip: 240 px, centered, from the ground line up to 4% above the limit."""
    x = frame_w / 2.0 - BACKING_W / 2.0
    y = BACKING_TOP * frame_h
    ground = GROUND * frame_h
    return x, y, BACKING_W, ground - y


def scene_at(frame: int, rows: tuple[Scene, ...] | None = None) -> Scene:
    """Settled plate at this frame. During a hand-off the incoming plate is used after 60%."""
    rows = rows or scenes_from_manifest()
    current = rows[0]
    for cur, nxt in zip(rows, rows[1:]):
        start = handoff_start_frame(break_fold(cur.world_m))
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
    """White paper against the local background just outside the outline, at three heights.

    Samples sit 6 px inside the fill and 6 px outside the outline, so the 12 px
    stroke itself is not the background and not the paper.
    """
    height, width = image_bgr.shape[:2]
    ratios = []
    for frac in (0.25, 0.50, 0.75):
        sy = int(round(paper.y + paper.h * frac))
        sy = min(height - 3, max(2, sy))
        inside_x = int(round(paper.x + 6))
        outside_x = int(round(paper.x - outline - 6))
        inside_x = min(width - 3, max(2, inside_x))
        outside_x = min(width - 3, max(2, outside_x))
        paper_rgb = np.median(image_bgr[sy - 2 : sy + 3, inside_x - 2 : inside_x + 3, ::-1], axis=(0, 1))
        back_rgb = np.median(image_bgr[sy - 2 : sy + 3, outside_x - 2 : outside_x + 3, ::-1], axis=(0, 1))
        ratios.append(_contrast(paper_rgb, back_rgb))
    return tuple(ratios)


def _paper_texture(width: int, height: int, folds: int, seed: int) -> np.ndarray:
    """Flat paper, a little grain, and up to six doubled bands. RGBA."""
    rng = np.random.default_rng(seed)
    bands = min(6, max(1, folds))
    image = np.empty((height, width, 4), np.float32)
    image[:, :, 3] = 255
    for index in range(bands):
        y0 = int(round(index * height / bands))
        y1 = int(round((index + 1) * height / bands))
        tone = PAPER_RGB if index % 2 == 0 else SEAM_RGB
        image[y0:y1, :, 0] = tone[0]
        image[y0:y1, :, 1] = tone[1]
        image[y0:y1, :, 2] = tone[2]
    noise = rng.normal(0.0, 2.5, size=(height, width, 1))
    image[:, :, :3] = np.clip(image[:, :, :3] + noise, 0, 255)
    return image.astype(np.uint8)


def _load_plate(plate_id: str) -> np.ndarray:
    path = ART_ROOT / "normalized" / f"{plate_id}.png"
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"missing normalized plate {path}. Run ingest first.")
    if image.shape[0] != FRAME_H or image.shape[1] != FRAME_W:
        raise SystemExit(f"{path} is {image.shape[1]}x{image.shape[0]}, expected {FRAME_W}x{FRAME_H}")
    return image


def _scaled_cutout(object_id: str, height_px: float) -> np.ndarray:
    path = ART_ROOT / "keyed" / f"{object_id}.png"
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim != 3 or image.shape[2] != 4:
        raise SystemExit(f"missing keyed cutout {path}")
    alpha = image[:, :, 3]
    ys, xs = np.nonzero(alpha > 16)
    if len(ys) == 0:
        raise SystemExit(f"{object_id} has no opaque pixels")
    crop = image[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    dest_h = max(1, int(round(height_px)))
    dest_w = max(1, int(round(crop.shape[1] * (dest_h / crop.shape[0]))))
    scaled = cv2.resize(crop.astype(np.float32), (dest_w, dest_h), interpolation=cv2.INTER_LANCZOS4)
    rgb = np.clip(scaled[:, :, :3], 0, 255).astype(np.uint8)
    # cv2 is BGRA. skia wants RGBA.
    return np.dstack([rgb[:, :, 2], rgb[:, :, 1], rgb[:, :, 0], np.clip(scaled[:, :, 3], 0, 255).astype(np.uint8)])


def _stamp(fold: int) -> str:
    for key, _distance, label in MILESTONES:
        if milestone_fold(key) == fold:
            return f"= {label}"
    return ""


def _fit_size(kind: str, text: str, max_width: float, hi: float, lo: float) -> float:
    size = hi
    while size > lo and text_width(kind, size, text) > max_width:
        size -= 2
    return size


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
        if scene.plate in BACKING_PLATES and frame >= 84:
            self._backing(canvas)
            notes.append("slate strip 240 px")
        elif scene.plate in BACKING_PLATES and frame < 84:
            self._card_mat(canvas)
            notes.append("slate mat behind the 700 px card")
        scale = px_per_m(scene.world_m)
        self._objects(canvas, scene, scale, notes)
        if frame < 84:
            paper = self._card(canvas)
        else:
            paper = self._stack(canvas, fold, scale)
            self._altitude(canvas, scene, scale, fold)
        self._stamp(canvas, frame, fold, scene)
        canvas.rect(0, PANEL_Y, FRAME_W, FRAME_H - PANEL_Y, OUTLINE)
        self._top(canvas, frame)
        self._bottom(canvas, frame, fold)
        image = bgr(canvas)
        ratios = edge_contrast(image, paper)
        name = "hook" if frame == 0 else f"fold_{fold}"
        return StillFrame(name, frame, fold, scene.plate, scene.name, image, paper, ratios, tuple(notes))

    def _backing(self, canvas: PaperCanvas) -> None:
        x, y, w, h = backing_box()
        canvas.soft_rect(x, y, w, h, SHADOW, 0.35, 10, dx=10, dy=14)
        canvas.rect(x, y, w, h, BACKING)

    def _card_mat(self, canvas: PaperCanvas) -> None:
        """Slate behind the opening card, 40 px past each side, same vertical span as the strip.

        A 240 px strip sits entirely behind a 700 px card, so the card edge would
        still land on the moon. The mat uses the same cardstock and shadow.
        """
        _x, y, _w, h = backing_box()
        w = CARD_W + 80.0
        x = FRAME_W / 2.0 - w / 2.0
        canvas.soft_rect(x, y, w, h, SHADOW, 0.35, 10, dx=10, dy=14)
        canvas.rect(x, y, w, h, BACKING)

    def _objects(self, canvas: PaperCanvas, scene: Scene, scale: float, notes: list[str]) -> None:
        ground = GROUND * FRAME_H
        for object_id, x_frac, meters in SCENE_OBJECTS.get(scene.name, ()):
            height_px = meters * scale
            cutout = _scaled_cutout(object_id, height_px)
            left = FRAME_W * x_frac - cutout.shape[1] / 2.0
            top = ground - cutout.shape[0]
            canvas.image(cutout, left, top)
            drawn = cutout.shape[0]
            error = abs(drawn - height_px) / height_px if height_px else 0.0
            notes.append(f"{object_id} {meters:g} m -> {drawn} px (scale error {error:.2%})")

    def _stack(self, canvas: PaperCanvas, fold: int, scale: float) -> PaperBox:
        meters = height_m(fold)
        height_px = max(8.0, meters * scale)
        ground = GROUND * FRAME_H
        top = ground - height_px
        x = FRAME_W / 2.0 - STACK_W / 2.0
        canvas.soft_rect(x - 10, PANEL_Y - 20, STACK_W + 20, 16, SHADOW, 0.35, 10)
        canvas.rect(x - OUTLINE_PX, top - OUTLINE_PX, STACK_W + OUTLINE_PX * 2, height_px + OUTLINE_PX, OUTLINE)
        texture = _paper_texture(int(STACK_W), max(1, int(round(height_px))), fold, seed=7)
        canvas.image(texture, x, top)
        label = height_label(fold)
        self._chip(canvas, label, FRAME_W / 2.0, top - 56, "mono")
        return PaperBox(x, top, STACK_W, height_px)

    def _card(self, canvas: PaperCanvas) -> PaperBox:
        ground = GROUND * FRAME_H
        top = ground - CARD_H
        half = CARD_W / 2.0
        left = FRAME_W / 2.0 - half
        angle = HOOK_PROGRESS * math.pi
        right_w = half * abs(math.cos(angle))
        canvas.soft_rect(left, ground - 12, CARD_W, 20, SHADOW, 0.35, 10)
        canvas.rect(left - OUTLINE_PX, top - OUTLINE_PX, half + OUTLINE_PX, CARD_H + OUTLINE_PX, OUTLINE)
        canvas.rect(FRAME_W / 2.0, top - OUTLINE_PX, right_w + OUTLINE_PX, CARD_H + OUTLINE_PX, OUTLINE)
        left_tex = _paper_texture(int(half), int(CARD_H), folds=1, seed=7)
        canvas.image(left_tex, left, top)
        right_tex = _paper_texture(max(1, int(round(right_w))), int(CARD_H), folds=1, seed=11)
        # The lifted half is still the front face, a little darker so the turn reads.
        right_tex = right_tex.astype(np.float32)
        right_tex[:, :, :3] *= 0.86
        canvas.image(np.clip(right_tex, 0, 255).astype(np.uint8), FRAME_W / 2.0, top)
        return PaperBox(left, top, half, CARD_H)

    def _altitude(self, canvas: PaperCanvas, scene: Scene, scale: float, fold: int) -> None:
        ground = GROUND * FRAME_H
        if scene.name == "atmosphere":
            y = ground - KARMAN_M * scale
            self._dashed(canvas, y, STACK_W)
            self._chip(canvas, "SPACE", SAFE_L + 80, y - 46, "word")
        if scene.name == "deep_space":
            y = ground - MOON_M * scale
            self._dashed(canvas, y, STACK_W)

    def _dashed(self, canvas: PaperCanvas, y: float, gap_w: float) -> None:
        gap_l = FRAME_W / 2.0 - gap_w / 2.0 - OUTLINE_PX - 8
        gap_r = FRAME_W / 2.0 + gap_w / 2.0 + OUTLINE_PX + 8
        for x0, x1 in ((SAFE_L, gap_l), (gap_r, SAFE_R)):
            x = x0
            while x < x1:
                canvas.rect(x, y - 3, min(16.0, x1 - x), 6, "#F2F4F8")
                x += 28

    def _stamp(self, canvas: PaperCanvas, frame: int, fold: int, scene: Scene) -> None:
        # The top slot already says THE MOON at fold 42, and SPACE is on the Karman line.
        if fold in (30, 42):
            return
        text = _stamp(fold)
        if not text or frame != fold_frame(fold):
            return
        if scene.name == "street":
            head = GROUND * FRAME_H - PERSON_M * px_per_m(scene.world_m)
            self._chip(canvas, text, 250, head - 64, "word")
        elif scene.name == "city":
            # Beside the tower, below the height chip so the two labels do not join.
            tip = GROUND * FRAME_H - BURJ_M * px_per_m(scene.world_m)
            self._chip(canvas, text, 900, tip + 150, "word")

    def _chip(self, canvas: PaperCanvas, text: str, cx: float, top: float, kind: str) -> None:
        size = _fit_size(kind, text, 420, 36, 22)
        width = text_width(kind, size, text)
        x = min(SAFE_R - width - 12, max(SAFE_L, cx - width / 2.0))
        canvas.rect(x - 12, top - 6, width + 24, size + 16, OUTLINE)
        canvas.text(text, x, top, kind, size, PAPER)

    def _top(self, canvas: PaperCanvas, frame: int) -> None:
        lines = top_lines(self.script, frame, self.hook)
        if not lines:
            return
        size = 110.0
        while size > 72 and max(text_width("mono" if any(ch.isdigit() for ch in line) else "word", size, line) for line in lines) > (SAFE_R - SAFE_L):
            size -= 2
        block_h = len(lines) * (size + 8)
        block_w = max(text_width("mono" if any(ch.isdigit() for ch in line) else "word", size, line) for line in lines)
        x = FRAME_W / 2.0 - block_w / 2.0
        y = 220.0
        # Flat plate when the art behind the words is bright.
        patch = bgr(canvas)[int(y) : int(y + block_h), int(x) : int(x + block_w)]
        if patch.size:
            median = np.median(patch.reshape(-1, 3)[:, ::-1], axis=0)
            if _contrast(np.array(PAPER_RGB, dtype=np.float32), median) < 3.0:
                canvas.rect(x - 24, y - 16, block_w + 48, block_h + 16, "#0E1117", 0.55)
        cursor = y
        for line in lines:
            kind = "mono" if any(ch.isdigit() for ch in line) else "word"
            width = text_width(kind, size, line)
            canvas.text(line, FRAME_W / 2.0 - width / 2.0, cursor, kind, size, PAPER)
            cursor += size + 8

    def _bottom(self, canvas: PaperCanvas, frame: int, fold: int) -> None:
        label = "FOLDS"
        label_w = text_width("word", 36, label)
        canvas.text(label, FRAME_W / 2.0 - label_w / 2.0, 1340, "word", 36, "#8A93A6")
        number = str(fold)
        num_w = text_width("mono", 130, number)
        canvas.text(number, FRAME_W / 2.0 - num_w / 2.0, 1380, "mono", 130, PAPER)
        sub = sub_line(self.script, frame, fold)
        if sub:
            kind = "mono" if any(ch.isdigit() for ch in sub) else "word"
            sub_w = text_width(kind, 34, sub)
            canvas.text(sub, FRAME_W / 2.0 - sub_w / 2.0, 1496, kind, 34, "#8A93A6")


CRITIQUE = """## Critique

All five stills clear 3:1 at the three edge samples. The samples are the white paper against the colour 6 px outside the 12 px outline.

The hook card is about 700 px wide, so a 240 px strip would sit entirely behind it and the card edge would still land on the moon. The opening still uses the same slate and the same shadow, widened to 780 px (40 px of slate past each side). The moon stays visible above that mat. The mug (0.095 m, 138 px) is hidden behind the card. The phone (0.15 m, 218 px) stands to the right at desk scale. The card itself is not true scale. A real sheet is 0.1 mm, and the pose is fold progress 0.35: the right half is foreshortened and darker.

Fold 15: the white stack is 3.3 m and about twice the person (168 px against 306 px). The slate column is much taller than that stack, because the strip runs up to the limit line. Compare the person to the white paper, not to the slate. The illustrated buildings are not to scale. The cream sky is the busy region, and the strip covers the center of it.

Fold 23: the stack is 838.9 m and the Burj cutout is 828 m, drawn at 601 px. They read as the same height, with the stack a little taller. The plate's cream tower is still visible beside the strip. The small red patch on the Burj is in the source art. The stack is 120 px wide on purpose. That width is not the building's width.

Fold 30: the atmosphere plate's center lane was already above 3:1, but the cream horizon beside the lower stack was about 1.2:1. The same 240 px strip is drawn there so the three edge samples pass. The stack top sits just above the 100 km line. The layered clouds are busy; the strip keeps the paper readable.

Fold 42: no strip. The background is the star field, and the edge samples are about 16:1. The stack is 439,805 km, past the dashed line at the Moon's center (384,400 km). The title crosses the plate's Moon at about 4.8:1. The bottom line is the paper length, 107,000 light-years, not the height.

p04 (mountains) uses the same 240 px strip in the compositor. None of these five frames lands on it.

Stills are the settled plate at zoom 1.0. There is no mid-scene drift.
"""


def render_heroes(hook: str = "A", out_dir: Path | None = None) -> list[StillFrame]:
    out_dir = out_dir or (ROOT / "out" / "stills")
    out_dir.mkdir(parents=True, exist_ok=True)
    drawer = DioramaStill(hook)
    frames = []
    lines = ["# Hero stills", "", "Stack-edge contrast is white paper against the local background 6 px outside the 12 px outline, at 25%, 50% and 75% of the paper height.", ""]
    for name, fold in HEROES:
        frame = 0 if fold == 0 else fold_frame(fold)
        still = drawer.render(frame)
        path = out_dir / f"{name}.png"
        cv2.imwrite(str(path), still.image)
        ratios = still.ratios
        flag = " PASS" if min(ratios) >= 3.0 else " FAIL"
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
    report = ROOT / "out" / "stills_contrast.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n\n" + CRITIQUE)
    print(f"wrote {report}")
    return frames
