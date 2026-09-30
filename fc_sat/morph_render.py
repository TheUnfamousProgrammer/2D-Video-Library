"""Pixel Morph frames. No bloom: dark vignette, sharp squares, hook on top.

Settled pictures are a nearest-neighbor upscale of the grid, so a rest frame
matches the prepared image. In-flight squares are ``(cell + 1) * (1 + lift *
sin(pi * E))``, drawn with LINE_AA and subpixel shift 4, furthest along on top.
Squares are rasterized inside the image box, so a lifted particle cannot spill.

The return leg restores each particle to its original color (blend by E, not
by recolor_strength * E). Outbound still uses recolor_strength. At the default
of 1.0 the two readings are the same formula, and the loop always closes on A.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from fc_sat.color import oklab_to_rgb_u8, rgb_u8_to_oklab
from fc_sat.hop_render import hook_alpha
from fc_sat.morph_assign import assign
from fc_sat.morph_config import MorphConfig
from fc_sat.morph_images import load_rgb, prepare_grid, reject_identical
from fc_sat.morph_motion import (
    arc_scales,
    cell_centers,
    leg_pose,
    phase_at,
    phase_bounds,
    phase_frames,
    rank_delays,
    total_frames,
)
from fc_sat.visual import composite_rgba, find_font, raster_text, vignette_bgr

SHIFT = 4
SHIFT_SCALE = 1 << SHIFT
_WORKER: "MorphRenderer | None" = None


def nearest_block(grid_rgb: np.ndarray, cell: int) -> np.ndarray:
    """Nearest-neighbor upscale of an RGB grid to BGR blocks of ``cell`` pixels."""
    bgr = np.ascontiguousarray(grid_rgb[..., ::-1])
    return cv2.resize(bgr, (grid_rgb.shape[1] * cell, grid_rgb.shape[0] * cell), interpolation=cv2.INTER_NEAREST)


def particle_blocks(grid_rgb: np.ndarray, cell: int) -> np.ndarray:
    """The same picture drawn one cell at a time. Used to check the blit path."""
    rows, cols = grid_rgb.shape[:2]
    canvas = np.zeros((rows * cell, cols * cell, 3), dtype=np.uint8)
    for row in range(rows):
        for col in range(cols):
            color = tuple(int(v) for v in grid_rgb[row, col, ::-1])
            cv2.rectangle(
                canvas,
                (col * cell, row * cell),
                ((col + 1) * cell - 1, (row + 1) * cell - 1),
                color,
                thickness=-1,
                lineType=cv2.LINE_8,
            )
    return canvas


def _crop_ink(image: np.ndarray) -> np.ndarray:
    alpha = image[..., 3]
    rows = np.where(alpha.max(axis=1) > 0)[0]
    cols = np.where(alpha.max(axis=0) > 0)[0]
    if rows.size == 0 or cols.size == 0:
        return image
    return np.ascontiguousarray(image[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1])


def hook_placement(cfg: MorphConfig) -> tuple[int, int, int, int, np.ndarray]:
    """Return ``(x, y, w, h, rgba)`` for the hook, kept above the image box.

    The request starts at 96 px. A line that wide does not fit in x [130, 950],
    and two lines do not fit between y=200 and the image box, so the size
    steps down until the opaque text fits. It is then centered as close to
    y=330 as that gap allows.
    """
    max_w = 820
    top_limit = 200
    bottom_limit = cfg.origin_y - 8
    available_h = bottom_limit - top_limit
    font = find_font()
    for px in range(96, 11, -2):
        image = _crop_ink(raster_text(cfg.hook, font, px, max_w, 2))
        height, width = int(image.shape[0]), int(image.shape[1])
        if width <= max_w and height <= available_h:
            x = int(round(cfg.width / 2.0 - width / 2.0))
            y = int(round(330.0 - height / 2.0))
            y = min(max(y, top_limit), bottom_limit - height)
            return x, y, width, height, image
    raise SystemExit("config field 'hook' does not fit above the image box inside the safe zone")


def layout_failures(cfg: MorphConfig) -> list[str]:
    """Image box and hook must sit inside x [130, 950] and y [200, 1536]."""
    failures: list[str] = []
    box = (cfg.origin_x, cfg.origin_y, cfg.origin_x + cfg.cols * cfg.cell, cfg.origin_y + cfg.rows * cfg.cell)
    if box[0] < 130 or box[2] > 950 or box[1] < 200 or box[3] > 1536:
        failures.append(
            f"image box ({box[0]},{box[1]})-({box[2]},{box[3]}) outside safe x[130,950] y[200,1536]"
        )
    x, y, w, h, _image = hook_placement(cfg)
    if x < 130 or x + w > 950 or y < 200 or y + h > 1536:
        failures.append(f"hook box ({x},{y})-({x + w},{y + h}) outside safe x[130,950] y[200,1536]")
    if y + h > cfg.origin_y:
        failures.append(f"hook bottom {y + h} overlaps the image box at y={cfg.origin_y}")
    return failures


class MorphRenderer:
    def __init__(self, cfg: MorphConfig, path_a, path_b, *, preview: bool = False):
        path_a = Path(path_a)
        path_b = Path(path_b)
        self.cfg = cfg
        self.preview = preview
        self.scale = 0.5 if preview else 1.0
        self.width = int(round(cfg.width * self.scale))
        self.height = int(round(cfg.height * self.scale))
        self.fps = 30 if preview else cfg.fps
        self.cell_px = max(1, int(round(cfg.cell * self.scale)))
        self.one_px = max(1, int(round(self.scale)))
        self.origin_x = cfg.origin_x * self.scale
        self.origin_y = cfg.origin_y * self.scale
        failures = layout_failures(cfg)
        if failures:
            raise SystemExit("layout outside the safe zone:\n" + "\n".join(failures))
        reject_identical(path_a, path_b)
        grid_a = prepare_grid(load_rgb(path_a), cfg.cols, cfg.rows, cfg.focus_a)
        grid_b = prepare_grid(load_rgb(path_b), cfg.cols, cfg.rows, cfg.focus_b)
        if np.array_equal(grid_a, grid_b):
            raise SystemExit("image A and image B are identical after crop")
        self.grid_a = grid_a
        self.grid_b = grid_b
        self.assignment = assign(grid_a, grid_b, cfg.spatial_weight)
        self.perm = self.assignment.permutation
        n = cfg.cols * cfg.rows
        self.lab_a = rgb_u8_to_oklab(grid_a).reshape(n, 3)
        self.lab_b = rgb_u8_to_oklab(grid_b).reshape(n, 3)
        self.target_lab = self.lab_b[self.perm]
        recolor = float(cfg.recolor_strength)
        self.arrived_lab = (1.0 - recolor) * self.lab_a + recolor * self.target_lab
        self.centers_a = cell_centers(cfg.cols, cfg.rows, self.cell_px, self.origin_x, self.origin_y)
        centers_b = self.centers_a.copy()
        self.centers_b = centers_b[self.perm]
        delay_cells = rank_delays(cfg.cols, cfg.rows, cfg.delay_frac)
        self.delay_ab = delay_cells
        self.delay_ba = delay_cells[self.perm]
        self.arc = arc_scales(n, cfg.seed)
        self.plan = phase_frames(cfg.timeline(), 60 if not preview else 60)
        self.bounds = phase_bounds(cfg.timeline())
        self.duration = cfg.duration()
        if preview:
            self.n_frames = max(2, int(round(self.duration * self.fps)))
        else:
            self.n_frames = total_frames(self.plan)
        self.box_w = cfg.cols * self.cell_px
        self.box_h = cfg.rows * self.cell_px
        self.x0 = int(round(self.origin_x))
        self.y0 = int(round(self.origin_y))
        self.background = vignette_bgr(self.width, self.height)
        self.up_a = nearest_block(grid_a, self.cell_px)
        self.up_b = nearest_block(grid_b, self.cell_px)
        self.up_hold_b = self.up_b if recolor >= 1.0 - 1e-9 else self._scatter(self.arrived_lab, self.perm)
        hx, hy, hw, hh, image = hook_placement(cfg)
        self.hook_rgba = image
        self.hook_xy = (hx, hy)
        self.hook_box = (hx, hy, hx + hw, hy + hh)
        self._inset = 0.75 * self.cell_px
        self._box = (
            self.origin_x,
            self.origin_y,
            self.origin_x + self.box_w,
            self.origin_y + self.box_h,
        )

    def _scatter(self, lab: np.ndarray, cell_ids: np.ndarray) -> np.ndarray:
        rgb = oklab_to_rgb_u8(lab)
        small = np.zeros((self.cfg.rows, self.cfg.cols, 3), dtype=np.uint8)
        small[cell_ids // self.cfg.cols, cell_ids % self.cfg.cols] = rgb[:, ::-1]
        return cv2.resize(small, (self.box_w, self.box_h), interpolation=cv2.INTER_NEAREST)

    def time_of(self, index: int) -> float:
        if self.preview:
            if index >= self.n_frames - 1:
                return self.duration
            return index / self.fps
        name, _local, tau = phase_at(index, self.plan)
        start, duration = self.bounds[name]
        return start + tau * duration

    def phase_of(self, index: int) -> tuple[str, float]:
        if not self.preview:
            name, _local, tau = phase_at(index, self.plan)
            return name, tau
        t = self.time_of(index)
        names = list(self.bounds)
        for name in names:
            start, duration = self.bounds[name]
            last = name == names[-1]
            if t < start + duration - 1e-9 or last:
                n60 = dict(self.plan)[name]
                if n60 <= 1:
                    return name, 0.0
                tau = min(1.0, max(0.0, ((t - start) * 60.0) / (n60 - 1)))
                return name, tau
        return names[-1], 1.0

    def frame_index(self, phase: str, tau: float) -> int:
        """Frame whose phase progress is nearest ``tau`` (full-rate plan)."""
        cursor = 0
        for name, count in self.plan:
            if name == phase:
                local = 0 if count <= 1 else int(round(float(tau) * (count - 1)))
                local = min(count - 1, max(0, local))
                if self.preview:
                    start, duration = self.bounds[name]
                    t = start + (local / max(count - 1, 1)) * duration
                    return min(self.n_frames - 1, int(round(t * self.fps)))
                return cursor + local
            cursor += count
        return 0

    def pose(self, index: int):
        name, tau = self.phase_of(index)
        return name, self._pose(name, tau)

    def _pose(self, name: str, tau: float):
        if name in ("hold_a_start", "hold_a_end"):
            return leg_pose(
                0.0, self.centers_a, self.centers_b, self.lab_a, self.target_lab, self.delay_ab, self.arc,
                delay_frac=self.cfg.delay_frac, arc_amp=self.cfg.arc_amp, recolor_strength=0.0,
                box=self._box, inset=self._inset,
            )
        if name == "hold_b":
            return leg_pose(
                1.0, self.centers_a, self.centers_b, self.lab_a, self.target_lab, self.delay_ab, self.arc,
                delay_frac=self.cfg.delay_frac, arc_amp=self.cfg.arc_amp,
                recolor_strength=self.cfg.recolor_strength, box=self._box, inset=self._inset,
            )
        if name == "morph_ab":
            return leg_pose(
                tau, self.centers_a, self.centers_b, self.lab_a, self.target_lab, self.delay_ab, self.arc,
                delay_frac=self.cfg.delay_frac, arc_amp=self.cfg.arc_amp,
                recolor_strength=self.cfg.recolor_strength, box=self._box, inset=self._inset,
            )
        return leg_pose(
            tau, self.centers_b, self.centers_a, self.arrived_lab, self.lab_a, self.delay_ba, -self.arc,
            delay_frac=self.cfg.delay_frac, arc_amp=self.cfg.arc_amp, recolor_strength=1.0,
            box=self._box, inset=self._inset,
        )

    def heaviest_index(self) -> int:
        """Frame with the most particles in flight. Ties keep the later frame."""
        best = 0
        best_count = -1
        if self.preview:
            candidates = range(0, self.n_frames, max(1, self.n_frames // 24))
        else:
            candidates = []
            cursor = 0
            for name, count in self.plan:
                if name.startswith("morph"):
                    step = max(1, count // 12)
                    candidates.extend(range(cursor, cursor + count, step))
                cursor += count
        for index in candidates:
            _name, pose = self.pose(index)
            flying = int(np.count_nonzero((pose.progress > 0.0) & (pose.progress < 1.0)))
            if flying >= best_count:
                best_count = flying
                best = int(index)
        return best

    def paint_roi(self, index: int) -> np.ndarray:
        name, tau = self.phase_of(index)
        if name in ("hold_a_start", "hold_a_end"):
            return self.up_a.copy()
        if name == "hold_b":
            return self.up_hold_b.copy()
        pose = self._pose(name, tau)
        if np.all(pose.progress <= 1e-8):
            return self.up_a.copy() if name == "morph_ab" else self.up_hold_b.copy()
        if np.all(pose.progress >= 1.0 - 1e-8):
            return self.up_hold_b.copy() if name == "morph_ab" else self.up_a.copy()
        y0, x0 = self.y0, self.x0
        roi = self.background[y0 : y0 + self.box_h, x0 : x0 + self.box_w].copy()
        self._paint_settled(roi, pose, name)
        self._paint_flight(roi, pose)
        return roi

    def _paint_settled(self, roi: np.ndarray, pose, name: str) -> None:
        at_start = np.flatnonzero(pose.progress <= 1e-8)
        at_end = np.flatnonzero(pose.progress >= 1.0 - 1e-8)
        if at_start.size == 0 and at_end.size == 0:
            return
        bgr = oklab_to_rgb_u8(pose.lab)[:, ::-1]
        if name == "morph_ab":
            start_ids = np.arange(self.perm.size, dtype=np.int32)
            end_ids = self.perm
        else:
            start_ids = self.perm
            end_ids = np.arange(self.perm.size, dtype=np.int32)
        small = np.zeros((self.cfg.rows, self.cfg.cols, 3), dtype=np.uint8)
        mask = np.zeros((self.cfg.rows, self.cfg.cols), dtype=np.uint8)
        cols = self.cfg.cols
        if at_start.size:
            cells = start_ids[at_start]
            small[cells // cols, cells % cols] = bgr[at_start]
            mask[cells // cols, cells % cols] = 1
        if at_end.size:
            cells = end_ids[at_end]
            small[cells // cols, cells % cols] = bgr[at_end]
            mask[cells // cols, cells % cols] = 1
        up = cv2.resize(small, (self.box_w, self.box_h), interpolation=cv2.INTER_NEAREST)
        up_mask = cv2.resize(mask, (self.box_w, self.box_h), interpolation=cv2.INTER_NEAREST) > 0
        roi[up_mask] = up[up_mask]

    def _paint_flight(self, roi: np.ndarray, pose) -> None:
        flying = np.flatnonzero((pose.progress > 1e-8) & (pose.progress < 1.0 - 1e-8))
        if flying.size == 0:
            return
        order = flying[np.argsort(pose.ease[flying], kind="mergesort")]
        colors = oklab_to_rgb_u8(pose.lab[order])[:, ::-1]
        base = float(self.cell_px + self.one_px)
        lift = float(self.cfg.lift)
        ox = self.origin_x
        oy = self.origin_y
        for k, index in enumerate(order):
            ease = float(pose.ease[index])
            size = base * (1.0 + lift * math.sin(math.pi * ease))
            color = (int(colors[k, 0]), int(colors[k, 1]), int(colors[k, 2]))
            _fill_square(roi, float(pose.position[index, 0]) - ox, float(pose.position[index, 1]) - oy, size, color)

    def particle_layer(self, index: int) -> np.ndarray:
        """Particles only. Pixels outside the image box stay 0."""
        layer = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        layer[self.y0 : self.y0 + self.box_h, self.x0 : self.x0 + self.box_w] = self.paint_roi(index)
        return layer

    def render(self, index: int) -> np.ndarray:
        frame = self.background.copy()
        frame[self.y0 : self.y0 + self.box_h, self.x0 : self.x0 + self.box_w] = self.paint_roi(index)
        self._draw_hook(frame, self.time_of(index))
        return frame

    def _draw_hook(self, frame: np.ndarray, t: float) -> None:
        alpha = hook_alpha(t, self.duration)
        if alpha <= 0:
            return
        image = self.hook_rgba
        if alpha < 1:
            image = image.copy()
            image[..., 3] = np.clip(image[..., 3].astype(np.float32) * alpha, 0, 255).astype(np.uint8)
        x = int(round(self.hook_xy[0] * self.scale))
        y = int(round(self.hook_xy[1] * self.scale))
        if self.preview:
            preview = cv2.resize(image, (max(1, image.shape[1] // 2), max(1, image.shape[0] // 2)), interpolation=cv2.INTER_AREA)
            composite_rgba(frame, preview, x, y)
        else:
            composite_rgba(frame, image, x, y)


def _fill_square(image: np.ndarray, cx: float, cy: float, size: float, color: tuple[int, int, int]) -> None:
    half = size * 0.5
    x0 = int(round((cx - half) * SHIFT_SCALE))
    y0 = int(round((cy - half) * SHIFT_SCALE))
    x1 = int(round((cx + half) * SHIFT_SCALE))
    y1 = int(round((cy + half) * SHIFT_SCALE))
    cv2.rectangle(image, (x0, y0), (x1, y1), color, thickness=-1, lineType=cv2.LINE_AA, shift=SHIFT)


def init_morph_worker(cfg: MorphConfig, path_a: str, path_b: str, preview: bool) -> None:
    global _WORKER
    cv2.setNumThreads(1)
    _WORKER = MorphRenderer(cfg, path_a, path_b, preview=preview)


def render_morph_chunk(indices: list[int]) -> list[np.ndarray]:
    if _WORKER is None:
        raise RuntimeError("morph worker was not initialized")
    return [_WORKER.render(index) for index in indices]


def contact_sheet(renderer: MorphRenderer, path) -> None:
    """Two rows (A to B, then B to A) at 0, 25, 50, 75, and 100 percent."""
    frames = []
    for phase in ("morph_ab", "morph_ba"):
        for tau in (0.0, 0.25, 0.5, 0.75, 1.0):
            frames.append(renderer.render(renderer.frame_index(phase, tau)))
    cell_w, cell_h = 270, 480
    sheet = np.zeros((cell_h * 2, cell_w * 5, 3), dtype=np.uint8)
    for index, frame in enumerate(frames):
        small = cv2.resize(frame, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
        row, col = divmod(index, 5)
        sheet[row * cell_h : (row + 1) * cell_h, col * cell_w : (col + 1) * cell_w] = small
    cv2.imwrite(str(path), sheet)
