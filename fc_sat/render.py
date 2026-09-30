"""Pure frame renderer. A frame depends only on sim state, the frame index, and config.

The bottom 20% and the right 12% of the frame are never written after the vignette
is copied in, so bloom cannot spill into the pixel safe zone. The ring stroke box
uses the centered thickness: outer radius = radius + thickness/2. At the defaults
that is 383 px, x from 157 to 923.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from fc_sat.color import palette_bgr
from fc_sat.config import Config
from fc_sat.sim import SimResult, ball_radius
from fc_sat.visual import apply_bloom, composite_rgba, find_font, raster_text, vignette_bgr

SHIFT = 4
SHIFT_SCALE = 1 << SHIFT


def safe_rect(cfg: Config) -> tuple[float, float, float, float]:
    """x0, x1, y0, y1 in pixels. Defaults are [130, 950] x [200, 1536]."""
    sx = cfg.width / 1080.0
    sy = cfg.height / 1920.0
    return (130.0 * sx, 950.0 * sx, 200.0 * sy, 1536.0 * sy)


def ring_layout_box(cfg: Config) -> tuple[float, float, float, float]:
    outer = cfg.ring_radius + cfg.ring_thickness / 2.0
    return (
        cfg.ring_cx - outer,
        cfg.ring_cy - outer,
        cfg.ring_cx + outer,
        cfg.ring_cy + outer,
    )


def ease_in_out_cubic(u: float) -> float:
    u = min(1.0, max(0.0, u))
    if u < 0.5:
        return 4.0 * u * u * u
    return 1.0 - ((-2.0 * u + 2.0) ** 3) / 2.0


def ease_out_cubic(u: float) -> float:
    u = min(1.0, max(0.0, u))
    return 1.0 - (1.0 - u) ** 3


def _pop_scale(age: float, amplitude: float, duration: float = 0.12) -> float:
    if age < 0.0 or age >= duration:
        return 1.0
    u = age / duration
    if u < 0.35:
        w = u / 0.35
        w = 1.0 - (1.0 - w) ** 2
        return 1.0 + amplitude * w
    w = (u - 0.35) / 0.65
    return 1.0 + amplitude * (1.0 - w * w)


def _centered_box(cx: float, cy: float, width: float, height: float) -> tuple[float, float, float, float]:
    return (cx - width / 2.0, cy - height / 2.0, cx + width / 2.0, cy + height / 2.0)


def layout_boxes(cfg: Config) -> dict[str, tuple[float, float, float, float]]:
    """Full-resolution boxes for the safe-zone math check, including pop and shadow."""
    font_path = find_font()
    max_width = int(safe_rect(cfg)[1] - safe_rect(cfg)[0] - 40)
    hook = raster_text(cfg.hook, font_path, 96, max_width, 2)
    boxes = {
        "hook": _centered_box(cfg.width / 2.0, 330.0, hook.shape[1], hook.shape[0]),
        "ring": ring_layout_box(cfg),
    }
    counter = raster_text(f"BALLS: {cfg.cap}", font_path, 72, max_width, 1)
    boxes["counter"] = _centered_box(
        cfg.width / 2.0,
        1400.0,
        counter.shape[1] * 1.28,
        counter.shape[0] * 1.28,
    )
    if cfg.show_wait_text:
        wait = raster_text(cfg.wait_text, font_path, 54, max_width, 1)
        boxes["wait"] = _centered_box(cfg.width / 2.0, 700.0, wait.shape[1], wait.shape[0])
    if cfg.show_cta:
        cta = raster_text(cfg.cta_text, font_path, 64, max_width, 2)
        boxes["cta"] = _centered_box(cfg.width / 2.0, 1180.0, cta.shape[1], cta.shape[0])
    return boxes


def assert_layout_safe(cfg: Config) -> list[str]:
    x0, x1, y0, y1 = safe_rect(cfg)
    failures = []
    for name, box in layout_boxes(cfg).items():
        if box[0] < x0 - 1e-3 or box[2] > x1 + 1e-3 or box[1] < y0 - 1e-3 or box[3] > y1 + 1e-3:
            failures.append(
                f"{name} box ({box[0]:.1f},{box[1]:.1f})-({box[2]:.1f},{box[3]:.1f}) "
                f"outside safe ({x0:.1f},{y0:.1f})-({x1:.1f},{y1:.1f})"
            )
    return failures


def _blit_circle(
    signal: np.ndarray,
    x: float,
    y: float,
    radius: float,
    color: tuple[int, int, int],
    thickness: int = -1,
) -> None:
    if radius <= 0.2:
        return
    pad = int(math.ceil(radius)) + thickness + 4 if thickness > 0 else int(math.ceil(radius)) + 4
    pad = max(pad, 2)
    patch = np.zeros((pad * 2 + 1, pad * 2 + 1, 3), dtype=np.uint8)
    fx = x - math.floor(x)
    fy = y - math.floor(y)
    center = (
        int(round((pad + fx) * SHIFT_SCALE)),
        int(round((pad + fy) * SHIFT_SCALE)),
    )
    rad = max(1, int(round(radius * SHIFT_SCALE)))
    thick = thickness if thickness < 0 else max(1, int(round(thickness)))
    cv2.circle(patch, center, rad, color, thick, lineType=cv2.LINE_AA, shift=SHIFT)
    left = int(math.floor(x)) - pad
    top = int(math.floor(y)) - pad
    src_x0 = 0
    src_y0 = 0
    if left < 0:
        src_x0 = -left
        left = 0
    if top < 0:
        src_y0 = -top
        top = 0
    right = min(signal.shape[1], left + patch.shape[1] - src_x0)
    bottom = min(signal.shape[0], top + patch.shape[0] - src_y0)
    if right <= left or bottom <= top:
        return
    src = patch[src_y0 : src_y0 + (bottom - top), src_x0 : src_x0 + (right - left)]
    signal[top:bottom, left:right] += src.astype(np.float32)


class Renderer:
    def __init__(self, cfg: Config, sim: SimResult, preview: bool = False) -> None:
        self.cfg = cfg
        self.sim = sim
        self.preview = preview
        self.scale = 0.5 if preview else 1.0
        self.fps, self.plan = cfg.frame_plan(preview=preview)
        self.n_frames = sum(n for _name, n in self.plan)
        self.width = int(round(cfg.width * self.scale))
        self.height = int(round(cfg.height * self.scale))
        sx0, sx1, sy0, sy1 = safe_rect(cfg)
        self.roi = (
            int(round(sx0 * self.scale)),
            int(round(sy0 * self.scale)),
            int(round(sx1 * self.scale)),
            int(round(sy1 * self.scale)),
        )
        self.background = vignette_bgr(self.width, self.height)
        self.font_path = find_font()
        self.colors = palette_bgr(cfg.palette_stops, max(cfg.cap, 1))
        self._text_cache: dict[tuple, np.ndarray] = {}
        self._phases = self._build_phases()
        self.glow = self._build_glow()
        self.pop = self._build_pop()
        full_plan = dict(cfg.frame_plan(preview=False)[1])
        self.full_growth_frames = full_plan["growth"]

    def _build_phases(self) -> list[tuple[str, int, int, float, float]]:
        seconds = dict(self.cfg.phase_durations())
        rows = []
        cursor = 0.0
        for name, n in self.plan:
            for local in range(n):
                progress = 0.0 if n == 1 else local / (n - 1)
                rows.append((name, local, n, progress, cursor + local / self.fps))
            cursor += seconds[name]
        return rows

    def _build_glow(self) -> np.ndarray:
        energy = np.zeros(self.n_frames, dtype=np.float64)
        if self.sim.events.size:
            times = self.sim.events["time"]
            speeds = self.sim.events["impact_speed"]
            frames = np.floor(times * self.fps).astype(np.int32)
            valid = (frames >= 0) & (frames < self.n_frames)
            np.add.at(energy, frames[valid], np.clip(speeds[valid] / self.cfg.max_speed, 0.0, 1.5))
        decay = math.exp(-(1.0 / self.fps) / 0.120)
        glow = np.zeros(self.n_frames, dtype=np.float64)
        level = 0.0
        for index in range(self.n_frames):
            level = level * decay + float(energy[index])
            glow[index] = min(level, 2.5)
        return glow

    def _build_pop(self) -> np.ndarray:
        counts = self.sim.counts
        milestones = set(int(row[1]) for row in np.atleast_2d(self.sim.milestones) if row.size >= 2)
        pop = np.ones(len(counts), dtype=np.float64)
        previous = int(counts[0]) if len(counts) else 1
        last_change = -10.0
        full_fps = self.cfg.fps
        for index, count in enumerate(counts):
            count = int(count)
            t = index / full_fps
            if count != previous:
                last_change = t
                previous = count
            amplitude = 0.28 if count in milestones else 0.12
            pop[index] = _pop_scale(t - last_change, amplitude)
        return pop

    def _label(self, text: str, px: int, max_lines: int) -> np.ndarray:
        key = (text, px, max_lines, self.width)
        cached = self._text_cache.get(key)
        if cached is None:
            max_width = max(40, self.roi[2] - self.roi[0] - int(20 * self.scale))
            cached = raster_text(text, self.font_path, px, max_width, max_lines)
            self._text_cache[key] = cached
        return cached

    def _growth_index(self, local: int) -> int:
        if not self.preview:
            return min(local, self.full_growth_frames - 1)
        mapped = int(round(local * (self.cfg.fps / self.fps)))
        return min(self.full_growth_frames - 1, max(0, mapped))

    def render(self, frame_index: int) -> np.ndarray:
        phase, local, n_phase, progress, t = self._phases[frame_index]
        frame = self.background.copy()
        x0, y0, x1, y1 = self.roi
        signal = np.zeros((y1 - y0, x1 - x0, 3), dtype=np.float32)
        self._draw_world(signal, x0, y0, phase, local, progress)
        self._bloom(signal)
        frame[y0:y1, x0:x1] = np.clip(
            frame[y0:y1, x0:x1].astype(np.float32) + signal, 0, 255
        ).astype(np.uint8)
        self._draw_text(frame, phase, local, progress, t)
        return frame

    def _draw_world(
        self,
        signal: np.ndarray,
        x0: int,
        y0: int,
        phase: str,
        local: int,
        progress: float,
    ) -> None:
        cfg = self.cfg
        scale = self.scale
        glow = float(self.glow[self._frame_index_of(phase, local)])
        if phase == "reset":
            glow *= 1.0 - progress
        beat_pulse = math.sin(math.pi * progress) if phase == "beat" else 0.0
        milestone_pulse = self._milestone_pulse(phase, local)
        intensity = 0.85 + 0.45 * min(glow, 1.5) + 0.18 * beat_pulse + 0.28 * milestone_pulse
        # #E8E8F0 is R,G,B so BGR is (0xF0, 0xE8, 0xE8), scaled by the glow.
        ring_color = (
            int(min(255, 0xF0 * intensity)),
            int(min(255, 0xE8 * intensity)),
            int(min(255, 0xE8 * intensity)),
        )
        center = ((cfg.ring_cx * scale) - x0, (cfg.ring_cy * scale) - y0)
        _blit_circle(
            signal,
            center[0],
            center[1],
            cfg.ring_radius * scale,
            ring_color,
            thickness=max(1, int(round(cfg.ring_thickness * scale))),
        )
        if phase == "beat":
            return
        positions, count, radius, scales = self._ball_state(phase, local, progress)
        if count <= 0 or radius <= 0:
            return
        order = range(count)
        for index in order:
            sx = float(scales[index]) if scales is not None else 1.0
            if sx <= 0.01:
                continue
            color = self.colors[int(self.sim.spawn_index[index]) % len(self.colors)]
            _blit_circle(
                signal,
                float(positions[index, 0]) * scale - x0,
                float(positions[index, 1]) * scale - y0,
                radius * sx * scale,
                (int(color[0]), int(color[1]), int(color[2])),
            )

    def _frame_index_of(self, phase: str, local: int) -> int:
        cursor = 0
        for name, n in self.plan:
            if name == phase:
                return cursor + local
            cursor += n
        return 0

    def _milestone_pulse(self, phase: str, local: int) -> float:
        if phase != "growth" or self.sim.milestones.size == 0:
            return 0.0
        index = self._growth_index(local)
        t = index / self.cfg.fps
        pulse = 0.0
        for row in np.atleast_2d(self.sim.milestones):
            age = t - float(row[0])
            if 0.0 <= age <= 0.20:
                pulse = max(pulse, 1.0 - age / 0.20)
        return pulse

    def _ball_state(self, phase: str, local: int, progress: float):
        sim = self.sim
        cfg = self.cfg
        if phase == "growth":
            index = self._growth_index(local)
            count = int(sim.counts[index])
            return sim.positions[index], count, float(sim.radii[index]), None
        if phase == "hold":
            return sim.hold_positions, sim.hold_count, sim.hold_radius, None
        if phase == "implode":
            ease = ease_in_out_cubic(progress)
            positions = sim.hold_positions.copy()
            count = sim.hold_count
            center = np.array([cfg.ring_cx, cfg.ring_cy], dtype=np.float32)
            positions[:count] = positions[:count] * (1.0 - ease) + center * ease
            return positions, count, sim.hold_radius * (1.0 - ease), None
        # reset: the single opening ball scales in at its exact start pose
        scale = ease_out_cubic(progress)
        positions = np.zeros_like(sim.hold_positions)
        positions[0] = sim.initial_pos
        return positions, 1, ball_radius(1, cfg), np.array([scale], dtype=np.float64)

    def _bloom(self, signal: np.ndarray) -> None:
        apply_bloom(signal, self.cfg.bloom_strength)

    def _draw_text(self, frame: np.ndarray, phase: str, local: int, progress: float, t: float) -> None:
        cfg = self.cfg
        scale = self.scale
        hook_opacity = self._hook_opacity(phase, progress, t)
        if hook_opacity > 0.01:
            hook = self._label(cfg.hook, max(12, int(round(96 * scale))), 2)
            _blit_scaled(
                frame,
                hook,
                cfg.width * scale / 2.0,
                330.0 * scale,
                1.0,
                hook_opacity,
            )
        count, opacity, pop = self._counter_state(phase, local, progress)
        if opacity > 0.01 and count is not None:
            label = self._label(f"BALLS: {count}", max(12, int(round(72 * scale))), 1)
            _blit_scaled(
                frame,
                label,
                cfg.width * scale / 2.0,
                1400.0 * scale,
                pop,
                opacity,
            )
        if cfg.show_wait_text and phase == "growth":
            index = self._growth_index(local)
            gt = index / cfg.fps
            start = 0.55 * cfg.growth_seconds
            if start <= gt <= start + 1.5:
                fade = min(1.0, (gt - start) / 0.2, (start + 1.5 - gt) / 0.2)
                wait = self._label(cfg.wait_text, max(12, int(round(54 * scale))), 1)
                _blit_scaled(frame, wait, cfg.width * scale / 2.0, 700.0 * scale, 1.0, fade)
        if cfg.show_cta and phase == "beat":
            fade = math.sin(math.pi * min(1.0, max(0.0, progress)))
            cta = self._label(cfg.cta_text, max(12, int(round(64 * scale))), 2)
            _blit_scaled(frame, cta, cfg.width * scale / 2.0, 1180.0 * scale, 1.0, fade)

    def _hook_opacity(self, phase: str, progress: float, t: float) -> float:
        if phase == "reset":
            return progress
        if t <= 2.6:
            return 1.0
        if t <= 3.0:
            return 1.0 - (t - 2.6) / 0.4
        return 0.0

    def _counter_state(self, phase: str, local: int, progress: float) -> tuple[int | None, float, float]:
        if phase == "growth":
            index = self._growth_index(local)
            return int(self.sim.counts[index]), 1.0, float(self.pop[index])
        if phase == "hold":
            return self.sim.hold_count, 1.0, 1.0
        if phase == "implode":
            return self.sim.hold_count, 1.0 - ease_in_out_cubic(progress), 1.0
        if phase == "reset":
            return 1, progress, 1.0
        return None, 0.0, 1.0


def _blit_scaled(
    frame: np.ndarray,
    rgba: np.ndarray,
    cx: float,
    cy: float,
    scale: float,
    opacity: float,
) -> None:
    image = rgba
    if abs(scale - 1.0) > 0.004:
        width = max(1, int(round(rgba.shape[1] * scale)))
        height = max(1, int(round(rgba.shape[0] * scale)))
        image = cv2.resize(rgba, (width, height), interpolation=cv2.INTER_LINEAR)
    if opacity < 0.999:
        image = image.copy()
        image[..., 3] = np.clip(image[..., 3].astype(np.float32) * opacity, 0, 255).astype(np.uint8)
    x = int(round(cx - image.shape[1] / 2.0))
    y = int(round(cy - image.shape[0] / 2.0))
    composite_rgba(frame, image, x, y)


def geometric_fill(cfg: Config, sim: SimResult) -> float:
    """Sum of disk areas over the collision-disk area. Overlaps are counted, so this can exceed 1."""
    radius = ball_radius(sim.final_count, cfg)
    return sim.final_count * radius * radius / (cfg.ring_radius ** 2)


def pixel_coverage(cfg: Config, sim: SimResult, frame: np.ndarray) -> float:
    """Fraction of the collision disk that is not the cached vignette, on a rendered frame."""
    background = vignette_bgr(frame.shape[1], frame.shape[0])
    scale = frame.shape[1] / cfg.width
    yy, xx = np.mgrid[0 : frame.shape[0], 0 : frame.shape[1]]
    dx = xx - cfg.ring_cx * scale
    dy = yy - cfg.ring_cy * scale
    inside = dx * dx + dy * dy <= (cfg.ring_radius * scale) ** 2
    delta = np.max(np.abs(frame.astype(np.int16) - background.astype(np.int16)), axis=2)
    changed = (delta > 8) & inside
    total = int(np.count_nonzero(inside))
    if total == 0:
        return 0.0
    return float(np.count_nonzero(changed)) / float(total)
