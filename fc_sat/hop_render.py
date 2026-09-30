"""Melody Hop frames.

Effects are clipped to the action box before bloom. Bloom tone-maps the full
additive layer, then that layer is added to the vignette. The right 12% and the
bottom 20% of the additive layer are cleared after bloom so those pixels stay
equal to the cached background.

Hook alpha is 1 on [0, 2.6], falls to 0 by 3.0, stays 0 until T-0.6, then rises
to 1 at T, and repeats with period T.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from fc_sat.color import palette_bgr
from fc_sat.hop_choreo import Choreography, build_choreography
from fc_sat.hop_config import HopConfig
from fc_sat.render import ease_out_cubic
from fc_sat.song import load_song
from fc_sat.visual import apply_bloom, composite_rgba, find_font, raster_text, vignette_bgr

SHIFT = 4
SHIFT_SCALE = 1 << SHIFT
ACTION = (170.0, 200.0, 910.0, 1536.0)  # x0, y0, x1 inclusive, y1 exclusive at y=1536


def action_bounds(scale: float = 1.0) -> tuple[int, int, int, int]:
    """Pixel box the additive layer may occupy before bloom. ``y1`` is exclusive."""
    x0 = int(round(ACTION[0] * scale))
    y0 = int(round(ACTION[1] * scale))
    x1 = int(round(ACTION[2] * scale)) + 1
    y1 = int(round(ACTION[3] * scale))
    return x0, y0, x1, y1


def hook_alpha(t: float, duration: float) -> float:
    if duration <= 0:
        return 1.0
    local = math.fmod(t, duration)
    if local < 0:
        local += duration
    if local >= duration - 1e-9:
        local = 0.0
    if local <= 2.6:
        return 1.0
    if local <= 3.0:
        return 1.0 - (local - 2.6) / 0.4
    if local < duration - 0.6:
        return 0.0
    return (local - (duration - 0.6)) / 0.6


def tone_mapped(color: np.ndarray) -> np.ndarray:
    """Same curve as bloom, so a palette color has a brightness ceiling."""
    value = np.asarray(color, dtype=np.float32)
    return 255.0 * (1.0 - np.exp(-value / 90.0))


def clip_action(signal: np.ndarray, scale: float) -> None:
    x0, y0, x1, y1 = action_bounds(scale)
    if y0 > 0:
        signal[:y0] = 0
    if y1 < signal.shape[0]:
        signal[y1:] = 0
    if x0 > 0:
        signal[:, :x0] = 0
    if x1 < signal.shape[1]:
        signal[:, x1:] = 0


def _shift_point(x: float, y: float) -> tuple[int, int]:
    return int(round(x * SHIFT_SCALE)), int(round(y * SHIFT_SCALE))


def _fill_round_rect(
    image: np.ndarray,
    x: float,
    y: float,
    width: float,
    height: float,
    radius: float,
    color: tuple[float, float, float] | tuple[int, int, int],
) -> None:
    radius = min(radius, width / 2.0, height / 2.0)
    if width <= 1 or height <= 1:
        return
    x0, y0 = _shift_point(x, y)
    x1, y1 = _shift_point(x + width, y + height)
    rad = max(1, int(round(radius * SHIFT_SCALE)))
    cv2.rectangle(image, (x0 + rad, y0), (x1 - rad, y1), color, -1, lineType=cv2.LINE_AA, shift=SHIFT)
    cv2.rectangle(image, (x0, y0 + rad), (x1, y1 - rad), color, -1, lineType=cv2.LINE_AA, shift=SHIFT)
    for cx, cy in (
        (x + radius, y + radius),
        (x + width - radius, y + radius),
        (x + radius, y + height - radius),
        (x + width - radius, y + height - radius),
    ):
        cv2.circle(image, _shift_point(cx, cy), rad, color, -1, lineType=cv2.LINE_AA, shift=SHIFT)


def _blit_ellipse(
    signal: np.ndarray,
    x: float,
    y: float,
    rx: float,
    ry: float,
    color: tuple[int, int, int],
    thickness: int = -1,
) -> None:
    if rx <= 0.3 or ry <= 0.3:
        return
    pad = int(math.ceil(max(rx, ry))) + (6 if thickness > 0 else 4)
    patch = np.zeros((pad * 2 + 1, pad * 2 + 1, 3), dtype=np.uint8)
    center = (int(round(pad * SHIFT_SCALE)), int(round(pad * SHIFT_SCALE)))
    axes = (max(1, int(round(rx * SHIFT_SCALE))), max(1, int(round(ry * SHIFT_SCALE))))
    thick = thickness if thickness < 0 else max(1, thickness)
    cv2.ellipse(patch, center, axes, 0, 0, 360, color, thick, lineType=cv2.LINE_AA, shift=SHIFT)
    left = int(round(x)) - pad
    top = int(round(y)) - pad
    _add_patch(signal, patch, left, top)


def _add_patch(signal: np.ndarray, patch: np.ndarray, left: int, top: int) -> None:
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


class HopRenderer:
    def __init__(self, cfg: HopConfig, preview: bool = False) -> None:
        self.cfg = cfg
        self.preview = preview
        self.scale = 0.5 if preview else 1.0
        self.fps = 30 if preview else 60
        self.width = int(round(cfg.width * self.scale))
        self.height = int(round(cfg.height * self.scale))
        song = load_song(cfg.song, octave_shift=cfg.octave_shift, bpm=cfg.bpm)
        self.dance: Choreography = build_choreography(
            song,
            bpm=cfg.bpm,
            span=cfg.span,
            pad_max_width=cfg.pad_max_width,
            pad_top_y=cfg.pad_top_y,
            pad_thickness=cfg.pad_thickness,
            corner=cfg.corner,
            ball_radius=cfg.ball_radius,
            h_ref=cfg.h_ref,
            h_min=cfg.h_min,
            h_max=cfg.h_max,
            exponent=cfg.exponent,
        )
        loops = max(1, cfg.repeats)
        if preview:
            self.n_frames = int(round(self.dance.grid.duration * self.fps)) * loops
        else:
            self.n_frames = self.dance.grid.n_frames * loops
        self.background = vignette_bgr(self.width, self.height)
        self.colors = palette_bgr(cfg.palette_stops, len(self.dance.layout.pads))
        self.ceilings = tone_mapped(self.colors.astype(np.float32))
        self.font_path = find_font()
        self._text: dict[tuple, np.ndarray] = {}
        self._static, self._labels = self._build_pads()
        self._glow = self._build_glow_mask()
        self._sparks = self._build_sparks()
        self._forbid_x = math.ceil(0.88 * self.width)
        self._forbid_y = math.ceil(0.80 * self.height)

    def frame_time(self, index: int) -> float:
        return index / self.fps

    def render(self, index: int) -> np.ndarray:
        signal = self.pre_bloom(self.frame_time(index))
        apply_bloom(signal, self.cfg.bloom_strength)
        self._apply_flash(signal, self.frame_time(index))
        if self._forbid_x < signal.shape[1]:
            signal[:, self._forbid_x :] = 0
        if self._forbid_y < signal.shape[0]:
            signal[self._forbid_y :] = 0
        frame = self.background.copy()
        frame[:] = np.clip(frame.astype(np.float32) + signal, 0, 255).astype(np.uint8)
        self._draw_hook(frame, self.frame_time(index))
        return frame

    def pre_bloom(self, t: float) -> np.ndarray:
        """Additive layer after the action-box clip and before bloom."""
        signal = self._static.copy()
        self._draw_glow(signal, t)
        self._draw_trail(signal, t)
        self._draw_impacts(signal, t)
        self._draw_ball(signal, t)
        clip_action(signal, self.scale)
        return signal

    def _scaled(self, value: float) -> float:
        return value * self.scale

    def _build_pads(self) -> tuple[np.ndarray, np.ndarray]:
        layer = np.zeros((self.height, self.width, 3), dtype=np.float32)
        labels = np.zeros((self.height, self.width), dtype=np.int16)
        scale = self.scale
        for pad, color in zip(self.dance.layout.pads, self.colors):
            x = self._scaled(pad.left)
            y = self._scaled(pad.top)
            width = self._scaled(pad.width)
            height = self._scaled(pad.thickness)
            body = tuple(int(channel) for channel in (np.asarray(color) * 0.55))
            _fill_round_rect(layer, x, y, width, height, self._scaled(pad.corner), body)
            edge = tuple(int(min(255, int(channel) + 40)) for channel in color)
            _fill_round_rect(layer, x, y, width, max(2.0, self._scaled(5)), self._scaled(pad.corner), edge)
            glow = tuple(int(channel) for channel in (np.asarray(color) * 0.40))
            _blit_ellipse(
                layer,
                self._scaled(pad.x),
                self._scaled(pad.top + pad.thickness + 10),
                self._scaled(pad.width * 0.55),
                self._scaled(14),
                glow,
            )
            mask = np.zeros((self.height, self.width), dtype=np.uint8)
            _fill_round_rect(mask, x, y, width, height, self._scaled(pad.corner), 255)
            labels[mask > 0] = pad.index + 1
        return layer, labels

    def _build_glow_mask(self) -> np.ndarray:
        yy, xx = np.mgrid[0 : self.height, 0 : self.width]
        cx = self._scaled(540.0)
        cy = self._scaled(self.cfg.pad_top_y)
        rx = max(self._scaled(420.0), 1.0)
        ry = max(self._scaled(160.0), 1.0)
        dist = np.sqrt(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2)
        mask = np.clip(1.0 - dist, 0.0, 1.0) ** 2
        return mask.astype(np.float32)

    def _build_sparks(self) -> list[tuple[np.ndarray, np.ndarray]]:
        rows = []
        count = self.cfg.sparks
        for note in range(len(self.dance.song.notes)):
            rng = np.random.default_rng([self.cfg.seed, note])
            if count <= 0:
                rows.append((np.zeros(0), np.zeros(0)))
                continue
            angles = rng.uniform(-20.0, 20.0, size=count) * (math.pi / 180.0)
            speeds = rng.uniform(250.0, 550.0, size=count)
            rows.append((angles.astype(np.float64), speeds.astype(np.float64)))
        return rows

    def _recent_color(self, t: float) -> tuple[np.ndarray, float]:
        pose = self.dance.pose(t)
        current = self.colors[pose.src].astype(np.float32)
        previous_note = (self._flight(t) - 1) % len(self.dance.song.notes)
        previous = self.colors[self.dance.pad_index(previous_note)].astype(np.float32)
        blend = 1.0 if pose.age >= 0.150 else pose.age / 0.150
        color = previous * (1.0 - blend) + current * blend
        strength = 0.25 * math.exp(-max(pose.age, 0.0) / 0.5)
        return color, strength

    def _flight(self, t: float) -> int:
        return self.dance._flight(self.dance.grid.duration and _positive_mod(t, self.dance.grid.duration))

    def _draw_glow(self, signal: np.ndarray, t: float) -> None:
        color, strength = self._recent_color(t)
        if strength <= 0:
            return
        for channel in range(3):
            signal[..., channel] += self._glow * (float(color[channel]) * strength)

    def _draw_trail(self, signal: np.ndarray, t: float) -> None:
        steps = max(1, int(round(self.cfg.trail_seconds * 60.0)))
        for step in range(steps, 0, -1):
            age = step / 60.0
            pose = self.dance.pose(t - age)
            fade = 1.0 - step / steps
            if fade <= 0:
                continue
            src = self.colors[pose.src].astype(np.float32)
            dst = self.colors[pose.dst].astype(np.float32)
            color = src * (1.0 - pose.u) + dst * pose.u
            tint = tuple(int(min(255, max(0, channel * fade))) for channel in color)
            radius = self._scaled(self.dance.layout.ball_r) * (0.25 + 0.55 * fade)
            _blit_ellipse(signal, self._scaled(pose.x), self._scaled(pose.y_arc), radius, radius, tint)

    def _draw_ball(self, signal: np.ndarray, t: float) -> None:
        pose = self.dance.pose(t)
        color = self.colors[pose.src].astype(np.float32)
        tint = tuple(int(channel) for channel in np.clip(color * 0.85, 0, 255))
        cx = self._scaled(pose.x)
        cy = self._scaled(pose.y)
        _blit_ellipse(
            signal,
            cx,
            cy,
            self._scaled(pose.rx) * 1.8,
            self._scaled(pose.ry) * 1.8,
            tint,
        )
        _blit_ellipse(signal, cx, cy, self._scaled(pose.rx), self._scaled(pose.ry), (245, 245, 250))

    def _events(self, t: float):
        grid = self.dance.grid
        duration = grid.duration
        for cycle in (-1, 0, 1):
            for index, onset in enumerate(grid.onset_times):
                age = t - (onset + cycle * duration)
                if 0.0 <= age <= 1.5:
                    yield index, age

    def _draw_impacts(self, signal: np.ndarray, t: float) -> None:
        for index, age in self._events(t):
            pad = self.dance.layout.pads[self.dance.pad_index(index)]
            color = self.colors[self.dance.pad_index(index)]
            cx = self._scaled(pad.x)
            cy = self._scaled(pad.top)
            if self.cfg.ring and age < 0.5:
                u = age / 0.5
                radius = self._scaled(20.0 + 50.0 * ease_out_cubic(u))
                alpha = 0.7 * (1.0 - u)
                tint = tuple(int(channel * alpha) for channel in color)
                _blit_ellipse(signal, cx, cy, radius, radius, tint, thickness=max(1, int(round(4 * self.scale))))
            if age < 0.45 and self.cfg.sparks:
                angles, speeds = self._sparks[index]
                life = age / 0.45
                radius = self._scaled(3.5 * (1.0 - life))
                tint = tuple(int(channel) for channel in color)
                for angle, speed in zip(angles, speeds):
                    vx = math.sin(float(angle)) * float(speed)
                    vy = -math.cos(float(angle)) * float(speed)
                    x = pad.x + vx * age
                    y = pad.top + vy * age + 0.5 * 2500.0 * age * age
                    _blit_ellipse(signal, self._scaled(x), self._scaled(y), radius, radius, tint)

    def _apply_flash(self, signal: np.ndarray, t: float) -> None:
        flashes = np.zeros(len(self.dance.layout.pads), dtype=np.float32)
        for index, age in self._events(t):
            flashes[self.dance.pad_index(index)] += math.exp(-age / 0.12)
        scale = self.scale
        for pad_index, amount in enumerate(flashes):
            if amount <= 0.001:
                continue
            pad = self.dance.layout.pads[pad_index]
            x0 = max(0, int(pad.left * scale) - 2)
            x1 = min(signal.shape[1], int(pad.right * scale) + 3)
            y0 = max(0, int(pad.top * scale) - 2)
            y1 = min(signal.shape[0], int((pad.top + pad.thickness) * scale) + 3)
            labels = self._labels[y0:y1, x0:x1]
            region = labels == (pad_index + 1)
            if not np.any(region):
                continue
            sub = signal[y0:y1, x0:x1]
            pixels = sub[region]
            ceiling = self.ceilings[pad_index]
            lifted = pixels + float(amount) * ceiling
            cap = np.maximum(pixels, ceiling)
            sub[region] = np.minimum(lifted, cap)

    def _label(self, text: str, px: int) -> np.ndarray:
        key = (text, px, self.width)
        cached = self._text.get(key)
        if cached is None:
            max_width = max(40, int(self.width * 0.72))
            cached = raster_text(text, self.font_path, px, max_width, 2)
            self._text[key] = cached
        return cached

    def _draw_hook(self, frame: np.ndarray, t: float) -> None:
        alpha = hook_alpha(t, self.dance.grid.duration)
        if alpha <= 0.01:
            return
        px = max(12, int(round(96 * self.scale)))
        image = self._label(self.cfg.hook, px)
        if alpha < 0.999:
            image = image.copy()
            image[..., 3] = np.clip(image[..., 3].astype(np.float32) * alpha, 0, 255).astype(np.uint8)
        x = int(round(self.width / 2.0 - image.shape[1] / 2.0))
        y = int(round(self._scaled(330.0) - image.shape[0] / 2.0))
        composite_rgba(frame, image, x, y)
        if self.cfg.show_note_names:
            for pad in self.dance.layout.pads:
                label = self._label(pad.name, max(12, int(round(28 * self.scale))))
                lx = int(round(self._scaled(pad.x) - label.shape[1] / 2.0))
                ly = int(round(self._scaled(pad.top + pad.thickness + 18)))
                composite_rgba(frame, label, lx, ly)


def _positive_mod(t: float, duration: float) -> float:
    local = math.fmod(t, duration)
    if local < 0:
        local += duration
    if local >= duration:
        return 0.0
    return local


_WORKER: HopRenderer | None = None


def init_hop_worker(cfg: HopConfig, preview: bool) -> None:
    global _WORKER
    cv2.setNumThreads(1)
    _WORKER = HopRenderer(cfg, preview=preview)


def render_hop_chunk(indices: list[int]) -> list[np.ndarray]:
    if _WORKER is None:
        raise RuntimeError("hop worker was not initialized")
    return [_WORKER.render(index) for index in indices]
