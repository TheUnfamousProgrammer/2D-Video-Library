"""Odd One Out frames.

Bloom is the shared halo only. The disc fill is composited after it, so a 5x5
patch at the center stays the color ``apply_look`` asked for. Text is drawn
after the bloom. Preview multiplies coordinates by 0.5 and runs at 30 fps.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from fc_sat.odd_config import OddConfig, Segment, build_timeline, timeline_duration, timeline_frames
from fc_sat.odd_diff import REVEAL_LABELS, apply_look, lab_to_bgr
from fc_sat.odd_layout import pip_centers
from fc_sat.odd_sim import Show
from fc_sat.render import _blit_circle
from fc_sat.visual import apply_bloom, composite_rgba, find_font, raster_text

_WORKER: "OddRenderer | None" = None


def _segment_at(timeline: tuple[Segment, ...], t: float) -> Segment:
    for segment in timeline:
        if segment.start_s <= t < segment.end_s - 1e-9:
            return segment
    return timeline[-1]


def _play_start(timeline: tuple[Segment, ...], level_id: int) -> float:
    for segment in timeline:
        if segment.kind == "play" and segment.level_id == level_id:
            return segment.start_s
    raise KeyError(level_id)


class OddRenderer:
    def __init__(self, cfg: OddConfig, show: Show, *, preview: bool = False, safe_overlay: bool = False) -> None:
        self.cfg = cfg
        self.show = show
        self.preview = preview
        self.safe_overlay = safe_overlay
        self.scale = 0.5 if preview else 1.0
        self.fps = 30 if preview else cfg.fps
        self.width = int(round(cfg.width * self.scale))
        self.height = int(round(cfg.height * self.scale))
        self.timeline = build_timeline(cfg)
        duration = timeline_duration(cfg)
        self.n_frames = int(round(duration * self.fps)) if preview else timeline_frames(cfg)
        self.font = find_font()
        self._text: dict[tuple, np.ndarray] = {}
        self.background = self._make_background()
        self.boxes: list[tuple[str, tuple[int, int, int, int]]] = []

    def s(self, value: float) -> float:
        return float(value) * self.scale

    def render(self, index: int) -> np.ndarray:
        return self.render_time(index / self.fps, highlight=False)

    def render_time(self, t: float, *, highlight: bool = False) -> np.ndarray:
        self.boxes = []
        frame = self.background.copy()
        segment = _segment_at(self.timeline, t)
        groups = self._groups(t, segment)
        signal = np.zeros((self.height, self.width, 3), dtype=np.float32)
        self._glow_field(signal)
        prepared = [self._prepare(group, t) for group in groups]
        for group in prepared:
            for item in group["items"]:
                color = tuple(int(channel * item["gain"] * 0.5) for channel in item["bgr"])
                _blit_circle(signal, item["x"], item["y"], item["radius"] * 1.9, color)
        apply_bloom(signal, self.cfg.bloom_strength)
        lit = np.clip(frame.astype(np.float32) + signal, 0, 255).astype(np.uint8)
        frame = lit
        for group in prepared:
            for item in group["items"]:
                self._paint_disc(frame, item)
                self._paint_tick(frame, item)
        reveal = next((group for group in prepared if group["reveal"]), None)
        if reveal is not None:
            odd = reveal["items"][reveal["odd_index"]]
            self._spotlight(frame, odd["x"], odd["y"])
            self._ring(frame, odd, reveal["reveal_t"])
            self._pill(frame, odd, reveal["label"])
        if highlight:
            for group in prepared:
                odd = group["items"][group["odd_index"]]
                cv2.circle(
                    frame,
                    (int(round(odd["x"])), int(round(odd["y"]))),
                    int(round(odd["radius"] + self.s(10))),
                    (0, 220, 255),
                    max(1, int(round(self.s(3)))),
                    lineType=cv2.LINE_AA,
                )
        if segment.kind == "outro":
            self._outro(frame)
        else:
            self._hud(frame, t, segment, prepared)
        if self.safe_overlay:
            self._guides(frame)
        return frame

    def _groups(self, t: float, segment: Segment) -> list[dict]:
        if segment.kind == "outro":
            return []
        if segment.kind in {"play", "reveal"}:
            return [self._one_group(segment.level_id, t, fade=1.0, reveal=segment.kind == "reveal", segment=segment)]
        # Wipe: the segment's level is the one that just finished. The next level fades in.
        ids = [level.id for level in self.cfg.levels]
        current = segment.level_id
        nxt = ids[ids.index(current) + 1]
        u = (t - segment.start_s) / max(segment.duration_s, 1e-6)
        out = self._one_group(current, t, fade=1.0 - u, reveal=False, segment=None)
        incoming = self._one_group(nxt, t, fade=u, reveal=False, segment=None)
        return [out, incoming]

    def _one_group(self, level_id: int, t: float, *, fade: float, reveal: bool, segment: Segment | None) -> dict:
        level = next(level for level in self.cfg.levels if level.id == level_id)
        sim = self.show.levels[level_id]
        sim_t = t - _play_start(self.timeline, level_id)
        reveal_t = 0.0 if segment is None or not reveal else t - segment.start_s
        exaggerate = 1.0 if reveal and reveal_t < 1.0 else 0.0
        pos = sim.at(sim_t)
        items = []
        for index in range(level.count):
            look = apply_look(
                level.difference,
                self.cfg.tier,
                is_odd=index == sim.odd_index,
                base_lab=sim.base_lab,
                odd_lab=sim.odd_lab,
                radius=float(sim.radii[index] if level.difference != "size" else self.cfg.radius),
                t=sim_t,
                spin_phase=float(sim.spin_phase[index]),
                pulse_phase=float(sim.pulse_phase[index]),
                exaggerate=exaggerate,
            )
            if level.difference == "size":
                # apply_look scales the base radius by the (possibly exaggerated) ratio.
                pass
            dim = reveal and index != sim.odd_index
            gain = fade * (0.25 if dim else 1.0)
            items.append(
                {
                    "x": self.s(float(pos[index, 0])),
                    "y": self.s(float(pos[index, 1])),
                    "radius": self.s(look["radius"]),
                    "bgr": lab_to_bgr(look["lab"]),
                    "gain": gain,
                    "tick": look["tick"],
                    "odd": index == sim.odd_index,
                }
            )
        return {
            "level": level,
            "items": items,
            "odd_index": sim.odd_index,
            "reveal": reveal,
            "reveal_t": reveal_t,
            "label": REVEAL_LABELS[level.difference],
            "sim_t": sim_t,
        }

    def _prepare(self, group: dict, t: float) -> dict:
        return group

    def _make_background(self) -> np.ndarray:
        height, width = self.height, self.width
        yy, xx = np.mgrid[0:height, 0:width]
        nx = (xx - (width - 1) / 2.0) / (width / 2.0)
        ny = (yy - (height - 1) / 2.0) / (height / 2.0)
        shade = np.clip(1.0 - 0.42 * np.power(nx * nx + ny * ny, 0.78), 0.42, 1.0)
        base = np.array([26.0, 15.0, 11.0], dtype=np.float32)  # BGR of #0B0F1A
        image = base * shade[..., None]
        step = max(8, int(round(54 * self.scale)))
        grid = ((xx % step == 0) | (yy % step == 0)).astype(np.float32)
        image += grid[..., None] * np.array([10.0, 8.0, 6.0], dtype=np.float32)
        return np.clip(image, 0, 255).astype(np.uint8)

    def _glow_field(self, signal: np.ndarray) -> None:
        mask = np.zeros(signal.shape[:2], dtype=np.uint8)
        x0 = int(round(self.s(self.cfg.field_x0)))
        y0 = int(round(self.s(self.cfg.field_y0)))
        x1 = int(round(self.s(self.cfg.field_x1)))
        y1 = int(round(self.s(self.cfg.field_y1)))
        rad = int(round(self.s(self.cfg.corner_radius)))
        thick = max(1, int(round(self.s(3))))
        _stroke_round(mask, x0, y0, x1, y1, rad, thick)
        color = np.array([210.0, 170.0, 120.0], dtype=np.float32)
        signal += mask.astype(np.float32)[..., None] * (color / 255.0) * 70.0

    def _paint_disc(self, frame: np.ndarray, item: dict) -> None:
        if item["gain"] <= 0.01 or item["radius"] < 0.5:
            return
        color = item["bgr"]
        _blend_circle(frame, item["x"], item["y"], item["radius"], color, min(1.0, item["gain"]))
        spec_r = max(1.0, item["radius"] * self.cfg.specular)
        _blend_circle(
            frame,
            item["x"] - item["radius"] * 0.32,
            item["y"] - item["radius"] * 0.32,
            spec_r,
            (255, 255, 255),
            0.9 * min(1.0, item["gain"]),
        )

    def _paint_tick(self, frame: np.ndarray, item: dict) -> None:
        tick = item["tick"]
        if tick is None or item["gain"] <= 0.01:
            return
        radius = item["radius"]
        angle = tick["angle"]
        x0 = item["x"] + math.cos(angle) * radius * tick["inner"]
        y0 = item["y"] + math.sin(angle) * radius * tick["inner"]
        x1 = item["x"] + math.cos(angle) * radius * tick["outer"]
        y1 = item["y"] + math.sin(angle) * radius * tick["outer"]
        white = 255 if tick.get("bright") else int(np.clip(255 * item["gain"], 0, 255))
        cv2.line(
            frame,
            (int(round(x0)), int(round(y0))),
            (int(round(x1)), int(round(y1))),
            (white, white, white),
            max(1, int(round(self.s(tick["width"])))),
            lineType=cv2.LINE_AA,
        )

    def _spotlight(self, frame: np.ndarray, x: float, y: float) -> None:
        yy, xx = np.ogrid[0 : self.height, 0 : self.width]
        dist = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
        radius = self.s(90)
        shade = 1.0 - 0.72 * np.clip((dist - radius) / max(self.s(160), 1.0), 0.0, 1.0)
        frame[:] = np.clip(frame.astype(np.float32) * shade[..., None], 0, 255).astype(np.uint8)

    def _ring(self, frame: np.ndarray, odd: dict, reveal_t: float) -> None:
        radius = self.s(120) * min(1.0, reveal_t / 0.35)
        if radius < 1:
            return
        cv2.circle(
            frame,
            (int(round(odd["x"])), int(round(odd["y"]))),
            int(round(radius)),
            (255, 255, 255),
            max(1, int(round(self.s(4)))),
            lineType=cv2.LINE_AA,
        )

    def _pill(self, frame: np.ndarray, odd: dict, label: str) -> None:
        px = max(12, int(round(self.s(36))))
        rgba = self._raster(label, px, int(self.s(520)))
        h, w = rgba.shape[:2]
        field = (
            int(round(self.s(self.cfg.field_x0 + 12))),
            int(round(self.s(self.cfg.field_y0 + 12))),
            int(round(self.s(self.cfg.field_x1 - 12))),
            int(round(self.s(self.cfg.field_y1 - 12))),
        )
        x = int(round(odd["x"] - w / 2))
        y = int(round(odd["y"] - odd["radius"] - h - self.s(14)))
        if y < field[1]:
            y = int(round(odd["y"] + odd["radius"] + self.s(14)))
        x = min(max(x, field[0]), field[2] - w)
        y = min(max(y, field[1]), field[3] - h)
        pad = int(round(self.s(8)))
        cv2.rectangle(
            frame,
            (x - pad, y - pad),
            (x + w + pad, y + h + pad),
            (24, 18, 14),
            -1,
            lineType=cv2.LINE_AA,
        )
        composite_rgba(frame, rgba, x, y)
        self.boxes.append(("pill", (x - pad, y - pad, x + w + pad, y + h + pad)))

    def _hud(self, frame: np.ndarray, t: float, segment: Segment, groups: list[dict]) -> None:
        level_id = segment.level_id if segment.kind != "wipe" else self._incoming_id(segment)
        level = next(item for item in self.cfg.levels if item.id == level_id)
        play_start = _play_start(self.timeline, level.id)
        elapsed = t - play_start
        if segment.kind == "wipe":
            elapsed = 0.0
        remaining = max(0.0, level.timer - max(elapsed, 0.0))
        caption = _caption(self.cfg, level.id, max(elapsed, 0.0), remaining, segment.kind)
        self._centered(
            frame,
            f"LEVEL {level.id}  {level.label}",
            (self.cfg.safe_x[0] + self.cfg.safe_x[1]) * 0.5,
            self.cfg.label_y,
            self.cfg.label_px,
            int(780),
            "label",
        )
        if caption and segment.kind != "reveal":
            self._centered(
                frame,
                caption,
                (self.cfg.caption_x[0] + self.cfg.caption_x[1]) * 0.5,
                self.cfg.caption_y,
                self.cfg.caption_px,
                int(self.cfg.caption_x[1] - self.cfg.caption_x[0]),
                "caption",
            )
        self._timer(frame, level.timer, remaining, elapsed)
        done = {item.id for item in self.cfg.levels if _reveal_started(self.timeline, item.id, t)}
        current = level.id
        self._pips(frame, done=done, current=current, outro=False)

    def _incoming_id(self, segment: Segment) -> int:
        ids = [level.id for level in self.cfg.levels]
        return ids[ids.index(segment.level_id) + 1]

    def _timer(self, frame: np.ndarray, timer: float, remaining: float, elapsed: float) -> None:
        x0 = int(round(self.s(self.cfg.timer_x0)))
        x1 = int(round(self.s(self.cfg.timer_x1)))
        y0 = int(round(self.s(self.cfg.timer_y)))
        y1 = int(round(self.s(self.cfg.timer_y + self.cfg.timer_h)))
        last = remaining <= self.cfg.heartbeat_window and elapsed >= 0
        frac = 0.0 if timer <= 0 else remaining / timer
        if last:
            throb = 0.5 + 0.5 * math.sin(elapsed * 2.0 * math.pi * 2.4)
            color = (40 + int(30 * throb), 40 + int(20 * throb), 220)
        elif frac > 0.5:
            color = _mix((40, 210, 90), (40, 210, 230), (1.0 - frac) / 0.5)
        else:
            color = _mix((40, 210, 230), (40, 50, 220), 1.0 - frac / 0.5)
        cv2.rectangle(frame, (x0, y0), (x1, y1), (32, 28, 24), -1, lineType=cv2.LINE_AA)
        fill = x0 + int(round((x1 - x0) * float(np.clip(frac, 0.0, 1.0))))
        if fill > x0:
            cv2.rectangle(frame, (x0, y0), (fill, y1), color, -1, lineType=cv2.LINE_AA)
        self.boxes.append(("timer", (x0, y0, x1, y1)))
        shown = int(math.ceil(timer - 1e-9))
        if elapsed < 0:
            shown = int(math.ceil(timer - 1e-9))
        elif remaining <= 1e-3:
            shown = 0
        else:
            shown = max(1, int(math.ceil(remaining - 1e-9)))
        self._centered(
            frame,
            str(shown),
            (self.cfg.timer_num_x0 + self.cfg.timer_num_x1) * 0.5,
            self.cfg.timer_y + self.cfg.timer_h * 0.5,
            self.cfg.timer_num_px,
            int(self.cfg.timer_num_x1 - self.cfg.timer_num_x0),
            "seconds",
        )

    def _outro(self, frame: np.ndarray) -> None:
        self._centered(
            frame,
            "How far did you get?",
            (self.cfg.safe_x[0] + self.cfg.safe_x[1]) * 0.5,
            self.cfg.outro_y,
            self.cfg.outro_px,
            int(760),
            "outro",
        )
        self._pips(frame, done={level.id for level in self.cfg.levels}, current=None, outro=True)
        self._centered(
            frame,
            self.cfg.cta,
            (self.cfg.cta_x[0] + self.cfg.cta_x[1]) * 0.5,
            self.cfg.cta_y,
            self.cfg.cta_px,
            int(self.cfg.cta_x[1] - self.cfg.cta_x[0]),
            "cta",
        )

    def _pips(self, frame: np.ndarray, *, done: set[int], current: int | None, outro: bool) -> None:
        centers = pip_centers(self.cfg, outro=outro, scale=self.scale)
        radius = max(2, int(round(self.s(self.cfg.pip_r))))
        for (level, (x, y)) in zip(self.cfg.levels, centers):
            center = (int(round(x)), int(round(y)))
            if level.id in done and level.id != current:
                cv2.circle(frame, center, radius, (230, 230, 230), -1, lineType=cv2.LINE_AA)
            elif level.id == current:
                cv2.circle(frame, center, radius, (230, 230, 230), max(2, radius // 4), lineType=cv2.LINE_AA)
            else:
                cv2.circle(frame, center, radius, (90, 80, 70), max(1, radius // 5), lineType=cv2.LINE_AA)
            self.boxes.append((f"pip{level.id}", (center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius)))

    def _centered(self, frame, text, cx, cy, px, max_width, name) -> None:
        rgba = self._raster(text, max(8, int(round(self.s(px)))), max(8, int(round(self.s(max_width)))))
        h, w = rgba.shape[:2]
        x = int(round(self.s(cx) - w / 2))
        y = int(round(self.s(cy) - h / 2))
        composite_rgba(frame, rgba, x, y)
        alpha = rgba[..., 3]
        ys, xs = np.where(alpha > 16)
        if len(xs):
            self.boxes.append((name, (x + int(xs.min()), y + int(ys.min()), x + int(xs.max()) + 1, y + int(ys.max()) + 1)))

    def _raster(self, text: str, px: int, max_width: int) -> np.ndarray:
        key = (text, px, max_width)
        cached = self._text.get(key)
        if cached is None:
            cached = raster_text(text, self.font, px, max_width, max_lines=1)
            self._text[key] = cached
        return cached

    def _guides(self, frame: np.ndarray) -> None:
        safe = (
            int(round(self.s(self.cfg.safe_x[0]))),
            int(round(self.s(self.cfg.safe_y[0]))),
            int(round(self.s(self.cfg.safe_x[1]))),
            int(round(self.s(self.cfg.safe_y[1]))),
        )
        field = (
            int(round(self.s(self.cfg.field_x0))),
            int(round(self.s(self.cfg.field_y0))),
            int(round(self.s(self.cfg.field_x1))),
            int(round(self.s(self.cfg.field_y1))),
        )
        cv2.rectangle(frame, (safe[0], safe[1]), (safe[2], safe[3]), (255, 80, 200), 2)
        cv2.rectangle(frame, (field[0], field[1]), (field[2], field[3]), (80, 220, 255), 2)

    def profile_ms(self) -> float:
        import time

        started = time.perf_counter()
        self.render(0)
        return (time.perf_counter() - started) * 1000.0


def _caption(cfg: OddConfig, level_id: int, elapsed: float, remaining: float, kind: str) -> str:
    if kind == "reveal":
        return ""
    if level_id == 4 and 0 <= remaining <= cfg.last_chance_seconds and elapsed > 0:
        return "Last chance"
    if level_id == 1 and elapsed < cfg.hook_seconds:
        return cfg.hook
    if level_id in cfg.pause_levels and 0 <= elapsed < cfg.pause_caption_seconds:
        return "Pausing won't help"
    return ""


def _reveal_started(timeline: tuple[Segment, ...], level_id: int, t: float) -> bool:
    for segment in timeline:
        if segment.kind == "reveal" and segment.level_id == level_id:
            return t >= segment.start_s - 1e-9
    return False


def _mix(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    t = float(np.clip(t, 0.0, 1.0))
    return tuple(int(a[i] * (1.0 - t) + b[i] * t) for i in range(3))


def _stroke_round(mask: np.ndarray, x0: int, y0: int, x1: int, y1: int, rad: int, thick: int) -> None:
    rad = max(0, min(rad, (x1 - x0) // 2, (y1 - y0) // 2))
    cv2.line(mask, (x0 + rad, y0), (x1 - rad, y0), 255, thick, lineType=cv2.LINE_AA)
    cv2.line(mask, (x0 + rad, y1), (x1 - rad, y1), 255, thick, lineType=cv2.LINE_AA)
    cv2.line(mask, (x0, y0 + rad), (x0, y1 - rad), 255, thick, lineType=cv2.LINE_AA)
    cv2.line(mask, (x1, y0 + rad), (x1, y1 - rad), 255, thick, lineType=cv2.LINE_AA)
    if rad > 0:
        cv2.ellipse(mask, (x0 + rad, y0 + rad), (rad, rad), 0, 180, 270, 255, thick, lineType=cv2.LINE_AA)
        cv2.ellipse(mask, (x1 - rad, y0 + rad), (rad, rad), 0, 270, 360, 255, thick, lineType=cv2.LINE_AA)
        cv2.ellipse(mask, (x1 - rad, y1 - rad), (rad, rad), 0, 0, 90, 255, thick, lineType=cv2.LINE_AA)
        cv2.ellipse(mask, (x0 + rad, y1 - rad), (rad, rad), 0, 90, 180, 255, thick, lineType=cv2.LINE_AA)


def _blend_circle(frame: np.ndarray, x: float, y: float, radius: float, color: tuple[int, int, int], alpha: float) -> None:
    if radius < 0.4 or alpha <= 0.01:
        return
    pad = int(math.ceil(radius)) + 2
    ix = int(math.floor(x))
    iy = int(math.floor(y))
    x0 = max(0, ix - pad)
    y0 = max(0, iy - pad)
    x1 = min(frame.shape[1], ix + pad + 1)
    y1 = min(frame.shape[0], iy + pad + 1)
    if x1 <= x0 or y1 <= y0:
        return
    mask = np.zeros((y1 - y0, x1 - x0), dtype=np.float32)
    cv2.circle(
        mask,
        (int(round(x - x0)), int(round(y - y0))),
        max(1, int(round(radius))),
        float(np.clip(alpha, 0.0, 1.0)),
        -1,
        lineType=cv2.LINE_AA,
    )
    region = frame[y0:y1, x0:x1].astype(np.float32)
    ink = np.array(color, dtype=np.float32)
    frame[y0:y1, x0:x1] = np.clip(region * (1.0 - mask[..., None]) + ink * mask[..., None], 0, 255).astype(np.uint8)


def init_odd_worker(cfg: OddConfig, show: Show, preview: bool, safe_overlay: bool) -> None:
    global _WORKER
    cv2.setNumThreads(1)
    _WORKER = OddRenderer(cfg, show, preview=preview, safe_overlay=safe_overlay)


def render_odd_chunk(indices: list[int]) -> list[np.ndarray]:
    if _WORKER is None:
        raise RuntimeError("odd render worker was not initialized")
    return [_WORKER.render(index) for index in indices]


def _fit(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def write_level_strips(renderer: OddRenderer, directory_stem) -> list[str]:
    """Six timer frames plus the reveal, and a debug strip with the odd item circled."""
    from pathlib import Path

    stem = Path(directory_stem)
    written: list[str] = []
    cell = (270, 480)
    for level in renderer.cfg.levels:
        play = next(segment for segment in renderer.timeline if segment.kind == "play" and segment.level_id == level.id)
        reveal = next(segment for segment in renderer.timeline if segment.kind == "reveal" and segment.level_id == level.id)
        times = [play.start_s + play.duration_s * (index / 5.0) for index in range(6)]
        times[-1] = min(times[-1], reveal.start_s - 0.5 / renderer.cfg.fps)
        times.append(reveal.start_s + min(0.45, reveal.duration_s * 0.35))
        row = [_fit(renderer.render_time(t), *cell) for t in times]
        debug = [_fit(renderer.render_time(t, highlight=True), *cell) for t in times]
        plain = stem.with_name(f"{stem.stem}.l{level.id}.strip.png")
        marked = stem.with_name(f"{stem.stem}.l{level.id}.debug.png")
        cv2.imwrite(str(plain), np.hstack(row))
        cv2.imwrite(str(marked), np.hstack(debug))
        written.extend([str(plain), str(marked)])
        print(f"strip L{level.id}: {plain.name}", flush=True)
    return written


def write_pause_test(renderer: OddRenderer, directory_stem) -> str | None:
    from pathlib import Path

    stem = Path(directory_stem)
    cells = []
    for level in renderer.cfg.levels:
        if level.difference not in {"spin", "pulse"}:
            continue
        play = next(segment for segment in renderer.timeline if segment.kind == "play" and segment.level_id == level.id)
        frame = renderer.render_time(play.start_s + play.duration_s * 0.5, highlight=False)
        cells.append(_fit(frame, 270, 480))
    if not cells:
        return None
    path = stem.with_name(f"{stem.stem}.pause.png")
    cv2.imwrite(str(path), np.hstack(cells))
    print(f"pause test: {path.name}", flush=True)
    return str(path)


def contact_sheet(renderer: OddRenderer, path) -> None:
    """One row per level: start, 40%, 80% of the timer, then the reveal."""
    from pathlib import Path

    cell = (270, 480)
    rows = []
    for level in renderer.cfg.levels:
        play = next(segment for segment in renderer.timeline if segment.kind == "play" and segment.level_id == level.id)
        reveal = next(segment for segment in renderer.timeline if segment.kind == "reveal" and segment.level_id == level.id)
        times = [
            play.start_s,
            play.start_s + 0.40 * play.duration_s,
            play.start_s + 0.80 * play.duration_s,
            reveal.start_s + min(0.45, reveal.duration_s * 0.35),
        ]
        rows.append(np.hstack([_fit(renderer.render_time(t), *cell) for t in times]))
        print(f"contact L{level.id}", flush=True)
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dest), np.vstack(rows))
    print(f"contact sheet: {dest}", flush=True)
