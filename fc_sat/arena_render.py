"""Flag Arena picture: time remap, camera, layout, and frames.

Video time is a monotone map of sim time. Hit-stops hold the picture.
Slow motion covers the eliminations that leave three and then two alive.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from fc_sat.arena_config import ArenaConfig
from fc_sat.arena_sim import Elim, SimResult
from fc_sat.color import dominant_colors
from fc_sat.visual import find_font, raster_text

SAFE = (130, 950, 200, 1536)
PLATFORM_BOX = (130, 950, 480, 1400)


@dataclass(frozen=True)
class Span:
    video0: float
    video1: float
    sim0: float
    sim1: float
    rate: float
    phase: str


@dataclass(frozen=True)
class Timeline:
    spans: tuple[Span, ...]
    t_win: float
    winner_video: float
    replay_video: float
    celebration_video: float
    duration: float
    hitstop: float


@dataclass(frozen=True)
class CameraView:
    zoom: float
    center_x: float
    center_y: float
    shake_x: float
    shake_y: float
    left: float
    top: float
    right: float
    bottom: float


@dataclass(frozen=True)
class TextBox:
    name: str
    x0: float
    y0: float
    x1: float
    y1: float


def slow_windows(result: SimResult, cfg: ArenaConfig, t_win: float) -> tuple[tuple[float, float], ...]:
    count = len(cfg.countries)
    raw: list[list[float]] = []
    for index, item in enumerate(result.elims):
        before = count - index
        if before not in (4, 3):
            continue
        start = max(0.0, item.time - cfg.slowmo_pre)
        end = min(t_win, item.time + cfg.slowmo_post)
        if end > start:
            raw.append([start, end])
    raw.sort()
    merged: list[list[float]] = []
    for start, end in raw:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return tuple((start, end) for start, end in merged)


def _inside(moment: float, windows: tuple[tuple[float, float], ...]) -> bool:
    return any(start - 1e-9 <= moment <= end + 1e-9 for start, end in windows)


def hit_stops(result: SimResult, cfg: ArenaConfig, windows: tuple[tuple[float, float], ...]) -> tuple[tuple[float, float, float], ...]:
    """(closing speed, sim time, seconds). Weakest closing speeds are dropped first."""
    holds: list[tuple[float, float, float]] = []
    frame = 1.0 / cfg.fps
    for impulse in result.impulses:
        if impulse.source not in {"ball", "boss"} or impulse.closing < cfg.hitstop_speed:
            continue
        if _inside(impulse.time, windows):
            continue
        holds.append((impulse.closing, impulse.time, cfg.hitstop_impact_frames * frame))
    for item in result.elims:
        if _inside(item.time, windows):
            continue
        holds.append((item.closing, item.time, cfg.hitstop_ko_frames * frame))
    holds.sort(key=lambda item: item[0])
    total = sum(item[2] for item in holds)
    while holds and total > cfg.hitstop_cap + 1e-9:
        total -= holds.pop(0)[2]
    holds.sort(key=lambda item: item[1])
    return tuple(holds)


def _emit_rated(spans: list[Span], video: float, start: float, end: float, windows: tuple[tuple[float, float], ...], cfg: ArenaConfig) -> float:
    points = [start, end]
    for left, right in windows:
        if start < left < end:
            points.append(left)
        if start < right < end:
            points.append(right)
    points = sorted(set(points))
    for left, right in zip(points, points[1:]):
        if right <= left + 1e-9:
            continue
        rate = cfg.slowmo_rate if _inside(0.5 * (left + right), windows) else 1.0
        length = (right - left) / max(rate, 1e-6)
        phase = "slow" if rate < 0.999 else "main"
        spans.append(Span(video, video + length, left, right, rate, phase))
        video += length
    return video


def build_timeline(result: SimResult, cfg: ArenaConfig) -> Timeline:
    t_win = result.t_win
    if t_win is None:
        if result.trace is not None and len(result.trace["xy"]):
            t_win = (len(result.trace["xy"]) - 1) * cfg.dt
        else:
            t_win = cfg.win_window[0]
    t_win = max(0.05, float(t_win))
    windows = slow_windows(result, cfg, t_win)
    holds = hit_stops(result, cfg, windows)
    spans: list[Span] = []
    video = 0.0
    cursor = 0.0
    hold_i = 0
    while cursor < t_win - 1e-9:
        next_hold = holds[hold_i][1] if hold_i < len(holds) else t_win
        end = min(t_win, next_hold)
        if end > cursor + 1e-9:
            video = _emit_rated(spans, video, cursor, end, windows, cfg)
        cursor = end
        if hold_i < len(holds) and abs(holds[hold_i][1] - cursor) <= 1e-6:
            dur = holds[hold_i][2]
            spans.append(Span(video, video + dur, cursor, cursor, 0.0, "hold"))
            video += dur
            hold_i += 1
            if hold_i < len(holds) and abs(holds[hold_i][1] - cursor) > 1e-6:
                continue
    winner_video = video
    replay_sim = min(cfg.replay_sim, t_win)
    replay_len = replay_sim / cfg.slowmo_rate
    spans.append(Span(video, video + replay_len, t_win - replay_sim, t_win, cfg.slowmo_rate, "replay"))
    video += replay_len
    replay_video = spans[-1].video0
    spans.append(Span(video, video + cfg.celebration, t_win, t_win, 0.0, "celebration"))
    celebration_video = video
    return Timeline(
        spans=tuple(spans),
        t_win=t_win,
        winner_video=winner_video,
        replay_video=replay_video,
        celebration_video=celebration_video,
        duration=spans[-1].video1 if spans else 0.0,
        hitstop=sum(item[2] for item in holds),
    )


def sample_timeline(timeline: Timeline, video_t: float) -> tuple[float, str]:
    moment = min(max(0.0, video_t), max(0.0, timeline.duration - 1e-6))
    chosen = timeline.spans[-1]
    for span in timeline.spans:
        if span.video0 - 1e-9 <= moment <= span.video1 + 1e-9:
            chosen = span
            break
    if chosen.video1 <= chosen.video0 or chosen.phase in {"celebration", "hold"}:
        return chosen.sim0, chosen.phase
    u = (moment - chosen.video0) / (chosen.video1 - chosen.video0)
    u = min(1.0, max(0.0, u))
    return chosen.sim0 + u * (chosen.sim1 - chosen.sim0), chosen.phase


def video_time(timeline: Timeline, sim_t: float) -> float:
    moment = min(max(0.0, sim_t), timeline.t_win)
    found = timeline.winner_video
    for span in timeline.spans:
        if span.phase in {"replay", "celebration"}:
            continue
        if span.phase == "hold" and abs(span.sim0 - moment) <= 1e-6:
            found = span.video1
            continue
        if span.sim1 > span.sim0 and span.sim0 - 1e-9 <= moment <= span.sim1 + 1e-9:
            u = (moment - span.sim0) / (span.sim1 - span.sim0)
            return span.video0 + u * (span.video1 - span.video0)
    return found


def event_video_times(timeline: Timeline, sim_t: float) -> tuple[float, ...]:
    times = [video_time(timeline, sim_t)]
    for span in timeline.spans:
        if span.phase != "replay" or span.sim1 <= span.sim0:
            continue
        if span.sim0 - 1e-9 <= sim_t <= span.sim1 + 1e-9:
            u = (sim_t - span.sim0) / (span.sim1 - span.sim0)
            times.append(span.video0 + u * (span.video1 - span.video0))
    return tuple(times)


def _spring(zoom: float, vel: float, target: float, dt: float, smooth: float) -> tuple[float, float]:
    wn = 4.0 / max(smooth, 1e-3)
    acc = wn * wn * (target - zoom) - 2.0 * wn * vel
    vel += acc * dt
    zoom += vel * dt
    return zoom, vel


def camera_view(cfg: ArenaConfig, time: float, hw: float, hh: float, *, shake: float = 0.0, punch: float = 1.0) -> CameraView:
    target = min(cfg.screen_hw / max(hw, 1.0), cfg.screen_hh / max(hh, 1.0))
    if time < cfg.punch_seconds:
        punch = cfg.punch_from + (cfg.punch_to - cfg.punch_from) * (time / cfg.punch_seconds)
    zoom = min(target * punch, cfg.screen_hw / max(hw, 1.0), cfg.screen_hh / max(hh, 1.0))
    left = 540.0 - hw * zoom
    right = 540.0 + hw * zoom
    top = 940.0 - hh * zoom
    bottom = 940.0 + hh * zoom
    return CameraView(zoom, 540.0 + shake, 940.0, shake, 0.0, left, top, right, bottom)


def _fill_round_rect(frame: np.ndarray, left: int, top: int, right: int, bottom: int, radius: int, color: tuple[int, int, int]) -> None:
    radius = int(max(1, min(radius, (right - left) // 2, (bottom - top) // 2)))
    cv2.rectangle(frame, (left + radius, top), (right - radius, bottom), color, -1, lineType=cv2.LINE_AA)
    cv2.rectangle(frame, (left, top + radius), (right, bottom - radius), color, -1, lineType=cv2.LINE_AA)
    for cx, cy in ((left + radius, top + radius), (right - radius, top + radius), (left + radius, bottom - radius), (right - radius, bottom - radius)):
        cv2.circle(frame, (cx, cy), radius, color, -1, lineType=cv2.LINE_AA)


def _stroke_round_rect(frame: np.ndarray, left: int, top: int, right: int, bottom: int, radius: int, color: tuple[int, int, int]) -> None:
    radius = int(max(1, min(radius, (right - left) // 2, (bottom - top) // 2)))
    cv2.line(frame, (left + radius, top), (right - radius, top), color, 2, lineType=cv2.LINE_AA)
    cv2.line(frame, (left + radius, bottom), (right - radius, bottom), color, 2, lineType=cv2.LINE_AA)
    cv2.line(frame, (left, top + radius), (left, bottom - radius), color, 2, lineType=cv2.LINE_AA)
    cv2.line(frame, (right, top + radius), (right, bottom - radius), color, 2, lineType=cv2.LINE_AA)
    cv2.ellipse(frame, (left + radius, top + radius), (radius, radius), 180, 0, 90, color, 2, lineType=cv2.LINE_AA)
    cv2.ellipse(frame, (right - radius, top + radius), (radius, radius), 270, 0, 90, color, 2, lineType=cv2.LINE_AA)
    cv2.ellipse(frame, (left + radius, bottom - radius), (radius, radius), 90, 0, 90, color, 2, lineType=cv2.LINE_AA)
    cv2.ellipse(frame, (right - radius, bottom - radius), (radius, radius), 0, 0, 90, color, 2, lineType=cv2.LINE_AA)


def world_to_screen(cfg: ArenaConfig, x: float, y: float, hw: float, hh: float, zoom: float) -> tuple[float, float]:
    cx, cy = cfg.center
    sx = 540.0 + (x - cx) * zoom
    sy = 940.0 + (y - cy) * zoom
    return sx, sy


def active_kill_lines(elims: tuple[Elim, ...], timeline: Timeline, video_t: float, fade: float, limit: int) -> tuple[Elim, ...]:
    shown = [item for item in elims if 0.0 <= video_t - video_time(timeline, item.time) <= fade]
    return tuple(shown[-limit:])


def layout_boxes(cfg: ArenaConfig, video_t: float, *, alive: int, phase: str) -> tuple[TextBox, ...]:
    boxes: list[TextBox] = []
    boxes.append(TextBox("alive", 390, 205, 690, 305))
    show_hook = video_t <= 3.0 or phase in {"final", "wait"}
    if show_hook:
        boxes.append(TextBox("hook", 180, 342, 900, 418))
    if 3.0 <= video_t <= 5.5:
        boxes.append(TextBox("sub", 200, 423, 880, 457))
    boxes.append(TextBox("kill0", 150, 1422, 900, 1458))
    boxes.append(TextBox("kill1", 150, 1468, 900, 1504))
    return tuple(boxes)


def boxes_overlap(a: TextBox, b: TextBox) -> bool:
    return a.x0 < b.x1 and b.x0 < a.x1 and a.y0 < b.y1 and b.y0 < a.y1


def layout_clear(boxes: tuple[TextBox, ...]) -> bool:
    plat = TextBox("platform", *PLATFORM_BOX)
    for index, box in enumerate(boxes):
        if boxes_overlap(box, plat):
            return False
        for other in boxes[index + 1 :]:
            if boxes_overlap(box, other):
                return False
        if box.x0 < SAFE[0] or box.x1 > SAFE[1] or box.y0 < SAFE[2] or box.y1 > SAFE[3]:
            return False
    return True


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _circle_sprite(path: Path, size: int = 128) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        image = np.zeros((size, size, 4), dtype=np.uint8)
        image[..., :3] = (40, 40, 40)
        image[..., 3] = 255
    else:
        image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    yy, xx = np.ogrid[:size, :size]
    mask = (xx - size / 2) ** 2 + (yy - size / 2) ** 2 <= (size / 2 - 1) ** 2
    image[..., 3] = np.where(mask, image[..., 3], 0)
    return image


class ArenaRenderer:
    def __init__(self, cfg: ArenaConfig, result: SimResult, *, preview: bool = False, safe_overlay: bool = False, reel: bool = False) -> None:
        if result.trace is None:
            raise RuntimeError("arena render needs a recorded sim trace")
        self.cfg = cfg
        self.result = result
        self.preview = preview
        self.reel = reel
        self.safe_overlay = safe_overlay
        if reel:
            self.width, self.height, self.fps = 360, 640, 15
            self.scale = 360 / cfg.width
        else:
            self.scale = 0.5 if preview else 1.0
            self.width = int(round(cfg.width * self.scale))
            self.height = int(round(cfg.height * self.scale))
            self.fps = 30 if preview else cfg.fps
        self.timeline = build_timeline(result, cfg)
        self.n_frames = max(1, int(round(self.timeline.duration * self.fps)))
        self.font = find_font()
        self.boxes: list[tuple[int, int, int, int]] = []
        self.hud_boxes: tuple[TextBox, ...] = ()
        self._trace = result.trace
        root = _repo_root()
        self._sprites: list[np.ndarray] = []
        self._colors: list[tuple] = []
        if not reel:
            for country in cfg.countries:
                path = root / "assets" / "flags" / "png" / f"{country.iso2.lower()}.png"
                sprite = _circle_sprite(path)
                self._sprites.append(sprite)
                self._colors.append(dominant_colors(sprite[..., :3][..., ::-1], k=3, seed=cfg.seed))
            self._cameo = _circle_sprite(root / "assets" / "cameos" / "ohio.png")
        self._zoom = 1.0
        self._zvel = 0.0

    def frame_at(self, video_t: float) -> int:
        return int(min(self.n_frames - 1, max(0, round(video_t * self.fps))))

    def render(self, index: int) -> np.ndarray:
        index = int(min(self.n_frames - 1, max(0, index)))
        video_t = index / self.fps
        sim_t, phase = sample_timeline(self.timeline, video_t)
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        frame[:] = (28, 22, 16)
        step = max(1, int(round(sim_t / self.cfg.dt)))
        step = min(step, len(self._trace["xy"]) - 1)
        hw = float(self._trace["hw"][step])
        hh = float(self._trace["hh"][step])
        self._zoom, self._zvel = _spring(self._zoom, self._zvel, 1.0, 1.0 / self.fps, self.cfg.camera_smooth)
        cam = camera_view(self.cfg, sim_t, hw, hh)
        self._draw_platform(frame, cam, hw, hh, sim_t)
        self._draw_balls(frame, step, cam, phase, sim_t)
        self._draw_hud(frame, video_t, sim_t, phase)
        if self.safe_overlay:
            x0, x1, y0, y1 = (int(v * self.scale) for v in (SAFE[0], SAFE[1], SAFE[2], SAFE[3]))
            cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 255), 1)
        return frame

    def profile_ms(self) -> float:
        index = self.frame_at(self.timeline.winner_video)
        started = time.perf_counter()
        self.render(index)
        return (time.perf_counter() - started) * 1000.0

    def _s(self, x: float, y: float, zoom: float) -> tuple[int, int]:
        sx, sy = world_to_screen(self.cfg, x, y, 1.0, 1.0, zoom)
        return int(round(sx * self.scale)), int(round(sy * self.scale))

    def _draw_platform(self, frame: np.ndarray, cam: CameraView, hw: float, hh: float, sim_t: float) -> None:
        cx, cy = self.cfg.center
        zoom = cam.zoom
        cr = max(2, int(round(0.35 * min(hw, hh) * zoom * self.scale)))
        left, top = self._s(cx - hw, cy - hh, zoom)
        right, bottom = self._s(cx + hw, cy + hh, zoom)
        color = (70, 58, 42)
        rim = (180, 140, 60)
        if sim_t >= 6.0 and int(sim_t * 4) % 2 == 0:
            color = (48, 36, 70)
            rim = (40, 40, 180)
        _fill_round_rect(frame, left, top, right, bottom, cr, color)
        _stroke_round_rect(frame, left, top, right, bottom, cr, rim)

    def _draw_balls(self, frame: np.ndarray, step: int, cam: CameraView, phase: str, sim_t: float) -> None:
        xy = self._trace["xy"][step]
        alive = self._trace["alive"][step]
        phases = self._trace["phase"][step]
        facing = self._trace["facing"][step]
        zoom = cam.zoom
        radius = max(2, int(round(self.cfg.ball_radius * zoom * self.scale)))
        gone = {item.index: item.time for item in self.result.elims}
        for index, country in enumerate(self.cfg.countries):
            shrink = 1.0
            if not alive[index]:
                when = gone.get(index, sim_t)
                age = sim_t - when
                if phase == "celebration" or (not self.reel and age > 0.9):
                    continue
                shrink = max(0.0, 1.0 - max(0.0, age) / 0.9)
            sx, sy = self._s(float(xy[index, 0]), float(xy[index, 1]), zoom)
            draw_r = max(1, int(round(radius * shrink)))
            if self.reel:
                color = (180, 220, 255) if alive[index] else (80, 80, 120)
                cv2.circle(frame, (sx, sy), draw_r, color, 1, lineType=cv2.LINE_AA)
                cv2.putText(frame, country.iso3, (sx - 14, sy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (240, 240, 240), 1, cv2.LINE_AA)
                if step > 0:
                    prev = self._trace["xy"][step - 1, index]
                    gain = 0.15 / self.cfg.dt
                    ex, ey = self._s(
                        float(xy[index, 0]) + (float(xy[index, 0]) - float(prev[0])) * gain,
                        float(xy[index, 1]) + (float(xy[index, 1]) - float(prev[1])) * gain,
                        zoom,
                    )
                    cv2.line(frame, (sx, sy), (ex, ey), (120, 180, 255), 1, cv2.LINE_AA)
                if phases[index] == 1:
                    length = int(40 * self.scale * zoom)
                    ex = int(sx + math.cos(float(facing[index])) * length)
                    ey = int(sy + math.sin(float(facing[index])) * length)
                    cv2.line(frame, (sx, sy), (ex, ey), (80, 220, 255), 1, cv2.LINE_AA)
            elif index < len(self._sprites):
                self._blit(frame, self._sprites[index], sx, sy, draw_r * 2)
                self._eyes(frame, sx, sy, draw_r, float(facing[index]), angry=phases[index] == 1)
                if phases[index] == 1 and alive[index]:
                    length = int(draw_r * 3)
                    ex = int(sx + math.cos(float(facing[index])) * length)
                    ey = int(sy + math.sin(float(facing[index])) * length)
                    cv2.line(frame, (sx, sy), (ex, ey), (80, 220, 255), 1, cv2.LINE_AA)

    def _blit(self, frame: np.ndarray, sprite: np.ndarray, sx: int, sy: int, size: int) -> None:
        size = max(2, size)
        image = cv2.resize(sprite, (size, size), interpolation=cv2.INTER_LINEAR)
        x0 = sx - size // 2
        y0 = sy - size // 2
        fh, fw = frame.shape[:2]
        xa, xb = max(0, x0), min(fw, x0 + size)
        ya, yb = max(0, y0), min(fh, y0 + size)
        if xa >= xb or ya >= yb:
            return
        crop = image[ya - y0 : yb - y0, xa - x0 : xb - x0]
        alpha = crop[..., 3:4].astype(np.float32) / 255.0
        frame[ya:yb, xa:xb] = (frame[ya:yb, xa:xb] * (1.0 - alpha) + crop[..., :3] * alpha).astype(np.uint8)

    def _eyes(self, frame: np.ndarray, sx: int, sy: int, radius: int, facing: float, *, angry: bool) -> None:
        spread = max(2, int(radius * 0.38))
        eye_r = max(1, int(radius * (0.16 if angry else 0.22)))
        for sign in (-1, 1):
            ex = int(sx + math.cos(facing) * radius * 0.15 + sign * spread * 0.5)
            ey = int(sy - radius * 0.15)
            cv2.circle(frame, (ex, ey), eye_r, (255, 255, 255), -1, lineType=cv2.LINE_AA)
            cv2.circle(frame, (ex, ey), max(1, eye_r // 2), (20, 20, 20), -1, lineType=cv2.LINE_AA)

    def _fit_text(self, text: str, slot: TextBox, thickness: int) -> tuple[TextBox, int, int, float]:
        """Largest Hershey size that stays inside the reserved slot, centered."""
        span_w = max(1.0, slot.x1 - slot.x0)
        span_h = max(1.0, slot.y1 - slot.y0)
        font_scale = 0.4
        for _ in range(12):
            (width, height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale * self.scale, thickness)
            full_w = width / max(self.scale, 1e-6)
            full_h = (height + baseline) / max(self.scale, 1e-6)
            if full_w <= span_w * 0.96 and full_h <= span_h * 0.96:
                font_scale *= 1.25
            else:
                font_scale /= 1.25
                break
        font_scale = max(0.35, font_scale)
        (width, height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale * self.scale, thickness)
        x = int(round(((slot.x0 + slot.x1) * 0.5) * self.scale - width / 2))
        y = int(round(((slot.y0 + slot.y1) * 0.5) * self.scale + height / 2))
        return self._glyph_box(text, x, y, font_scale * self.scale, thickness), x, y, font_scale * self.scale

    def _glyph_box(self, text: str, x: int, y: int, font_scale: float, thickness: int) -> TextBox:
        (width, height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        scale = max(self.scale, 1e-6)
        return TextBox(text, x / scale, (y - height) / scale, (x + width) / scale, (y + baseline) / scale)

    def _draw_label(self, frame: np.ndarray, text: str, slot: TextBox, thickness: int, color: tuple[int, int, int]) -> TextBox:
        box, x, y, font_scale = self._fit_text(text, slot, thickness)
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
        return box

    def _draw_hud(self, frame: np.ndarray, video_t: float, sim_t: float, phase: str) -> None:
        alive = sum(1 for item in self.result.elims if item.time > sim_t)
        alive = len(self.cfg.countries) - (len(self.result.elims) - alive)
        slots = layout_boxes(self.cfg, video_t, alive=alive, phase=phase)
        boxes: list[TextBox] = []
        by_name = {slot.name: slot for slot in slots}
        if "alive" in by_name:
            boxes.append(self._draw_label(frame, f"ALIVE {alive}", by_name["alive"], 2, (240, 240, 240)))
        if "hook" in by_name:
            hook = self.cfg.hook[: self.cfg.hook_max_chars]
            if phase == "final":
                hook = "FINAL 2" if alive <= 2 else "FINAL 3"
            elif phase == "wait":
                hook = "wait for it..."
            boxes.append(self._draw_label(frame, hook, by_name["hook"], 2, (230, 230, 230)))
        if "sub" in by_name:
            boxes.append(self._draw_label(frame, self.cfg.sub_text, by_name["sub"], 1, (200, 200, 200)))
        lines = active_kill_lines(self.result.elims, self.timeline, video_t, self.cfg.kill_fade, 2)
        for row, item in enumerate(lines):
            slot = by_name.get(f"kill{row}")
            if slot is None:
                continue
            killer = self.cfg.countries[item.killer].iso3 if item.killer is not None and item.killer >= 0 else item.cause.upper()
            victim = self.cfg.countries[item.index].iso3
            boxes.append(self._draw_label(frame, f"{killer} > {victim}"[:34], slot, 1, (240, 240, 240)))
        if phase == "celebration" and self.result.winner is not None:
            name = self.cfg.countries[self.result.winner].name.upper()
            slot = TextBox("winner", 180, 342, 900, 418)
            boxes.append(self._draw_label(frame, f"{name} WINS"[:24], slot, 2, (250, 250, 250)))
        if phase == "replay":
            slot = TextBox("replay", 180, 342, 900, 418)
            boxes.append(self._draw_label(frame, "REPLAY", slot, 2, (250, 220, 160)))
        self.hud_boxes = tuple(boxes)


def contact_sheet(renderer: ArenaRenderer, path: Path) -> None:
    times = [0.5, 2.0, renderer.timeline.t_win * 0.5, renderer.timeline.winner_video, renderer.timeline.celebration_video + 0.4]
    tiles = [renderer.render(renderer.frame_at(moment)) for moment in times]
    sheet = np.concatenate(tiles, axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), sheet)


def write_ko_strips(renderer: ArenaRenderer, directory: Path) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    moments = [item.time for item in renderer.result.elims[:3]]
    if renderer.result.t_win is not None:
        moments.append(renderer.result.t_win)
    paths = []
    for index, moment in enumerate(moments):
        tiles = []
        for step in range(8):
            sim_t = moment + (step - 3) * 0.1
            video_t = video_time(renderer.timeline, max(0.0, sim_t))
            tiles.append(renderer.render(renderer.frame_at(video_t)))
        sheet = np.concatenate(tiles, axis=1)
        path = directory / f"ko_{index}.png"
        cv2.imwrite(str(path), sheet)
        paths.append(path)
    return paths


def write_sim_cache(path: Path, result: SimResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        elims=np.array([(item.time, item.index, item.closing) for item in result.elims], dtype=np.float64),
    )


def read_sim_cache(path: Path) -> SimResult:
    raise RuntimeError("v2 renders from the live sim, not the v1 cache")


def init_arena_worker(cfg: ArenaConfig, cache_path: str, preview: bool, safe_overlay: bool) -> None:
    del cfg, cache_path, preview, safe_overlay


def render_arena_chunk(indices: list[int]) -> list[np.ndarray]:
    del indices
    return []
