"""Flag Arena picture: time remap, camera, and BGR frames.

Video time is a monotone piecewise-linear map of sim time. Slow motion is
0.35x on the 4-to-3 and 3-to-2 eliminations. Replay and celebration are
extra spans after the winner. The camera keeps a steady zone at 400 px
when shake is forced off.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from fc_sat.arena_config import ArenaConfig, render_phrase
from fc_sat.arena_sim import (
    Elim,
    Meteor,
    SimResult,
    alive_at,
    sweeper_angle,
    zone_radius,
)
from fc_sat.color import dominant_colors
from fc_sat.visual import apply_bloom, composite_rgba, find_font, raster_text, vignette_bgr

_WORKER: "ArenaRenderer | None" = None


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


@dataclass(frozen=True)
class CameraView:
    zoom: float
    screen_radius: float
    center_x: float
    center_y: float
    shake_x: float
    shake_y: float

    def circle_bounds(self) -> tuple[float, float, float, float]:
        left = self.center_x + self.shake_x - self.screen_radius
        right = self.center_x + self.shake_x + self.screen_radius
        top = self.center_y + self.shake_y - self.screen_radius
        bottom = self.center_y + self.shake_y + self.screen_radius
        return left, top, right, bottom


def slow_windows(result: SimResult, cfg: ArenaConfig, t_win: float) -> tuple[tuple[float, float], ...]:
    """Sim intervals around the eliminations that leave 3 and then 2 alive."""
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


def build_timeline(result: SimResult, cfg: ArenaConfig) -> Timeline:
    t_win = result.t_win
    if t_win is None:
        if result.trace is not None and len(result.trace["xy"]):
            t_win = (len(result.trace["xy"]) - 1) * cfg.dt
        else:
            t_win = cfg.win_window[0]
    t_win = max(0.05, float(t_win))
    windows = slow_windows(result, cfg, t_win)
    cuts = {0.0, t_win}
    for start, end in windows:
        cuts.add(start)
        cuts.add(end)
    ordered = [point for point in sorted(cuts) if 0.0 <= point <= t_win]
    spans: list[Span] = []
    video = 0.0
    for start, end in zip(ordered, ordered[1:]):
        if end <= start:
            continue
        slow = any(window[0] - 1e-9 <= start and end <= window[1] + 1e-9 for window in windows)
        rate = cfg.slowmo_rate if slow else 1.0
        length = (end - start) / rate
        spans.append(Span(video, video + length, start, end, rate, "slow" if slow else "main"))
        video += length
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
        duration=video + cfg.celebration,
    )


def sample_timeline(timeline: Timeline, video_t: float) -> tuple[float, str]:
    moment = min(max(0.0, video_t), timeline.duration)
    chosen = timeline.spans[-1]
    for span in timeline.spans:
        if span.video0 - 1e-9 <= moment <= span.video1 + 1e-9:
            chosen = span
            break
    if chosen.phase == "celebration" or chosen.video1 <= chosen.video0:
        return chosen.sim0, chosen.phase
    u = (moment - chosen.video0) / (chosen.video1 - chosen.video0)
    u = min(1.0, max(0.0, u))
    return chosen.sim0 + u * (chosen.sim1 - chosen.sim0), chosen.phase


def video_time(timeline: Timeline, sim_t: float) -> float:
    """Main-phase video time. Monotone in sim time."""
    moment = min(max(0.0, sim_t), timeline.t_win)
    for span in timeline.spans:
        if span.phase == "replay" or span.phase == "celebration":
            continue
        if span.sim0 - 1e-9 <= moment <= span.sim1 + 1e-9:
            if span.sim1 <= span.sim0:
                return span.video0
            u = (moment - span.sim0) / (span.sim1 - span.sim0)
            return span.video0 + u * (span.video1 - span.video0)
    return timeline.winner_video


def event_video_times(timeline: Timeline, sim_t: float) -> tuple[float, ...]:
    """Video times where a sim event is heard: the main map, plus replay."""
    times = [video_time(timeline, sim_t)]
    for span in timeline.spans:
        if span.phase != "replay":
            continue
        if span.sim0 - 1e-9 <= sim_t <= span.sim1 + 1e-9 and span.sim1 > span.sim0:
            u = (sim_t - span.sim0) / (span.sim1 - span.sim0)
            times.append(span.video0 + u * (span.video1 - span.video0))
    return tuple(times)


def _spring_step(zoom: float, vel: float, target: float, dt: float, wn: float) -> tuple[float, float]:
    acc = wn * wn * (target - zoom) - 2.0 * wn * vel
    vel = vel + acc * dt
    zoom = zoom + vel * dt
    return zoom, vel


def camera_view(cfg: ArenaConfig, time: float, *, shake: float = 0.0) -> CameraView:
    """Critically damped zoom. ``shake`` is the offset amplitude in pixels."""
    dt = 1.0 / 240.0
    steps = max(1, int(math.ceil(max(time, 0.0) / dt)))
    wn = 5.8 / cfg.camera_smooth
    zone0 = zone_radius(cfg, 0.0)
    punch0 = cfg.punch_from
    zoom = punch0 * cfg.screen_radius / zone0
    vel = 0.0
    zone = zone0
    for step in range(steps):
        now = min(time, (step + 1) * dt)
        zone = zone_radius(cfg, now)
        punch = cfg.punch_to
        if now < cfg.punch_seconds and cfg.punch_seconds > 0:
            u = now / cfg.punch_seconds
            punch = cfg.punch_from + (cfg.punch_to - cfg.punch_from) * u
        target = punch * cfg.screen_radius / max(zone, 1.0)
        zoom, vel = _spring_step(zoom, vel, target, dt, wn)
        if zoom > target:
            zoom = target
            vel = 0.0
    shake_x = 0.0
    shake_y = 0.0
    if shake:
        shake_x = shake * math.sin(2.0 * math.pi * cfg.shake_hz * time)
        shake_y = shake * math.cos(2.0 * math.pi * cfg.shake_hz * time)
    return CameraView(
        zoom=zoom,
        screen_radius=zone * zoom,
        center_x=cfg.center[0],
        center_y=cfg.center[1],
        shake_x=shake_x,
        shake_y=shake_y,
    )


def active_kill_lines(elims: tuple[Elim, ...], timeline: Timeline, video_t: float, fade: float, limit: int) -> tuple[Elim, ...]:
    shown = []
    for item in elims:
        age = video_t - video_time(timeline, item.time)
        if 0.0 <= age <= fade:
            shown.append(item)
    return tuple(shown[-limit:])


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def write_sim_cache(path: Path, result: SimResult) -> None:
    if result.trace is None:
        raise RuntimeError("arena render cache needs a recorded trace")
    path.parent.mkdir(parents=True, exist_ok=True)
    trace = result.trace
    np.savez_compressed(
        path,
        xy=trace["xy"],
        angle=trace["angle"],
        alive=trace["alive"],
        cameo_xy=trace["cameo_xy"],
        cameo_alive=trace["cameo_alive"],
        elim_time=np.array([item.time for item in result.elims], dtype=np.float64),
        elim_index=np.array([item.index for item in result.elims], dtype=np.int32),
        elim_cause=np.array([item.cause for item in result.elims], dtype="U16"),
        meteor_time=np.array([item.time for item in result.meteors], dtype=np.float64),
        meteor_x=np.array([item.x for item in result.meteors], dtype=np.float64),
        meteor_y=np.array([item.y for item in result.meteors], dtype=np.float64),
        t_win=np.float64(np.nan if result.t_win is None else result.t_win),
        winner=np.int32(-1 if result.winner is None else result.winner),
        duel_start=np.float64(np.nan if result.duel_start is None else result.duel_start),
        cameo_exit=np.float64(np.nan if result.cameo_exit is None else result.cameo_exit),
        seed=np.int32(result.seed),
        backend=np.array(result.backend),
    )


def read_sim_cache(path: Path) -> SimResult:
    data = np.load(path, allow_pickle=False)
    t_win = float(data["t_win"])
    winner = int(data["winner"])
    duel = float(data["duel_start"])
    cameo_exit = float(data["cameo_exit"])
    causes = [str(item) for item in data["elim_cause"]]
    elims = tuple(
        Elim(float(when), int(index), cause)
        for when, index, cause in zip(data["elim_time"], data["elim_index"], causes)
    )
    meteors = tuple(
        Meteor(float(when), float(x), float(y))
        for when, x, y in zip(data["meteor_time"], data["meteor_x"], data["meteor_y"])
    )
    return SimResult(
        seed=int(data["seed"]),
        backend=str(data["backend"]),
        elims=elims,
        meteors=meteors,
        near_misses=(),
        t_win=None if math.isnan(t_win) else t_win,
        winner=None if winner < 0 else winner,
        duel_start=None if math.isnan(duel) else duel,
        cameo_exit=None if math.isnan(cameo_exit) else cameo_exit,
        trace={
            "xy": data["xy"],
            "angle": data["angle"],
            "alive": data["alive"],
            "cameo_xy": data["cameo_xy"],
            "cameo_alive": data["cameo_alive"],
        },
    )


def _circle_sprite(path: Path, size: int = 256) -> np.ndarray:
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
    rgb = image[..., 2::-1]
    alpha = image[..., 3]
    yy, xx = np.ogrid[:size, :size]
    mask = (xx - size / 2) ** 2 + (yy - size / 2) ** 2 <= (size / 2 - 1) ** 2
    alpha = np.where(mask, alpha, 0).astype(np.uint8)
    return np.dstack([rgb, alpha])


def _lerp_color(a: tuple[int, int, int], b: tuple[int, int, int], u: float) -> tuple[int, int, int]:
    u = min(1.0, max(0.0, u))
    return tuple(int(a[i] + (b[i] - a[i]) * u) for i in range(3))


class ArenaRenderer:
    def __init__(self, cfg: ArenaConfig, result: SimResult, *, preview: bool = False, safe_overlay: bool = False) -> None:
        if result.trace is None:
            raise RuntimeError("arena render needs a recorded sim trace")
        self.cfg = cfg
        self.result = result
        self.preview = preview
        self.safe_overlay = safe_overlay
        self.scale = 0.5 if preview else 1.0
        self.width = int(round(cfg.width * self.scale))
        self.height = int(round(cfg.height * self.scale))
        self.fps = 30 if preview else cfg.fps
        self.timeline = build_timeline(result, cfg)
        self.n_frames = max(1, int(round(self.timeline.duration * self.fps)))
        self.font = find_font()
        self.boxes: list[tuple[int, int, int, int]] = []
        self._text_cache: dict[tuple, np.ndarray] = {}
        self._trace = result.trace
        self._dt = cfg.dt
        self._elim_at = {item.index: item for item in result.elims}
        root = _repo_root()
        self._sprites = []
        self._colors = []
        for country in cfg.countries:
            path = root / "assets" / "flags" / "png" / f"{country.iso2.lower()}.png"
            sprite = _circle_sprite(path)
            self._sprites.append(sprite)
            self._colors.append(dominant_colors(sprite[..., :3], k=3, seed=cfg.seed + len(self._sprites)))
        cameo_path = root / "assets" / "cameos" / "ohio.png"
        self._cameo_sprite = _circle_sprite(cameo_path)
        self._background = vignette_bgr(self.width, self.height)
        self._hit_times = self._impact_times()
        self._confetti = self._build_confetti()
        self._cameras = self._build_cameras()

    def frame_at(self, video_t: float) -> int:
        return int(min(self.n_frames - 1, max(0, round(video_t * self.fps))))

    def render(self, index: int) -> np.ndarray:
        index = int(min(self.n_frames - 1, max(0, index)))
        video_t = index / self.fps
        sim_t, phase = sample_timeline(self.timeline, video_t)
        cam = self._cameras[index]
        self.boxes = []
        frame = self._background.copy()
        self._draw_grid(frame, video_t)
        self._draw_floor(frame, sim_t, cam, phase)
        xy, ang, alive = self._sample(sim_t)
        if phase == "celebration":
            self._draw_celebration(frame, video_t, cam)
        else:
            self._draw_balls(frame, sim_t, video_t, xy, ang, alive, cam, phase)
            self._draw_cameo(frame, sim_t, cam)
        self._draw_hud(frame, sim_t, video_t, phase)
        signal = frame.astype(np.float32)
        apply_bloom(signal, self.cfg.bloom_strength)
        frame = np.clip(signal, 0, 255).astype(np.uint8)
        if self.safe_overlay:
            self._draw_safe(frame)
        return frame

    def profile_ms(self) -> float:
        index = self.frame_at(self.timeline.winner_video)
        started = time.perf_counter()
        self.render(index)
        return (time.perf_counter() - started) * 1000.0

    def _impact_times(self) -> list[np.ndarray]:
        xy = self._trace["xy"]
        if len(xy) < 3:
            return [np.zeros(0) for _ in self.cfg.countries]
        vel = np.diff(xy, axis=0) / self._dt
        speed = np.linalg.norm(vel, axis=2)
        drop = np.zeros_like(speed)
        drop[1:] = speed[:-1] - speed[1:]
        hit = (drop > 180.0) & (speed > self.cfg.spark_speed)
        times = []
        for ball in range(xy.shape[1]):
            times.append(np.flatnonzero(hit[:, ball]).astype(np.float64) * self._dt)
        return times

    def _build_confetti(self) -> np.ndarray:
        rng = np.random.Generator(np.random.PCG64(self.cfg.seed + 17))
        count = self.cfg.confetti
        winner = 0 if self.result.winner is None else self.result.winner
        palette = self._colors[winner]
        colors = palette[rng.integers(0, len(palette), size=count)]
        angle = rng.random(count) * math.tau
        speed = rng.uniform(80.0, 420.0, size=count)
        spin = rng.uniform(-6.0, 6.0, size=count)
        delay = rng.uniform(0.0, 0.35, size=count)
        return np.column_stack([angle, speed, spin, delay, colors])

    def _build_cameras(self) -> list[CameraView]:
        cfg = self.cfg
        dt = 1.0 / self.fps
        wn = 5.8 / cfg.camera_smooth
        zone0 = zone_radius(cfg, 0.0)
        zoom = cfg.punch_from * cfg.screen_radius / zone0
        vel = 0.0
        views: list[CameraView] = []
        shake_events = [video_time(self.timeline, item.time) for item in self.result.meteors]
        shake_events.extend(video_time(self.timeline, item.time) for item in self.result.elims)
        duel = self._duelists()
        for index in range(self.n_frames):
            video_t = index * dt
            sim_t, phase = sample_timeline(self.timeline, video_t)
            zone = zone_radius(cfg, sim_t)
            punch = cfg.punch_to
            if phase == "main" or phase == "slow":
                if video_t < cfg.punch_seconds and cfg.punch_seconds > 0:
                    u = video_t / cfg.punch_seconds
                    punch = cfg.punch_from + (cfg.punch_to - cfg.punch_from) * u
            extra = cfg.replay_zoom if phase == "replay" else 1.0
            target = punch * extra * cfg.screen_radius / max(zone, 1.0)
            zoom, vel = _spring_step(zoom, vel, target, dt, wn)
            if zoom > target:
                zoom = target
                vel = 0.0
            look_x, look_y = cfg.center
            if phase == "replay" and duel is not None:
                xy, _, _ = self._sample(sim_t)
                look_x = float(xy[duel[0], 0] + xy[duel[1], 0]) * 0.5
                look_y = float(xy[duel[0], 1] + xy[duel[1], 1]) * 0.5
            shake = 0.0
            for when in shake_events:
                age = video_t - when
                if 0.0 <= age <= cfg.shake_decay * 3:
                    shake += cfg.shake_px * math.exp(-age / cfg.shake_decay)
            shake = min(cfg.shake_px, shake)
            views.append(
                CameraView(
                    zoom=zoom,
                    screen_radius=zone * zoom / max(extra, 1e-6),
                    center_x=cfg.center[0],
                    center_y=cfg.center[1],
                    shake_x=shake * math.sin(2.0 * math.pi * cfg.shake_hz * video_t),
                    shake_y=shake * math.cos(2.0 * math.pi * cfg.shake_hz * video_t),
                )
            )
            views[-1] = CameraView(
                zoom=views[-1].zoom,
                screen_radius=views[-1].screen_radius,
                center_x=look_x,
                center_y=look_y,
                shake_x=views[-1].shake_x,
                shake_y=views[-1].shake_y,
            )
        return views

    def _duelists(self) -> tuple[int, int] | None:
        if self.result.winner is None:
            return None
        other = self.result.elims[-1].index if self.result.elims else self.result.winner
        return self.result.winner, other

    def _sample(self, sim_t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        xy = self._trace["xy"]
        n = len(xy)
        pos = min(max(sim_t / self._dt, 0.0), n - 1)
        i0 = int(math.floor(pos))
        i1 = min(i0 + 1, n - 1)
        u = pos - i0
        mixed = (1.0 - u) * xy[i0] + u * xy[i1]
        a0 = self._trace["angle"][i0]
        a1 = self._trace["angle"][i1]
        da = (a1 - a0 + np.pi) % (2.0 * np.pi) - np.pi
        alive = self._trace["alive"][i0]
        return mixed, a0 + da * u, alive

    def _world(self, x: float, y: float, cam: CameraView) -> tuple[float, float]:
        sx = self.width * 0.5 + (x - cam.center_x) * cam.zoom * self.scale + cam.shake_x * self.scale
        sy = self.height * 0.5 + (y - cam.center_y) * cam.zoom * self.scale + cam.shake_y * self.scale
        return sx, sy

    def _draw_grid(self, frame: np.ndarray, video_t: float) -> None:
        step = max(8, int(80 * self.scale))
        offset = int((video_t * 18) % step)
        color = (28, 36, 48)
        for x in range(-step + offset, self.width, step):
            cv2.line(frame, (x, 0), (x, self.height), color, 1, cv2.LINE_AA)
        for y in range(-step + offset, self.height, step):
            cv2.line(frame, (0, y), (self.width, y), color, 1, cv2.LINE_AA)

    def _draw_floor(self, frame: np.ndarray, sim_t: float, cam: CameraView, phase: str) -> None:
        cfg = self.cfg
        zone = zone_radius(cfg, sim_t)
        cx, cy = self._world(cfg.center[0], cfg.center[1], cam)
        radius = max(1, int(round(zone * cam.zoom * self.scale)))
        cv2.circle(frame, (int(cx), int(cy)), radius, (32, 28, 24), -1, cv2.LINE_AA)
        ring = 40.0
        while ring < zone:
            cv2.circle(frame, (int(cx), int(cy)), max(1, int(ring * cam.zoom * self.scale)), (48, 42, 36), 1, cv2.LINE_AA)
            ring += 40.0
        alive = alive_at(self.result.elims, len(cfg.countries), sim_t)
        u = alive / len(cfg.countries)
        cyan = (255, 220, 40)
        magenta = (180, 40, 220)
        red = (40, 40, 220)
        color = _lerp_color(magenta, cyan, (u - 0.5) * 2) if u > 0.5 else _lerp_color(red, magenta, u * 2)
        cv2.circle(frame, (int(cx), int(cy)), radius, color, max(2, int(8 * self.scale)), cv2.LINE_AA)
        upcoming = _upcoming_radius(cfg, sim_t)
        if upcoming is not None and phase != "celebration":
            warn = max(1, int(upcoming * cam.zoom * self.scale))
            cv2.circle(frame, (int(cx), int(cy)), warn, (40, 40, 180), max(1, int(3 * self.scale)), cv2.LINE_AA)
        if sim_t >= cfg.sweeper_start and phase != "celebration":
            angle = sweeper_angle(cfg, sim_t)
            length = max(1.0, zone - cfg.sweeper_inset) * cam.zoom * self.scale
            x1 = int(cx + math.cos(angle) * length)
            y1 = int(cy + math.sin(angle) * length)
            cv2.line(frame, (int(cx), int(cy)), (x1, y1), (210, 230, 255), max(2, int(cfg.sweeper_thickness * cam.zoom * self.scale * 0.35)), cv2.LINE_AA)
        for meteor in self.result.meteors:
            if meteor.time - cfg.meteor_reticle <= sim_t <= meteor.time + 0.35:
                mx, my = self._world(meteor.x, meteor.y, cam)
                arm = int(28 * self.scale)
                cv2.line(frame, (int(mx) - arm, int(my)), (int(mx) + arm, int(my)), (80, 80, 255), 2, cv2.LINE_AA)
                cv2.line(frame, (int(mx), int(my) - arm), (int(mx), int(my) + arm), (80, 80, 255), 2, cv2.LINE_AA)
        if phase == "replay":
            shade = frame.astype(np.float32)
            shade[::4] *= 0.72
            frame[:] = shade.astype(np.uint8)

    def _draw_balls(self, frame, sim_t, video_t, xy, ang, alive, cam, phase) -> None:
        cfg = self.cfg
        order = sorted(range(len(cfg.countries)), key=lambda index: xy[index, 1])
        for index in order:
            self._draw_one_ball(frame, index, sim_t, video_t, xy[index], float(ang[index]), bool(alive[index]), cam, phase)

    def _draw_one_ball(self, frame, index, sim_t, video_t, xy, angle, alive, cam, phase) -> None:
        cfg = self.cfg
        item = self._elim_at.get(index)
        shrink = 1.0
        falling = False
        if not alive:
            if item is None or sim_t > item.time + cfg.fall_seconds:
                return
            falling = True
            shrink = 1.0 - (sim_t - item.time) / cfg.fall_seconds
        zone = zone_radius(cfg, sim_t)
        dist = math.hypot(float(xy[0]) - cfg.center[0], float(xy[1]) - cfg.center[1])
        radius = cfg.ball_radius * shrink * cam.zoom * self.scale
        if radius < 1.0:
            return
        sx, sy = self._world(float(xy[0]), float(xy[1]), cam)
        if falling:
            sy += (1.0 - shrink) * 80.0 * self.scale
        shadow = int(radius)
        cv2.circle(frame, (int(sx), int(sy + cfg.shadow_offset_radii * radius)), max(1, shadow), (12, 10, 10), -1, cv2.LINE_AA)
        side = max(2, int(radius * 2))
        sprite = cv2.resize(self._sprites[index], (side, side), interpolation=cv2.INTER_LINEAR)
        composite_rgba(frame, sprite, int(sx - side / 2), int(sy - side / 2))
        cv2.circle(frame, (int(sx), int(sy)), max(1, int(radius)), (16, 16, 16), max(1, int(3 * self.scale)), cv2.LINE_AA)
        self._draw_eyes(frame, index, sim_t, sx, sy, radius, xy, zone, dist, falling, phase)
        self._draw_name(frame, index, video_t, sim_t, sx, sy, radius, alive)

    def _draw_eyes(self, frame, index, sim_t, sx, sy, radius, xy, zone, dist, falling, phase) -> None:
        cfg = self.cfg
        span = cfg.blink[0] + (cfg.blink[1] - cfg.blink[0]) * ((index * 17 + cfg.seed) % 100) / 100.0
        blinking = (sim_t + index * 0.31) % span < 0.1
        scared = dist > zone - cfg.scared_margin
        hit = False
        times = self._hit_times[index]
        if len(times):
            age = sim_t - times[times <= sim_t][-1:] 
            hit = len(age) > 0 and float(age[0]) <= cfg.hit_seconds
        winner = phase == "celebration" or (self.result.winner == index and self.result.t_win is not None and sim_t >= self.result.t_win)
        eye = radius * cfg.eye_frac
        gap = radius * 0.28
        shift = 4
        scale = 1 << shift
        for side in (-1.0, 1.0):
            ex = int((sx + side * gap) * scale)
            ey = int((sy - radius * 0.12) * scale)
            if blinking and not falling and not winner:
                cv2.line(frame, (ex - int(eye * scale), ey), (ex + int(eye * scale), ey), (255, 255, 255), max(1, int(2 * self.scale)), cv2.LINE_AA, shift)
                continue
            cv2.circle(frame, (ex, ey), max(1, int(eye * scale)), (255, 255, 255), -1, cv2.LINE_AA, shift)
            pupil = eye * (0.35 if scared else 0.5)
            look = 0.0 if falling else math.atan2(float(xy[1]) - cfg.center[1], float(xy[0]) - cfg.center[0])
            if falling:
                look = sim_t * 9.0 + side
            px = ex + int(math.cos(look) * eye * 0.35 * scale)
            py = ey + int(math.sin(look) * eye * 0.35 * scale)
            cv2.circle(frame, (px, py), max(1, int(pupil * scale)), (8, 8, 8), -1, cv2.LINE_AA, shift)
            if hit:
                cv2.line(frame, (ex - int(eye * scale), ey - int(eye * 1.4 * scale)), (ex + int(eye * scale), ey - int(eye * 0.7 * scale)), (8, 8, 8), max(1, int(2 * self.scale)), cv2.LINE_AA, shift)
            if winner:
                cv2.ellipse(frame, (ex, ey - int(eye * 0.2 * scale)), (int(eye * scale), int(eye * 0.7 * scale)), 0, 200, 340, (8, 8, 8), max(1, int(2 * self.scale)), cv2.LINE_AA, shift)

    def _draw_name(self, frame, index, video_t, sim_t, sx, sy, radius, alive) -> None:
        if not alive:
            return
        cfg = self.cfg
        country = cfg.countries[index]
        label = None
        alpha = 1.0
        if video_t <= cfg.name_seconds + cfg.name_fade:
            label = country.iso3
            if video_t > cfg.name_seconds:
                alpha = 1.0 - (video_t - cfg.name_seconds) / cfg.name_fade
        else:
            alive_now = alive_at(self.result.elims, len(cfg.countries), sim_t)
            if alive_now <= cfg.last_named:
                label = country.name
        if not label or alpha <= 0.02:
            return
        self._text(frame, label, cfg.name_px, sx, sy + radius + 4 * self.scale, alpha=alpha, align="center", max_width=int(220 * self.scale))

    def _draw_cameo(self, frame, sim_t, cam) -> None:
        trace = self._trace
        if "cameo_xy" not in trace or sim_t + 1e-6 < self.cfg.cameo_enter:
            return
        n = len(trace["cameo_xy"])
        pos = min(max(sim_t / self._dt, 0.0), n - 1)
        i0 = int(math.floor(pos))
        if not bool(trace["cameo_alive"][i0]):
            return
        i1 = min(i0 + 1, n - 1)
        u = pos - i0
        xy = (1.0 - u) * trace["cameo_xy"][i0] + u * trace["cameo_xy"][i1]
        if float(xy[0]) == 0.0 and float(xy[1]) == 0.0:
            return
        radius = self.cfg.cameo_radius * cam.zoom * self.scale
        sx, sy = self._world(float(xy[0]), float(xy[1]), cam)
        side = max(2, int(radius * 2))
        sprite = cv2.resize(self._cameo_sprite, (side, side), interpolation=cv2.INTER_LINEAR)
        composite_rgba(frame, sprite, int(sx - side / 2), int(sy - side / 2))

    def _draw_celebration(self, frame, video_t, cam) -> None:
        cfg = self.cfg
        local = video_t - self.timeline.celebration_video
        winner = 0 if self.result.winner is None else self.result.winner
        radius = cfg.ball_radius * cfg.winner_scale * (cfg.screen_radius / 190.0) * self.scale
        sx = self.width * 0.5
        sy = self.height * 0.42
        side = max(2, int(radius * 2))
        sprite = cv2.resize(self._sprites[winner], (side, side), interpolation=cv2.INTER_LINEAR)
        composite_rgba(frame, sprite, int(sx - side / 2), int(sy - side / 2))
        self._draw_eyes(frame, winner, cfg.win_window[1], sx, sy, radius, np.array(cfg.center), 130.0, 0.0, False, "celebration")
        name = cfg.countries[winner].name.upper()
        self._text(frame, f"{name} WINS", 86, sx, sy + radius + 16 * self.scale, max_lines=1)
        self._text(frame, "Comment your country", 48, self.width * 0.5, cfg.cta_y * self.scale, max_lines=1)
        self._podium(frame, sy + radius + 150 * self.scale)
        self._confetti_draw(frame, local)
        self._text(frame, f"seed {cfg.seed}", 22, self.width * 0.5, 1500 * self.scale, max_lines=1)

    def _podium(self, frame, y: float) -> None:
        places = list(self.result.placements)
        if len(places) < 2:
            return
        show = places[-3:-1] if len(places) >= 3 else places[:-1]
        for offset, index in zip((-90, 90), show):
            side = max(2, int(54 * self.scale))
            sprite = cv2.resize(self._sprites[index], (side, side), interpolation=cv2.INTER_LINEAR)
            composite_rgba(frame, sprite, int(self.width * 0.5 + offset * self.scale - side / 2), int(y))

    def _confetti_draw(self, frame, local: float) -> None:
        if local < 0:
            return
        rows = self._confetti
        for row in rows[::2]:
            age = local - float(row[3])
            if age < 0:
                continue
            angle = float(row[0]) + float(row[2]) * age
            dist = float(row[1]) * age
            x = int(self.width * 0.5 + math.cos(angle) * dist * self.scale)
            y = int(self.height * 0.42 + math.sin(angle) * dist * self.scale * 0.65 + 40 * age * age)
            if 0 <= x < self.width and 0 <= y < self.height:
                color = (int(row[6]), int(row[5]), int(row[4]))
                cv2.circle(frame, (x, y), max(1, int(3 * self.scale)), color, -1, cv2.LINE_AA)

    def _draw_hud(self, frame, sim_t, video_t, phase) -> None:
        cfg = self.cfg
        alive = alive_at(self.result.elims, len(cfg.countries), sim_t)
        pop = 1.0
        for item in self.result.elims:
            age = video_t - video_time(self.timeline, item.time)
            if 0.0 <= age <= cfg.counter_pop:
                u = age / cfg.counter_pop
                pop = 1.0 + (cfg.counter_pop_scale - 1.0) * math.sin(math.pi * min(1.0, u))
                break
        self._text(frame, str(alive if phase != "celebration" else 1), cfg.counter_px * pop, self.width * 0.5, cfg.counter_y * self.scale, max_lines=1, max_width=int(400 * self.scale))
        if video_t <= cfg.hook_until + cfg.hook_fade:
            alpha = 1.0 if video_t <= cfg.hook_until else 1.0 - (video_t - cfg.hook_until) / cfg.hook_fade
            self._text(frame, cfg.hook, cfg.hook_px, self.width * 0.5, cfg.hook_y * self.scale, alpha=alpha, max_lines=cfg.hook_lines)
        if cfg.fact_window[0] <= video_t <= cfg.fact_window[1]:
            self._text(frame, cfg.fact_text, 32, self.width * 0.5, (cfg.hook_y + 160) * self.scale, max_lines=2)
        cameo_video = video_time(self.timeline, cfg.cameo_enter)
        if cfg.wait_start <= sim_t < cfg.cameo_enter:
            self._text(frame, cfg.wait_text, 54, self.width * 0.5, cfg.hook_y * self.scale, max_lines=1)
        if alive == 3 and phase != "celebration":
            self._text(frame, "FINAL 3", 72, self.width * 0.5, (cfg.hook_y + 200) * self.scale, max_lines=1)
        if alive == 2 and phase == "slow":
            self._text(frame, "FINAL 2", 72, self.width * 0.5, (cfg.hook_y + 200) * self.scale, max_lines=1)
        if phase == "replay":
            self._text(frame, "REPLAY", 36, cfg.safe_x[0] * self.scale + 8, 220 * self.scale, align="left", max_lines=1)
        lines = active_kill_lines(self.result.elims, self.timeline, video_t, cfg.kill_fade, cfg.kill_lines)
        for slot, item in enumerate(lines):
            country = cfg.countries[item.index]
            phrase = self._phrase(item)
            y = cfg.kill_y[min(slot, 2)] * self.scale
            self._text(frame, phrase, cfg.kill_px, cfg.kill_x * self.scale, y, align="left", max_width=int(760 * self.scale), max_lines=1)
            side = max(2, int(22 * self.scale))
            sprite = cv2.resize(self._sprites[item.index], (side, side), interpolation=cv2.INTER_LINEAR)
            composite_rgba(frame, sprite, int(cfg.kill_x * self.scale - side - 6), int(y))
        if phase != "celebration":
            for item in self.result.elims:
                place = len(cfg.countries) - self.result.elims.index(item)
                age = video_t - video_time(self.timeline, item.time)
                if place <= 5 and 0.0 <= age <= 1.4:
                    self._text(frame, f"#{place} {cfg.countries[item.index].name.upper()}", 42, self.width * 0.5, 1180 * self.scale, max_lines=1)
        _ = cameo_video

    def _phrase(self, item: Elim) -> str:
        cfg = self.cfg
        country = cfg.countries[item.index]
        event = "meteor" if item.cause == "meteor" else "self_fall" if item.cause == "self" else "eliminated"
        lines = cfg.memes.lines(event)
        template = lines[(cfg.seed + item.index) % len(lines)]
        return render_phrase(template, code=country.iso2, name=country.name, cameo=cfg.cameo_name)

    def _text(
        self,
        frame: np.ndarray,
        text: str,
        font_px: float,
        x: float,
        y: float,
        *,
        alpha: float = 1.0,
        align: str = "center",
        max_width: int | None = None,
        max_lines: int = 2,
    ) -> None:
        if alpha <= 0.02:
            return
        safe_w = int((self.cfg.safe_x[1] - self.cfg.safe_x[0]) * self.scale) - 36
        width = safe_w if max_width is None else min(max_width, safe_w)
        px = max(12, int(font_px * self.scale))
        key = (text, px, max(40, width), max_lines)
        rgba = self._text_cache.get(key)
        if rgba is None:
            rgba = raster_text(text, self.font, px, max(40, width), max_lines)
            self._text_cache[key] = rgba
        if alpha < 0.999:
            rgba = rgba.copy()
            rgba[..., 3] = (rgba[..., 3].astype(np.float32) * alpha).astype(np.uint8)
        h, w = rgba.shape[:2]
        if align == "left":
            left = int(round(x))
            top = int(round(y - h / 2))
        else:
            left = int(round(x - w / 2))
            top = int(round(y))
        sx0 = int(math.ceil(self.cfg.safe_x[0] * self.scale))
        sx1 = int(math.floor(self.cfg.safe_x[1] * self.scale))
        sy0 = int(math.ceil(self.cfg.safe_y[0] * self.scale))
        sy1 = int(math.floor(self.cfg.safe_y[1] * self.scale))
        left = min(max(left, sx0), max(sx0, sx1 - w))
        top = min(max(top, sy0), max(sy0, sy1 - h))
        self.boxes.append((left, top, left + w, top + h))
        composite_rgba(frame, rgba, left, top)

    def _draw_safe(self, frame: np.ndarray) -> None:
        sx0 = int(self.cfg.safe_x[0] * self.scale)
        sx1 = int(self.cfg.safe_x[1] * self.scale)
        sy0 = int(self.cfg.safe_y[0] * self.scale)
        sy1 = int(self.cfg.safe_y[1] * self.scale)
        cv2.rectangle(frame, (sx0, sy0), (sx1, sy1), (80, 220, 80), 2, cv2.LINE_AA)
        zx0, zy0, zx1, zy1 = [int(v * self.scale) for v in (*self.cfg.zone_x, *self.cfg.zone_y)]
        cv2.rectangle(frame, (zx0, zy0), (zx1, zy1), (80, 180, 240), 2, cv2.LINE_AA)


def _upcoming_radius(cfg: ArenaConfig, time: float) -> float | None:
    keys = cfg.zone_keyframes
    for (t0, r0), (t1, r1) in zip(keys, keys[1:]):
        if r1 < r0 - 1.0 and t0 - cfg.warning_lead <= time < t1:
            return r1
    return None


def contact_sheet(renderer: ArenaRenderer, path: Path) -> None:
    timeline = renderer.timeline
    moments = [
        0.0,
        4.0,
        10.0,
        16.0,
        22.0,
        timeline.winner_video,
        timeline.replay_video + 0.2,
        timeline.celebration_video + 0.4,
    ]
    frames = [renderer.render(renderer.frame_at(moment)) for moment in moments]
    cols = 4
    rows = 2
    cell_w, cell_h = 270, 480
    sheet = np.zeros((cell_h * rows, cell_w * cols, 3), dtype=np.uint8)
    for index, frame in enumerate(frames):
        small = cv2.resize(frame, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
        row, col = divmod(index, cols)
        sheet[row * cell_h : (row + 1) * cell_h, col * cell_w : (col + 1) * cell_w] = small
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), sheet)


def init_arena_worker(cfg: ArenaConfig, cache_path: str, preview: bool, safe_overlay: bool) -> None:
    global _WORKER
    cv2.setNumThreads(1)
    _WORKER = ArenaRenderer(cfg, read_sim_cache(Path(cache_path)), preview=preview, safe_overlay=safe_overlay)


def render_arena_chunk(indices: list[int]) -> list[np.ndarray]:
    if _WORKER is None:
        raise RuntimeError("arena render worker was not initialized")
    return [_WORKER.render(index) for index in indices]
