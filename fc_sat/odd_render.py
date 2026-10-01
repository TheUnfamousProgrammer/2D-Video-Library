"""Flat Odd One Out frames. No bloom, no glow, no vignette, no motion."""

from __future__ import annotations

import math

import cv2
import numpy as np

from fc_sat.color import hex_to_rgb
from fc_sat.odd_config import OddConfig, Segment, build_timeline
from fc_sat.odd_diff import apply_look, paint_item, reveal_label
from fc_sat.odd_layout import hud_boxes
from fc_sat.odd_sim import Show
from fc_sat.visual import composite_rgba, find_font, raster_text_flat


def _bgr(value: str) -> tuple[int, int, int]:
    rgb = np.clip(hex_to_rgb(value) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return int(rgb[2]), int(rgb[1]), int(rgb[0])


def _segment_at(timeline: tuple[Segment, ...], t: float) -> Segment:
    for segment in timeline:
        if segment.start_s <= t < segment.end_s - 1e-9:
            return segment
    return timeline[-1]


def _rounded_outline(frame: np.ndarray, box: tuple[int, int, int, int], radius: int, color: tuple[int, int, int], width: int) -> None:
    x0, y0, x1, y1 = box
    radius = max(1, min(radius, (x1 - x0) // 2, (y1 - y0) // 2))
    width = max(1, width)
    cv2.rectangle(frame, (x0 + radius, y0), (x1 - radius, y0 + width), color, -1)
    cv2.rectangle(frame, (x0 + radius, y1 - width), (x1 - radius, y1), color, -1)
    cv2.rectangle(frame, (x0, y0 + radius), (x0 + width, y1 - radius), color, -1)
    cv2.rectangle(frame, (x1 - width, y0 + radius), (x1, y1 - radius), color, -1)
    cv2.ellipse(frame, (x0 + radius, y0 + radius), (radius, radius), 180, 0, 90, color, width, lineType=cv2.LINE_8)
    cv2.ellipse(frame, (x1 - radius, y0 + radius), (radius, radius), 270, 0, 90, color, width, lineType=cv2.LINE_8)
    cv2.ellipse(frame, (x0 + radius, y1 - radius), (radius, radius), 90, 0, 90, color, width, lineType=cv2.LINE_8)
    cv2.ellipse(frame, (x1 - radius, y1 - radius), (radius, radius), 0, 0, 90, color, width, lineType=cv2.LINE_8)


class OddRenderer:
    def __init__(self, cfg: OddConfig, show: Show, *, preview: bool = False, safe_overlay: bool = False) -> None:
        self.cfg = cfg
        self.show = show
        self.preview = preview
        self.safe_overlay = safe_overlay
        self.scale = 0.5 if preview else 1.0
        self.fps = 30 if preview else cfg.fps
        self.width = int(cfg.width * self.scale)
        self.height = int(cfg.height * self.scale)
        duration = show.timeline[-1].end_s
        self.n_frames = int(round(duration * self.fps))
        self.font = find_font()
        self.background = _bgr(cfg.background)
        self.outline = _bgr(cfg.outline_color)
        self.timer_color = _bgr(cfg.timer_color)
        self.timer_red = _bgr(cfg.timer_red)
        self.ring = _bgr(cfg.ring_color)
        self.timeline = show.timeline
        self.boxes: list[tuple[str, tuple[float, float, float, float]]] = []

    def render(self, index: int) -> np.ndarray:
        return self.render_time(index / self.fps)

    def render_time(self, t: float) -> np.ndarray:
        segment = _segment_at(self.timeline, t)
        frame = np.empty((self.height, self.width, 3), dtype=np.uint8)
        frame[:] = self.background
        self.boxes = []
        if segment.kind == "play":
            level = self.show.levels[segment.level_id]
            local = t - segment.start_s
            self._draw_level(frame, level, pop=self._pop(local), reveal_u=None)
            remaining = max(0.0, level.level.timer - local)
            self._draw_hud(frame, level.level.id, self._caption(level.level.id, local, reveal=False), remaining)
        elif segment.kind == "reveal":
            level = self.show.levels[segment.level_id]
            local = t - segment.start_s
            self._draw_level(frame, level, pop=1.0, reveal_u=local)
            self._draw_hud(frame, level.level.id, reveal_label(level.level.difference, self.cfg.detail_mode), 0.0)
        elif segment.kind == "dissolve":
            outgoing = self.show.levels[segment.level_id]
            incoming_id = self._next_level(segment.level_id)
            incoming = self.show.levels[incoming_id]
            u = (t - segment.start_s) / max(segment.duration_s, 1e-6)
            a = np.empty_like(frame)
            b = np.empty_like(frame)
            a[:] = self.background
            b[:] = self.background
            self._draw_level(a, outgoing, pop=1.0, reveal_u=self.cfg.reveal_seconds)
            self._draw_level(b, incoming, pop=self._pop(0.0), reveal_u=None)
            mixed = a.astype(np.float32) * (1.0 - u) + b.astype(np.float32) * u
            frame[:] = mixed.astype(np.uint8)
            self._draw_hud(frame, incoming.level.id, self._caption(incoming.level.id, 0.0, reveal=False), incoming.level.timer)
        else:
            last = self.cfg.levels[-1]
            level = self.show.levels[last.id]
            self._draw_level(frame, level, pop=self.cfg.fade_alpha, reveal_u=self.cfg.reveal_seconds, uniform=True)
            self._draw_outro(frame)
        if self.safe_overlay:
            self._guides(frame)
        return frame

    def _next_level(self, level_id: int) -> int:
        ids = [level.id for level in self.cfg.levels]
        return ids[ids.index(level_id) + 1]

    def _pop(self, local: float) -> float:
        if self.cfg.pop_seconds <= 0:
            return 1.0
        u = min(1.0, max(0.0, local / self.cfg.pop_seconds))
        return self.cfg.pop_floor + (1.0 - self.cfg.pop_floor) * u

    def _caption(self, level_id: int, local: float, *, reveal: bool) -> str:
        if reveal:
            level = self.show.levels[level_id]
            return reveal_label(level.level.difference, self.cfg.detail_mode)
        if level_id == self.cfg.levels[0].id and local < self.cfg.hook_seconds:
            return self.cfg.hook
        return ""

    def _draw_level(self, frame: np.ndarray, sim, *, pop: float, reveal_u: float | None, uniform: bool = False) -> None:
        self._outline(frame)
        grow = None
        others = 1.0
        if reveal_u is not None and not uniform:
            others = 1.0 + (self.cfg.fade_alpha - 1.0) * min(1.0, reveal_u / max(self.cfg.fade_seconds, 1e-6))
            grow = min(1.0, reveal_u / max(self.cfg.ring_grow, 1e-6))
        elif reveal_u is not None:
            grow = 1.0
        for index in range(sim.level.count):
            look = apply_look(
                sim.level.difference,
                self.cfg.tier,
                is_odd=index == sim.odd_index,
                base_lab=sim.base_lab,
                odd_lab=sim.odd_lab,
                size=sim.level.size,
                params=sim.diff_params,
            )
            if uniform or reveal_u is None or index == sim.odd_index:
                alpha = pop
            else:
                alpha = pop * others
            cx, cy = sim.centers[index]
            paint_item(
                frame,
                look,
                cx,
                cy,
                corner_frac=self.cfg.corner_frac,
                alpha=float(alpha),
                scale=self.scale,
            )
        if grow is not None:
            cx, cy = sim.centers[sim.odd_index]
            radius = (sim.level.size / 2.0) * (self.cfg.ring_from + (self.cfg.ring_to - self.cfg.ring_from) * grow)
            self._ring(frame, cx, cy, radius)

    def _outline(self, frame: np.ndarray) -> None:
        box = tuple(int(round(v * self.scale)) for v in (self.cfg.field_x0, self.cfg.field_y0, self.cfg.field_x1, self.cfg.field_y1))
        _rounded_outline(
            frame,
            box,
            int(round(self.cfg.field_radius * self.scale)),
            self.outline,
            max(1, int(round(self.cfg.outline_px * self.scale))),
        )

    def _ring(self, frame: np.ndarray, cx: float, cy: float, radius: float) -> None:
        thickness = max(1, int(round(self.cfg.ring_px * self.scale)))
        cv2.circle(
            frame,
            (int(round(cx * self.scale)), int(round(cy * self.scale))),
            max(1, int(round(radius * self.scale))),
            self.ring,
            thickness,
            lineType=cv2.LINE_8,
        )

    def _draw_hud(self, frame: np.ndarray, level_id: int, caption: str, remaining: float) -> None:
        level = next(item for item in self.cfg.levels if item.id == level_id)
        self._centered(frame, f"LEVEL {level_id}", self.cfg.label_px, self.width / self.scale * 0.5, self.cfg.label_y, self.cfg.safe_x[1] - self.cfg.safe_x[0], "label")
        if caption:
            cx = (self.cfg.caption_x[0] + self.cfg.caption_x[1]) * 0.5
            self._centered(frame, caption, self.cfg.caption_px, cx, self.cfg.caption_y, self.cfg.caption_x[1] - self.cfg.caption_x[0], "caption")
        self._timer(frame, remaining, level.timer)
        seconds = "0" if remaining <= 1e-3 else str(int(math.ceil(remaining - 1e-6)))
        num_cx = (self.cfg.timer_num_x0 + self.cfg.timer_num_x1) * 0.5
        num_cy = self.cfg.timer_y + self.cfg.timer_h * 0.5
        self._centered(
            frame,
            seconds,
            self.cfg.timer_num_px,
            num_cx,
            num_cy,
            self.cfg.timer_num_x1 - self.cfg.timer_num_x0,
            "seconds",
        )

    def _timer(self, frame: np.ndarray, remaining: float, timer: float) -> None:
        x0 = int(round(self.cfg.timer_x0 * self.scale))
        x1 = int(round(self.cfg.timer_x1 * self.scale))
        y0 = int(round(self.cfg.timer_y * self.scale))
        y1 = int(round((self.cfg.timer_y + self.cfg.timer_h) * self.scale))
        track = (42, 47, 58)
        cv2.rectangle(frame, (x0, y0), (x1, y1), track, -1)
        frac = 0.0 if timer <= 0 else max(0.0, min(1.0, remaining / timer))
        fill = x0 + int(round((x1 - x0) * frac))
        color = self.timer_red if remaining <= self.cfg.red_seconds else self.timer_color
        if fill > x0:
            cv2.rectangle(frame, (x0, y0), (fill, y1), color, -1)

    def _draw_outro(self, frame: np.ndarray) -> None:
        self._centered(
            frame,
            "How far did you get?",
            self.cfg.outro_px,
            self.cfg.width * 0.5,
            self.cfg.outro_y,
            self.cfg.safe_x[1] - self.cfg.safe_x[0],
            "question",
        )
        self._centered(
            frame,
            self.cfg.cta,
            self.cfg.cta_px,
            self.cfg.width * 0.5,
            self.cfg.cta_y,
            self.cfg.safe_x[1] - self.cfg.safe_x[0],
            "cta",
        )

    def _centered(self, frame, text: str, px: int, cx: float, cy: float, max_width: float, name: str) -> None:
        raster = raster_text_flat(text, self.font, max(8, int(round(px * self.scale))), max(8, int(round(max_width * self.scale))), 1)
        x = int(round(cx * self.scale - raster.shape[1] / 2))
        y = int(round(cy * self.scale - raster.shape[0] / 2))
        composite_rgba(frame, raster, x, y)
        self.boxes.append((name, (x, y, x + raster.shape[1], y + raster.shape[0])))

    def _guides(self, frame: np.ndarray) -> None:
        safe = tuple(int(round(v * self.scale)) for v in (*self.cfg.safe_x, *self.cfg.safe_y))
        field = tuple(int(round(v * self.scale)) for v in (self.cfg.field_x0, self.cfg.field_x1, self.cfg.field_y0, self.cfg.field_y1))
        cv2.rectangle(frame, (safe[0], safe[2]), (safe[1], safe[3]), (255, 0, 255), 1)
        cv2.rectangle(frame, (field[0], field[2]), (field[1], field[3]), (255, 255, 0), 1)

    def profile_ms(self) -> float:
        import time

        started = time.perf_counter()
        self.render(0)
        return (time.perf_counter() - started) * 1000.0


def init_odd_worker(cfg: OddConfig, show: Show, preview: bool, safe_overlay: bool) -> None:
    global _RENDERER
    _RENDERER = OddRenderer(cfg, show, preview=preview, safe_overlay=safe_overlay)


def render_odd_chunk(indices: list[int]) -> list[np.ndarray]:
    return [_RENDERER.render(index) for index in indices]


_RENDERER: OddRenderer | None = None


def _settled_time(cfg: OddConfig, level_id: int) -> float:
    for segment in build_timeline(cfg):
        if segment.kind == "play" and segment.level_id == level_id:
            return segment.start_s + cfg.pop_seconds + 0.05
    raise RuntimeError(f"no play segment for level {level_id}")


def _reveal_time(cfg: OddConfig, level_id: int) -> float:
    for segment in build_timeline(cfg):
        if segment.kind == "reveal" and segment.level_id == level_id:
            return segment.start_s + min(0.45, segment.duration_s * 0.4)
    raise RuntimeError(f"no reveal segment for level {level_id}")


def write_level_pngs(renderer: OddRenderer, output) -> dict[int, dict[str, str]]:
    """A clean puzzle and a ringed answer for each level. No highlight on the clean file."""
    from pathlib import Path

    output = Path(output)
    written: dict[int, dict[str, str]] = {}
    for level in renderer.cfg.levels:
        clean = renderer.render_time(_settled_time(renderer.cfg, level.id))
        answer = renderer.render_time(_reveal_time(renderer.cfg, level.id))
        clean_path = output.with_name(f"{output.stem}.l{level.id}.png")
        answer_path = output.with_name(f"{output.stem}.l{level.id}.answer.png")
        cv2.imwrite(str(clean_path), clean)
        cv2.imwrite(str(answer_path), answer)
        written[level.id] = {"clean": str(clean_path), "answer": str(answer_path)}
        print(f"level {level.id} clean {clean_path.name} answer {answer_path.name}", flush=True)
    return written


def contact_sheet(renderer: OddRenderer, path) -> None:
    cells = []
    for level in renderer.cfg.levels:
        for moment in (_settled_time(renderer.cfg, level.id), _reveal_time(renderer.cfg, level.id)):
            frame = renderer.render_time(moment)
            cells.append(cv2.resize(frame, (270, 480), interpolation=cv2.INTER_AREA))
    columns = 2
    rows = int(math.ceil(len(cells) / columns))
    sheet = np.zeros((rows * 480, columns * 270, 3), dtype=np.uint8)
    for index, cell in enumerate(cells):
        row, col = divmod(index, columns)
        sheet[row * 480 : (row + 1) * 480, col * 270 : (col + 1) * 270] = cell
    cv2.imwrite(str(path), sheet)
    print(f"contact {path}", flush=True)


def write_ladder(cfg: OddConfig, dest) -> list:
    """Clean puzzle stills for every type and rung. No highlight and no video."""
    from dataclasses import replace
    from pathlib import Path

    from fc_sat.odd_sim import simulate_show

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    by_id = {level.id: level for level in cfg.levels}
    kinds = []
    for kind, level_id in (("hue", 1), ("tilt", 2), ("detail", 3)):
        level = by_id.get(level_id)
        if level is None or level.difference != kind:
            raise RuntimeError(f"ladder needs level {level_id} ({kind})")
        kinds.append((kind, level))
    written = []
    answers = ["# Ladder answers", ""]
    figures = []
    for kind, template in kinds:
        for rung in range(1, 6):
            level = replace(template, rung=rung)
            one = replace(cfg, levels=(level,), seed=cfg.seed + 100 * rung + level.id)
            show = simulate_show(one)
            renderer = OddRenderer(one, show, preview=False)
            frame = renderer.render_time(_settled_time(one, level.id))
            name = f"{kind}_rung{rung}_{level.count}items.png"
            path = dest / name
            cv2.imwrite(str(path), frame)
            sim = show.levels[level.id]
            answers.append(f"- {name}: row {sim.row + 1}, column {sim.col + 1} ({sim.phrase})")
            figures.append(f'<figure><img src="{name}" alt=""><figcaption>{name}</figcaption></figure>')
            written.append(path)
            print(f"ladder {name} row {sim.row + 1} col {sim.col + 1}", flush=True)
    answers.append("")
    (dest / "ANSWERS.md").write_text("\n".join(answers), encoding="utf-8")
    html = (
        '<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        "<title>Odd One Out ladder</title>\n<style>\n"
        "body { margin: 24px; background: #14171F; color: #fff; font-family: sans-serif; }\n"
        ".grid { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 12px; }\n"
        "figure { margin: 0; }\nimg { width: 100%; height: auto; }\n"
        "figcaption { font-size: 13px; margin-top: 4px; }\n"
        "</style>\n</head>\n<body>\n<h1>Odd One Out ladder</h1>\n<div class=\"grid\">\n"
        + "\n".join(figures)
        + "\n</div>\n</body>\n</html>\n"
    )
    (dest / "index.html").write_text(html, encoding="utf-8")
    return written
