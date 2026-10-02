"""Musical events and the retention map. Audio plays this list; it does not re-derive it from pictures."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from fc_sat.polycircle_claims import ClaimBook, evaluate, load_claims
from fc_sat.polycircle_config import PolyConfig, load_config
from fc_sat.polycircle_format import format_count, format_gap
from fc_sat.polycircle_geometry import (
    PHI_F,
    edge_length,
    gap,
    rotation_angles,
    solve_rotation,
    zoom_at,
)
from fc_sat.polycircle_schedule import Schedule, build_schedule, snap_frame

ROOT = Path(__file__).resolve().parents[1]
CHORDS = ("Am", "F", "C", "G")


def _beat(beat: float, bpm: float, fps: int) -> int:
    return snap_frame(beat, bpm, fps)[0]


@dataclass
class Timeline:
    config: PolyConfig
    schedule: Schedule
    angles: np.ndarray
    rotation_scale: float
    rotation_turns: int
    phi0: float
    phi_f: float
    speed0: float
    speed700: float
    kicks: list[int]
    claps: list[int]
    hats: list[int]
    open_hats: list[int]
    snares: list[int]
    arp: list[int]
    bass: list[int]
    bells: list[int]
    bell_degrees: list[int]
    impacts: list[int]
    ticks: list[int]
    chime: int
    chords: list[str] = field(default_factory=list)

    @property
    def n_frames(self) -> int:
        return self.schedule.n_frames

    @property
    def fps(self) -> int:
        return self.config.fps

    def seconds(self, frame: int) -> float:
        return frame / self.fps

    def to_json(self) -> dict:
        solution = {
            "scale": self.rotation_scale,
            "quarter_turns": self.rotation_turns,
            "phi0": self.phi0,
            "phi_f": self.phi_f,
            "speed0_deg_s": self.speed0,
            "speed700_deg_s": self.speed700,
        }
        return {
            "fps": self.fps,
            "bpm": self.config.bpm,
            "n_frames": self.n_frames,
            "sample_rate": self.config.sample_rate,
            "max_snap_error_frames": self.schedule.max_snap_error,
            "rotation": solution,
            "adds": [
                {"frame": item.frame, "n_after": item.n_after, "interval": item.interval, "edge": item.edge}
                for item in self.schedule.adds
            ],
            "doublings": [
                {"frame": item.frame, "n_before": item.n_before, "n_after": item.n_after}
                for item in self.schedule.doubles
            ],
            "kicks": self.kicks,
            "claps": self.claps,
            "hats": self.hats,
            "open_hats": self.open_hats,
            "snares": self.snares,
            "arp": self.arp,
            "bass": self.bass,
            "bells": self.bells,
            "bell_degrees": self.bell_degrees,
            "impacts": self.impacts,
            "ticks": self.ticks,
            "chime": self.chime,
            "chords": self.chords,
            "silent_gap": [744, 768],
            "tail": [1806, 1824],
        }


def _unique(frames: list[int], n_frames: int) -> list[int]:
    kept = sorted({frame for frame in frames if 0 <= frame < n_frames})
    return kept


def counter_number(timeline: Timeline, frame: int) -> int | None:
    """Integer on the counter, or None when the counter is the infinity sign."""
    if 1632 <= frame < 1800:
        return None
    if 1440 <= frame <= 1488:
        log0 = math.log10(6_291_456)
        log1 = math.log10(61)
        span = log0 - log1
        u = (frame - 1440) / (1488 - 1440)
        step = math.floor((u * span) / 0.1 + 1e-12)
        if frame >= 1488:
            return 61
        shown = int(round(10 ** (log0 - step * 0.1)))
        return max(61, shown)
    if 1488 < frame < 1800:
        return 61
    return timeline.schedule.n(frame)


def _events(schedule: Schedule, bpm: float, fps: int) -> dict[str, list[int]]:
    n_frames = schedule.n_frames

    def beat(value: float) -> int:
        return _beat(value, bpm, fps)

    kicks = []
    for index in range(75):
        if index in (30, 31) or 60 <= index <= 67:
            continue
        kicks.append(beat(index))

    claps = []
    for index in range(8, 75):
        if index % 4 not in (1, 3):
            continue
        if index == 31 or 60 <= index <= 67:
            continue
        claps.append(beat(index))

    hats = [beat(index + 0.5) for index in range(16)]
    for step in range(16 * 4, 31 * 4):
        hats.append(beat(step / 4.0))
    for step in range(32 * 4, 60 * 4):
        hats.append(beat(step / 4.0))
    for step in range(68 * 4, 75 * 4):
        frame = beat(step / 4.0)
        if frame < 1800:
            hats.append(frame)
    hats = [frame for frame in hats if not 744 <= frame < 768]

    open_hats = [beat(index) for index in (32, 36, 40, 44, 48, 52, 56, 68, 72)]

    snares = list(range(576, 608, 12)) + list(range(608, 640, 6)) + list(range(640, 672, 3))

    arp = [item.frame for item in schedule.adds if item.frame != 768]
    for step in range(32 * 2, 60 * 2):
        arp.append(beat(step / 2.0))
    for index in range(60, 68):
        arp.append(beat(index))
    for step in range(68 * 2, 75 * 2):
        frame = beat(step / 2.0)
        if frame < 1800:
            arp.append(frame)

    bass = [beat(index + 0.5) for index in range(8, 16)]
    bass += [beat(index + 0.5) for index in range(32, 60)]
    bass += [beat(index + 0.5) for index in range(68, 75) if beat(index + 0.5) < 1800]
    bass.append(beat(72))

    bells = [item.frame for item in schedule.doubles]
    impacts = [768, 960, 1632]
    ticks = _tick_frames()
    return {
        "kicks": _unique(kicks, n_frames),
        "claps": _unique(claps, n_frames),
        "hats": _unique(hats, n_frames),
        "open_hats": _unique(open_hats, n_frames),
        "snares": _unique(snares, n_frames),
        "arp": _unique(arp, n_frames),
        "bass": _unique(bass, n_frames),
        "bells": bells,
        "impacts": impacts,
        "ticks": ticks,
    }


def _tick_frames() -> list[int]:
    log0 = math.log10(6_291_456)
    log1 = math.log10(61)
    span = log0 - log1
    frames = []
    previous = 0
    for frame in range(1440, 1488):
        u = (frame - 1440) / (1488 - 1440)
        step = math.floor((u * span) / 0.1 + 1e-12)
        if step != previous:
            frames.append(frame)
            previous = step
    return frames


def build_timeline(config: PolyConfig | None = None) -> Timeline:
    config = config or load_config()
    schedule = build_schedule(config.bpm, config.fps, config.frames)
    solution = solve_rotation(config.fps)
    angles = rotation_angles(solution, config.frames, config.fps)
    events = _events(schedule, config.bpm, config.fps)
    degrees = [min(14, index * 2) for index in range(len(schedule.doubles))]
    return Timeline(
        config=config,
        schedule=schedule,
        angles=angles,
        rotation_scale=solution.scale,
        rotation_turns=solution.turns,
        phi0=solution.phi0,
        phi_f=solution.phi_f,
        speed0=solution.speed0_deg_s,
        speed700=solution.speed700_deg_s,
        kicks=events["kicks"],
        claps=events["claps"],
        hats=events["hats"],
        open_hats=events["open_hats"],
        snares=events["snares"],
        arp=events["arp"],
        bass=events["bass"],
        bells=events["bells"],
        bell_degrees=degrees,
        impacts=events["impacts"],
        ticks=events["ticks"],
        chime=1488,
        chords=[CHORDS[bar % 4] for bar in range(19)],
    )


def write_timeline(timeline: Timeline, path: Path | None = None) -> Path:
    path = path or (ROOT / "out" / "timeline.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(timeline.to_json(), indent=2))
    return path


def load_timeline_json(path: Path | None = None) -> dict:
    path = path or (ROOT / "out" / "timeline.json")
    return json.loads(path.read_text())


def retention_rows(timeline: Timeline) -> list[tuple[float, int, str, str]]:
    """Time, frame, event, and what the viewer should feel. Times are frame / fps."""
    cues = [
        (0, "hook", "The question is already on screen, and the square is already moving."),
        (24, "first side", "The promise is real: a side just appeared, with the kick."),
        (192, "eighth notes", "It is speeding up. Counting is still possible."),
        (384, "sixteenth notes", "Counting slips. The shape is rounding off."),
        (576, "thirty-second notes", "The roll starts. Something is about to give."),
        (744, "silent gap", "The beat drops out. One note is left."),
        (768, "drop", "It looks finished. The caption says it looks like a circle."),
        (864, "spin stops", "The shape freezes. The eye gets a still picture."),
        (876, "zoom starts", "The camera moves in. The circle claim is about to break."),
        (948, "gap label", "The edge is still straight. The gap has a number."),
        (960, "doublings", "Each beat pulls the edge closer, and it is still not a circle."),
        (1152, "zoom out", "The number races. The shape shrinks back to the full frame."),
        (1344, "gap at full size", "The caption says the gap is still there, now far below a pixel."),
        (1440, "counter rewinds", "The huge number climbs back down toward something you can check."),
        (1488, "61", "This is the screen answer: 61 sides, and you cannot tell."),
        (1632, "NEVER", "The math answer is the opposite of the screen answer."),
        (1728, "only closer", "More sides only get closer. They do not arrive."),
        (1800, "snap", "The square is back. The loop is about to close."),
        (1823, "loop", "The last frame is the first frame. The next kick can land clean."),
    ]
    return [(timeline.seconds(frame), frame, label, feeling) for frame, label, feeling in cues]


def retention_markdown(timeline: Timeline) -> str:
    lines = [
        "# Retention map",
        "",
        "Times are `frame / 60`. Read a YouTube retention graph against the feeling column.",
        "NEVER is at 27.2 s because bar 18 starts at frame 1632, not at 25.6 s.",
        "",
        "| time | frame | event | what the viewer should feel |",
        "| --- | --- | --- | --- |",
    ]
    for seconds, frame, label, feeling in retention_rows(timeline):
        lines.append(f"| {seconds:.1f} s | {frame} | {label} | {feeling} |")
    lines.append("")
    return "\n".join(lines)


def facts_markdown(timeline: Timeline, book: ClaimBook | None = None) -> str:
    book = book or load_claims()
    results = evaluate(book)
    radius = timeline.config.radius
    lines = [
        "# Polycircle facts",
        "",
        "Every value below is computed. Defaults: 150 bpm, 1824 frames, radius 370 px, zoom 36.",
        f"Maximum beat snap error: {timeline.schedule.max_snap_error:.3e} frames (limit 0.5).",
        "",
        "## Gap",
        "",
        f"First n with gap <= 0.5 px at R={radius:g}: {book.get('n_fool').value}.",
        f"gap(60) = {gap(60, radius):.6f} px. gap(61) = {gap(61, radius):.6f} px.",
        f"gap(96) = {gap(96, radius):.6f} px. At 36x that is {gap(96, radius) * 36:.4f} px, shown as {format_gap(gap(96, radius) * 36)}.",
        f"Edge at n=96: {edge_length(96, radius):.4f} px ({edge_length(96, radius) * 36:.2f} px at 36x).",
        "",
        "| k | n | gap at 36x | on screen |",
        "| --- | --- | --- | --- |",
    ]
    for k in range(9):
        n_sides = 96 * (2 ** k)
        value = gap(n_sides, radius) * 36
        lines.append(f"| {k} | {format_count(n_sides)} | {value:.6g} | {format_gap(value)} |")
    end_n = 96 * (2 ** 16)
    lines += [
        "",
        f"After 16 doublings, n = {format_count(end_n)} and the zoom-1 gap is {format_gap(gap(end_n, radius))} px.",
        "",
        "## Schedule",
        "",
        "| frame | n |",
        "| --- | --- |",
    ]
    for frame in (0, 192, 384, 576, 672, 768, 960, 1128, 1320, 1488, 1800, 1823):
        lines.append(f"| {frame} | {format_count(timeline.schedule.n(frame))} |")
    lines += [
        "",
        f"Adds: {len(timeline.schedule.adds)}. Doublings: {len(timeline.schedule.doubles)}.",
        f"n reaches 61 at frame {book.get('n_61_frame').value}.",
        "",
        "## Rotation",
        "",
        "Unscaled speed rises linearly from 30 deg/s at frame 0 to 150 deg/s at frame 700,",
        "then easeInOutCubic down to 0 at frame 864. A single scale makes the integral",
        "an integer number of quarter-turns, so the square at frame 0 matches the freeze.",
        "",
        f"- scale: {timeline.rotation_scale:.6f}",
        f"- quarter-turns: {timeline.rotation_turns}",
        f"- speed at frame 0: {timeline.speed0:.3f} deg/s",
        f"- speed at frame 700: {timeline.speed700:.3f} deg/s",
        f"- phi_f: {timeline.phi_f:.6f} (target {PHI_F:.6f})",
        f"- phi_0: {timeline.phi0:.6f}",
        "",
        "## Camera",
        "",
        "Zoom is 1 until frame 876, eases in log space to 36 by frame 948, holds, then",
        "eases back to 1 from frame 1152 to frame 1280. At zoom 1 the view is the world.",
        f"At zoom 36 the top of the circle sits on the camera anchor (540, 910). Zoom there is {zoom_at(954):.1f}.",
        "",
        "## Bar colors",
        "",
        "The brief's hues are kept. OKLab L is matched so the spread is at most 0.03.",
        "Drawn colors: " + " ".join(timeline.config.bar_colors),
        "",
        "## 10 metre question",
        "",
        f"Radius 5 m (5000 mm), tolerance 1 mm: {book.get('n_metre').value} sides.",
        "",
        "## Claims",
        "",
    ]
    failed = [claim_id for claim_id, ok, _detail in results if not ok]
    lines.append(f"{sum(1 for _, ok, _ in results if ok)}/{len(results)} passed.")
    if failed:
        lines.append("Failed: " + ", ".join(failed))
    lines.append("")
    return "\n".join(lines)
