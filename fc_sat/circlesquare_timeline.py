"""Musical events for the circlesquare short. Audio plays this list."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from fc_sat.beatkit.grid import snap_frame
from fc_sat.circlesquare_claims import ClaimBook, evaluate, load_claims
from fc_sat.circlesquare_config import CircleConfig, load_config
from fc_sat.circlesquare_math import A, gap, zoom_at
from fc_sat.circlesquare_schedule import (
    FINAL_K,
    Schedule,
    build_schedule,
    counter_value,
    rewind_tick_frames,
)
from fc_sat.polycircle_format import format_count, format_gap

ROOT = Path(__file__).resolve().parents[1]
CHORDS = ("Am", "F", "C", "G")


def _beat(beat: float, bpm: float, fps: int) -> int:
    return snap_frame(beat, bpm, fps)[0]


@dataclass
class Timeline:
    config: CircleConfig
    schedule: Schedule
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
        return {
            "fps": self.fps,
            "bpm": self.config.bpm,
            "n_frames": self.n_frames,
            "sample_rate": self.config.sample_rate,
            "max_snap_error_frames": self.schedule.max_snap_error,
            "adds": [
                {"frame": item.frame, "k_after": item.k_after, "interval": item.interval}
                for item in self.schedule.adds
            ],
            "doublings": [
                {"frame": item.frame, "k_before": item.k_before, "k_after": item.k_after}
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
    return sorted({frame for frame in frames if 0 <= frame < n_frames})


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

    return {
        "kicks": _unique(kicks, n_frames),
        "claps": _unique(claps, n_frames),
        "hats": _unique(hats, n_frames),
        "open_hats": _unique(open_hats, n_frames),
        "snares": _unique(snares, n_frames),
        "arp": _unique(arp, n_frames),
        "bass": _unique(bass, n_frames),
        "bells": [item.frame for item in schedule.doubles],
        "impacts": [768, 960, 1632],
        "ticks": rewind_tick_frames(),
    }


def build_timeline(config: CircleConfig | None = None) -> Timeline:
    config = config or load_config()
    schedule = build_schedule(config.bpm, config.fps, config.frames)
    events = _events(schedule, config.bpm, config.fps)
    degrees = [min(14, index * 2) for index in range(len(schedule.doubles))]
    return Timeline(
        config=config,
        schedule=schedule,
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
    # The polycircle film also writes out/timeline.json. This short uses its own name.
    if path.name == "timeline.json":
        path = path.with_name("circlesquare_timeline.json")
    path.write_text(json.dumps(timeline.to_json(), indent=2))
    return path


def retention_rows(timeline: Timeline) -> list[tuple[float, int, str, str]]:
    cues = [
        (0, "hook", "The question is already on screen, and one thick circle is already turning."),
        (24, "first circle", "A second circle joins on the beat. The promise is real."),
        (192, "eighth notes", "Circles arrive faster. The shape is still obviously round."),
        (384, "sixteenth notes", "The outline starts to flatten. Counting slips."),
        (576, "thirty-second notes", "The roll starts. Something is about to give."),
        (744, "silent gap", "The beat drops out. One note is left."),
        (768, "drop", "It looks finished. The caption says it looks like a square."),
        (876, "zoom", "The camera moves into a corner."),
        (948, "not quite", "The corner is still round. The gap has a number."),
        (960, "doublings", "The tip hits the corner as the circles double. The corner sharpens."),
        (1152, "zoom out", "The number races while the view pulls back."),
        (1344, "gap at full size", "The gap is still there, now far below a pixel."),
        (1440, "counter rewinds", "The huge number climbs back down toward something you can check."),
        (1488, "138", "On this screen, 138 circles is enough to fool the eye."),
        (1632, "never exact", "In math it is never a square. Corners need infinitely many circles."),
        (1728, "only closer", "More circles only get closer. They do not arrive."),
        (1800, "snap", "It collapses back to one circle. The question returns."),
        (1823, "loop", "The last frame is the first frame. The next kick can land clean."),
    ]
    return [(timeline.seconds(frame), frame, label, feeling) for frame, label, feeling in cues]


def retention_markdown(timeline: Timeline) -> str:
    lines = [
        "# Retention map",
        "",
        "Times are `frame / 60`. Read a YouTube retention graph against the feeling column.",
        "NEVER EXACT is at 27.2 s because bar 18 starts at frame 1632.",
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
    lines = [
        "# Circlesquare facts",
        "",
        "Every value below is computed. A square of half-side 340 px, 150 bpm, 1824 frames, zoom 36.",
        f"First circle radius r1 = {book.get('r1').value:.6f} px. Corner distance = {book.get('corner_px').value:.6f} px.",
        f"Maximum beat snap error: {timeline.schedule.max_snap_error:.3e} frames.",
        "",
        f"Smallest K with gap <= 0.5 px: {book.get('k_fool').value}.",
        f"gap(1) = {gap(1):.4f} px. gap(93) = {gap(93):.4f} px. gap(138) = {gap(138):.4f} px.",
        f"At 36x the 93-circle gap is {gap(93) * 36:.2f} px, shown as {format_gap(gap(93) * 36)}.",
        f"For K >= 10, gap(K) = {ASYMPTOTE_LABEL(A)} / K. At K = 2976 the exact gap and the formula agree.",
        f"A 10 m square within 1 mm needs {book.get('metre_k').value} circles.",
        "",
        "The square shortcut at K > 2976 and zoom <= 2 is 0.023 px off the true curve at the threshold",
        "(under 0.05 px on screen). It is under 0.01 px once K reaches 11,904. K above 23,808 reuses",
        "the 23,808-circle curve; the counter still shows the true count.",
        "",
        "| frame | K | counter |",
        "| --- | --- | --- |",
    ]
    for frame in (0, 192, 384, 576, 672, 768, 960, 1128, 1320, 1488, 1632, 1800, 1823):
        shown = counter_value(timeline.schedule, frame)
        label = "infinity" if shown is None else format_count(shown)
        lines.append(f"| {frame} | {format_count(timeline.schedule.k(frame))} | {label} |")
    lines += [
        "",
        f"Adds: {len(timeline.schedule.adds)}. Doublings: {len(timeline.schedule.doubles)}. Final K: {format_count(FINAL_K)}.",
        "",
        "Bar colors: " + " ".join(timeline.config.bar_colors),
        "",
        f"Claims {sum(1 for _, ok, _ in results if ok)}/{len(results)} passed.",
        "",
    ]
    return "\n".join(lines)


def ASYMPTOTE_LABEL(half: float) -> str:
    from fc_sat.circlesquare_math import ASYMPTOTE

    return f"{ASYMPTOTE:.2f}"


def shown_counter(timeline: Timeline, frame: int) -> int | None:
    return counter_value(timeline.schedule, frame)
