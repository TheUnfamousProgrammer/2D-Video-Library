"""Musical events for the scale short. Audio plays this list.

The score is the polycircle short's: the same accelerating arp into the silent beat, the drop
at 12.8 s, the breakdown and the impact at 27.2 s. On top of it, every landing rings a bell
whose pitch climbs with the size of the thing the camera lands on.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from fc_sat.beatkit.grid import snap_frame
from fc_sat.scale_scene import Scene, build_scene
from fc_sat.scale_world import SNAP

ROOT = Path(__file__).resolve().parents[1]
BPM = 150.0
FPS = 60
N_FRAMES = 1824
SAMPLE_RATE = 48000
SEED = 29
CHORDS = ("Am", "F", "C", "G")
DROP = 768
IMPACTS = (DROP, 960, 1632)
CHIME = 1700
TOP_DEGREE = 14
# The polycircle short's build: quarters, eighths, sixteenths, then every 3 frames, then 3 last beats.
ARP_BUILD = (
    list(range(24, 192, 24))
    + list(range(192, 384, 12))
    + list(range(384, 576, 6))
    + list(range(576, 672, 3))
    + [672, 696, 720, 744]
)


@dataclass
class Timeline:
    scene: Scene
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
    lands: list[int]
    max_snap_error: float
    chords: list[str] = field(default_factory=list)

    n_frames: int = N_FRAMES
    fps: int = FPS
    bpm: float = BPM
    sample_rate: int = SAMPLE_RATE
    seed: int = SEED

    def seconds(self, frame: int) -> float:
        return frame / self.fps

    def to_json(self) -> dict:
        return {
            "fps": self.fps,
            "bpm": self.bpm,
            "n_frames": self.n_frames,
            "sample_rate": self.sample_rate,
            "max_snap_error_frames": self.max_snap_error,
            "lands": self.lands,
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
            "silent_gap": [744, DROP],
            "tail": [1806, self.n_frames],
        }


def _beat(beat: float) -> int:
    return snap_frame(beat, BPM, FPS)[0]


def _unique(frames, n_frames: int = N_FRAMES) -> list[int]:
    return sorted({int(frame) for frame in frames if 0 <= frame < n_frames})


def _events() -> dict[str, list[int]]:
    kicks = [_beat(index) for index in range(75) if index not in (30, 31) and not 60 <= index <= 67]
    claps = [_beat(index) for index in range(8, 75) if index % 4 in (1, 3) and index != 31 and not 60 <= index <= 67]
    hats = [_beat(index + 0.5) for index in range(16)]
    for step in list(range(16 * 4, 31 * 4)) + list(range(32 * 4, 60 * 4)):
        hats.append(_beat(step / 4.0))
    for step in range(68 * 4, 75 * 4):
        if _beat(step / 4.0) < SNAP[0]:
            hats.append(_beat(step / 4.0))
    hats = [frame for frame in hats if not 744 <= frame < DROP]
    open_hats = [_beat(index) for index in (32, 36, 40, 44, 48, 52, 56, 68, 72)]
    snares = list(range(576, 608, 12)) + list(range(608, 640, 6)) + list(range(640, 672, 3))
    arp = list(ARP_BUILD)
    for step in range(32 * 2, 60 * 2):
        arp.append(_beat(step / 2.0))
    for index in range(60, 68):
        arp.append(_beat(index))
    for step in range(68 * 2, 75 * 2):
        if _beat(step / 2.0) < SNAP[0]:
            arp.append(_beat(step / 2.0))
    bass = [_beat(index + 0.5) for index in range(8, 16)]
    bass += [_beat(index + 0.5) for index in range(32, 60)]
    bass += [_beat(index + 0.5) for index in range(68, 75) if _beat(index + 0.5) < SNAP[0]]
    bass.append(_beat(72))
    return {
        "kicks": _unique(kicks),
        "claps": _unique(claps),
        "hats": _unique(hats),
        "open_hats": _unique(open_hats),
        "snares": _unique(snares),
        "arp": _unique(arp),
        "bass": _unique(bass),
    }


def bell_degree(size_m: float, smallest: float, biggest: float) -> int:
    """Pitch climbs with log size: the smallest thing is the lowest bell, the biggest the highest."""
    share = (math.log10(size_m) - math.log10(smallest)) / (math.log10(biggest) - math.log10(smallest))
    return int(round(TOP_DEGREE * min(1.0, max(0.0, share))))


def build_timeline(scene: Scene | None = None) -> Timeline:
    scene = scene or build_scene()
    events = _events()
    items = scene.items
    lands = [item.land for item in items]
    smallest, biggest = items[0].size_m, items[-1].size_m
    rung = [item for item in items if item.land > 0]
    sixteenth = 60.0 * FPS / BPM / 4.0
    error = max(abs(frame - round(frame / sixteenth) * sixteenth) for frame in lands)
    return Timeline(
        scene=scene,
        kicks=events["kicks"],
        claps=events["claps"],
        hats=events["hats"],
        open_hats=events["open_hats"],
        snares=events["snares"],
        arp=events["arp"],
        bass=events["bass"],
        bells=[item.land for item in rung],
        bell_degrees=[bell_degree(item.size_m, smallest, biggest) for item in rung],
        impacts=list(IMPACTS),
        ticks=[],
        chime=CHIME,
        lands=lands,
        max_snap_error=error,
        chords=[CHORDS[bar % 4] for bar in range(19)],
    )


def write_timeline(timeline: Timeline, path: Path | None = None) -> Path:
    path = path or (ROOT / "out" / "scale_timeline.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(timeline.to_json(), indent=2))
    return path


def retention_rows(timeline: Timeline) -> list[tuple[float, int, str, str]]:
    cues = [
        (0, "hook", "A proton fills the screen. The promise: smallest to biggest, all of it."),
        (96, "first zoom", "The proton shrinks to a dot beside an atom 60,000x wider. The counter's zeros start to fall away."),
        (384, "speed up", "Landings come twice as often. The arp doubles."),
        (696, "wait for it", "A cat, the arp at full speed, then the silent beat."),
        (768, "drop", "YOU ARE HERE. The viewer is the object."),
        (960, "city", "Statues, towers, the space station. The second impact."),
        (1152, "space", "The ground falls away. Moon, Earth, Jupiter, the Sun, a star 900 Suns wide."),
        (1440, "breakdown", "The music thins out. A nebula, our galaxy, then the supercluster it lives in. The counter runs out of room."),
        (1632, "impact", "The observable universe."),
        (1700, "edge", "AND THAT'S ONLY THE PART WE CAN SEE."),
        (1800, "snap", "Back to the proton. Frame 1823 is frame 0."),
    ]
    return [(timeline.seconds(frame), frame, label, feeling) for frame, label, feeling in cues]


def retention_markdown(timeline: Timeline) -> str:
    lines = [
        "# Retention map",
        "",
        "Times are `frame / 60`. Read a YouTube retention graph against the feeling column.",
        "The drop is at 12.8 s (frame 768) and the universe lands on the impact at 27.2 s (frame 1632).",
        "",
        "| time | frame | event | what the viewer should feel |",
        "| --- | --- | --- | --- |",
    ]
    for seconds, frame, label, feeling in retention_rows(timeline):
        lines.append(f"| {seconds:.1f} s | {frame} | {label} | {feeling} |")
    lines.append("")
    return "\n".join(lines)


def facts_markdown(timeline: Timeline, sizes: dict[str, dict]) -> str:
    scene = timeline.scene
    lines = [
        "# Scale facts",
        "",
        "Every size is cited in configs/scale_claims.yaml. The counter is that size in plain metres.",
        f"Largest distance from a landing to the sixteenth-note grid: {timeline.max_snap_error:.3e} frames.",
        "",
        "| land | object | counter | friendly line | x bigger than the last | source |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    previous = None
    for placed, counter, display in zip(scene.placed, scene.counters, scene.displays):
        item = placed.item
        ratio = "" if previous is None else f"{item.size_m / previous:,.1f}"
        source = str(sizes[item.id].get("source", "")).split(";")[0]
        lines.append(f"| {item.land} | {item.name} | {counter} | {display} | {ratio} | {source} |")
        previous = item.size_m
    lines.append("")
    return "\n".join(lines)
