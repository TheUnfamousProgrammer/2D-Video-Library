"""Musical events for the deepest-to-highest short: the scale short's score, with a bell on every
landing. On the dive the bells fall with depth; on the climb they rise with height."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from fc_sat.dive_scene import DiveScene, build_scene
from fc_sat.scale_timeline import BPM, CHIME, CHORDS, FPS, IMPACTS, N_FRAMES, SAMPLE_RATE, _events

ROOT = Path(__file__).resolve().parents[1]
SEED = 31
TOP_DEGREE = 14


@dataclass
class DiveTimeline:
    scene: DiveScene
    events: dict
    bells: list[int]
    bell_degrees: list[int]
    lands: list[int]
    max_snap_error: float
    chords: list[str] = field(default_factory=list)
    n_frames: int = N_FRAMES
    fps: int = FPS
    seed: int = SEED

    @property
    def kicks(self) -> list[int]:
        return self.events["kicks"]

    @property
    def impacts(self) -> list[int]:
        return list(IMPACTS)

    def to_json(self) -> dict:
        return {
            "fps": self.fps,
            "bpm": BPM,
            "n_frames": self.n_frames,
            "sample_rate": SAMPLE_RATE,
            "max_snap_error_frames": self.max_snap_error,
            "lands": self.lands,
            **self.events,
            "bells": self.bells,
            "bell_degrees": self.bell_degrees,
            "impacts": list(IMPACTS),
            "ticks": [],
            "chime": CHIME,
            "chords": self.chords,
            "silent_gap": [744, 768],
            "tail": [1806, self.n_frames],
        }


def _degree(share: float) -> int:
    return int(round(TOP_DEGREE * min(1.0, max(0.0, share))))


def build_timeline(scene: DiveScene | None = None) -> DiveTimeline:
    scene = scene or build_scene()
    stops = list(scene.stops)
    down = [s for s in stops if s.act == "down" and s.value_m > 0]
    up = [s for s in stops if s.act == "up"]
    bells, degrees = [], []
    for stop in stops[1:]:
        if stop.act == "down":
            lo, hi = math.log10(down[0].value_m), math.log10(down[-1].value_m)
            share = 1.0 - (math.log10(stop.value_m) - lo) / (hi - lo)
            degree = 3 + _degree(share) // 2  # falls from about 10 to 3
        else:
            lo, hi = math.log10(up[0].value_m), math.log10(up[-1].value_m)
            degree = _degree((math.log10(stop.value_m) - lo) / (hi - lo))
        bells.append(stop.land)
        degrees.append(degree)
    sixteenth = 60.0 * FPS / BPM / 4.0
    lands = [s.land for s in stops]
    error = max(abs(f - round(f / sixteenth) * sixteenth) for f in lands)
    return DiveTimeline(scene, _events(), bells, degrees, lands, error, [CHORDS[bar % 4] for bar in range(19)])


def write_timeline(timeline: DiveTimeline, path: Path | None = None) -> Path:
    path = path or (ROOT / "out" / "dive_timeline.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(timeline.to_json(), indent=2))
    return path


def facts_markdown(timeline: DiveTimeline, values: dict[str, dict]) -> str:
    lines = ["# Dive facts", "", "Every value is cited in configs/dive_claims.yaml.", "", "| land | stop | counter | friendly line | source |", "| --- | --- | --- | --- | --- |"]
    for stop, counter, display in zip(timeline.scene.stops, timeline.scene.counters, timeline.scene.displays):
        lines.append(f"| {stop.land} | {stop.name} | {counter} | {display} | {values[stop.id].get('source', '')} |")
    lines.append("")
    return "\n".join(lines)
