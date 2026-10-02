"""Frame-snapped Collatz timeline. Hook A is the default; S9 can slide later."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
TIMELINE_PATH = ROOT / "configs" / "collatz_timeline.yaml"


class TimelineError(ValueError):
    pass


def snap(seconds: float, fps: int) -> float:
    return round(seconds * fps) / fps


def frame_of(seconds: float, fps: int) -> int:
    return int(round(seconds * fps))


@dataclass
class Scene:
    id: str
    name: str
    start: float
    end: float
    spec: dict

    def rel(self, offset: float, fps: int) -> float:
        return snap(self.start + offset, fps)


@dataclass
class Timeline:
    fps: int
    width: int
    height: int
    scenes: list[Scene]
    duration: float
    hook: str
    spec: dict
    shifted_by: float = 0.0

    @property
    def n_frames(self) -> int:
        return frame_of(self.duration, self.fps)

    def scene(self, scene_id: str) -> Scene:
        for scene in self.scenes:
            if scene.id == scene_id:
                return scene
        raise KeyError(scene_id)

    def at(self, seconds: float) -> Scene:
        for scene in self.scenes:
            if scene.start <= seconds < scene.end or (scene.id == "S9" and seconds <= scene.end):
                if seconds < scene.end or scene is self.scenes[-1]:
                    return scene
        return self.scenes[-1]


def load_timeline_spec(path: Path | None = None) -> dict:
    return yaml.safe_load((path or TIMELINE_PATH).read_text())


def build_timeline(
    spec: dict | None = None,
    *,
    hook: str = "A",
    hook_end: float | None = None,
    last_vo_end: float | None = None,
) -> Timeline:
    spec = spec or load_timeline_spec()
    fps = int(spec["fps"])
    cursor = 0.0
    scenes: list[Scene] = []
    for body in spec["scenes"]:
        start = snap(cursor, fps)
        end = snap(cursor + float(body["duration"]), fps)
        scenes.append(Scene(body["id"], body["name"], start, end, body))
        cursor = end
    shifted = 0.0
    if hook_end is not None:
        s2 = next(scene for scene in scenes if scene.id == "S2")
        if hook_end > s2.start:
            shifted = snap(hook_end - s2.start, fps)
            if shifted > float(spec["hook_shift_max_s"]) + 1e-9:
                raise TimelineError(
                    f"hook {hook} ends {hook_end:.2f}s, which needs a {shifted:.2f}s shift "
                    f"(max {spec['hook_shift_max_s']}s)"
                )
            for scene in scenes:
                if scene.id == "S1":
                    scene.end = snap(scene.end + shifted, fps)
                    continue
                scene.start = snap(scene.start + shifted, fps)
                scene.end = snap(scene.end + shifted, fps)
    s8 = next(scene for scene in scenes if scene.id == "S8")
    s9 = next(scene for scene in scenes if scene.id == "S9")
    nominal = s9.start
    if last_vo_end is not None:
        pushed = snap(max(nominal, last_vo_end + float(spec["bridge_gap_s"])), fps)
    else:
        pushed = nominal
    if pushed > s9.start:
        s9.start = pushed
        s9.end = snap(pushed + float(spec["s9_duration"]), fps)
        s8.end = s9.start
    duration = s9.end
    if duration > float(spec["hard_cap_s"]) + 1e-9:
        raise TimelineError(f"duration {duration:.2f}s exceeds the {spec['hard_cap_s']}s cap")
    return Timeline(fps, int(spec["width"]), int(spec["height"]), scenes, duration, hook, spec, shifted)


def s2_step_frames(timeline: Timeline) -> list[int]:
    scene = timeline.scene("S2")
    return [frame_of(scene.rel(offset, timeline.fps), timeline.fps) for offset in scene.spec["step_offsets"]]


def ride_step_time(timeline: Timeline, step: int) -> float:
    scene = timeline.scene("S5")
    frames = int(scene.spec["frames_per_step"])
    return snap(scene.start + step * frames / timeline.fps, timeline.fps)


def proof_end(timeline: Timeline, n: int) -> float:
    from fc_sat.collatz_math import steps

    scene = timeline.scene("S3")
    frames = int(scene.spec["frames_per_step"])
    for run in scene.spec["runs"]:
        if int(run["n"]) == n:
            count = steps(n)
            return snap(scene.rel(float(run["offset"]), timeline.fps) + count * frames / timeline.fps, timeline.fps)
    raise KeyError(n)


def retention_rows(timeline: Timeline, script_lines) -> list[tuple[str, float, str]]:
    by_id = {line.id: line for line in script_lines}
    shift = timeline.shifted_by
    def moved(line_id: str) -> float:
        line = by_id[line_id]
        base = line.offset
        s2 = timeline.scene("S2").start - shift
        if base >= s2 - 1e-9:
            base += shift
        return snap(base, timeline.fps)

    return [
        ("hook", 0.0, "hook"),
        ("rule", timeline.scene("S2").start, "rule"),
        ("first_arrival", timeline.scene("S2").rel(timeline.scene("S2").spec["step_offsets"][-1], timeline.fps), "first arrival chime"),
        ("second_arrival", proof_end(timeline, 9), "second arrival"),
        ("interrupt", timeline.scene("S4").start, "pattern interrupt (silence)"),
        ("stamp_27", timeline.scene("S5").start, "stamp of 27"),
        ("step_count", moved("L9"), "step-count line"),
        ("cross_1000", ride_step_time(timeline, int(timeline.scene("S5").spec["cross_step"])), "step 36 crosses 1,000"),
        ("peak", ride_step_time(timeline, int(timeline.scene("S5").spec["peak_step"])), "peak"),
        ("arrival", timeline.scene("S5").end, "arrival"),
        ("funnel", timeline.scene("S6").rel(float(timeline.scene("S6").spec["relayout_s"]), timeline.fps), "funnel starts"),
        ("resolve", timeline.scene("S7").start, "resolve"),
        ("punchline", timeline.scene("S8").start, "punchline"),
        ("bridge", timeline.scene("S9").start, "loop bridge"),
    ]


def retention_markdown(rows: list[tuple[str, float, str]]) -> str:
    lines = ["# Retention map", "", "Read the YouTube retention graph against these beats.", ""]
    for _name, when, purpose in rows:
        lines.append(f"- {when:.2f}s {purpose}")
    lines.append("")
    lines.append(
        "Step 36 is the frame when the ride, at 5 frames per step from the 27 stamp, "
        "first draws a value above 1,000. That is 16.30s on hook A, not 17.30s."
    )
    lines.append("")
    return "\n".join(lines)


def facts_markdown(book) -> str:
    from fc_sat.collatz_format import format_int

    lines = ["# Collatz facts", ""]
    for claim in book.claims.values():
        shown = claim.value
        if isinstance(shown, int) and not isinstance(shown, bool) and shown > 999:
            shown = f"{shown} ({format_int(shown)})"
        lines.append(f"- `{claim.id}`: {claim.text} Value: {shown}.")
    lines.append("")
    return "\n".join(lines)
