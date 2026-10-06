"""Musical events for the coinspin short. Audio plays this list.

Every upright face is a bell, every sideways or upside-down face before the SAT act is a
pluck, the bolts click, and the "ohhh" frame (the trip-around spin landing) gets the chime.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from fc_sat.beatkit.grid import snap_frame
from fc_sat.coinspin_claims import ClaimBook, evaluate, load_claims
from fc_sat.coinspin_config import CoinConfig, load_config
from fc_sat.coinspin_math import BOLTS, CARRY, DROP, RESIZE, SAT_END, quarter_frames, segment_spins, upright_frames
from fc_sat.coinspin_schedule import counter_at

ROOT = Path(__file__).resolve().parents[1]
CHORDS = ("Am", "F", "C", "G")
# Pentatonic degree (0 is A4, 5 is A5, 10 is A6) for each bell.
BELL_DEGREES = {288: 5, DROP: 10, 1032: 6, 1224: 8, 1320: 10, 1452: 5, 1512: 6, 1572: 7, SAT_END: 10, 1740: 8}
EXTRA_PLUCKS = (924, 1056, 1272, 1296, 1368, 1376, 1384, 1680)
CHIME = CARRY[1]


def _beat(beat: float, bpm: float, fps: int) -> int:
    return snap_frame(beat, bpm, fps)[0]


@dataclass
class Timeline:
    config: CoinConfig
    max_snap_error: float
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
    uprights: list[int]
    chords: list[str] = field(default_factory=list)

    @property
    def n_frames(self) -> int:
        return self.config.frames

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
            "max_snap_error_frames": self.max_snap_error,
            "uprights": self.uprights,
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
            "tail": [1806, 1824],
        }


def _unique(frames, n_frames: int) -> list[int]:
    return sorted({int(frame) for frame in frames if 0 <= frame < n_frames})


def _events(bpm: float, fps: int, n_frames: int) -> dict[str, list[int]]:
    def beat(value: float) -> int:
        return _beat(value, bpm, fps)

    kicks = [beat(index) for index in range(75) if index not in (30, 31) and not 60 <= index <= 67]
    claps = [beat(index) for index in range(8, 75) if index % 4 in (1, 3) and index != 31 and not 60 <= index <= 67]
    hats = [beat(index + 0.5) for index in range(16)]
    for step in list(range(16 * 4, 31 * 4)) + list(range(32 * 4, 60 * 4)):
        hats.append(beat(step / 4.0))
    for step in range(68 * 4, 75 * 4):
        if beat(step / 4.0) < 1800:
            hats.append(beat(step / 4.0))
    hats = [frame for frame in hats if not 744 <= frame < DROP]
    open_hats = [beat(index) for index in (32, 36, 40, 44, 48, 52, 56, 68, 72)]
    snares = list(range(576, 608, 12)) + list(range(608, 640, 6)) + list(range(640, 672, 3))

    arp = [frame for frame in quarter_frames() if frame < RESIZE[0]]
    arp += list(EXTRA_PLUCKS)
    for step in range(40 * 2, 56 * 2):
        arp.append(beat(step / 2.0))
    for index in range(60, 68):
        arp.append(beat(index))
    for step in range(68 * 2, 75 * 2):
        if beat(step / 2.0) < 1800:
            arp.append(beat(step / 2.0))
    arp = [frame for frame in arp if frame not in BELL_DEGREES and not 744 < frame < DROP]

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
        "impacts": [DROP, 960, SAT_END],
        "ticks": list(BOLTS),
    }


def build_timeline(config: CoinConfig | None = None) -> Timeline:
    config = config or load_config()
    events = _events(config.bpm, config.fps, config.frames)
    uprights = upright_frames()
    missing = [frame for frame in uprights if frame not in BELL_DEGREES]
    if missing:
        raise SystemExit(f"uprights without a bell: {missing}")
    # Uprights and the act 1/2 quarter turns must sit on the sixteenth-note grid.
    sixteenth = 60.0 * config.fps / config.bpm / 4.0
    error = 0.0
    for frame in uprights + [frame for frame in quarter_frames() if frame < RESIZE[0]]:
        error = max(error, abs(frame - round(frame / sixteenth) * sixteenth))
    bells = sorted(BELL_DEGREES)
    return Timeline(
        config=config,
        max_snap_error=error,
        kicks=events["kicks"],
        claps=events["claps"],
        hats=events["hats"],
        open_hats=events["open_hats"],
        snares=events["snares"],
        arp=events["arp"],
        bass=events["bass"],
        bells=bells,
        bell_degrees=[BELL_DEGREES[frame] for frame in bells],
        impacts=events["impacts"],
        ticks=events["ticks"],
        chime=CHIME,
        uprights=uprights,
        chords=[CHORDS[bar % 4] for bar in range(19)],
    )


def write_timeline(timeline: Timeline, path: Path | None = None) -> Path:
    path = path or (ROOT / "out" / "coinspin_timeline.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(timeline.to_json(), indent=2))
    return path


def retention_rows(timeline: Timeline) -> list[tuple[float, int, str, str]]:
    cues = [
        (0, "hook", "The question names both coins and what to count: the face coming back upright."),
        (144, "the guess", "Same size coins, so just 1 spin? The face is already turning fast."),
        (288, "halfway", "Freeze at the bottom. Half the edge is painted and the face is upright: 1 spin already."),
        (384, "other half", "The second half repeats the first. A faint copy marks the finish."),
        (744, "silent beat", "It stops just short of home."),
        (768, "drop", "2 spins. Two upright faces on screen at once."),
        (876, "why", "The 2 becomes ? + ? = 2. The painted edge peels off into a flat road."),
        (1032, "rolling", "Along the flat edge it spins once: the 1 everyone expected."),
        (1104, "bolted", "Bolted to a rod, it cannot roll, yet its face still turns as it is carried around."),
        (1224, "ohhh", "1 + 1 = 2: the trip around adds a spin all by itself."),
        (1344, "SAT", "The 1982 SAT asked this with a grey coin 3x wider. Its answer was 3."),
        (1572, "not back yet", "The count hits 3 with a quarter of the trip still to go."),
        (1632, "4", "4 upright faces in a cross. The SAT's 3 is struck out."),
        (1680, "rule", "3 + 1 = 4. Each trip around adds 1 spin."),
        (1800, "snap", "Back to the question. Frame 1823 is frame 0."),
    ]
    return [(timeline.seconds(frame), frame, label, feeling) for frame, label, feeling in cues]


def retention_markdown(timeline: Timeline) -> str:
    lines = [
        "# Retention map",
        "",
        "Times are `frame / 60`. Read a YouTube retention graph against the feeling column.",
        "The drop is at 12.8 s (frame 768) and the SAT answer lands on the impact at 27.2 s (frame 1632).",
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
        "# Coinspin facts",
        "",
        "Every value below is read from gold_pose, the function the renderer draws, or cited.",
        f"Largest distance from an upright or a quarter turn to the sixteenth-note grid: {timeline.max_snap_error:.3e} frames.",
        "",
        "| segment | frames | spins |",
        "| --- | --- | --- |",
        f"| lap around a same-size coin | 12-768 | {segment_spins(12, DROP):g} |",
        f"| along the grey edge laid flat | 936-1032 | {segment_spins(936, 1032):g} |",
        f"| bolted and carried around | 1128-1224 | {segment_spins(1128, 1224):g} |",
        f"| SAT lap, grey coin 3x wider | 1384-1632 | {segment_spins(1384, SAT_END):g} |",
        "",
        "Upright frames (bells): " + ", ".join(str(frame) for frame in timeline.uprights) + ".",
        "",
        "| frame | counter |",
        "| --- | --- |",
    ]
    for frame in (0, 288, 768, 876, 1032, 1224, 1368, 1452, 1512, 1572, 1632, 1680, 1800):
        counter = counter_at(frame)
        if counter.mode == "equation":
            a, b, c = counter.terms
            shown = f"{a.text} + {b.text} = {c.text}"
        else:
            shown = counter.terms[0].text
        if counter.chip is not None:
            shown += f" (chip: {counter.chip.label} {counter.chip.value})"
        lines.append(f"| {frame} | {shown} |")
    lines += ["", f"Claims {sum(1 for _, ok, _ in results if ok)}/{len(results)} passed.", ""]
    return "\n".join(lines)
