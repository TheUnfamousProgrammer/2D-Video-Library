"""Musical events for the coinspin short. Audio plays this list."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from fc_sat.beatkit.grid import snap_frame
from fc_sat.coinspin_claims import ClaimBook, evaluate, load_claims
from fc_sat.coinspin_config import CoinConfig, load_config
from fc_sat.coinspin_math import DOUBLING_FRAMES, DOUBLING_RATIOS, LAPS, YEAR, sidereal_day
from fc_sat.coinspin_schedule import counter_at, quarter_frames, spin_frames, uncounted_spin_frames
from fc_sat.polycircle_format import format_count

ROOT = Path(__file__).resolve().parents[1]
CHORDS = ("Am", "F", "C", "G")
# Pentatonic degrees (0 is A4) for each spin bell, then a motif for the doublings.
SPIN_DEGREES = (3, 5, 4, 5, 6, 5, 6, 7, 10)
POST_DROP_DEGREES = (8, 7, 6)
DOUBLING_MOTIF = (7, 8, 9, 10, 9, 8, 7, 5)
DROP = 768


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
    spins: list[int]
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
            "laps": [
                {"start": lap.start, "end": lap.end, "ratio": lap.ratio, "counted": lap.counted} for lap in LAPS
            ],
            "doublings": [
                {"frame": frame, "ratio": ratio} for frame, ratio in zip(DOUBLING_FRAMES, DOUBLING_RATIOS)
            ],
            "spins": self.spins,
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


def _unique(frames: list[int], n_frames: int) -> list[int]:
    return sorted({frame for frame in frames if 0 <= frame < n_frames})


def _events(bpm: float, fps: int, n_frames: int) -> dict[str, list[int]]:
    def beat(value: float) -> int:
        return _beat(value, bpm, fps)

    kicks = [beat(index) for index in range(75) if index not in (30, 31) and not 60 <= index <= 67]
    claps = [
        beat(index)
        for index in range(8, 75)
        if index % 4 in (1, 3) and index != 31 and not 60 <= index <= 67
    ]
    hats = [beat(index + 0.5) for index in range(16)]
    for step in list(range(16 * 4, 31 * 4)) + list(range(32 * 4, 60 * 4)):
        hats.append(beat(step / 4.0))
    for step in range(68 * 4, 75 * 4):
        frame = beat(step / 4.0)
        if frame < 1800:
            hats.append(frame)
    hats = [frame for frame in hats if not 744 <= frame < DROP]
    open_hats = [beat(index) for index in (32, 36, 40, 44, 48, 52, 56, 68, 72)]
    snares = list(range(576, 608, 12)) + list(range(608, 640, 6)) + list(range(640, 672, 3))

    # Before the drop every quarter turn of the arrow is a pluck. After it the
    # groove keeps the eighth-note pattern of the other 150 bpm films.
    arp = [frame for frame in quarter_frames() if frame < DROP]
    for step in range(32 * 2, 60 * 2):
        arp.append(beat(step / 2.0))
    for index in range(60, 68):
        arp.append(beat(index))
    for step in range(68 * 2, 75 * 2):
        frame = beat(step / 2.0)
        if frame < 1800:
            arp.append(frame)
    arp = [frame for frame in arp if frame != DROP]

    bass = [beat(index + 0.5) for index in range(8, 16)]
    bass += [beat(index + 0.5) for index in range(32, 60)]
    bass += [beat(index + 0.5) for index in range(68, 75) if beat(index + 0.5) < 1800]
    bass.append(beat(72))

    ticks = list(range(YEAR[0], YEAR[1], 6))
    return {
        "kicks": _unique(kicks, n_frames),
        "claps": _unique(claps, n_frames),
        "hats": _unique(hats, n_frames),
        "open_hats": _unique(open_hats, n_frames),
        "snares": _unique(snares, n_frames),
        "arp": _unique(arp, n_frames),
        "bass": _unique(bass, n_frames),
        "impacts": [DROP, DOUBLING_FRAMES[0], 1632],
        "ticks": _unique(ticks, n_frames),
    }


def _bells() -> tuple[list[int], list[int]]:
    spins = spin_frames()
    if len(spins) != len(SPIN_DEGREES):
        raise SystemExit(f"{len(spins)} counted spins but {len(SPIN_DEGREES)} spin bells")
    post = [frame for frame in uncounted_spin_frames() if frame < DOUBLING_FRAMES[0]]
    if len(post) != len(POST_DROP_DEGREES):
        raise SystemExit(f"{len(post)} post-drop spins but {len(POST_DROP_DEGREES)} bells")
    frames = list(spins) + post + list(DOUBLING_FRAMES)
    degrees = list(SPIN_DEGREES) + list(POST_DROP_DEGREES)
    degrees += [DOUBLING_MOTIF[index % len(DOUBLING_MOTIF)] for index in range(len(DOUBLING_FRAMES))]
    return frames, degrees


def build_timeline(config: CoinConfig | None = None) -> Timeline:
    config = config or load_config()
    events = _events(config.bpm, config.fps, config.frames)
    # Spins, quarter turns, and doublings must sit on the sixteenth-note grid.
    sixteenth = 60.0 * config.fps / config.bpm / 4.0
    error = 0.0
    for frame in spin_frames() + list(DOUBLING_FRAMES) + quarter_frames():
        nearest = round(frame / sixteenth) * sixteenth
        error = max(error, abs(frame - nearest))
    bells, degrees = _bells()
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
        bell_degrees=degrees,
        impacts=events["impacts"],
        ticks=events["ticks"],
        chime=YEAR[1],
        spins=spin_frames(),
        chords=[CHORDS[bar % 4] for bar in range(19)],
    )


def write_timeline(timeline: Timeline, path: Path | None = None) -> Path:
    path = path or (ROOT / "out" / "coinspin_timeline.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(timeline.to_json(), indent=2))
    return path


def retention_rows(timeline: Timeline) -> list[tuple[float, int, str, str]]:
    cues = [
        (0, "hook", "The question is on screen and the gold coin is already rolling. Everyone guesses 1."),
        (96, "first spin", "The counter says 1 while the coin is only halfway around. Something is off."),
        (192, "two", "One lap, two spins. The guess was wrong."),
        (216, "bigger coin", "A coin twice as wide. The arrow spins faster and the counter lands on 3."),
        (384, "SAT", "The 1982 SAT asks the same question. Five answer choices appear on the beat."),
        (672, "they said 3", "The counter hits 3, the test's answer, but the coin still has a quarter lap to go."),
        (744, "silent gap", "The beat drops out for the last few degrees."),
        (768, "drop", "Four. Not one of the choices."),
        (876, "why", "3 from rolling along the rim, plus 1 from the trip around."),
        (960, "doublings", "The big coin doubles on every beat. The +1 stays."),
        (1344, "earth", "The coins become the Sun and Earth."),
        (1392, "year", "One orbit. Days and spins race."),
        (1488, "366", "365 days, 366 spins. One spin is 23 h 56 m 4 s."),
        (1632, "never", "The +1 never goes away."),
        (1728, "the trip", "The orbit is traced in gold. The extra spin is the trip around."),
        (1800, "snap", "Back to two coins. The question returns."),
        (1823, "loop", "The last frame is the first frame. The next kick lands clean."),
    ]
    return [(timeline.seconds(frame), frame, label, feeling) for frame, label, feeling in cues]


def retention_markdown(timeline: Timeline) -> str:
    lines = [
        "# Retention map",
        "",
        "Times are `frame / 60`. Read a YouTube retention graph against the feeling column.",
        "The drop is at 12.8 s (frame 768) and NEVER is at 27.2 s (frame 1632), on bar lines.",
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
    day = sidereal_day(365.24219)
    lines = [
        "# Coinspin facts",
        "",
        "Every value below is computed by the rolling function the renderer draws, or cited.",
        f"Largest distance from a spin, quarter turn, or doubling to the sixteenth-note grid: {timeline.max_snap_error:.3e} frames.",
        "",
        "| lap | frames | big coin | spins | spin frames |",
        "| --- | --- | --- | --- | --- |",
    ]
    for index, lap in enumerate(LAPS, start=1):
        span = lap.end - lap.start
        frames = [lap.start + span * spin // (lap.ratio + 1) for spin in range(1, lap.ratio + 2)]
        lines.append(
            f"| {index} | {lap.start}-{lap.end} | {lap.ratio}x | {lap.ratio + 1} | {', '.join(map(str, frames))} |"
        )
    lines += [
        "",
        f"Inside a ring 3x wider the coin spins {book.get('inside_spins').value} times.",
        f"Doubling ratios: {', '.join(format_count(ratio) for ratio in DOUBLING_RATIOS)}.",
        f"Sidereal day from 365.24219 solar days: {day:.4f} s.",
        "",
        "| frame | counter |",
        "| --- | --- |",
    ]
    for frame in (0, 96, 192, 264, 360, 480, 672, 768, 876, 960, 1343, 1440, 1488, 1632, 1800, 1823):
        counter = counter_at(frame)
        if counter.mode == "sky":
            shown = f"days {counter.days}, spins {counter.spins}"
        elif counter.mode == "formula":
            shown = f"{format_count(counter.value)} + {counter.plus}"
        else:
            shown = str(counter.value)
        lines.append(f"| {frame} | {shown} |")
    lines += [
        "",
        "Trace colors: " + " ".join(timeline.config.trace_colors),
        "",
        f"Claims {sum(1 for _, ok, _ in results if ok)}/{len(results)} passed.",
        "",
    ]
    return "\n".join(lines)
