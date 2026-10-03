"""Frame-exact timeline. Audio events are listed here so the mix does not invent them."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from fc_sat.beatkit.grid import FRAMES_PER_BEAT, N_FRAMES
from fc_sat.paperfold_claims import ClaimBook, evaluate, load_claims
from fc_sat.paperfold_config import PaperConfig, load_config
from fc_sat.paperfold_format import milky_way_label, space_km_label, stamp_lines
from fc_sat.paperfold_math import milestone_folds
from fc_sat.paperfold_schedule import SEGMENTS, FoldEvent, bar_start, build_folds, max_snap_error, segment_at

ROOT = Path(__file__).resolve().parents[1]

# Am, F, C, G, one chord per bar. Semitones above A.
CHORDS = ((0, 3, 7), (5, 9, 12), (3, 7, 10), (7, 11, 14))
# A natural minor, three octaves, then it wraps.
MINOR = (0, 2, 3, 5, 7, 8, 10)


@dataclass
class Timeline:
    config: PaperConfig
    folds: list[FoldEvent]
    kicks: list[int]
    hats: list[int]
    claps: list[int]
    bass: list[int]
    snares: list[int]
    folds_sfx: list[int]
    arp: list[int]
    arp_degrees: list[int]
    bells: list[int]
    ticks: list[int]
    impacts: list[int]
    pickup: int
    thump: int
    tape: int
    max_snap_error: float

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
            "frames": self.n_frames,
            "folds": [{"n": item.n, "frame": item.frame} for item in self.folds],
            "kicks": self.kicks,
            "hats": self.hats,
            "claps": self.claps,
            "bass": self.bass,
            "snares": self.snares,
            "folds_sfx": self.folds_sfx,
            "arp": self.arp,
            "arp_degrees": self.arp_degrees,
            "bells": self.bells,
            "ticks": self.ticks,
            "impacts": self.impacts,
            "pickup": self.pickup,
            "thump": self.thump,
            "tape": self.tape,
            "chords": [list(chord) for chord in CHORDS],
            "segments": [{"name": name, "start": start, "end": end} for name, start, end in SEGMENTS],
        }


def _silent(frame: int) -> bool:
    """Beat 4 of bar 6, and the reality breakdown. The tape-stop frame is handled apart."""
    if 552 <= frame < 576:
        return True
    if 1248 <= frame < 1440:
        return True
    return False


def _kicks() -> list[int]:
    frames = []
    for frame in range(0, N_FRAMES, FRAMES_PER_BEAT):
        if _silent(frame) or frame == 1800:
            continue
        frames.append(frame)
    return frames


def _offbeat_hats() -> list[int]:
    """Eighth-note offbeats in bars 1-2, then sixteenths once the pad enters."""
    early = [frame for frame in range(12, 192, 24)]
    late = []
    for frame in range(384, N_FRAMES, 6):
        if _silent(frame) or frame >= 1800:
            continue
        if 672 <= frame < 864:
            continue
        late.append(frame)
    return early + late


def _claps() -> list[int]:
    frames = []
    for bar in (3, 4, 5, 6, 7, 10, 11, 12, 13, 16, 17, 18, 19):
        start = bar_start(bar)
        for beat in (2, 4):
            frame = start + (beat - 1) * FRAMES_PER_BEAT
            if _silent(frame) or frame >= 1800:
                continue
            frames.append(frame)
    return frames


def _bass() -> list[int]:
    """Offbeat quarter notes: the '&' of each beat, in the bars that have a bass."""
    frames = []
    for bar in (3, 4, 5, 6, 7, 10, 11, 12, 13, 16, 17, 18, 19):
        start = bar_start(bar)
        for beat in range(4):
            frame = start + beat * FRAMES_PER_BEAT + 12
            if _silent(frame) or frame >= 1800:
                continue
            frames.append(frame)
    return frames


def _snares() -> list[int]:
    """Accelerating roll from frame 480 up to the silent gap. Spacing halves twice."""
    frames = []
    gap = 12
    frame = 480
    while frame < 552:
        frames.append(frame)
        if frame >= 516:
            gap = 6
        if frame >= 534:
            gap = 3
        frame += gap
    return frames


def build_timeline(config: PaperConfig | None = None) -> Timeline:
    config = config or load_config()
    folds = build_folds()
    sfx = [item.frame for item in folds]
    # Step up the A minor scale and wrap every 3 octaves (21 notes).
    degrees = [index % (len(MINOR) * 3) for index in range(len(sfx))]
    bells = [item.frame for item in folds if item.n >= 31]
    return Timeline(
        config=config,
        folds=folds,
        kicks=_kicks(),
        hats=_offbeat_hats(),
        claps=_claps(),
        bass=_bass(),
        snares=_snares(),
        folds_sfx=sfx,
        arp=list(sfx),
        arp_degrees=degrees,
        bells=bells,
        ticks=[696, 744, 792],
        impacts=[576, 1152, 1440, 1632],
        pickup=1128,
        thump=1536,
        tape=1800,
        max_snap_error=max_snap_error(config.bpm, config.fps),
    )


def write_timeline(timeline: Timeline, path: Path | None = None) -> Path:
    path = path or (ROOT / "out" / "timeline.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(timeline.to_json(), indent=2))
    return path


def retention_rows(timeline: Timeline) -> list[tuple[float, int, str, str]]:
    """Time, frame, event, feeling. Times are frame / 60.

    The brief lists 12.4 s as the silent gap. That time is frame 744, which is
    the Earth-wrap stamp. This film's silent gap is frames 552-575 (9.2 s),
    the beat before the space drop. Both rows are here so a retention graph
    can be read against the picture that is actually on screen.
    """
    cues = [
        (0, "hook", "A big sheet is already mid-fold. The question is on screen. Nothing fades in."),
        (12, "first fold", "The first fold lands on the eighth note, with the kick."),
        (84, "tower", "Fold 7 lands. The ghost copy has doubled the stack, and the tower view is fully up."),
        (168, "fold 14", "The hook's last fold. The tower is the subject now."),
        (216, "person approaching", "A person-sized line is coming down to meet the stack."),
        (408, "Burj", "The building is beside the tower. The stack just passed it."),
        (432, "person passed", "Human scale is gone. The next labels are buildings and mountains."),
        (504, "Everest", "The mountain stamp lands. The climb is still speeding up."),
        (552, "silent gap", "The beat drops out. One fold note is left, then space."),
        (576, "space drop", "The background goes dark. The caption says that's space."),
        (672, "the catch", "The tower freezes. The question stops being about height."),
        (744, "wraps the Earth", "The paper-needed marker jumps past one trip around the Earth. This is 12.4 s, not the silent gap."),
        (792, "400 times to the Sun", "Thirty folds of paper is hundreds of Earth-Sun distances."),
        (864, "theory restarts", "The filter opens. In theory, the folding continues."),
        (1128, "Moon pass", "The stack reaches the Moon. The stamp hits."),
        (1152, "second drop", "The paper needed is as wide as the Milky Way."),
        (1248, "real record", "The tower shrinks to something a person can stand next to."),
        (1440, "answer", "In math, 42. The beat comes back on that number."),
        (1536, "real life", "In real life, 12. A softer hit."),
        (1632, "outro", "Paper stops. Math doesn't."),
        (1800, "collapse", "Six frames back to the flat sheet. The hook text returns so the loop matches."),
    ]
    return [(timeline.seconds(frame), frame, label, feeling) for frame, label, feeling in cues]


def retention_markdown(timeline: Timeline) -> str:
    lines = [
        "# Retention map",
        "",
        "Times are `frame / 60`. Read a YouTube retention graph against the feeling column.",
        "The silent gap is 9.2 s (frames 552-575), the beat before the space drop.",
        "12.4 s is the Earth-wrap stamp at frame 744, not silence.",
        "The top-down view runs through frame 72. The change into the tower is frames 72-84.",
        "1.4 s (frame 84) is fold 7, the first tower doubling, and the tower is fully up there.",
        "2.8 s is fold 14, the last fold of the hook.",
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
    passed = milestone_folds()
    lines = [
        "# Paperfold facts",
        "",
        "Every value below is computed. Defaults: 0.1 mm paper, single-direction folds,",
        "Gallivan's formula, 150 bpm, 1824 frames, 60 fps, 1080x1920.",
        "The posted polycircle file was that full raster with fps=30 and -shortest, which kept 912 frames.",
        "Preview and hooks are 540x960 at 30 fps. Full is 1080x1920 at 60 fps, 1824 frames.",
        "The sheet-to-tower move is frames 72-84. Fold 7 at frame 84 is the first tower doubling.",
        f"Maximum beat snap error: {timeline.max_snap_error:.3e} frames (limit 0.5).",
        "",
        "Claims live in `configs/paperfold_claims.yaml`. `configs/claims.yaml` belongs to Collatz.",
        "Hook B is three lines because `REACH THE MOON?` is 15 characters.",
        "The multiplication sign is the font glyph, because JetBrains Mono ExtraBold has it.",
        "Britney Gallivan's name stays: Guinness records 12 folds on 27 January 2002 with 1,219 m of tissue.",
        "Everest is 8,848.86 m (2020), rounded to 8,849 m. The pass fold is the same either way.",
        "Earth's circumference is the WGS84 equatorial 40,075 km, not the mean-radius 40,030 km.",
        "",
        "## Milestone pass folds",
        "",
        "| milestone | fold |",
        "| --- | --- |",
    ]
    for name, fold in passed.items():
        lines.append(f"| {name} | {fold} |")
    lines += [
        "",
        f"Folds to the Sun (1 AU): {book.get('sun_folds').value}.",
        f"Space stamp: {space_km_label()}. Milky Way stamp: {milky_way_label()}.",
        f"30-fold paper stamp: {' / '.join(stamp_lines(30))}.",
        "",
        "## Schedule",
        "",
        "| fold | frame | segment |",
        "| --- | --- | --- |",
    ]
    for item in timeline.folds:
        if item.n in (1, 6, 7, 14, 15, 23, 27, 29, 30, 31, 42) or item.frame in (216, 408, 504, 552):
            lines.append(f"| {item.n} | {item.frame} | {segment_at(item.frame)} |")
    lines += [
        "",
        f"Folds: {len(timeline.folds)}. Fold 30 is frame {timeline.folds[29].frame}. Fold 42 is frame {timeline.folds[41].frame}.",
        "",
        "## Segments",
        "",
    ]
    for name, start, end in SEGMENTS:
        lines.append(f"- {name}: frames {start}-{end - 1} ({start / 60:.1f} s)")
    lines += ["", "## Claims", ""]
    failed = [claim_id for claim_id, ok, _detail in results if not ok]
    lines.append(f"{sum(1 for _, ok, _ in results if ok)}/{len(results)} passed.")
    if failed:
        lines.append("Failed: " + ", ".join(failed))
    lines.append("")
    return "\n".join(lines)
