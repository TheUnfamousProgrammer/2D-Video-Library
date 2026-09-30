"""Public-domain melody parser.

Tokens are ``NAME:BEATS``. A bar ``|`` is cosmetic and ignored. ``NAME`` is
``A``–``G`` with an optional ``#`` or ``b`` and an octave, or ``R`` for a rest.
MIDI is ``12 * (octave + 1) + semitone``. Frequency is ``440 * 2 ** ((midi - 69) / 12)``.

Leading rests are dropped. Any other rest adds its beats to the previous
flight. A trailing rest lengthens the flight that wraps from the last note
back onto note 0. ``octave_shift`` changes synthesis frequency only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

_TOKEN = re.compile(r"^([A-GR])([#b]?)(-?\d+)?:(-?[0-9]*\.?[0-9]+)$")
_SEMITONE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


class SongError(ValueError):
    """A melody token or the whole loop failed validation."""

    def __init__(self, index: int, message: str) -> None:
        self.index = index
        super().__init__(f"token {index}: {message}")


@dataclass(frozen=True)
class Note:
    name: str
    midi: int
    beats: float
    token_index: int


@dataclass(frozen=True)
class Song:
    title: str
    composer: str
    bpm: float
    beats_per_bar: int
    notes_source: str
    notes: tuple[Note, ...]
    flight_beats: tuple[float, ...]
    onset_beats: tuple[float, ...]
    total_beats: float
    octave_shift: int = 1

    @property
    def pitches(self) -> tuple[str, ...]:
        """Distinct pitch names, low MIDI to high, first spelling kept."""
        best: dict[int, str] = {}
        for note in self.notes:
            best.setdefault(note.midi, note.name)
        return tuple(best[midi] for midi in sorted(best))

    def frequency(self, midi: int) -> float:
        return midi_frequency(midi, self.octave_shift)


def midi_frequency(midi: int, octave_shift: int = 0) -> float:
    return 440.0 * 2.0 ** ((midi + 12 * octave_shift - 69) / 12.0)


def note_midi(name: str) -> int:
    """MIDI number for a pitch name. ``D#5`` and ``Eb5`` are the same number."""
    letter = name[0].upper()
    accidental = 0
    rest = name[1:]
    if rest[:1] in {"#", "b"}:
        accidental = 1 if rest[0] == "#" else -1
        rest = rest[1:]
    if letter not in _SEMITONE or not rest or not _signed_int(rest):
        raise ValueError(name)
    return 12 * (int(rest) + 1) + _SEMITONE[letter] + accidental


def parse_notes(source: str) -> tuple[tuple[Note, ...], tuple[float, ...], tuple[float, ...], float]:
    """Return notes, flight beats, onset beats, and total beats."""
    raw = source.split()
    parsed: list[tuple[str, float, int]] = []
    for index, token in enumerate(raw):
        if token == "|":
            continue
        match = _TOKEN.match(token)
        if match is None or (match.group(1) != "R" and match.group(3) is None):
            raise SongError(index, f"unknown note name '{token}'")
        if match.group(1) == "R" and (match.group(2) or match.group(3) is not None):
            raise SongError(index, f"unknown note name '{token}'")
        try:
            beats = float(match.group(4))
        except ValueError as exc:
            raise SongError(index, f"unknown note name '{token}'") from exc
        if beats <= 0.0 or beats != beats:
            raise SongError(index, f"beats must be positive, got '{token}'")
        name = "R" if match.group(1) == "R" else f"{match.group(1)}{match.group(2)}{match.group(3)}"
        parsed.append((name, beats, index))
    if not parsed:
        raise SongError(0, "song is empty")

    while parsed and parsed[0][0] == "R":
        parsed.pop(0)
    if not parsed:
        raise SongError(0, "song is empty")

    notes: list[Note] = []
    flights: list[float] = []
    for name, beats, index in parsed:
        if name == "R":
            if not flights:
                continue
            flights[-1] += beats
            continue
        try:
            midi = note_midi(name)
        except ValueError as exc:
            raise SongError(index, f"unknown note name '{name}'") from exc
        notes.append(Note(name=name, midi=midi, beats=beats, token_index=index))
        flights.append(beats)

    if len(notes) < 1:
        raise SongError(0, "song is empty")
    distinct = {note.midi for note in notes}
    if len(distinct) < 2:
        raise SongError(notes[-1].token_index, f"song has {len(distinct)} distinct pitches; need 2 to 10")
    if len(distinct) > 10:
        seen: set[int] = set()
        offender = notes[-1].token_index
        for note in notes:
            seen.add(note.midi)
            if len(seen) > 10:
                offender = note.token_index
                break
        raise SongError(offender, f"song has {len(distinct)} distinct pitches; need 2 to 10")

    onset = []
    cursor = 0.0
    for flight in flights:
        onset.append(cursor)
        cursor += flight
    return tuple(notes), tuple(flights), tuple(onset), cursor


def load_song(path: str | Path, *, octave_shift: int = 1, bpm: float | None = None) -> Song:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SongError(0, "song file must be a mapping")
    for field in ("title", "composer", "notes"):
        value = data.get(field)
        if not isinstance(value, str) or not value.strip():
            raise SongError(0, f"field '{field}' must be a non-empty string")
    file_bpm = data.get("bpm")
    if isinstance(file_bpm, bool) or not isinstance(file_bpm, (int, float)) or float(file_bpm) <= 0:
        raise SongError(0, "field 'bpm' must be a positive number")
    bars = data.get("beats_per_bar")
    if isinstance(bars, bool) or not isinstance(bars, int) or bars <= 0:
        raise SongError(0, "field 'beats_per_bar' must be a positive integer")
    notes, flights, onset, total = parse_notes(str(data["notes"]))
    used_bpm = float(file_bpm if bpm is None else bpm)
    if used_bpm <= 0:
        raise SongError(notes[-1].token_index, "field 'bpm' must be a positive number")
    _reject_long_loop(total, used_bpm, notes[-1].token_index)
    return Song(
        title=str(data["title"]).strip(),
        composer=str(data["composer"]).strip(),
        bpm=used_bpm,
        beats_per_bar=int(bars),
        notes_source=str(data["notes"]),
        notes=notes,
        flight_beats=flights,
        onset_beats=onset,
        total_beats=total,
        octave_shift=int(octave_shift),
    )


def loop_seconds(total_beats: float, bpm: float) -> float:
    """Unsnapped loop length ``beats * 60 / bpm``."""
    return total_beats * 60.0 / bpm


def _reject_long_loop(total_beats: float, bpm: float, index: int) -> None:
    seconds = loop_seconds(total_beats, bpm)
    if seconds > 60.0:
        raise SongError(index, f"loop is {seconds:.3f}s, longer than 60s")


def _signed_int(text: str) -> bool:
    if text[:1] in {"+", "-"}:
        text = text[1:]
    return text.isdigit()
