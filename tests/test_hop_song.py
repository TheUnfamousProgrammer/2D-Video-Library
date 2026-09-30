from pathlib import Path

import pytest

from fc_sat.hop_choreo import build_choreography, build_grid, build_layout, pad_press
from fc_sat.song import SongError, load_song, midi_frequency, note_midi, parse_notes

ODE = Path("songs/ode_to_joy.yaml")


def _tokens(source: str) -> list[str]:
    return [token for token in source.split() if token != "|"]


def test_ode_counts_come_from_the_note_string():
    song = load_song(ODE)
    tokens = _tokens(song.notes_source)
    note_tokens = [token for token in tokens if not token.startswith("R:")]
    assert len(song.notes) == len(note_tokens)
    assert song.total_beats == pytest.approx(sum(float(token.split(":")[1]) for token in note_tokens))
    assert song.total_beats == pytest.approx(32.0)
    assert song.pitches == ("C4", "D4", "E4", "F4", "G4")
    assert song.octave_shift == 1
    assert song.notes[0].midi == note_midi("E4")


def test_frequency_table_and_enharmonics():
    assert midi_frequency(note_midi("A4")) == pytest.approx(440.0)
    assert round(midi_frequency(note_midi("C4")), 4) == 261.6256
    assert note_midi("D#5") == note_midi("Eb5")
    assert midi_frequency(note_midi("D#5")) == midi_frequency(note_midi("Eb5"))
    shifted = load_song(ODE, octave_shift=1)
    assert shifted.frequency(note_midi("C4")) == pytest.approx(midi_frequency(note_midi("C4"), 1))
    assert shifted.notes[0].midi == note_midi("E4")


def test_parser_rejects_bad_tokens_and_pitch_counts():
    with pytest.raises(SongError, match=r"token 1"):
        parse_notes("C4:1 H4:1 D4:1")
    with pytest.raises(SongError, match=r"token 0"):
        parse_notes("C4:0 D4:1")
    with pytest.raises(SongError, match=r"token 2"):
        parse_notes("C4:1 D4:1 E4:-1")
    names = ["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5", "D5", "E5", "F5"]
    with pytest.raises(SongError, match=r"token 10"):
        parse_notes(" ".join(f"{name}:1" for name in names))
    with pytest.raises(SongError, match=r"token 3"):
        parse_notes("C4:1 C4:1 C4:1 C4:1")
    with pytest.raises(SongError, match=r"token 0"):
        parse_notes("R:1 R:2")
    with pytest.raises(SongError, match=r"token 0"):
        parse_notes("")


def test_leading_rest_is_dropped_and_trailing_rest_extends_the_wrap():
    notes, flights, onset, total = parse_notes("R:2 C4:1 D4:1 E4:1")
    assert [note.name for note in notes] == ["C4", "D4", "E4"]
    assert onset[0] == 0
    assert total == pytest.approx(3.0)
    notes, flights, onset, total = parse_notes("C4:1 D4:1 R:2")
    assert flights == pytest.approx((1.0, 3.0))
    assert total == pytest.approx(4.0)
    assert onset == pytest.approx((0.0, 1.0))
    notes, flights, _onset, total = parse_notes("C4:1 R:1 D4:1")
    assert flights == pytest.approx((2.0, 1.0))
    assert total == pytest.approx(3.0)


def test_loop_longer_than_60_seconds_names_a_token(tmp_path: Path):
    path = tmp_path / "long.yaml"
    path.write_text(
        "title: Long\ncomposer: Test\nbpm: 60\nbeats_per_bar: 4\nnotes: 'C4:30 D4:31'\n",
        encoding="utf-8",
    )
    with pytest.raises(SongError, match=r"token 1"):
        load_song(path)


def test_ode_grid_snaps_without_moving():
    song = load_song(ODE)
    grid = build_grid(song)
    assert grid.t0 == pytest.approx(16.0)
    assert grid.n_frames == 960
    assert grid.duration == pytest.approx(16.0)
    assert grid.spb == pytest.approx(0.5)
    assert grid.onset_frames[0] == 0
    assert grid.onset_times[0] == 0
    assert all(abs(t * 60 - round(t * 60)) < 1e-9 for t in grid.onset_times)
    assert all(b > a for a, b in zip(grid.onset_frames, grid.onset_frames[1:]))
    assert grid.onset_samples == tuple(frame * 800 for frame in grid.onset_frames)
    assert sum(grid.flight_frames) == grid.n_frames
    assert min(grid.flight_frames) == 15
    assert len(grid.onset_samples) == len(song.notes)


def test_short_flight_is_rejected():
    notes, flights, onset, total = parse_notes("C4:0.05 D4:1")
    from fc_sat.song import Song

    song = Song(
        title="Short",
        composer="Test",
        bpm=120,
        beats_per_bar=4,
        notes_source="C4:0.05 D4:1",
        notes=notes,
        flight_beats=flights,
        onset_beats=onset,
        total_beats=total,
    )
    with pytest.raises(Exception, match=r"shorter than 6"):
        build_grid(song)


def test_layout_for_five_and_ten_pitches():
    song = load_song(ODE)
    layout = build_layout(song)
    assert layout.centers == pytest.approx((260.0, 400.0, 540.0, 680.0, 820.0))
    assert layout.width == pytest.approx(112.0)
    assert layout.pads[0].left == pytest.approx(204.0)
    assert layout.pads[-1].right == pytest.approx(876.0)
    assert layout.ball_r == pytest.approx(34.0)
    assert layout.y_contact == pytest.approx(1180.0 - 34.0)

    names = ["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5", "D5", "E5"]
    notes, flights, onset, total = parse_notes(" ".join(f"{name}:1" for name in names))
    from fc_sat.song import Song

    wide = Song(
        title="Wide",
        composer="Test",
        bpm=120,
        beats_per_bar=4,
        notes_source="",
        notes=notes,
        flight_beats=flights,
        onset_beats=onset,
        total_beats=total,
    )
    ten = build_layout(wide)
    assert len(ten.pads) == 10
    assert ten.width == pytest.approx(56.0)
    assert ten.ball_r <= 0.42 * ten.width + 1e-9


def test_arc_contact_periodicity_and_press():
    song = load_song(ODE)
    dance = build_choreography(song)
    grid = dance.grid
    for index, t in enumerate(grid.onset_times):
        pose = dance.pose(t)
        pad = dance.layout.pads[dance.pad_index(index)]
        assert pose.x == pytest.approx(pad.x, abs=1e-6)
        assert pose.bottom == pytest.approx(pad.top, abs=1e-6)
        assert pose.pad_top == pytest.approx(pose.bottom, abs=1e-6)
        assert pose.age == pytest.approx(0.0, abs=1e-9)
    for index, t in enumerate(grid.onset_times):
        if index + 1 < len(grid.onset_times):
            nxt = grid.onset_times[index + 1]
        else:
            nxt = grid.duration
        before = dance.pose(nxt - 1e-6)
        after = dance.pose(nxt)
        assert before.x == pytest.approx(after.x, abs=0.02)
        assert before.y_arc == pytest.approx(after.y_arc, abs=0.02)
    assert max(dance.heights) <= 600.0 + 1e-9
    edges = (dance.layout.pads[0].left, dance.layout.pads[-1].right)
    for frame in range(0, grid.n_frames, 7):
        pose = dance.pose(frame / 60.0)
        assert edges[0] - 1e-6 <= pose.x <= edges[1] + 1e-6
    t = 1.25
    again = dance.pose(t + grid.duration)
    first = dance.pose(t)
    assert again == first
    assert dance.pose(grid.duration) == dance.pose(0.0)
    assert pad_press(0.0) == 0.0
    assert 8.0 < pad_press(0.030) < 12.0
