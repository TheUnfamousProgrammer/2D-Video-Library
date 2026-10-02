"""Layout, audio, voice, post kit, and the full-render gate."""

from __future__ import annotations

import numpy as np
import pytest

from fc_sat.audio import SR, loudness_loop
from fc_sat.collatz_accum import Accum
from fc_sat.collatz_audio import C3, duck, frequency, pitch_index
from fc_sat.collatz_claims import load_claims
from fc_sat.collatz_layout import FUNNEL_SCALE, Film, funnel_x, inside_safe, overlaps
from fc_sat.collatz_post import render_postkit
from fc_sat.collatz_script import load_script
from fc_sat.collatz_timeline import build_timeline
from fc_sat.collatz_verify import style_law_errors
from fc_sat.collatz_voice import FitError, choose_take, discover_model, without_ipa, ElevenLabsClient
from make_collatz import main
from tools.collatz_reply import main as reply_main
from tools.collatz_reply import reply_text


def test_layout_stays_inside_the_safe_zone():
    film = Film(build_timeline(), load_script(), load_claims())
    for index in range(0, film.timeline.n_frames, 3):
        plan = film.plan(index)
        assert len(plan.texts) <= 4
        for box in plan.texts:
            assert inside_safe(box)
        for left in plan.texts:
            for right in plan.texts:
                if left is right:
                    continue
                assert not overlaps(left, right)


def test_funnel_geometry():
    film = Film(build_timeline(), load_script(), load_claims())
    lines = film.funnel_polylines()
    assert len(lines) == 999
    assert FUNNEL_SCALE == pytest.approx(860 / 178)
    assert {round(line[-1][0], 5) for line in lines} == {970.0}
    assert funnel_x(0) == pytest.approx(970)
    assert funnel_x(178) == pytest.approx(110)


def test_accumulation_matches_redraw():
    lines = [[(2, 2), (12, 8), (20, 4)], [(1, 16), (18, 3), (24, 12)], [(4, 10), (8, 18)]]
    running = Accum(30, 22)
    for count in range(1, len(lines) + 1):
        running.add_polyline(lines[count - 1], (200, 180, 160), 2, 0.1)
        fresh = Accum(30, 22)
        for line in lines[:count]:
            fresh.add_polyline(line, (200, 180, 160), 2, 0.1)
        assert np.allclose(running.color, fresh.color)
        assert np.allclose(running.cover, fresh.cover)


def test_pitch_map():
    assert pitch_index(27) == 5
    assert frequency(27) == pytest.approx(C3 * 2)
    assert frequency(9232) == pytest.approx(C3 * 2 ** (31 / 12))


def test_mix_length():
    from fc_sat.collatz_audio import mix

    timeline = build_timeline()
    audio, _lufs, true_peak, _sync = mix(timeline)
    assert len(audio) == timeline.n_frames * 800
    assert true_peak <= -1 + 1e-3
    assert float(np.max(np.abs(audio))) <= 1


def test_ducking_and_loudness_loop():
    bus = np.ones((SR // 2, 2))
    voice = np.zeros_like(bus)
    voice[SR // 10 : SR // 5] = 0.4
    ducked = duck(bus, voice, SR, 7)
    assert ducked[SR // 8, 0] < 0.7
    assert ducked[10, 0] > 0.9
    tone = (np.random.default_rng(1).standard_normal((SR, 2)) * 0.05)
    audio, lufs, peak = loudness_loop(tone, SR)
    assert abs(lufs + 14) <= 0.5
    assert peak <= -1 + 1e-6
    assert len(audio) == SR


def test_fit_and_pronunciation():
    assert choose_take([1.2, 0.8], 1.0, 1.4, True) == ("take", 1, 1.0)
    assert choose_take([1.4, 1.3], 1.0, 0.9, True)[0] == "alt"
    kind, _index, rate = choose_take([1.05], 1.0, 1.4, True)
    assert kind == "atempo" and rate <= 1.06
    with pytest.raises(FitError):
        choose_take([1.2], 1.0, 1.4, False)
    pron = {"Collatz": [{"id": "plain", "text": "Collatz"}, {"id": "respell", "text": "Koll-ahts"}, {"id": "ipa", "text": "/ˈkɒlæts/"}]}
    assert without_ipa("the /ˈkɒlæts/ problem", pron) == "the Koll-ahts problem"


def test_model_discovery_and_retry(tmp_path, monkeypatch):
    assert discover_model([{"model_id": "eleven_v4", "name": "Eleven v4", "can_do_text_to_speech": True}]) == "eleven_v4"
    with pytest.raises(SystemExit):
        discover_model([
            {"model_id": "a_v4", "name": "A", "can_do_text_to_speech": True},
            {"model_id": "b_v4", "name": "B", "can_do_text_to_speech": True},
        ])
    calls = {"n": 0}

    def transport(_method, body):
        calls["n"] += 1
        if calls["n"] == 1:
            return 400, {"detail": "style is not supported"}, b""
        return 200, {}, b"RIFF"

    monkeypatch.setattr("fc_sat.collatz_voice.CACHE", tmp_path)
    client = ElevenLabsClient(transport, "eleven_v4", "voice")
    audio = client.synthesize("Hello.", {"stability": 0.5, "similarity_boost": 0.75, "style": 0.2}, seed=3)
    assert audio == b"RIFF"
    assert calls["n"] == 2
    client.synthesize("Hello.", {"stability": 0.5, "similarity_boost": 0.75, "style": 0.2}, seed=3)
    assert calls["n"] == 2


def test_postkit_and_reply():
    text, errors = render_postkit(load_claims(), reply=True)
    assert errors == []
    assert "Collatz conjecture" in text and "#shorts" in text
    assert "111" in reply_text(27)
    assert "0 steps" in reply_text(1)
    assert reply_main(["-1"]) == 2
    assert reply_main(["nope"]) == 2


def test_full_refused_and_style_law():
    with pytest.raises(SystemExit, match="approved"):
        main(["full"])
    assert style_law_errors() == []
