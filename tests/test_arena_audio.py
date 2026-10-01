import numpy as np

from fc_sat.arena_audio import render_arena_mix
from fc_sat.arena_config import load_arena_config
from fc_sat.arena_render import build_timeline
from fc_sat.arena_sim import Elim, SimResult
from fc_sat.arena_voice import (
    HEADER,
    announcer_text,
    load_voice,
    raise_if_unknown_model,
    voice_payload,
)
from fc_sat.audio import SR, finish_broadcast


def _paced(cfg) -> SimResult:
    times = [2.0, 3.0, 4.5, 6.0, 7.2, 8.5, 10.0, 12.0, 14.0, 16.5, 18.0, 20.0, 22.5, 24.0, 26.0, 27.2]
    elims = tuple(Elim(when, index, "self" if index % 3 else "meteor") for index, when in enumerate(times))
    return SimResult(
        seed=cfg.seed,
        backend="test",
        elims=elims,
        meteors=(),
        near_misses=(),
        t_win=28.0,
        winner=31,
        duel_start=24.5,
        cameo_exit=18.0,
        trace=None,
    )


def test_finish_broadcast_keeps_the_ending_audible():
    n = SR * 3
    t = np.arange(n) / SR
    tone = 0.2 * np.sin(2 * np.pi * 220.0 * t)
    stereo = np.column_stack([tone, tone])
    mastered, lufs, peak = finish_broadcast(stereo)
    tail = mastered[-int(0.08 * SR) : -int(0.01 * SR), 0]
    assert float(np.max(np.abs(tail))) > 1e-3
    assert abs(lufs + 14.0) <= 0.5
    assert peak <= -1.0 + 1e-6


def test_mix_length_loudness_and_peak():
    cfg = load_arena_config("configs/arena_default.yaml")
    result = _paced(cfg)
    timeline = build_timeline(result, cfg)
    n_frames = int(round(timeline.duration * cfg.fps))
    mix = render_arena_mix(cfg, result, n_frames, fps=cfg.fps, timeline=timeline)
    assert mix.n_samples == n_frames * (SR // cfg.fps)
    assert len(mix.full) == mix.n_samples
    assert len(mix.sfx_only) == mix.n_samples
    assert abs(mix.lufs + 14.0) <= 1.5
    assert mix.true_peak <= -1.0 + 1e-6
    assert float(np.max(np.abs(mix.full))) <= 1.0


def test_announcer_header_and_winner_shout():
    cfg = load_arena_config("configs/arena_default.yaml")
    result = _paced(cfg)
    timeline = build_timeline(result, cfg)
    text = announcer_text(cfg, result, timeline)
    assert text.startswith(HEADER)
    assert "eleven_v4" not in text
    assert "[excited] Final three!" in text
    assert "[shouting] VIETNAM WINS!" in text
    assert text.count("[shocked]") == 3


def test_voice_model_id_comes_from_config(monkeypatch):
    cfg = load_arena_config("configs/arena_default.yaml")
    from dataclasses import replace

    cfg = replace(cfg, voice_model_id="studio-model-from-yaml")
    payload = voice_payload(cfg, "[excited] Final three!")
    assert payload["model_id"] == "studio-model-from-yaml"
    seen = {}

    def fetch(voice_id, body, api_key):
        seen["model"] = body["model_id"]
        return 400, b'{"detail":"unknown model id"}'

    monkeypatch.setenv("ELEVENLABS_API_KEY", "test-key")
    monkeypatch.setenv("ELEVENLABS_VOICE_ID", "voice-1")
    try:
        load_voice(cfg, _paced(cfg), build_timeline(_paced(cfg), cfg), 1000, fetch=fetch)
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("unknown model should raise")
    assert seen["model"] == "studio-model-from-yaml"
    assert "studio-model-from-yaml" in message
    assert "ElevenLabs model docs" in message


def test_voice_skips_without_a_key(capsys, monkeypatch):
    cfg = load_arena_config("configs/arena_default.yaml")
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    monkeypatch.delenv("ELEVENLABS_VOICE_ID", raising=False)
    result = _paced(cfg)
    audio = load_voice(cfg, result, build_timeline(result, cfg), 800)
    assert audio is None
    err = capsys.readouterr().err
    assert err.count("announcer audio skipped") == 1


def test_unknown_model_helper_names_the_configured_id():
    try:
        raise_if_unknown_model(400, "model_id is invalid", "eleven_custom")
    except RuntimeError as exc:
        assert "eleven_custom" in str(exc)
    else:
        raise AssertionError("expected a model error")
