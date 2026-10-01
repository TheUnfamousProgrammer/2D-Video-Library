"""ElevenLabs announcer. The model id is read from config; nothing is hardcoded.

Without an API key this writes the paste-ready script and skips the request.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

from fc_sat.arena_config import ArenaConfig
from fc_sat.arena_render import Timeline, video_time
from fc_sat.arena_sim import SimResult
from fc_sat.audio import SR

HEADER = (
    "target model: Eleven v4 (verify the model id in the ElevenLabs docs)\n"
    "Voice: energetic designed caster. Not a whisper or PVC character. Stability: Natural or Creative.\n"
    "Generate one line at a time. Square-bracket tags are audio directions, not spoken words.\n"
)


def announcer_lines(cfg: ArenaConfig, result: SimResult, timeline: Timeline) -> tuple[tuple[float, str], ...]:
    lines: list[tuple[float, str]] = []
    final_three = None
    count = len(cfg.countries)
    for index, item in enumerate(result.elims):
        if count - index - 1 == 3:
            final_three = item.time
            break
    lines.append((video_time(timeline, final_three or timeline.t_win), "[excited] Final three!"))
    for item in result.elims[-3:]:
        name = cfg.countries[item.index].name
        lines.append((video_time(timeline, item.time), f"[shocked] {name} is out!"))
    if result.winner is not None:
        winner = cfg.countries[result.winner].name.upper()
        lines.append((timeline.winner_video, f"[shouting] {winner} WINS!"))
    return tuple(lines)


def announcer_text(cfg: ArenaConfig, result: SimResult, timeline: Timeline) -> str:
    body = [HEADER.rstrip("\n"), ""]
    for when, line in announcer_lines(cfg, result, timeline):
        body.append(f"# {when:.2f}s")
        body.append(line)
    return "\n".join(body) + "\n"


def write_announcer(path: Path, cfg: ArenaConfig, result: SimResult, timeline: Timeline) -> None:
    path.write_text(announcer_text(cfg, result, timeline), encoding="utf-8")


def voice_payload(cfg: ArenaConfig, text: str) -> dict[str, str]:
    """Request body. ``model_id`` is the config value, not a constant in this module."""
    return {"text": text, "model_id": cfg.voice_model_id}


def raise_if_unknown_model(status: int, body: str, model_id: str) -> None:
    if status < 400:
        return
    lowered = body.lower()
    if "model" in lowered and any(word in lowered for word in ("unknown", "not found", "invalid", "does not exist", "not available")):
        raise RuntimeError(
            f"ElevenLabs rejected voice.model_id {model_id!r}. "
            "Check the ElevenLabs model docs and update voice.model_id in the arena config."
        )


def _cache_path(voice_id: str, model_id: str, text: str) -> Path:
    digest = hashlib.sha256(f"{voice_id}|{model_id}|{text}".encode()).hexdigest()[:24]
    return Path(".cache") / "voice" / f"{digest}.bin"


def fetch_clip(cfg: ArenaConfig, text: str, voice_id: str, api_key: str, fetch) -> bytes:
    cached = _cache_path(voice_id, cfg.voice_model_id, text)
    if cached.is_file():
        return cached.read_bytes()
    payload = voice_payload(cfg, text)
    status, body = fetch(voice_id, payload, api_key)
    if isinstance(body, bytes):
        message = body.decode("utf-8", errors="replace")
        raw = body
    else:
        message = str(body)
        raw = message.encode()
    raise_if_unknown_model(status, message, cfg.voice_model_id)
    if status >= 400:
        raise RuntimeError(f"ElevenLabs voice request failed ({status}): {message[:300]}")
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_bytes(raw)
    return raw


def load_voice(
    cfg: ArenaConfig,
    result: SimResult,
    timeline: Timeline,
    n_samples: int,
    *,
    fetch=None,
) -> np.ndarray | None:
    api_key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
    voice_id = os.environ.get("ELEVENLABS_VOICE_ID", "").strip() or cfg.voice_id.strip()
    if not api_key or not voice_id:
        print("announcer audio skipped: ELEVENLABS_API_KEY or voice id is unset", file=sys.stderr)
        return None
    if fetch is None:
        fetch = _http_fetch
    bed = np.zeros(n_samples, dtype=np.float64)
    for when, line in announcer_lines(cfg, result, timeline):
        raw = fetch_clip(cfg, line, voice_id, api_key, fetch)
        clip = _decode_clip(raw)
        if clip is None or not len(clip):
            continue
        start = int(round(when * SR))
        n = min(len(clip), n_samples - start)
        if start >= 0 and n > 0:
            bed[start : start + n] += clip[:n]
    stereo = np.column_stack([bed, bed])
    return stereo


def _decode_clip(raw: bytes) -> np.ndarray | None:
    if raw.startswith(b"RIFF"):
        import io

        from scipy.io import wavfile

        _sr, data = wavfile.read(io.BytesIO(raw))
        audio = np.asarray(data, dtype=np.float64)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        peak = np.max(np.abs(audio)) or 1.0
        if peak > 1.5:
            audio = audio / 32768.0
        return audio
    return None


def _http_fetch(voice_id: str, payload: dict, api_key: str) -> tuple[int, bytes]:
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"xi-api-key": api_key, "Content-Type": "application/json", "Accept": "audio/mpeg"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()
