"""Eleven v4 narration with character timestamps. The key is read from .env and never printed."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / ".cache" / "rule37_vo"
MODEL = "eleven_v4"


def _key() -> str:
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))
    key = os.environ.get("ELEVENLABS_API_KEY", "")
    if not key:
        raise SystemExit("ELEVENLABS_API_KEY is not set")
    return key


def synth(text: str, voice_id: str, *, stability: float = 0.5, similarity: float = 0.75, seed: int = 1,
          previous_text: str | None = None, next_text: str | None = None) -> tuple[Path, dict]:
    """Return (mp3 path, alignment). Cached by every request field so a rerun costs no credits."""
    body = {
        "text": text,
        "model_id": MODEL,
        "language_code": "en",
        "seed": seed,
        "voice_settings": {"stability": stability, "similarity_boost": similarity},
    }
    if previous_text:
        body["previous_text"] = previous_text
    if next_text:
        body["next_text"] = next_text
    digest = hashlib.sha256(json.dumps([voice_id, body], sort_keys=True).encode()).hexdigest()[:20]
    CACHE.mkdir(parents=True, exist_ok=True)
    mp3, meta = CACHE / f"{digest}.mp3", CACHE / f"{digest}.json"
    if mp3.exists() and meta.exists():
        return mp3, json.loads(meta.read_text())
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/with-timestamps?output_format=mp3_44100_128"
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"xi-api-key": _key(), "Content-Type": "application/json"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                payload = json.loads(resp.read())
            break
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:400]
            if exc.code in (429, 500, 502, 503) and attempt < 3:
                time.sleep(2 + 3 * attempt)
                continue
            raise SystemExit(f"ElevenLabs HTTP {exc.code}: {detail}")
    mp3.write_bytes(base64.b64decode(payload["audio_base64"]))
    align = payload.get("normalized_alignment") or payload.get("alignment") or {}
    meta.write_text(json.dumps({"text": text, "voice": voice_id, "alignment": payload.get("alignment"),
                                "normalized_alignment": payload.get("normalized_alignment")}))
    return mp3, json.loads(meta.read_text())
