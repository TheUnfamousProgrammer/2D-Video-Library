"""ElevenLabs v4 client, take fitting, and the manual-script fallback.

The API key is read from the environment and is never printed.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache" / "vo"
PICKS_PATH = ROOT / "configs" / "vo_picks.yaml"
PRON_PATH = ROOT / "configs" / "collatz_pronunciation.yaml"


class VoiceError(RuntimeError):
    pass


class FitError(VoiceError):
    pass


def load_dotenv() -> None:
    path = ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def key_present() -> bool:
    load_dotenv()
    return bool(os.environ.get("ELEVENLABS_API_KEY")) and bool(os.environ.get("ELEVENLABS_VOICE_ID"))


def discover_model(models: list[dict]) -> str:
    hits = []
    for model in models:
        ident = f"{model.get('model_id', '')} {model.get('name', '')}".lower()
        if "v4" not in ident:
            continue
        if model.get("can_do_text_to_speech") is False:
            continue
        if any(word in ident for word in ("scribe", "speech to text", "speech-to-text")):
            continue
        hits.append(model.get("model_id") or model.get("name"))
    if len(hits) != 1:
        print("ElevenLabs v4 text-to-speech models:")
        for model in models:
            print(f"  {model.get('model_id')}  {model.get('name')}")
        raise SystemExit("need exactly one v4 text-to-speech model; refusing to guess")
    return hits[0]


def cache_key(model: str, voice: str, text: str, settings: dict, seed: int) -> str:
    payload = json.dumps(
        {"model": model, "voice": voice, "text": text, "settings": settings, "seed": seed},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def choose_take(durations: list[float], slot: float, alt_duration: float, allow_atempo: bool, cap: float = 1.06):
    for index, duration in enumerate(durations):
        if duration <= slot + 1e-6:
            return ("take", index, 1.0)
    if alt_duration <= slot + 1e-6:
        return ("alt", None, 1.0)
    if allow_atempo:
        for index, duration in enumerate(durations):
            rate = duration / slot
            if rate <= cap + 1e-6:
                return ("atempo", index, rate)
    raise FitError(
        f"no take fits {slot:.2f}s (takes {durations}, alt {alt_duration:.2f}s, atempo {'on' if allow_atempo else 'off'})"
    )


def without_ipa(text: str, variants: dict) -> str:
    import re

    if "/" not in text:
        return text
    repl = next(item["text"] for item in variants["Collatz"] if item["id"] == "respell")
    return re.sub(r"/[^/]+/", repl, text)


def apply_pronunciation(text: str, picks: dict, variants: dict) -> str:
    for word, choice in picks.items():
        options = variants.get(word, [])
        replacement = next((item["text"] for item in options if item["id"] == choice), None)
        if replacement and replacement != word:
            text = text.replace(word, replacement)
    return text


def load_pronunciation() -> dict:
    return yaml.safe_load(PRON_PATH.read_text())


def load_picks() -> dict:
    if not PICKS_PATH.exists():
        return {}
    return dict((yaml.safe_load(PICKS_PATH.read_text()) or {}).get("picks") or {})


def save_picks(updates: dict) -> dict:
    current = load_picks()
    # A generation never calls this. vo-select is the only writer, and it
    # replaces only the keys the user named.
    current.update(updates)
    PICKS_PATH.write_text(yaml.safe_dump({"picks": current}, sort_keys=False))
    return current


def parse_selection(tokens: list[str]) -> dict[str, int]:
    parsed = {}
    for token in tokens:
        if "=" not in token:
            raise SystemExit(f"expected L1=2, got {token}")
        key, value = token.split("=", 1)
        parsed[key] = int(value)
    return parsed


class ElevenLabsClient:
    def __init__(self, transport, model: str, voice_id: str) -> None:
        self.transport = transport
        self.model = model
        self.voice_id = voice_id

    def synthesize(self, text: str, settings: dict, seed: int, previous: str | None = None, nxt: str | None = None) -> bytes:
        digest = cache_key(self.model, self.voice_id, text, settings, seed)
        path = CACHE / f"{digest}.bin"
        if path.exists():
            return path.read_bytes()
        body = {
            "text": text,
            "model_id": self.model,
            "language_code": "en",
            "apply_text_normalization": "off",
            "voice_settings": {
                "stability": settings.get("stability", 0.5),
                "similarity_boost": settings.get("similarity_boost", 0.75),
                **{key: settings[key] for key in ("style", "speed") if key in settings},
            },
            "seed": seed,
        }
        if previous:
            body["previous_text"] = previous
        if nxt:
            body["next_text"] = nxt
        if settings.get("output_format"):
            body["output_format"] = settings["output_format"]
        audio = self._post(body)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(audio)
        (CACHE / f"{digest}.json").write_text(json.dumps({"text": text, "seed": seed, "settings": settings}))
        return audio

    def _post(self, body: dict) -> bytes:
        delay = 1.0
        dropped = False
        for _attempt in range(5):
            status, payload, raw = self.transport("POST", body)
            if status == 200:
                return raw
            text = payload if isinstance(payload, str) else json.dumps(payload)
            if status == 400 and not dropped:
                changed = False
                for field in ("style", "speed", "previous_text", "next_text"):
                    if field not in text.lower():
                        continue
                    if field in body:
                        body.pop(field, None)
                        changed = True
                    if field in body.get("voice_settings", {}):
                        body["voice_settings"].pop(field, None)
                        changed = True
                if changed:
                    print(f"voice setting rejected ({text[:180]}); retrying without it")
                    dropped = True
                    continue
            if status == 429 or status >= 500:
                time.sleep(delay)
                delay *= 2
                continue
            raise VoiceError(f"ElevenLabs HTTP {status}: {text[:300]}")
        raise VoiceError("ElevenLabs request failed after retries")


def stem_from_dir(folder: Path, lines, n_samples: int, sr: int = 48000, shift_s: float = 0.0) -> np.ndarray:
    """Place user wavs named L1.wav ... on the timeline. Missing files are skipped."""
    import subprocess

    import numpy as np

    from fc_sat.encode import find_ffmpeg

    folder = Path(folder)
    if not folder.is_dir():
        raise SystemExit(f"--vo-dir is not a directory: {folder}")
    stem = np.zeros((n_samples, 2), dtype=np.float64)
    found = 0
    ffmpeg = find_ffmpeg()
    for line in lines:
        path = folder / f"{line.id}.wav"
        if not path.exists():
            print(f"vo missing {path.name}")
            continue
        proc = subprocess.run(
            [ffmpeg, "-v", "error", "-i", str(path), "-ac", "2", "-ar", str(sr), "-f", "f32le", "-"],
            check=False,
            capture_output=True,
        )
        if proc.returncode != 0:
            raise SystemExit(f"could not read {path.name}: {proc.stderr.decode()[:200]}")
        audio = np.frombuffer(proc.stdout, dtype=np.float32).reshape(-1, 2).astype(np.float64)
        start_s = line.offset + (shift_s if line.offset >= 2.6 else 0.0)
        start = int(round(start_s * sr))
        end = min(n_samples, start + len(audio))
        if start < n_samples:
            stem[start:end] += audio[: end - start]
        found += 1
        print(f"vo {line.id} at {start_s:.2f}s ({len(audio) / sr:.2f}s)")
    if found == 0:
        raise SystemExit(f"no L1.wav ... files in {folder}")
    return stem


def write_manual_script(lines, model_note: str, path: Path) -> None:
    rows = [
        "# Script for manual TTS",
        "",
        model_note,
        "",
        "Name the files L1.wav through L14.wav and pass `--vo-dir` to `preview` or `audio`.",
        "Do not put the API key in the files.",
        "",
    ]
    for line in lines:
        rows.append(f"## {line.id}")
        rows.append(f"- tag candidates: {', '.join(line.tag_candidates) or '(none)'}")
        rows.append(f"- chosen tag: {line.tag or '(none)'}")
        rows.append(f"- stability: {line.stability if line.stability is not None else 0.5}")
        rows.append("- similarity_boost: 0.75")
        rows.append("- seeds: base+0, base+1, base+2")
        rows.append(f"- slot_max_s: {line.slot_max_s}")
        rows.append(f"- text: {line.spoken()}")
        rows.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows))


def write_audition(lines, path: Path) -> None:
    parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'><title>Collatz VO audition</title>",
        "<style>body{font-family:sans-serif;background:#12151C;color:#F2F4F8;padding:24px} article{margin:16px 0;padding:12px;background:#1B2030}</style>",
        "</head><body><h1>Collatz takes</h1>",
    ]
    pron = load_pronunciation()
    parts.append("<h2>Pronunciation</h2>")
    for word, options in pron["variants"].items():
        parts.append(f"<p>{word}: " + ", ".join(f"{item['id']} = {item['text']}" for item in options) + "</p>")
    for line in lines:
        parts.append(f"<article><h3>{line.id}</h3><p>{line.spoken()}</p>")
        for seed in range(3):
            rel = f"{line.id}_s{seed}.wav"
            parts.append(
                f"<p>seed {seed} <audio controls src='{rel}'></audio> slot {line.slot_max_s}s</p>"
            )
        parts.append("</article>")
    parts.append("</body></html>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts))
