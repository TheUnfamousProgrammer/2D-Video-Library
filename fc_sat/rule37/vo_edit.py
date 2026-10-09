"""Tighten the narration: squash pauses, lift tempo, and remap word times onto the new clock."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SR = 48000
TEMPO = 1.07


@dataclass
class Word:
    text: str
    t0: float
    t1: float


def decode(path: Path, sr: int = SR) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).astype(np.float64)


def words_from(meta: dict) -> list[Word]:
    a = meta["alignment"]
    out, cur, t0, t1 = [], "", 0.0, 0.0
    for ch, s, e in zip(a["characters"], a["character_start_times_seconds"], a["character_end_times_seconds"]):
        if ch.isspace():
            if cur:
                out.append(Word(cur, t0, t1))
            cur = ""
            continue
        if not cur:
            t0 = s
        cur += ch
        t1 = e
    if cur:
        out.append(Word(cur, t0, t1))
    return [w for w in out if not (w.text.startswith("[") and w.text.endswith("]"))]


def _silences(x: np.ndarray, sr: int, min_len: float = 0.14) -> list[tuple[int, int]]:
    hop = sr // 100
    n = len(x) // hop
    rms = np.sqrt(np.mean(x[: n * hop].reshape(n, hop) ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(rms / rms.max())
    quiet = db < -38
    runs, i = [], 0
    while i < n:
        if quiet[i]:
            j = i
            while j < n and quiet[j]:
                j += 1
            if (j - i) * hop >= min_len * sr:
                runs.append((i * hop, j * hop))
            i = j
        else:
            i += 1
    return runs


def tighten(x: np.ndarray, sr: int = SR) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Squash each pause to 0.11 s + 30 % of the excess. Returns audio and (old_t, new_t) anchors."""
    pieces, anchors, pos_old, pos_new = [], [(0.0, 0.0)], 0, 0
    fade = int(0.006 * sr)
    for a, b in _silences(x, sr):
        if a == 0:
            continue
        run = b - a
        keep = int(0.11 * sr + 0.30 * (run - 0.11 * sr))
        head = keep // 2
        cut0, cut1 = a + head, b - (keep - head)
        seg = x[pos_old:cut0].copy()
        pieces.append(seg)
        pos_new += len(seg)
        anchors.append((cut0 / sr, pos_new / sr))
        anchors.append((cut1 / sr, pos_new / sr))
        pos_old = cut1
    tail = x[pos_old:].copy()
    pieces.append(tail)
    anchors.append((len(x) / sr, (pos_new + len(tail)) / sr))
    for p in pieces[1:]:
        p[:fade] *= np.linspace(0, 1, fade)
    for p in pieces[:-1]:
        p[-fade:] *= np.linspace(1, 0, fade)
    return np.concatenate(pieces), anchors


def remap(t: float, anchors: list[tuple[float, float]]) -> float:
    olds = np.array([a for a, _ in anchors])
    news = np.array([b for _, b in anchors])
    return float(np.interp(t, olds, news)) / TEMPO


def build(mp3: Path, meta: dict, out_wav: Path) -> tuple[np.ndarray, list[Word]]:
    import soundfile as sf

    x = decode(mp3)
    y, anchors = tighten(x)
    tmp = out_wav.with_suffix(".pre.wav")
    sf.write(tmp, y.astype(np.float32), SR)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(tmp), "-filter:a", f"atempo={TEMPO}", "-ar", str(SR),
                    str(out_wav)], check=True)
    tmp.unlink()
    z, _ = sf.read(out_wav)
    words = [Word(w.text, remap(w.t0, anchors), remap(w.t1, anchors)) for w in words_from(meta)]
    out_wav.with_suffix(".words.json").write_text(json.dumps([w.__dict__ for w in words], indent=0))
    return z, words
