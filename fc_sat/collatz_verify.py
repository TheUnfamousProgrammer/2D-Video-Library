"""Checks for a finished Collatz mp4 and for the renderer source."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from fc_sat.audio import true_peak_db
from fc_sat.encode import find_ffmpeg, find_ffprobe
from fc_sat.verify import photosensitivity_hot_count

ROOT = Path(__file__).resolve().parents[1]
STYLE_FILES = ("fc_sat/collatz_render.py", "fc_sat/collatz_draw.py")
FORBIDDEN = ("blur", "bloom", "glow", "gradient", "shadow")


def style_law_errors() -> list[str]:
    errors = []
    for relative in STYLE_FILES:
        text = (ROOT / relative).read_text().lower()
        for word in FORBIDDEN:
            if word in text:
                errors.append(f"{relative} contains {word}")
    return errors


def _probe(path: Path) -> dict:
    import json

    ffprobe = find_ffprobe(find_ffmpeg())
    proc = subprocess.run(
        [ffprobe, "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise SystemExit(f"ffprobe failed:\n{proc.stderr}")
    return json.loads(proc.stdout or "{}")


def verify_movie(path: Path) -> list[str]:
    import json

    import pyloudnorm as pyln
    from scipy.io import wavfile

    errors = []
    info = _probe(path)
    video = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), None)
    audio = next((s for s in info.get("streams", []) if s.get("codec_type") == "audio"), None)
    if video is None or audio is None:
        return ["missing video or audio stream"]
    if video.get("codec_name") != "h264":
        errors.append(f"video codec {video.get('codec_name')}")
    duration = float(info.get("format", {}).get("duration", 0))
    if not 28 <= duration <= 36:
        errors.append(f"duration {duration:.3f}s outside [28, 36]")
    if duration > 40:
        errors.append(f"duration {duration:.3f}s over 40")
    ffmpeg = find_ffmpeg()
    wav = path.with_suffix(".verify.wav")
    subprocess.run(
        [ffmpeg, "-y", "-v", "error", "-i", str(path), "-ac", "2", str(wav)],
        check=False,
    )
    if wav.exists():
        sr, data = wavfile.read(wav)
        if data.dtype != np.float32 and data.dtype != np.float64:
            data = data.astype(np.float64) / np.iinfo(data.dtype).max
        if data.ndim == 1:
            data = np.stack([data, data], axis=1)
        meter = pyln.Meter(sr)
        lufs = float(meter.integrated_loudness(data))
        peak = true_peak_db(data)
        if abs(lufs + 14) > 1:
            errors.append(f"loudness {lufs:.2f} LUFS")
        if peak > -1 + 1e-3:
            errors.append(f"true peak {peak:.2f} dBTP")
        if float(np.max(np.abs(data))) > 1:
            errors.append("clipped samples")
        wav.unlink()
    errors.extend(style_law_errors())
    report = path.parent / "claims_report.md"
    if report.exists() and "FAIL" in report.read_text():
        errors.append("claims report has a failure")
    _ = json
    return errors


def hot_count(luma: np.ndarray, fps: int) -> int:
    return photosensitivity_hot_count(luma, fps)
