"""Checks for the finished short. Failures are listed; the caller decides what to re-render."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
from scipy.signal import butter, sosfilt

from fc_sat.audio import SR, true_peak_db
from fc_sat.beatkit.grid import HOP
from fc_sat.beatkit.probes import frame_rate_errors
from fc_sat.encode import find_ffmpeg, find_ffprobe
from fc_sat.linedraw.audio import sidechain
from fc_sat.linedraw.choreo import N_FRAMES
from fc_sat.linedraw.render import panel_fraction
from fc_sat.verify import photosensitivity_hot_count, ssim_u8


def _decode_audio(path: Path) -> tuple[int, np.ndarray]:
    wav = path.with_suffix(".verify.wav")
    subprocess.run(
        [find_ffmpeg(), "-y", "-v", "error", "-i", str(path), "-ac", "2", "-ar", "48000", str(wav)],
        check=False,
    )
    from scipy.io import wavfile

    rate, data = wavfile.read(wav)
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float64) / np.iinfo(data.dtype).max
    else:
        data = data.astype(np.float64)
    if data.ndim == 1:
        data = np.stack([data, data], axis=1)
    return int(rate), data


def _band_energy(mono: np.ndarray, rate: int, low: float, high: float) -> float:
    spectrum = np.fft.rfft(mono)
    freqs = np.fft.rfftfreq(len(mono), 1.0 / rate)
    mask = (freqs >= low) & (freqs < high)
    return float(np.sum(np.abs(spectrum[mask]) ** 2))


def audio_quality(audio: np.ndarray, rate: int = SR) -> tuple[list[str], dict[str, float]]:
    """Drop contrast, silence, phone-speaker harmonics, clicks, and mono correlation."""
    errors = []
    stats: dict[str, float] = {}
    meter = pyln.Meter(rate)
    build = audio[564 * HOP : 744 * HOP]
    drop = audio[768 * HOP : 948 * HOP]
    build_lufs = float(meter.integrated_loudness(build))
    drop_lufs = float(meter.integrated_loudness(drop))
    stats["build_lufs"] = build_lufs
    stats["drop_lufs"] = drop_lufs
    stats["drop_contrast"] = drop_lufs - build_lufs
    if not 4.0 <= stats["drop_contrast"] <= 7.0:
        errors.append(f"drop contrast {stats['drop_contrast']:.2f} LU")
    silent = audio[744 * HOP : 768 * HOP]
    peak = float(np.max(np.abs(silent))) if len(silent) else 1.0
    stats["silent_db"] = 20.0 * np.log10(peak + 1e-12)
    if stats["silent_db"] > -60.0:
        errors.append(f"silent beat {stats['silent_db']:.1f} dBFS")
    tail = audio[1806 * HOP :]
    tail_peak = float(np.max(np.abs(tail))) if len(tail) else 1.0
    stats["tail_db"] = 20.0 * np.log10(tail_peak + 1e-12)
    if stats["tail_db"] > -50.0:
        errors.append(f"tape-stop tail {stats['tail_db']:.1f} dBFS")
    bass = audio[768 * HOP : 1344 * HOP, 0]
    fundamental = _band_energy(bass, rate, 40.0, 90.0)
    harmonics = _band_energy(bass, rate, 100.0, 300.0)
    filtered = sosfilt(butter(2, 250.0, btype="high", fs=rate, output="sos"), bass)
    harmonics_after = _band_energy(filtered, rate, 100.0, 300.0)
    stats["harmonic_delta_db"] = 10.0 * np.log10((harmonics + 1e-12) / (fundamental + 1e-12))
    stats["harmonic_after_db"] = 10.0 * np.log10((harmonics_after + 1e-12) / (fundamental + 1e-12))
    if stats["harmonic_delta_db"] < -18.0:
        errors.append(f"808 harmonics {stats['harmonic_delta_db']:.1f} dB vs the fundamental")
    left = audio[:, 0] - np.mean(audio[:, 0])
    right = audio[:, 1] - np.mean(audio[:, 1])
    stats["correlation"] = float(np.sum(left * right) / (np.sqrt(np.sum(left * left) * np.sum(right * right)) + 1e-12))
    if stats["correlation"] < 0.3:
        errors.append(f"mono correlation {stats['correlation']:.2f}")
    jump = np.max(np.abs(np.diff(audio, axis=0)), axis=1)
    # Drum onsets are allowed to jump. 864 and 960 land on kicks. These are the fade,
    # the breakdown, and the tape stop.
    boundary_samples = [frame * HOP for frame in (744, 1344, 1800)]
    worst = 0.0
    for sample in boundary_samples:
        window = jump[max(0, sample - 8) : sample + 8]
        if len(window):
            worst = max(worst, float(np.max(window)))
    stats["boundary_jump"] = worst
    if worst >= 0.05:
        errors.append(f"section click {worst:.3f}")
    return errors, stats


def verify(root: Path, video: Path | None = None) -> tuple[int, str]:
    video = video or (root / "linedraw.mp4")
    meta = json.loads((root / "picture.json").read_text())
    errors: list[str] = []
    lines: list[str] = ["# Verify", ""]
    if not video.exists():
        errors.append("missing video")
    else:
        probe = frame_rate_errors(find_ffprobe(find_ffmpeg()), video, rate="60/1", frames=N_FRAMES)
        errors.extend(probe)
        info = subprocess.run(
            [find_ffprobe(find_ffmpeg()), "-v", "error", "-print_format", "json", "-show_streams", str(video)],
            check=False,
            capture_output=True,
            text=True,
        )
        payload = json.loads(info.stdout or "{}")
        video_stream = next(stream for stream in payload["streams"] if stream["codec_type"] == "video")
        audio_stream = next(stream for stream in payload["streams"] if stream["codec_type"] == "audio")
        if int(video_stream["width"]) != 1080 or int(video_stream["height"]) != 1920:
            errors.append(f"size {video_stream['width']}x{video_stream['height']}")
        if audio_stream.get("codec_name") != "aac" or int(audio_stream.get("sample_rate", 0)) != 48000 or int(audio_stream.get("channels", 0)) != 2:
            errors.append(f"audio stream {audio_stream.get('codec_name')} {audio_stream.get('sample_rate')} {audio_stream.get('channels')}")
        rate, decoded = _decode_audio(video)
        if decoded.shape[0] < N_FRAMES * HOP - SR // 10:
            errors.append(f"decoded audio length {decoded.shape[0]}")
        lufs = float(pyln.Meter(rate).integrated_loudness(decoded[: N_FRAMES * HOP]))
        peak = true_peak_db(decoded[: N_FRAMES * HOP])
        lines.append(f"- loudness {lufs:.2f} LUFS, true peak {peak:.2f} dBTP")
        if abs(lufs + 14.0) > 1.0:
            errors.append(f"loudness {lufs:.2f} LUFS")
        if peak > -1.0:
            errors.append(f"true peak {peak:.2f} dBTP")
        if float(np.max(np.abs(decoded))) > 1.0:
            errors.append("clipping")
        window = decoded[: int(0.05 * rate), 0]
        onset = int(np.argmax(np.abs(window) >= 0.05 * (np.max(np.abs(window)) or 1.0)))
        lines.append(f"- onset sample {onset}")
        if onset > int(0.001 * rate):
            errors.append(f"onset at sample {onset}")
        quality, stats = audio_quality(decoded[: N_FRAMES * HOP], rate)
        errors.extend(quality)
        for key, value in stats.items():
            lines.append(f"- {key} {value:.3f}")
    if panel_fraction() < 0.18:
        errors.append(f"panel fraction {panel_fraction():.3f}")
    else:
        lines.append(f"- panel fraction {panel_fraction():.3f}")
    luma_path = root / "luma.npy"
    if luma_path.exists():
        luma = np.load(luma_path)
        hot = photosensitivity_hot_count(luma, 60)
        lines.append(f"- luma jumps over 0.10 in one second: {hot}")
        if hot > 3:
            errors.append(f"photosensitivity {hot}")
    lines.append("")
    if errors:
        lines.append("## Failures")
        lines.extend(f"- {error}" for error in errors)
    else:
        lines.append("All checked gates passed.")
    lines.append("")
    report = "\n".join(lines)
    (root / "verify_report.md").write_text(report)
    _ = meta
    print(report)
    return (1 if errors else 0), report


def sidechain_drops() -> bool:
    bus = np.ones((SR, 2), dtype=np.float64)
    ducked = sidechain(bus, [0])
    return float(ducked[int(0.012 * SR), 0]) < 0.6 and float(ducked[int(0.2 * SR), 0]) > 0.9
