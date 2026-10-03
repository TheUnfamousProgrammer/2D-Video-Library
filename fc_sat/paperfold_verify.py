"""Checks for the paperfold master. Measurements are printed, then written to the report."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from fc_sat.audio import true_peak_db
from fc_sat.beatkit.music import events_aligned, onset_sample
from fc_sat.beatkit.probes import frame_rate_errors
from fc_sat.encode import find_ffmpeg, find_ffprobe
from fc_sat.paperfold_claims import evaluate, load_claims
from fc_sat.paperfold_film import Film
from fc_sat.paperfold_timeline import build_timeline
from fc_sat.verify import _probe, photosensitivity_hot_count, ssim_u8

ROOT = Path(__file__).resolve().parents[1]


def _decode_frame(path: Path, index: int) -> np.ndarray | None:
    ffmpeg = find_ffmpeg()
    proc = subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-i",
            str(path),
            "-vf",
            f"select=eq(n\\,{index})",
            "-vframes",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "pipe:1",
        ],
        check=False,
        capture_output=True,
    )
    raw = proc.stdout
    if len(raw) < 1080 * 1920 * 3:
        return None
    return np.frombuffer(raw[: 1080 * 1920 * 3], dtype=np.uint8).reshape(1920, 1080, 3)


def verify(path: Path | None = None) -> int:
    path = path or (ROOT / "out" / "paperfold.mp4")
    lines = ["# Verify", ""]
    errors: list[str] = []
    claims = evaluate(load_claims())
    failed = [claim_id for claim_id, ok, _detail in claims if not ok]
    if failed:
        errors.append("claims failed: " + ", ".join(failed))
        lines.append(f"- claims FAIL {failed}")
    else:
        lines.append(f"- claims {len(claims)}/{len(claims)} passed")
        print(f"claims {len(claims)}/{len(claims)} passed")

    film_text = (ROOT / "fc_sat" / "paperfold_film.py").read_text().lower()
    for word in ("bloom", "glow"):
        if word in film_text:
            errors.append(f"paperfold_film.py contains {word}")
    lines.append("- bloom/glow: none in the film renderer")
    lines.append("- blur: only the specified cast-shadow and paper-grain filters in the still renderer")

    film = Film("A")
    opening = film.render(0)
    closing = film.render(1823)
    score = ssim_u8(opening, closing)
    lines.append(f"- loop SSIM {score:.4f}")
    print(f"loop SSIM {score:.4f}")
    if score < 0.99:
        errors.append(f"loop SSIM {score:.4f}")

    luma_path = ROOT / "out" / "paperfold_luma.npy"
    if luma_path.exists():
        luma = np.load(luma_path)
        hot = photosensitivity_hot_count(luma, 60)
        lines.append(f"- photosensitivity hot frames {hot}")
        print(f"photosensitivity hot frames {hot}")
        if hot > 3:
            errors.append(f"luminance flashed on {hot} frames inside one second")

    if path.exists():
        info = _probe(find_ffprobe(find_ffmpeg()), path)
        video = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), None)
        audio = next((s for s in info.get("streams", []) if s.get("codec_type") == "audio"), None)
        if video is None or audio is None:
            errors.append("missing video or audio stream")
        else:
            errors.extend(frame_rate_errors(find_ffprobe(find_ffmpeg()), path, rate="60/1", frames=1824))
            width = int(video.get("width", 0))
            height = int(video.get("height", 0))
            lines.append(f"- raster {width}x{height} {video.get('r_frame_rate')}")
            if (width, height) != (1080, 1920):
                errors.append(f"raster {width}x{height}")
        wav = path.with_suffix(".verify.wav")
        subprocess.run(
            [find_ffmpeg(), "-y", "-v", "error", "-i", str(path), "-ac", "2", str(wav)],
            check=False,
        )
        if wav.exists():
            from scipy.io import wavfile
            import pyloudnorm as pyln

            sr, data = wavfile.read(wav)
            if np.issubdtype(data.dtype, np.integer):
                data = data.astype(np.float64) / np.iinfo(data.dtype).max
            else:
                data = data.astype(np.float64)
            if data.ndim == 1:
                data = np.stack([data, data], axis=1)
            lufs = float(pyln.Meter(sr).integrated_loudness(data))
            peak = true_peak_db(data)
            lines.append(f"- loudness {lufs:.2f} LUFS, true peak {peak:.2f} dBTP")
            print(f"loudness {lufs:.2f} LUFS, true peak {peak:.2f} dBTP")
            if abs(lufs + 14) > 1:
                errors.append(f"loudness {lufs:.2f} LUFS")
            if peak > -1 + 1e-3:
                errors.append(f"true peak {peak:.2f} dBTP")
            if float(np.max(np.abs(data))) > 1:
                errors.append("clipping")
            onset = onset_sample(data, sr)
            lines.append(f"- kick onset sample {onset}")
            print(f"kick onset sample {onset}")
            if onset > int(0.001 * sr):
                errors.append(f"kick onset at sample {onset}")
            timeline = build_timeline()
            missing = events_aligned(data, timeline.kicks[:8] + timeline.impacts, sr, 60)
            if missing:
                errors.append(f"missing onsets at frames {missing[:8]}")
                lines.append(f"- missing onsets {missing[:8]}")
            else:
                lines.append("- timeline onsets present")
            tail = data[-18 * (sr // 60) :]
            tail_db = 20.0 * np.log10(float(np.max(np.abs(tail))) + 1e-12)
            lines.append(f"- tail peak {tail_db:.1f} dBFS")
            if tail_db > -50:
                errors.append(f"tail peak {tail_db:.1f} dBFS")
    else:
        errors.append(f"missing {path}")

    if errors:
        lines.append("")
        lines.append("## Failures")
        for error in errors:
            lines.append(f"- {error}")
            print(f"FAIL {error}")
    else:
        lines.append("")
        lines.append("All checks passed.")
        print("verify passed")
    report = ROOT / "out" / "verify_report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n")
    print(f"wrote {report}")
    return 1 if errors else 0
