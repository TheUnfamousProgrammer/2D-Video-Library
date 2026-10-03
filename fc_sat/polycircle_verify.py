"""Checks for polycircle. A failed check prints the measurement and saves the worst frame."""

from __future__ import annotations

import math
import subprocess
from pathlib import Path

import numpy as np

from fc_sat.audio import true_peak_db
from fc_sat.beatkit.probes import frame_rate_errors
from fc_sat.encode import find_ffmpeg, find_ffprobe
from fc_sat.polycircle_audio import events_aligned, onset_sample
from fc_sat.polycircle_claims import evaluate, load_claims
from fc_sat.polycircle_geometry import gap, is_convex, monotone_angles, points_from
from fc_sat.polycircle_render import PolyRenderer
from fc_sat.polycircle_timeline import build_timeline
from fc_sat.verify import _probe, photosensitivity_hot_count, ssim_u8

ROOT = Path(__file__).resolve().parents[1]


def style_law_errors() -> list[str]:
    forbidden = ("blur", "bloom", "glow", "gradient", "shadow")
    errors = []
    for relative in ("fc_sat/polycircle_render.py", "fc_sat/polycircle_draw.py"):
        text = (ROOT / relative).read_text().lower()
        for word in forbidden:
            if word in text:
                errors.append(f"{relative} contains {word}")
    return errors


def _luma_column(frame: np.ndarray, x: int) -> np.ndarray:
    column = frame[:, x].astype(np.float64)
    return 0.0722 * column[:, 0] + 0.7152 * column[:, 1] + 0.2126 * column[:, 2]


def measure_center_gap(frame: np.ndarray) -> float:
    """Distance between the two bright ridges on the center column, in pixels."""
    luma = _luma_column(frame, frame.shape[1] // 2)
    y0, y1 = 880, 960
    region = luma[y0:y1]
    peaks = []
    for index in range(1, len(region) - 1):
        if region[index] >= region[index - 1] and region[index] >= region[index + 1] and region[index] > 40:
            if peaks and index - peaks[-1][0] < 4:
                if region[index] > peaks[-1][1]:
                    peaks[-1] = (index, float(region[index]))
                continue
            peaks.append((index, float(region[index])))
    peaks.sort(key=lambda item: item[1], reverse=True)
    if len(peaks) < 2:
        return float("nan")
    first, second = sorted((peaks[0][0], peaks[1][0]))
    return float(second - first)


def supersampled_gap(n_sides: int = 61, radius: float = 370.0) -> float:
    """4x drawing of the top edge. Returns the gap in original pixels."""
    from fc_sat.polycircle_draw import bgr, make_canvas
    from fc_sat.polycircle_geometry import regular_angles

    scale = 4
    span = int(radius * 2 * scale + 80)
    canvas = make_canvas(span, span)
    canvas.fill("#0E1117")
    center = np.array([span / 2.0, span / 2.0])
    phi = -math.pi / 2 - math.pi / n_sides
    angles = regular_angles(n_sides, phi)
    radii = np.full(n_sides, radius * scale)
    pts = [(float(center[0] + r * math.cos(a)), float(center[1] + r * math.sin(a))) for a, r in zip(angles, radii)]
    canvas.stroke(pts, "#FFD166", 2.0 * scale, 1.0, closed=True)
    x = int(round(center[0]))
    luma = _luma_column(bgr(canvas), x)
    above = luma[: int(center[1])]
    bright = np.where(above > float(above.max()) * 0.5)[0]
    if bright.size == 0:
        return float("nan")
    y_mid = (float(bright[0]) + float(bright[-1])) / 2.0
    circle_top = center[1] - radius * scale
    return (y_mid - circle_top) / scale


def geometry_errors(worst: Path) -> list[str]:
    errors = []
    timeline = build_timeline()
    renderer = PolyRenderer(timeline, width=1080, height=1920)
    frame = renderer.render(954)
    measured = measure_center_gap(frame)
    expected = gap(96, 370.0) * 36.0
    print(f"frame 954 center gap {measured:.2f} px, expected {expected:.2f}")
    if not math.isfinite(measured) or abs(measured - expected) > 1.5:
        errors.append(f"frame 954 gap {measured:.2f} px, expected {expected:.2f}")
        _save(frame, worst)
    try:
        sampled = supersampled_gap(61, 370.0)
    except Exception as exc:
        sampled = float("nan")
        errors.append(f"supersampled gap failed: {exc}")
    print(f"n=61 supersampled gap {sampled:.3f} px")
    if not math.isfinite(sampled) or not 0.25 <= sampled <= 0.55:
        errors.append(f"n=61 gap {sampled:.3f} px is outside 0.25..0.55")
    opening = renderer.render(0)
    closing = renderer.render(1823)
    seam = ssim_u8(opening, closing)
    print(f"loop seam SSIM {seam:.4f}")
    if seam < 0.99:
        errors.append(f"loop seam SSIM {seam:.4f}")
        _save(closing, worst)
    for frame_index in (0, 24, 100, 200, 400, 600, 800, 1000, 1200, 1500, 1823):
        _zoom, window, _morph, angles, radii = renderer._geometry(frame_index)
        if len(angles) >= 2:
            steps = np.diff(angles)
            if np.any(steps <= 1e-9) or np.any(steps >= math.pi):
                errors.append(f"frame {frame_index} angles are not monotone")
        if window is None and len(angles) >= 3 and not monotone_angles(angles):
            errors.append(f"frame {frame_index} closed polygon is not monotone")
        if window is None and len(angles) >= 3:
            pts = points_from(renderer.center, angles, radii)
            if not is_convex(pts):
                errors.append(f"frame {frame_index} polygon is not convex")
    return errors


def _save(frame: np.ndarray, path: Path) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame[:, :, ::-1]).save(path)
    print(f"wrote {path}")


def flash_errors() -> list[str]:
    timeline = build_timeline()
    renderer = PolyRenderer(timeline, width=270, height=480)
    luma = []
    for frame in range(0, timeline.n_frames, 2):
        image = renderer.render(frame)
        rgb = image.astype(np.float64)
        y = (0.0722 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.2126 * rgb[:, :, 2]).mean() / 255.0
        luma.append(y)
        if frame and frame % 600 == 0:
            print(f"luma {frame}", flush=True)
    # Sampled at 30 fps, so a 1 second window is 30 samples.
    hot = photosensitivity_hot_count(np.array(luma), 30)
    print(f"luma jumps over 0.10 in a 1s window: {hot}")
    if hot > 3:
        return [f"luminance flashed on {hot} frames inside one second"]
    return []


def probe_errors(path: Path) -> list[str]:
    errors = []
    info = _probe(find_ffprobe(find_ffmpeg()), path)
    video = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), None)
    audio = next((s for s in info.get("streams", []) if s.get("codec_type") == "audio"), None)
    if video is None or audio is None:
        return ["missing video or audio stream"]
    if video.get("codec_name") != "h264":
        errors.append(f"video codec {video.get('codec_name')}")
    if str(video.get("pix_fmt")) != "yuv420p":
        errors.append(f"pix_fmt {video.get('pix_fmt')}")
    duration = float(info["format"].get("duration", 0))
    print(f"duration {duration:.3f}s")
    if abs(duration - 30.4) > 0.05:
        errors.append(f"duration {duration:.3f}s")
    vdur = float(video.get("duration") or duration)
    adur = float(audio.get("duration") or duration)
    if abs(vdur - adur) > 0.020:
        errors.append(f"A/V durations differ by {abs(vdur - adur) * 1000:.1f} ms")
    errors.extend(frame_rate_errors(find_ffprobe(find_ffmpeg()), path, rate="60/1", frames=1824))
    wav = path.with_suffix(".verify.wav")
    subprocess.run([find_ffmpeg(), "-y", "-v", "error", "-i", str(path), "-ac", "2", str(wav)], check=False)
    if not wav.exists():
        return errors + ["could not extract audio"]
    from scipy.io import wavfile

    sr, data = wavfile.read(wav)
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float64) / np.iinfo(data.dtype).max
    else:
        data = data.astype(np.float64)
    if data.ndim == 1:
        data = np.stack([data, data], axis=1)
    import pyloudnorm as pyln

    lufs = float(pyln.Meter(sr).integrated_loudness(data))
    peak = true_peak_db(data)
    print(f"loudness {lufs:.2f} LUFS, true peak {peak:.2f} dBTP")
    if abs(lufs + 14) > 1:
        errors.append(f"loudness {lufs:.2f} LUFS")
    if peak > -1 + 1e-3:
        errors.append(f"true peak {peak:.2f} dBTP")
    if float(np.max(np.abs(data))) > 1:
        errors.append("clipping")
    onset = onset_sample(data, sr)
    print(f"kick onset sample {onset}")
    if onset > int(0.001 * sr):
        errors.append(f"kick onset at sample {onset}")
    timeline = build_timeline()
    missing = events_aligned(data, timeline.kicks + timeline.arp + timeline.bells, sr, 60)
    if missing:
        errors.append(f"missing onsets at frames {missing[:8]}")
    return errors


def verify(path: Path | None = None) -> int:
    errors = []
    claims = evaluate(load_claims())
    failed = [claim_id for claim_id, ok, _detail in claims if not ok]
    if failed:
        errors.append("claims failed: " + ", ".join(failed))
    else:
        print(f"claims {len(claims)}/{len(claims)} passed")
    errors.extend(style_law_errors())
    errors.extend(geometry_errors(ROOT / "out" / "polycircle_worst.png"))
    errors.extend(flash_errors())
    if path is not None and path.exists():
        errors.extend(probe_errors(path))
    if errors:
        for error in errors:
            print(f"FAIL {error}")
        return 1
    print("verify passed")
    return 0
