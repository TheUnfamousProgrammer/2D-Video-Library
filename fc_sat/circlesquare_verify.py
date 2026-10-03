"""Checks for the circlesquare short. Failures name the measurement."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from fc_sat.audio import true_peak_db
from fc_sat.beatkit.probes import frame_rate_errors
from fc_sat.circlesquare_audio import events_aligned, onset_sample
from fc_sat.circlesquare_claims import evaluate, lint_text, load_claims
from fc_sat.circlesquare_math import R1, ZMAX, gap, zoom_at
from fc_sat.circlesquare_render import CircleRenderer, inside_safe, machinery_alpha, overlaps
from fc_sat.circlesquare_timeline import build_timeline
from fc_sat.encode import find_ffmpeg, find_ffprobe
from fc_sat.verify import _probe, photosensitivity_hot_count, ssim_u8

ROOT = Path(__file__).resolve().parents[1]
_CLAIM_IDS = ["screen_counts", "screen_gaps", "zoom_max", "gap_px", "k_fool", "k_drop", "final_k"]


def style_law_errors() -> list[str]:
    forbidden = ("blur", "bloom", "glow", "gradient", "shadow")
    errors = []
    for relative in ("fc_sat/circlesquare_render.py", "fc_sat/circlesquare_draw.py"):
        text = (ROOT / relative).read_text().lower()
        for word in forbidden:
            if word in text:
                errors.append(f"{relative} contains {word}")
    return errors


def _world_tip(renderer: CircleRenderer, frame: int) -> np.ndarray:
    from fc_sat.circlesquare_math import square_complex, theta_at

    zoom = 1.0 if renderer._tease(frame) else zoom_at(frame)
    samples = renderer._shape(frame, zoom)
    if samples is None:
        samples = square_complex(4096)
    theta = 0.0 if frame >= 1823 else theta_at(frame)
    index = int(round((theta / (2.0 * np.pi)) % 1.0 * len(samples))) % len(samples)
    value = samples[index]
    return np.array([float(np.real(value)), float(np.imag(value))], dtype=np.float64)


def continuity_errors() -> list[str]:
    """World-space tip travel. One frame of the orbit is about 26 px; a scene cut is not."""
    timeline = build_timeline()
    renderer = CircleRenderer(timeline, width=270, height=480)
    errors = []
    previous = _world_tip(renderer, 0)
    for frame in range(1, timeline.n_frames):
        tip = _world_tip(renderer, frame)
        delta = float(np.hypot(*(tip - previous)))
        limit = 160.0 if _designed(frame) else 40.0
        if delta > limit:
            errors.append(f"tip jumped {delta:.1f} px at frame {frame}")
            if len(errors) > 6:
                break
        previous = tip
    return errors


def _designed(frame: int) -> bool:
    if 876 <= frame <= 948 or 1152 <= frame <= 1280 or 1800 <= frame <= 1806 or frame >= 1822:
        return True
    if frame % 24 == 0 and frame >= 960:
        return True
    # Eight frames of blend before each doubling, and the add ramps.
    if 952 <= frame <= 1128 or 1144 <= frame <= 1320:
        return True
    return False


def layout_errors() -> list[str]:
    book = load_claims()
    renderer = CircleRenderer(width=1080, height=1920, hook="A")
    errors = []
    for frame in range(0, 1824, 1):
        boxes = renderer.plan(frame)
        visible = [box for box in boxes if box.alpha > 0.2 and box.text]
        if len(visible) > 5:
            errors.append(f"frame {frame} has {len(visible)} text elements")
        for box in visible:
            if not inside_safe(box, 1.0):
                errors.append(f"frame {frame} {box.text!r} leaves the safe zone")
            errors.extend(lint_text(box.text, _CLAIM_IDS, book, f"frame {frame}"))
        for index, left in enumerate(visible):
            for right in visible[index + 1 :]:
                if overlaps(left, right):
                    errors.append(f"frame {frame} overlap {left.text!r} / {right.text!r}")
        if len(errors) > 8:
            break
    return errors


def geometry_errors(worst: Path) -> list[str]:
    errors = []
    timeline = build_timeline()
    renderer = CircleRenderer(timeline, width=1080, height=1920)
    opening = renderer.render(0)
    closing = renderer.render(1823)
    seam = ssim_u8(opening, closing)
    print(f"loop seam SSIM {seam:.4f}")
    if seam < 0.99:
        errors.append(f"loop seam SSIM {seam:.4f}")
        _save(closing, worst)
    measured = _pixel_gap(renderer.render(954))
    expected = gap(93) * ZMAX
    print(f"frame 954 corner gap {measured:.2f} px, vector {expected:.2f}")
    if not math.isfinite(measured) or abs(measured - expected) > 3.0:
        errors.append(f"frame 954 gap {measured:.2f} px, expected {expected:.2f}")
    radius = _pixel_radius(opening)
    print(f"frame 0 radius {radius:.2f} px, r1 {R1:.2f}")
    if not math.isfinite(radius) or abs(radius - R1) > 1.0:
        errors.append(f"frame 0 radius {radius:.2f} px, expected {R1:.2f}")
    for frame in (876, 954, 1128, 1488, 1700):
        if machinery_alpha(frame) > 0.02:
            errors.append(f"machinery alpha {machinery_alpha(frame):.2f} at frame {frame}")
    return errors


def _pixel_gap(frame: np.ndarray) -> float:
    """Centerline gap from the two reference edges, in screen pixels, near the corner."""
    rgb = frame[:, :, ::-1]
    orange = (rgb[:, :, 0] > 180) & (rgb[:, :, 1] > 80) & (rgb[:, :, 1] < 190) & (rgb[:, :, 2] < 120)
    distances = []
    for x in range(490, 536):
        ys = np.flatnonzero(orange[:, x])
        ys = ys[(ys > 905) & (ys < 980)]
        if ys.size < 2:
            continue
        y = float(np.median(ys))
        distances.append(min(540.0 - x, y - 905.0))
    if not distances:
        return float("nan")
    return float(max(distances))


def _pixel_radius(frame: np.ndarray) -> float:
    rgb = frame[:, :, ::-1]
    row = rgb[905].astype(np.float64).mean(axis=1)
    left = int(np.argmax(row[:500]))
    right = 580 + int(np.argmax(row[580:]))
    return float(right - left) / 2.0


def flash_errors() -> list[str]:
    timeline = build_timeline()
    renderer = CircleRenderer(timeline, width=270, height=480)
    luma = []
    for frame in range(0, timeline.n_frames, 2):
        image = renderer.render(frame)
        rgb = image.astype(np.float64)
        y = (0.0722 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.2126 * rgb[:, :, 2]).mean() / 255.0
        luma.append(y)
        if frame and frame % 600 == 0:
            print(f"luma {frame}", flush=True)
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
    width = int(video.get("width") or 0)
    height = int(video.get("height") or 0)
    if (width, height) != (1080, 1920):
        errors.append(f"size {width}x{height}")
    errors.extend(frame_rate_errors(find_ffprobe(find_ffmpeg()), path, rate="60/1", frames=1824))
    wav = path.with_suffix(".verify.wav")
    import subprocess

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
    missing = events_aligned(data, timeline.kicks[:8] + timeline.impacts, sr, 60)
    if missing:
        errors.append(f"missing onsets at frames {missing[:8]}")
    tail = data[-18 * (sr // 60) :]
    tail_db = 20.0 * math.log10(float(np.max(np.abs(tail))) + 1e-12)
    print(f"tail peak {tail_db:.1f} dBFS")
    if tail_db > -50:
        errors.append(f"tail peak {tail_db:.1f} dBFS")
    return errors


def _save(frame: np.ndarray, path: Path) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame[:, :, ::-1]).save(path)


def verify(path: Path | None = None) -> int:
    errors = []
    claims = evaluate(load_claims())
    failed = [claim_id for claim_id, ok, _detail in claims if not ok]
    if failed:
        errors.append("claims failed: " + ", ".join(failed))
    else:
        print(f"claims {len(claims)}/{len(claims)} passed")
    errors.extend(style_law_errors())
    errors.extend(layout_errors())
    errors.extend(continuity_errors())
    errors.extend(geometry_errors(ROOT / "out" / "circlesquare_worst.png"))
    errors.extend(flash_errors())
    if path is not None and path.exists():
        errors.extend(probe_errors(path))
    report = ["# Verify", ""]
    if errors:
        report.append(f"{len(errors)} failed.")
        for error in errors:
            print(f"FAIL {error}")
            report.append(f"- FAIL {error}")
    else:
        report.append("All checks passed.")
        print("verify passed")
    report.append("")
    (ROOT / "out" / "verify_report.md").write_text("\n".join(report) + "\n")
    return 1 if errors else 0
