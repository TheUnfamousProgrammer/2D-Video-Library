"""Checks for the coinspin short. Failures name the measurement."""

from __future__ import annotations

import math
import subprocess
from pathlib import Path

import numpy as np

from fc_sat.audio import true_peak_db
from fc_sat.beatkit.probes import frame_rate_errors
from fc_sat.coinspin_audio import events_aligned, onset_sample
from fc_sat.coinspin_claims import evaluate, lint_text, load_claims
from fc_sat.coinspin_math import CX, CY, DOUBLING_FRAMES, EARTH_MORPH, GROWS, LAPS, SNAP, STAGE, TAU, scene_at, theta_at
from fc_sat.coinspin_render import CoinRenderer, inside_safe, overlaps
from fc_sat.coinspin_schedule import counter_at, spin_frames
from fc_sat.coinspin_timeline import build_timeline
from fc_sat.encode import find_ffmpeg, find_ffprobe
from fc_sat.verify import _probe, photosensitivity_hot_count, ssim_u8

ROOT = Path(__file__).resolve().parents[1]
MAX_TEXT = 6
_CLAIM_IDS = [
    "screen_counts",
    "equal_spins",
    "ratio_two",
    "sat_year",
    "sat_ratio",
    "sat_choices",
    "sat_intended",
    "sat_spins",
    "sat_rolling",
    "plus_one",
    "doubling_ratios",
    "tropical_year",
    "sidereal_year",
    "sidereal_whole",
    "sidereal_day_s",
]


def style_law_errors() -> list[str]:
    forbidden = ("blur", "bloom", "glow", "gradient", "shadow")
    errors = []
    for relative in ("fc_sat/coinspin_render.py", "fc_sat/circlesquare_draw.py"):
        text = (ROOT / relative).read_text().lower()
        for word in forbidden:
            if word in text:
                errors.append(f"{relative} contains {word}")
    return errors


def layout_errors(step: int = 1) -> list[str]:
    book = load_claims()
    renderer = CoinRenderer(width=1080, height=1920, hook="A")
    errors = []
    for frame in range(0, 1824, step):
        boxes = renderer.plan(frame)
        visible = [box for box in boxes if box.alpha > 0.2 and box.text]
        if len(visible) > MAX_TEXT:
            errors.append(f"frame {frame} has {len(visible)} text elements")
        for box in visible:
            if not inside_safe(box, 1.0):
                errors.append(f"frame {frame} {box.text!r} leaves the safe zone")
            errors.extend(lint_text(box.text, _CLAIM_IDS, book, f"frame {frame}"))
        for index, left in enumerate(visible):
            for right in visible[index + 1 :]:
                if overlaps(left, right):
                    errors.append(f"frame {frame} overlap {left.text!r} / {right.text!r}")
        bottom = [box for box in visible if box.y >= 600]
        if bottom:
            plate_top = min(box.y for box in bottom) - 16.0
            scene = scene_at(frame)
            lowest = max(scene.fixed_center[1] + scene.fixed_px, scene.rolling_center[1] + scene.rolling_px)
            if lowest > plate_top:
                errors.append(f"frame {frame} coins reach y {lowest:.0f}, under the counter plate at {plate_top:.0f}")
        if len(errors) > 8:
            break
    return errors


def spin_sync_errors() -> list[str]:
    """Every bell lands on the frame the arrow points straight up and the counter steps."""
    errors = []
    timeline = build_timeline()
    for frame in spin_frames():
        lap = next(lap for lap in LAPS if lap.start < frame <= lap.end)
        turns = (lap.ratio + 1.0) * (frame - lap.start) / float(lap.end - lap.start)
        if abs(turns - round(turns)) > 1e-9:
            errors.append(f"spin bell at frame {frame} lands {turns:.3f} turns into the lap")
        if frame not in timeline.bells:
            errors.append(f"no bell at spin frame {frame}")
        before = counter_at(frame - 1).value
        after = counter_at(frame).value
        if after != (before or 0) + 1:
            errors.append(f"counter goes {before} -> {after} at spin frame {frame}")
    for lap in LAPS:
        if not lap.counted:
            continue
        for frame in range(lap.start + 1, lap.end):
            if frame in spin_frames():
                continue
            if counter_at(frame).value != counter_at(frame - 1).value:
                errors.append(f"counter steps at frame {frame} without a spin")
            angle = theta_at(frame) % TAU
            if counter_at(frame).value != int(theta_at(frame) // TAU):
                errors.append(f"counter {counter_at(frame).value} disagrees with the arrow at frame {frame} ({angle:.3f})")
    return errors[:8]


def contact_errors() -> list[str]:
    """While coins roll, the rolling coin touches the rim and never sinks into it."""
    errors = []
    for frame in range(0, EARTH_MORPH[0]):
        scene = scene_at(frame)
        gap = math.hypot(scene.rolling_center[0] - CX, scene.rolling_center[1] - CY)
        want = scene.fixed_px + scene.rolling_px
        if abs(gap - want) > 1e-6:
            errors.append(f"frame {frame}: centers {gap:.3f} px apart, radii sum {want:.3f}")
        extent = gap + scene.rolling_px
        if extent > STAGE + 1e-6:
            errors.append(f"frame {frame}: coins reach {extent:.1f} px from center, stage is {STAGE:g}")
        if len(errors) > 6:
            break
    return errors


def _designed(frame: int) -> bool:
    if any(first <= frame <= landing for first, landing, _a, _b in GROWS):
        return True
    if frame in (216, 384, 768, DOUBLING_FRAMES[0]):
        return True
    if EARTH_MORPH[0] <= frame <= EARTH_MORPH[1] or SNAP[0] <= frame <= SNAP[1]:
        return True
    return any(landing <= frame <= landing + 8 for landing in DOUBLING_FRAMES)


def continuity_errors() -> list[str]:
    """Rolling-coin center travel per frame. A lap moves it a few px; a cut is not allowed."""
    errors = []
    previous = scene_at(0).rolling_center
    for frame in range(1, 1824):
        center = scene_at(frame).rolling_center
        delta = math.hypot(center[0] - previous[0], center[1] - previous[1])
        limit = 120.0 if _designed(frame) else 45.0
        if delta > limit:
            errors.append(f"rolling coin jumped {delta:.1f} px at frame {frame}")
            if len(errors) > 6:
                break
        previous = center
    return errors


def geometry_errors(worst: Path) -> list[str]:
    errors = []
    renderer = CoinRenderer(width=1080, height=1920)
    opening = renderer.render(0)
    closing = renderer.render(1823)
    seam = ssim_u8(opening, closing)
    print(f"loop seam SSIM {seam:.4f}")
    if seam < 0.99:
        errors.append(f"loop seam SSIM {seam:.4f}")
        _save(closing, worst)
    scene = scene_at(0)
    measured = _gold_radius(opening, scene.rolling_center)
    print(f"frame 0 rolling coin radius {measured:.2f} px, expected {scene.rolling_px:.2f}")
    if not math.isfinite(measured) or abs(measured - scene.rolling_px) > 1.5:
        errors.append(f"frame 0 rolling coin radius {measured:.2f} px, expected {scene.rolling_px:.2f}")
    return errors


def _gold_radius(frame: np.ndarray, center: tuple[float, float]) -> float:
    """Half the gold run along the rolling coin's center row, rim included."""
    row = frame[int(round(center[1]))].astype(np.int32)
    blue, green, red = row[:, 0], row[:, 1], row[:, 2]
    gold = (red > 150) & (green > 100) & (blue < 110) & (red - blue > 90)
    xs = np.flatnonzero(gold)
    if xs.size == 0:
        return float("nan")
    return (float(xs.max()) - float(xs.min()) + 1.0) / 2.0


def flash_errors() -> list[str]:
    timeline = build_timeline()
    renderer = CoinRenderer(timeline, width=270, height=480)
    luma = []
    for frame in range(0, timeline.n_frames, 2):
        image = renderer.render(frame).astype(np.float64)
        y = (0.0722 * image[:, :, 0] + 0.7152 * image[:, :, 1] + 0.2126 * image[:, :, 2]).mean() / 255.0
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
    ffprobe = find_ffprobe(find_ffmpeg())
    info = _probe(ffprobe, path)
    video = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), None)
    audio = next((s for s in info.get("streams", []) if s.get("codec_type") == "audio"), None)
    if video is None or audio is None:
        return ["missing video or audio stream"]
    width = int(video.get("width") or 0)
    height = int(video.get("height") or 0)
    if (width, height) != (1080, 1920):
        errors.append(f"size {width}x{height}")
    errors.extend(frame_rate_errors(ffprobe, path, rate="60/1", frames=1824))
    wav = path.with_suffix(".verify.wav")
    subprocess.run([find_ffmpeg(), "-y", "-v", "error", "-i", str(path), "-ac", "2", str(wav)], check=False)
    if not wav.exists():
        return errors + ["could not extract audio"]
    from scipy.io import wavfile

    sr, data = wavfile.read(wav)
    wav.unlink(missing_ok=True)
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
    missing = events_aligned(data, timeline.kicks[:8] + timeline.impacts + timeline.spins, sr, 60)
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
    errors.extend(spin_sync_errors())
    errors.extend(contact_errors())
    errors.extend(continuity_errors())
    errors.extend(layout_errors())
    errors.extend(geometry_errors(ROOT / "out" / "coinspin_worst.png"))
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
    out = ROOT / "out"
    out.mkdir(parents=True, exist_ok=True)
    (out / "coinspin_verify_report.md").write_text("\n".join(report) + "\n")
    return 1 if errors else 0
