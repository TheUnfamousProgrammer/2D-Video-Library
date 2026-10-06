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
from fc_sat.coinspin_math import (
    GLIDE_OUT,
    RESIZE,
    RETURN,
    SNAP,
    TAU,
    carry_spins,
    film_lap_spins,
    gold_pose,
    road_spins,
    sat_lap_spins,
    stage_scale,
    upright_frames,
)
from fc_sat.coinspin_render import ARROWS, CoinRenderer, inside_safe, overlaps
from fc_sat.coinspin_schedule import count_changes
from fc_sat.coinspin_timeline import build_timeline
from fc_sat.encode import find_ffmpeg, find_ffprobe
from fc_sat.verify import _probe, photosensitivity_hot_count, ssim_u8

ROOT = Path(__file__).resolve().parents[1]
MAX_TEXT = 6
_CLAIM_IDS = [
    "unit",
    "halfway_spins",
    "equal_spins",
    "road_spins",
    "trip_spins",
    "sat_year",
    "sat_ratio",
    "sat_intended",
    "sat_spins",
    "sat_rolling",
    "screen_counts",
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


def physics_errors() -> list[str]:
    """Contact and no slipping on the rim, distance over radius on the road, a rigid rod."""
    errors = []
    for name, fn, want in (
        ("same-size lap", film_lap_spins, 2),
        ("flat road", road_spins, 1),
        ("bolted carry", carry_spins, 1),
        ("SAT lap", sat_lap_spins, 4),
    ):
        try:
            got = fn()
        except ValueError as exc:
            errors.append(f"{name}: {exc}")
            continue
        if got != want:
            errors.append(f"{name} spins {got}, expected {want}")
    return errors


def count_sync_errors() -> list[str]:
    """A count changes only on a frame where the face is exactly upright, and that frame rings."""
    errors = []
    timeline = build_timeline()
    uprights = upright_frames()
    changes = count_changes()
    if changes != uprights:
        errors.append(f"counts change on {changes}, faces come upright on {uprights}")
    for frame in uprights:
        turns = gold_pose(frame).theta / TAU
        if abs(turns - round(turns)) > 1e-9:
            errors.append(f"frame {frame}: face is {turns % 1:.4f} of a turn off upright")
        if frame not in timeline.bells:
            errors.append(f"no bell on upright frame {frame}")
    return errors


def _designed(frame: int) -> bool:
    spans = (GLIDE_OUT, RETURN, RESIZE, SNAP, (743, 769))
    return any(start <= frame <= end for start, end in spans)


def continuity_errors() -> list[str]:
    errors = []
    previous = gold_pose(0).center
    for frame in range(1, 1824):
        center = gold_pose(frame).center
        delta = math.hypot(center[0] - previous[0], center[1] - previous[1])
        limit = 40.0 if _designed(frame) else 16.0
        if delta > limit:
            errors.append(f"gold coin jumped {delta:.1f} px at frame {frame}")
            if len(errors) > 6:
                break
        previous = center
    return errors


def _stage_extent(frame: int) -> tuple[float, float]:
    """Highest and lowest y the coins and their marks reach, in design pixels."""
    pose = gold_pose(frame)
    s = stage_scale(frame)
    reach = pose.radius + 31.0
    if any(start <= frame < gone for start, _done, _fade, gone in ARROWS):
        reach = max(reach, pose.radius + 31.0)
    if frame in range(768, 780) or frame in range(1632, 1644):
        reach = max(reach, pose.radius + 78.0)
    top = 915.0 + s * (pose.center[1] - reach - 915.0)
    bottom = 915.0 + s * (pose.center[1] + pose.radius + 4.0 - 915.0)
    if 384 <= frame < 900:
        bottom = max(bottom, 915.0 + 240.0 + 120.0)
    return top, bottom


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
        top, bottom = _stage_extent(frame)
        tops = [box for box in visible if box.y < 700]
        bottoms = [box for box in visible if box.y >= 700]
        if tops and top < max(box.bottom() for box in tops) + 24.0:
            errors.append(f"frame {frame}: the stage reaches y {top:.0f}, under the top card")
        if bottoms and bottom > min(box.y for box in bottoms) - 16.0:
            errors.append(f"frame {frame}: the stage reaches y {bottom:.0f}, under the counter plate")
        if len(errors) > 8:
            break
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
    pose = gold_pose(0)
    measured = _gold_radius(opening, pose.center)
    print(f"frame 0 gold coin radius {measured:.2f} px, expected {pose.radius:.2f}")
    if not math.isfinite(measured) or abs(measured - pose.radius) > 1.5:
        errors.append(f"frame 0 gold coin radius {measured:.2f} px, expected {pose.radius:.2f}")
    if abs(pose.theta) > 1e-12:
        errors.append("frame 0 face is not upright")
    return errors


def _gold_radius(frame: np.ndarray, center: tuple[float, float]) -> float:
    """Half the gold run along a row 40% of a radius below the center (clear of the eyes)."""
    row = frame[int(round(center[1] - 0.45 * 120.0))].astype(np.int32)
    blue, green, red = row[:, 0], row[:, 1], row[:, 2]
    gold = (red > 150) & (green > 100) & (blue < 110) & (red - blue > 90)
    xs = np.flatnonzero(gold)
    if xs.size == 0:
        return float("nan")
    half_chord = (float(xs.max()) - float(xs.min()) + 1.0) / 2.0
    return math.hypot(half_chord, 0.45 * 120.0)


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
    if (int(video.get("width") or 0), int(video.get("height") or 0)) != (1080, 1920):
        errors.append(f"size {video.get('width')}x{video.get('height')}")
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
    missing = events_aligned(data, timeline.kicks[:8] + timeline.impacts + timeline.uprights, sr, 60)
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
    errors.extend(physics_errors())
    errors.extend(count_sync_errors())
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
