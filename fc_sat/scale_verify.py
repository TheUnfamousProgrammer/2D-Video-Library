"""Checks for the scale short. Failures name the measurement."""

from __future__ import annotations

import math
import subprocess
from pathlib import Path

import numpy as np

from fc_sat.audio import true_peak_db
from fc_sat.beatkit.probes import frame_rate_errors
from fc_sat.encode import find_ffmpeg, find_ffprobe
from fc_sat.scale_audio import events_aligned, onset_sample
from fc_sat.scale_claims import display_errors, evaluate, lint_text, load_claims
from fc_sat.scale_render import ScaleRenderer, inside_safe, overlaps
from fc_sat.scale_scene import build_scene, load_sizes
from fc_sat.scale_timeline import build_timeline
from fc_sat.scale_world import BASELINE_Y, SNAP, camera_at, check_order, focus_index, punch
from fc_sat.verify import _probe, photosensitivity_hot_count, ssim_u8

ROOT = Path(__file__).resolve().parents[1]
MAX_TEXT = 8
# Clear space between the featured object and either text plate, in design pixels.
PLATE_PAD = 22.0
CLEARANCE = 8.0


def style_law_errors() -> list[str]:
    forbidden = ("blur", "bloom", "glow", "gradient", "shadow")
    errors = []
    for relative in ("fc_sat/scale_render.py", "fc_sat/circlesquare_draw.py"):
        text = (ROOT / relative).read_text().lower()
        for word in forbidden:
            if word in text:
                errors.append(f"{relative} contains {word}")
    return errors


def fact_errors() -> list[str]:
    errors = []
    results = evaluate(load_claims())
    errors += [f"claim {claim_id}: {detail}" for claim_id, ok, detail in results if not ok]
    sizes = load_sizes()
    errors += display_errors(sizes)
    provisional = [object_id for object_id, body in sizes.items() if "provisional" in str(body.get("source", "")).lower()]
    if provisional:
        errors.append("sizes still provisional: " + ", ".join(provisional))
    errors += check_order(build_scene().items)
    return errors


def art_errors() -> list[str]:
    scene = build_scene()
    missing = [placed.item.id for placed, has in zip(scene.placed, scene.has_art) if not has]
    return ["no cutout yet for " + ", ".join(missing)] if missing else []


def sync_errors() -> list[str]:
    """Every landing is on the sixteenth-note grid and rings a bell (frame 0 is the loop start)."""
    timeline = build_timeline()
    errors = []
    if timeline.max_snap_error > 1e-9:
        errors.append(f"a landing is {timeline.max_snap_error:.3f} frames off the grid")
    for frame in timeline.lands[1:]:
        if frame not in timeline.bells:
            errors.append(f"no bell on landing frame {frame}")
    return errors


def camera_errors() -> list[str]:
    """The camera arrives exactly on each landing, only ever zooms out, and the loop closes."""
    placed = list(build_scene().placed)
    errors = []
    for index, p in enumerate(placed):
        camera = camera_at(placed, p.item.land)
        if abs(camera.scale / p.scale - 1.0) > 1e-9 or abs(camera.x_m - p.x_m) > 1e-9 * max(1.0, abs(p.x_m)):
            errors.append(f"camera is not framed on {p.item.id} at its landing frame {p.item.land}")
        if camera.focus != index:
            errors.append(f"focus at frame {p.item.land} is {camera.focus}, expected {index}")
    previous = camera_at(placed, 0).scale
    for frame in range(1, SNAP[0]):
        scale = camera_at(placed, frame).scale
        if scale > previous * (1.0 + 1e-12):
            errors.append(f"camera zooms back in at frame {frame}")
            break
        previous = scale
    start, end = camera_at(placed, 0), camera_at(placed, 1823)
    if (start.x_m, start.scale) != (end.x_m, end.scale):
        errors.append("camera at frame 1823 is not the camera at frame 0")
    return errors


def _object_extent(renderer: ScaleRenderer, frame: int) -> tuple[float, float]:
    placed = renderer.placed
    camera = camera_at(placed, frame)
    bump = punch(placed, frame) if frame < SNAP[0] else 1.0
    _x, y, _w, h = renderer._rect(placed[focus_index(placed, frame)], camera, bump)
    return y, y + h


def layout_errors(step: int = 1) -> list[str]:
    book = load_claims()
    renderer = ScaleRenderer(width=1080, height=1920, hook="A")
    errors = []
    held = _held_frames(renderer)
    for frame in range(0, 1824, step):
        boxes = renderer.plan(frame)
        visible = [box for box in boxes if box.alpha > 0.2 and box.text]
        if len(visible) > MAX_TEXT:
            errors.append(f"frame {frame} has {len(visible)} text elements")
        focus = renderer.placed[focus_index(renderer.placed, frame)].item.id
        for box in visible:
            if not inside_safe(box, 1.0):
                errors.append(f"frame {frame} {box.text!r} leaves the safe zone")
        # A counter wrapped onto two rows is one number: lint the rows joined back together.
        words = [box.text for box in visible if box.kind != "mono"] + ["".join(box.text for box in visible if box.kind == "mono")]
        for text in words:
            errors.extend(lint_text(text, [focus], book, f"frame {frame}"))
        for index, left in enumerate(visible):
            for right in visible[index + 1 :]:
                if overlaps(left, right):
                    errors.append(f"frame {frame} overlap {left.text!r} / {right.text!r}")
        if frame in held:
            top, bottom = _object_extent(renderer, frame)
            tops = [box for box in visible if box.y < 700]
            bottoms = [box for box in visible if box.y >= 700]
            if tops and top < max(box.bottom() for box in tops) + PLATE_PAD + CLEARANCE:
                errors.append(f"frame {frame}: {focus} reaches y {top:.0f}, under the top card")
            if bottoms and bottom > min(box.y for box in bottoms) - PLATE_PAD - CLEARANCE:
                errors.append(f"frame {frame}: {focus} reaches y {bottom:.0f}, under the counter plate")
        if len(errors) > 8:
            break
    return errors


def _held_frames(renderer: ScaleRenderer) -> set[int]:
    """Frames where the camera sits on one object (not mid-move, not the snap)."""
    held = set()
    for frame in range(SNAP[0]):
        if camera_at(renderer.placed, frame).blend == 0.0:
            held.add(frame)
    return held


def geometry_errors(worst: Path) -> list[str]:
    renderer = ScaleRenderer(width=1080, height=1920)
    opening = renderer.render(0)
    closing = renderer.render(1823)
    seam = ssim_u8(opening, closing)
    print(f"loop seam SSIM {seam:.4f}")
    if seam < 0.99:
        _save(closing, worst)
        return [f"loop seam SSIM {seam:.4f}"]
    return []


def flash_errors() -> list[str]:
    renderer = ScaleRenderer(width=270, height=480)
    luma = []
    for frame in range(0, 1824, 2):
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
    missing = events_aligned(data, timeline.kicks[:8] + timeline.impacts + timeline.bells, sr, 60)
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


def verify(path: Path | None = None, *, allow_placeholders: bool = False) -> int:
    errors = []
    errors.extend(fact_errors())
    art = art_errors()
    if allow_placeholders:
        for line in art:
            print(f"note: {line}")
    else:
        errors.extend(art)
    errors.extend(style_law_errors())
    errors.extend(sync_errors())
    errors.extend(camera_errors())
    errors.extend(layout_errors())
    errors.extend(geometry_errors(ROOT / "out" / "scale_worst.png"))
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
    (out / "scale_verify_report.md").write_text("\n".join(report) + "\n")
    return 1 if errors else 0
