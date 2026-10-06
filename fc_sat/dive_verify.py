"""Checks for the deepest-to-highest short. Failures name the measurement."""

from __future__ import annotations

import math
import subprocess
from pathlib import Path

import numpy as np

from fc_sat.audio import true_peak_db
from fc_sat.beatkit.probes import frame_rate_errors
from fc_sat.dive_render import DiveRenderer
from fc_sat.dive_scene import build_scene, load_values
from fc_sat.dive_timeline import build_timeline
from fc_sat.dive_world import CLAIMS_PATH, SNAP, camera_at, check_order, focus_index
from fc_sat.encode import find_ffmpeg, find_ffprobe
from fc_sat.scale_audio import events_aligned, onset_sample
from fc_sat.scale_claims import display_errors, evaluate, lint_text, load_claims
from fc_sat.scale_render import inside_safe, overlaps
from fc_sat.verify import _probe, photosensitivity_hot_count, ssim_u8

ROOT = Path(__file__).resolve().parents[1]
MAX_TEXT = 8


def style_law_errors() -> list[str]:
    errors = []
    for relative in ("fc_sat/dive_render.py", "fc_sat/circlesquare_draw.py"):
        text = (ROOT / relative).read_text().lower()
        errors += [f"{relative} contains {word}" for word in ("blur", "bloom", "glow", "gradient", "shadow") if word in text]
    return errors


def fact_errors() -> list[str]:
    errors = [f"claim {cid}: {detail}" for cid, ok, detail in evaluate(load_claims(CLAIMS_PATH)) if not ok]
    values = load_values()
    errors += display_errors(values)
    provisional = [k for k, v in values.items() if "provisional" in str(v.get("source", "")).lower()]
    if provisional:
        errors.append("values still provisional: " + ", ".join(provisional))
    errors += check_order(list(build_scene().stops))
    return errors


def art_errors() -> list[str]:
    scene = build_scene()
    missing = [s.art for s, has in zip(scene.stops, scene.has_art) if not has]
    return ["no cutout yet for " + ", ".join(missing)] if missing else []


def camera_errors() -> list[str]:
    stops = list(build_scene().stops)
    timeline = build_timeline()
    errors = []
    if timeline.max_snap_error > 1e-9:
        errors.append("a landing is off the sixteenth-note grid")
    for frame in timeline.lands[1:]:
        if frame not in timeline.bells:
            errors.append(f"no bell on landing {frame}")
    for index, stop in enumerate(stops):
        camera = camera_at(stops, stop.land)
        if abs(camera.span / stop.span - 1.0) > 1e-9 or camera.focus != index:
            errors.append(f"camera is not framed on {stop.id} at frame {stop.land}")
    previous = None
    for frame in range(SNAP[0]):
        camera = camera_at(stops, frame)
        if previous is not None and camera.act == previous.act and camera.span < previous.span * (1 - 1e-12):
            errors.append(f"camera zooms back in at frame {frame}")
            break
        previous = camera
    if camera_at(stops, 0) != camera_at(stops, 1823):
        errors.append("camera at frame 1823 is not the camera at frame 0")
    return errors


def layout_errors() -> list[str]:
    book = load_claims(CLAIMS_PATH)
    renderer = DiveRenderer()
    errors = []
    for frame in range(1824):
        visible = [b for b in renderer.plan(frame) if b.alpha > 0.2 and b.text]
        if len(visible) > MAX_TEXT:
            errors.append(f"frame {frame} has {len(visible)} text elements")
        focus = renderer.stops[focus_index(renderer.stops, frame)].id
        for box in visible:
            if not inside_safe(box, 1.0):
                errors.append(f"frame {frame} {box.text!r} leaves the safe zone")
        words = [b.text for b in visible if b.kind != "mono"] + ["".join(b.text for b in visible if b.kind == "mono")]
        for text in words:
            errors.extend(lint_text(text, [focus], book, f"frame {frame}"))
        for i, left in enumerate(visible):
            for right in visible[i + 1 :]:
                if overlaps(left, right):
                    errors.append(f"frame {frame} overlap {left.text!r} / {right.text!r}")
        if len(errors) > 8:
            break
    return errors


def geometry_errors() -> list[str]:
    renderer = DiveRenderer()
    seam = ssim_u8(renderer.render(0), renderer.render(1823))
    print(f"loop seam SSIM {seam:.4f}")
    return [] if seam >= 0.99 else [f"loop seam SSIM {seam:.4f}"]


def flash_errors() -> list[str]:
    renderer = DiveRenderer(width=270, height=480)
    luma = []
    for frame in range(0, 1824, 2):
        image = renderer.render(frame).astype(np.float64)
        luma.append((0.0722 * image[:, :, 0] + 0.7152 * image[:, :, 1] + 0.2126 * image[:, :, 2]).mean() / 255.0)
    hot = photosensitivity_hot_count(np.array(luma), 30)
    print(f"luma jumps over 0.10 in a 1s window: {hot}")
    return [f"luminance flashed on {hot} frames inside one second"] if hot > 3 else []


def probe_errors(path: Path) -> list[str]:
    errors = []
    ffprobe = find_ffprobe(find_ffmpeg())
    info = _probe(ffprobe, path)
    video = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), None)
    if video is None or not any(s.get("codec_type") == "audio" for s in info.get("streams", [])):
        return ["missing video or audio stream"]
    if (int(video.get("width") or 0), int(video.get("height") or 0)) != (1080, 1920):
        errors.append(f"size {video.get('width')}x{video.get('height')}")
    errors.extend(frame_rate_errors(ffprobe, path, rate="60/1", frames=1824))
    wav = path.with_suffix(".verify.wav")
    subprocess.run([find_ffmpeg(), "-y", "-v", "error", "-i", str(path), "-ac", "2", str(wav)], check=False)
    from scipy.io import wavfile
    import pyloudnorm as pyln

    sr, data = wavfile.read(wav)
    wav.unlink(missing_ok=True)
    data = data.astype(np.float64) / np.iinfo(data.dtype).max if np.issubdtype(data.dtype, np.integer) else data.astype(np.float64)
    lufs = float(pyln.Meter(sr).integrated_loudness(data))
    peak = true_peak_db(data)
    print(f"loudness {lufs:.2f} LUFS, true peak {peak:.2f} dBTP")
    if abs(lufs + 14) > 1:
        errors.append(f"loudness {lufs:.2f} LUFS")
    if peak > -1 + 1e-3:
        errors.append(f"true peak {peak:.2f} dBTP")
    if onset_sample(data, sr) > int(0.001 * sr):
        errors.append("kick onset late")
    timeline = build_timeline()
    missing = events_aligned(data, timeline.kicks[:8] + timeline.impacts + timeline.bells, sr, 60)
    if missing:
        errors.append(f"missing onsets at frames {missing[:8]}")
    tail_db = 20.0 * math.log10(float(np.max(np.abs(data[-18 * (sr // 60) :]))) + 1e-12)
    if tail_db > -50:
        errors.append(f"tail peak {tail_db:.1f} dBFS")
    return errors


def verify(path: Path | None = None, *, allow_placeholders: bool = False) -> int:
    errors = fact_errors()
    art = art_errors()
    if allow_placeholders:
        for line in art:
            print(f"note: {line}")
    else:
        errors += art
    errors += style_law_errors() + camera_errors() + layout_errors() + geometry_errors() + flash_errors()
    if path is not None and path.exists():
        errors += probe_errors(path)
    for error in errors:
        print(f"FAIL {error}")
    print("verify passed" if not errors else f"{len(errors)} failed")
    (ROOT / "out").mkdir(exist_ok=True)
    (ROOT / "out" / "dive_verify_report.md").write_text("# Verify\n\n" + ("\n".join(f"- FAIL {e}" for e in errors) or "All checks passed.") + "\n")
    return 1 if errors else 0
