"""Checks for an Odd One Out master.

Pixel checks run on the file and again on a crf-30 720x1280 re-encode.
A failure prints the measured number and writes that frame. Thresholds stay put.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pyloudnorm as pyln

from fc_sat.audio import true_peak_db
from fc_sat.encode import find_ffmpeg, find_ffprobe
from fc_sat.odd_config import OddConfig, build_timeline, timeline_frames
from fc_sat.odd_diff import measure_detail, measure_hue, measure_tilt
from fc_sat.odd_layout import math_layout_failures
from fc_sat.verify import (
    Check,
    _add,
    _decode_audio,
    _duration,
    _iter_decoded,
    _luma_mean,
    _probe,
    _stream,
    photosensitivity_hot_count,
)

HARSH_W = 720
HARSH_H = 1280


def harsh_reencode(src: Path, dest: Path) -> None:
    ffmpeg = find_ffmpeg()
    dest.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(src),
        "-vf",
        f"scale={HARSH_W}:{HARSH_H}:flags=bicubic,scale=out_color_matrix=bt709:out_range=tv",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "30",
        "-preset",
        "veryfast",
        "-an",
        str(dest),
    ]
    proc = subprocess.run(command, check=False, capture_output=True)
    if proc.returncode != 0:
        raise SystemExit(proc.stderr.decode("utf-8", "replace"))


def _bg(cfg: OddConfig) -> tuple[int, int, int]:
    from fc_sat.odd_render import _bgr

    return _bgr(cfg.background)


def score_frame(frame: np.ndarray, cfg: OddConfig, sim, scale: float) -> tuple[bool, str]:
    """Detectability for one settled puzzle frame. ``scale`` maps full-res sim coords onto ``frame``."""
    kind = sim.level.difference
    odd = int(sim.odd_index)
    centers = [(float(p[0]) * scale, float(p[1]) * scale) for p in sim.centers]
    odd_xy = centers[odd]
    others = [xy for index, xy in enumerate(centers) if index != odd]
    sample = others[:12]
    if kind in ("hue", "hue_subtle"):
        measured = measure_hue(frame, odd_xy, sample)
        nominal = float(sim.diff_params["distance"])
        ok = measured >= 0.70 * nominal
        return ok, f"oklab {measured:.3f} vs 70% of {nominal:.3f}"
    if kind == "tilt":
        cells = tuple(
            (box[0] * scale, box[1] * scale, box[2] * scale, box[3] * scale) for box in sim.cells
        )
        stats = measure_tilt(frame, list(cells), odd, _bg(cfg))
        nominal = float(cfg.tier.tilt_degrees)
        ok = stats["delta"] >= 0.60 * nominal
        return ok, f"angle delta {stats['delta']:.1f} deg vs 60% of {nominal:.0f} (odd {stats['odd']:.1f}, median {stats['median']:.1f})"
    dot = float(sim.diff_params["dot_radius"]) * scale
    stats = measure_detail(frame, odd_xy, others, dot)
    ok = stats["median"] > 0.2 and stats["ratio"] < 0.20
    return ok, f"white fraction odd {stats['odd']:.3f} is {stats['ratio']:.3f} of median {stats['median']:.3f} (need < 0.20)"


def harsh_measure_show(cfg: OddConfig, show) -> dict[int, str]:
    """Re-encode one settled frame per level at crf 30 / 720x1280 and measure it."""
    from fc_sat.odd_render import OddRenderer, _settled_time

    renderer = OddRenderer(cfg, show, preview=False)
    ffmpeg = find_ffmpeg()
    found: dict[int, str] = {}
    with tempfile.TemporaryDirectory(prefix="odd-harsh-") as tmp:
        root = Path(tmp)
        for level in cfg.levels:
            frame = renderer.render_time(_settled_time(cfg, level.id))
            png = root / f"l{level.id}.png"
            mp4 = root / f"l{level.id}.mp4"
            cv2.imwrite(str(png), frame)
            command = [
                ffmpeg,
                "-y",
                "-loop",
                "1",
                "-i",
                str(png),
                "-frames:v",
                "1",
                "-vf",
                f"scale={HARSH_W}:{HARSH_H}:flags=bicubic,scale=out_color_matrix=bt709:out_range=tv",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "30",
                "-preset",
                "veryfast",
                str(mp4),
            ]
            proc = subprocess.run(command, check=False, capture_output=True)
            if proc.returncode != 0:
                raise SystemExit(proc.stderr.decode("utf-8", "replace"))
            decoded = next(_iter_decoded(ffmpeg, mp4, HARSH_W, HARSH_H))
            ok, detail = score_frame(decoded, cfg, show.levels[level.id], HARSH_W / cfg.width)
            found[level.id] = ("pass " if ok else "FAIL ") + detail
            print(f"harsh L{level.id} {found[level.id]}", flush=True)
            if not ok:
                cv2.imwrite(str(png.with_name(f"L{level.id}.worst.png")), decoded)
    return found


def sim_checks(cfg: OddConfig, show) -> list[Check]:
    checks: list[Check] = []
    previous = None
    for level in cfg.levels:
        sim = show.levels[level.id]
        _add(checks, f"L{level.id} one odd", 0 <= sim.odd_index < level.count, f"index {sim.odd_index} of {level.count}")
        cell = (sim.row, sim.col)
        repeated = previous is not None and cell == previous
        _add(checks, f"L{level.id} new cell", not repeated, f"row {sim.row + 1} col {sim.col + 1} {sim.phrase}")
        previous = cell
        if level.difference in ("hue", "hue_subtle"):
            cvd = sim.diff_params.get("cvd") or {}
            floor = cfg.tier.hue_subtle_cvd if level.difference == "hue_subtle" else cfg.tier.cvd_min_distance
            worst = min(float(cvd.get(kind, 0.0)) for kind in ("protan", "deutan", "tritan"))
            _add(
                checks,
                f"L{level.id} cvd",
                worst + 1e-9 >= floor,
                " ".join(f"{kind}={float(cvd.get(kind, 0.0)):.3f}" for kind in ("protan", "deutan", "tritan")),
            )
    return checks


def sequence_checks(cfg: OddConfig) -> list[Check]:
    checks: list[Check] = []
    timeline = build_timeline(cfg)
    ok = True
    detail = []
    for level in cfg.levels:
        play = next(segment for segment in timeline if segment.kind == "play" and segment.level_id == level.id)
        reveal = next(segment for segment in timeline if segment.kind == "reveal" and segment.level_id == level.id)
        aligned = play.end_frame == reveal.start_frame and abs(play.end_s - reveal.start_s) < 1e-9
        ok = ok and aligned
        detail.append(f"L{level.id}@{reveal.start_s:.3f}s")
    _add(checks, "reveal at timer 0", ok, ", ".join(detail))
    labels = [level.label for level in cfg.levels]
    _add(checks, "label sequence", labels == [f"LEVEL {level.id}" for level in cfg.levels], " ".join(labels))
    return checks


def layout_frame_checks(cfg: OddConfig, show) -> list[Check]:
    from fc_sat.odd_layout import field_box, layout_failures
    from fc_sat.odd_render import OddRenderer

    checks: list[Check] = []
    math_bad = math_layout_failures(cfg)
    _add(checks, "layout math", not math_bad, "inside" if not math_bad else "; ".join(math_bad[:6]))
    renderer = OddRenderer(cfg, show, preview=False)
    field = field_box(cfg)
    safe = (cfg.safe_x[0], cfg.safe_y[0], cfg.safe_x[1], cfg.safe_y[1])
    indexes = {0, max(0, renderer.n_frames - 1)}
    for segment in show.timeline:
        indexes.add(segment.start_frame)
        indexes.add(min(renderer.n_frames - 1, segment.start_frame + max(0, segment.n_frames // 2)))
    rendered: list[str] = []
    too_many = False
    for index in sorted(indexes):
        renderer.render(min(index, renderer.n_frames - 1))
        if len(renderer.boxes) > 4:
            too_many = True
            rendered.append(f"frame {index} has {len(renderer.boxes)} texts")
        rendered.extend(layout_failures(renderer.boxes, field, safe))
    _add(checks, "text count", not too_many, "at most 4" if not too_many else "; ".join(rendered[:4]))
    _add(checks, "layout frames", not rendered, f"{len(indexes)} frames inside" if not rendered else "; ".join(rendered[:6]))
    return checks


def sidecar_check(path: Path, show) -> Check:
    npz_path = path.with_suffix(".sim.npz")
    if not npz_path.is_file():
        return Check("sim sidecar", False, f"missing {npz_path.name}")
    data = np.load(npz_path)
    bad = []
    for level in show.cfg.levels:
        key = str(level.id)
        if f"odd_{key}" not in data:
            bad.append(f"L{level.id} missing")
            continue
        odd = int(data[f"odd_{key}"][0])
        if odd != int(show.levels[level.id].odd_index):
            bad.append(f"L{level.id} file {odd} sim {show.levels[level.id].odd_index}")
    return Check("sim sidecar", not bad, "matches this seed" if not bad else "; ".join(bad))


def container_checks(path: Path, cfg: OddConfig) -> tuple[list[Check], dict | None, dict | None]:
    checks: list[Check] = []
    ffmpeg = find_ffmpeg()
    info = _probe(find_ffprobe(ffmpeg), path)
    video = _stream(info, "video")
    audio = _stream(info, "audio")
    if video is None:
        _add(checks, "video stream", False, "missing")
        return checks, None, None
    width = int(video.get("width", 0))
    height = int(video.get("height", 0))
    _add(checks, "resolution", width == 1080 and height == 1920, f"{width}x{height}")
    _add(checks, "fps", str(video.get("r_frame_rate")) == "60/1", str(video.get("r_frame_rate")))
    _add(checks, "h264", video.get("codec_name") == "h264", str(video.get("codec_name")))
    profile = str(video.get("profile", ""))
    _add(checks, "profile high", profile.lower().startswith("high"), profile or "missing")
    _add(checks, "yuv420p", video.get("pix_fmt") == "yuv420p", str(video.get("pix_fmt")))
    _add(
        checks,
        "bt709 tags",
        video.get("color_space") == "bt709"
        and video.get("color_transfer") == "bt709"
        and video.get("color_primaries") == "bt709"
        and video.get("color_range") == "tv",
        f"space={video.get('color_space')} trc={video.get('color_transfer')} "
        f"primaries={video.get('color_primaries')} range={video.get('color_range')}",
    )
    if audio is None:
        _add(checks, "audio stream", False, "missing")
    else:
        _add(checks, "aac", audio.get("codec_name") == "aac", str(audio.get("codec_name")))
        _add(
            checks,
            "audio format",
            str(audio.get("sample_rate")) == "48000" and int(audio.get("channels", 0)) == 2,
            f"{audio.get('sample_rate')} Hz {audio.get('channels')} ch",
        )
    expected = timeline_frames(cfg) / float(cfg.fps)
    vdur = _duration(video, info)
    adur = _duration(audio, info) if audio else None
    if vdur is None:
        _add(checks, "duration", False, "video duration missing")
    else:
        _add(checks, "duration window", 20.0 <= vdur <= 30.0, f"{vdur:.3f}s")
        _add(checks, "duration vs timeline", abs(vdur - expected) <= 0.020, f"{vdur:.4f}s vs {expected:.4f}s")
        if adur is not None:
            _add(checks, "av sync", abs(vdur - adur) <= 0.020, f"video {vdur:.4f}s audio {adur:.4f}s")
        else:
            _add(checks, "av sync", False, "audio duration missing")
    size_mb = path.stat().st_size / (1024 * 1024)
    _add(checks, "file size", size_mb < 100.0, f"{size_mb:.2f} MB")
    return checks, video, audio


def _play_frame(cfg: OddConfig, level_id: int, fraction: float) -> int:
    for segment in build_timeline(cfg):
        if segment.kind == "play" and segment.level_id == level_id:
            return segment.start_frame + int(round((segment.n_frames - 1) * fraction))
    raise RuntimeError(f"no play segment for level {level_id}")


def verify_odd_file(path: Path, cfg: OddConfig, show=None) -> list[Check]:
    """Container, loudness, layout, sim constraints, and pixel detectability.

    The harsh re-encode is ``{stem}.harsh.mp4``. The hard tier is longer than 30 s,
    so the duration window fails on purpose. The upload file is the three-level cut.
    """
    from fc_sat.odd_render import OddRenderer, _settled_time
    from fc_sat.odd_sim import simulate_show

    path = Path(path)
    print(f"verify {path.name}", flush=True)
    checks, video, audio = container_checks(path, cfg)
    if show is None:
        print("simulating for verify", flush=True)
        show = simulate_show(cfg)
    checks.append(sidecar_check(path, show))
    checks.extend(sequence_checks(cfg))
    checks.extend(sim_checks(cfg, show))
    print("layout frames", flush=True)
    checks.extend(layout_frame_checks(cfg, show))
    if audio is not None:
        samples = _decode_audio(find_ffmpeg(), path)
        meter = pyln.Meter(48000)
        mono = samples.mean(axis=1) if samples.ndim == 2 else samples
        lufs = float(meter.integrated_loudness(samples))
        peak = true_peak_db(samples)
        sample_peak = float(np.max(np.abs(samples))) if samples.size else 1.0
        _add(checks, "lufs", abs(lufs + 14.0) <= 1.0, f"{lufs:.2f} LUFS")
        _add(checks, "true peak", peak <= -1.0 + 1e-3, f"{peak:.2f} dBTP")
        _add(checks, "no clipping", sample_peak < 1.0, f"sample peak {sample_peak:.4f}")
        expected_n = timeline_frames(cfg) * 800
        delta_s = abs(len(mono) - expected_n) / 48000.0
        _add(checks, "audio length", delta_s <= 0.020, f"{len(mono)} samples vs {expected_n} ({delta_s * 1000:.1f} ms)")
    width = int(video.get("width", 0)) if video else 0
    height = int(video.get("height", 0)) if video else 0
    if width == cfg.width and height == cfg.height:
        ffmpeg = find_ffmpeg()
        wanted = {}
        for level in cfg.levels:
            wanted[_play_frame(cfg, level.id, 0.5)] = level.id
        wanted[0] = wanted.get(0, 0)
        hook_max = 0.0
        lumas = []
        kept = {}
        cap_y = int(round(cfg.caption_y))
        cap_px = int(cfg.caption_px)
        cap_x0 = int(cfg.caption_x[0])
        cap_x1 = int(cfg.caption_x[1])
        for index, frame in enumerate(_iter_decoded(ffmpeg, path, width, height)):
            lumas.append(_luma_mean(frame))
            if index == 0:
                band = frame[max(0, cap_y - cap_px // 2) : cap_y + cap_px // 2, cap_x0:cap_x1]
                if band.size:
                    luma = 0.2126 * band[..., 2] + 0.7152 * band[..., 1] + 0.0722 * band[..., 0]
                    hook_max = float(luma.max() / 255.0)
            if index in wanted and wanted[index] != 0:
                kept[wanted[index]] = frame.copy()
        _add(checks, "hook on frame 0", hook_max >= 0.45, f"caption-band max luma {hook_max:.3f}")
        hot = photosensitivity_hot_count(np.asarray(lumas, dtype=np.float64), cfg.fps)
        _add(checks, "photosensitivity", hot <= 3, f"max jumps >0.10 in 1s: {hot}")
        renderer = OddRenderer(cfg, show, preview=False)
        glow = _glow_ok(renderer.render_time(_settled_time(cfg, cfg.levels[0].id)), cfg, show.levels[cfg.levels[0].id])
        _add(checks, "no glow", glow, "pixel outside the disc matches the background" if glow else "halo outside an item")
        for level in cfg.levels:
            frame = kept.get(level.id)
            if frame is None:
                _add(checks, f"L{level.id} {level.difference}", False, "play frame was not decoded")
                continue
            ok, detail = score_frame(frame, cfg, show.levels[level.id], 1.0)
            _add(checks, f"L{level.id} {level.difference}", ok, detail)
            if not ok:
                dest = path.with_name(f"{path.stem}.L{level.id}.worst.png")
                cv2.imwrite(str(dest), frame)
                print(f"saved worst frame {dest}", flush=True)
        harsh = path.with_name(path.stem + ".harsh.mp4")
        print(f"harsh re-encode -> {harsh.name}", flush=True)
        harsh_reencode(path, harsh)
        kept = {}
        for index, frame in enumerate(_iter_decoded(ffmpeg, harsh, HARSH_W, HARSH_H)):
            if index in wanted and wanted[index] != 0:
                kept[wanted[index]] = frame.copy()
        scale = HARSH_W / cfg.width
        for level in cfg.levels:
            frame = kept.get(level.id)
            if frame is None:
                _add(checks, f"harsh L{level.id}", False, "frame missing")
                continue
            ok, detail = score_frame(frame, cfg, show.levels[level.id], scale)
            _add(checks, f"harsh L{level.id} {level.difference}", ok, detail)
            if not ok:
                dest = path.with_name(f"{path.stem}.L{level.id}.harsh.worst.png")
                cv2.imwrite(str(dest), frame)
                print(f"saved worst frame {dest}", flush=True)
    else:
        _add(checks, "detectability", False, f"need {cfg.width}x{cfg.height}, got {width}x{height}")
    return checks


def _glow_ok(frame: np.ndarray, cfg: OddConfig, sim) -> bool:
    bg = np.array(_bg(cfg), dtype=np.int16)
    cx, cy = sim.centers[0]
    x = int(round(cx + sim.level.size / 2.0 + 6))
    y = int(round(cy))
    if not (0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]):
        return False
    return int(np.max(np.abs(frame[y, x].astype(np.int16) - bg))) <= 3
