"""Post-render checks against the finished mp4.

Run: python -m fc_sat.verify out.mp4 --config configs/default.yaml

Pixel safe-zone tolerances are fixed: mean absolute RGB error <= 8/255 and
99th-percentile luma delta <= 0.06 against the renderer's vignette at the same
pixels. Do not raise them to force a pass. If a full render fails, report the
measured values and inspect the saved frame first.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import yaml

from fc_sat.audio import true_peak_db
from fc_sat.config import Config, load_config
from fc_sat.encode import find_ffmpeg, find_ffprobe
from fc_sat.hop_audio import true_peak_fft
from fc_sat.hop_config import HopConfig, load_hop_config
from fc_sat.hop_render import HopRenderer, layout_failures
from fc_sat.render import assert_layout_safe, vignette_bgr
from fc_sat.sim import load_or_simulate

SAFE_MEAN_ABS = 8.0 / 255.0
SAFE_P99_LUMA = 0.06
BLACK_LUMA = 0.008


@dataclass
class Check:
    name: str
    ok: bool
    detail: str


def ssim_u8(a: np.ndarray, b: np.ndarray) -> float:
    """Global SSIM on BT.709 luma. Identical frames score 1."""
    if a.shape != b.shape:
        return 0.0

    def luma(image: np.ndarray) -> np.ndarray:
        rgb = image.astype(np.float64)
        return 0.0722 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.2126 * rgb[..., 2]

    x = luma(a)
    y = luma(b)
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    mux, muy = float(x.mean()), float(y.mean())
    varx = float(x.var())
    vary = float(y.var())
    cov = float(((x - mux) * (y - muy)).mean())
    return ((2 * mux * muy + c1) * (2 * cov + c2)) / ((mux * mux + muy * muy + c1) * (varx + vary + c2))


def photosensitivity_hot_count(luma: np.ndarray, fps: int) -> int:
    """Most frame-to-frame jumps above 0.10 inside any 1-second window."""
    if luma.size < 2:
        return 0
    hot = np.abs(np.diff(luma.astype(np.float64))) > 0.10
    window = max(1, int(round(fps)))
    if hot.size <= window:
        return int(hot.sum())
    kernel = np.ones(window, dtype=np.int32)
    rolled = np.convolve(hot.astype(np.int32), kernel, mode="valid")
    return int(rolled.max())


def _probe(ffprobe: str, path: Path) -> dict:
    proc = subprocess.run(
        [ffprobe, "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise SystemExit(f"ffprobe failed:\n{proc.stderr}")
    return json.loads(proc.stdout or "{}")


def _stream(info: dict, codec_type: str) -> dict | None:
    for stream in info.get("streams", []):
        if stream.get("codec_type") == codec_type:
            return stream
    return None


def _duration(stream: dict, info: dict) -> float | None:
    for source in (stream, info.get("format", {})):
        raw = source.get("duration")
        if raw is not None:
            try:
                return float(raw)
            except (TypeError, ValueError):
                continue
    return None


def _extract_frame(ffmpeg: str, path: Path, index: int, width: int, height: int) -> np.ndarray:
    proc = subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-i",
            str(path),
            "-vf",
            f"select=eq(n\\,{index})",
            "-frames:v",
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
    needed = width * height * 3
    if proc.returncode != 0 or len(proc.stdout) < needed:
        raise SystemExit(f"could not extract frame {index}: {proc.stderr.decode('utf-8', 'replace')}")
    return np.frombuffer(proc.stdout[:needed], dtype=np.uint8).reshape(height, width, 3).copy()


def _iter_decoded(ffmpeg: str, path: Path, width: int, height: int):
    proc = subprocess.Popen(
        [ffmpeg, "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.stdout is not None
    needed = width * height * 3
    try:
        while True:
            buf = proc.stdout.read(needed)
            if len(buf) < needed:
                break
            yield np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 3)
    finally:
        if proc.stdout:
            proc.stdout.close()
        proc.wait()


def _luma_mean(frame: np.ndarray) -> float:
    b = frame[..., 0].astype(np.float64)
    g = frame[..., 1].astype(np.float64)
    r = frame[..., 2].astype(np.float64)
    return float((0.2126 * r + 0.7152 * g + 0.0722 * b).mean() / 255.0)


def _bgr_luma(pixels: np.ndarray) -> np.ndarray:
    return (0.2126 * pixels[:, 2] + 0.7152 * pixels[:, 1] + 0.0722 * pixels[:, 0]) / 255.0


def _decode_audio(ffmpeg: str, path: Path) -> np.ndarray:
    proc = subprocess.run(
        [ffmpeg, "-v", "error", "-i", str(path), "-vn", "-ac", "2", "-ar", "48000", "-f", "f32le", "pipe:1"],
        check=False,
        capture_output=True,
    )
    if proc.returncode != 0 or not proc.stdout:
        raise SystemExit(f"could not decode audio: {proc.stderr.decode('utf-8', 'replace')}")
    samples = np.frombuffer(proc.stdout, dtype=np.float32).astype(np.float64)
    if samples.size % 2:
        samples = samples[:-1]
    return samples.reshape(-1, 2)


def _add(checks: list[Check], name: str, ok: bool, detail: str) -> None:
    checks.append(Check(name, bool(ok), detail))


def verify_file(path: Path, cfg: Config, *, use_cache: bool = True) -> list[Check]:
    path = Path(path)
    checks: list[Check] = []
    ffmpeg = find_ffmpeg()
    ffprobe = find_ffprobe(ffmpeg)
    info = _probe(ffprobe, path)
    video = _stream(info, "video")
    audio = _stream(info, "audio")
    if video is None:
        _add(checks, "video stream", False, "missing")
        return checks
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
    expected = cfg.n_frames(preview=False) / cfg.fps
    vdur = _duration(video, info)
    adur = _duration(audio, info) if audio else None
    if vdur is None:
        _add(checks, "duration", False, "video duration missing")
    else:
        _add(checks, "duration vs config", abs(vdur - expected) <= 0.020, f"{vdur:.4f}s vs {expected:.4f}s")
        if adur is not None:
            _add(checks, "av sync", abs(vdur - adur) <= 0.020, f"video {vdur:.4f}s audio {adur:.4f}s")
        else:
            _add(checks, "av sync", False, "audio duration missing")
    size_mb = path.stat().st_size / (1024 * 1024)
    _add(checks, "file size", size_mb < 100.0, f"{size_mb:.2f} MB")

    luma = np.empty(0, dtype=np.float64)
    worst_mean = 0.0
    worst_p99 = -1.0
    worst_index = 0
    worst_frame = None
    first = None
    last = None
    if width > 0 and height > 0:
        vignette = vignette_bgr(width, height) if (width, height) == (1080, 1920) else None
        mask = None
        vig_pix = None
        vig_luma = None
        if vignette is not None:
            x_cut = int(math.ceil(width * 0.88))
            y_cut = int(math.ceil(height * 0.80))
            mask = np.zeros((height, width), dtype=bool)
            mask[:, x_cut:] = True
            mask[y_cut:, :] = True
            vig_pix = vignette[mask].astype(np.float32)
            vig_luma = _bgr_luma(vig_pix)
        means = []
        for index, frame in enumerate(_iter_decoded(ffmpeg, path, width, height)):
            if first is None:
                first = frame.copy()
            last = frame
            means.append(_luma_mean(frame))
            if mask is not None and vig_pix is not None and vig_luma is not None:
                pix = frame[mask].astype(np.float32)
                mean_abs = float(np.mean(np.abs(pix - vig_pix)) / 255.0)
                p99 = float(np.quantile(_bgr_luma(pix) - vig_luma, 0.99))
                if mean_abs >= worst_mean or p99 >= worst_p99:
                    worst_frame = frame.copy()
                    worst_index = index
                worst_mean = max(worst_mean, mean_abs)
                worst_p99 = max(worst_p99, p99)
        luma = np.array(means, dtype=np.float64)
        if last is not None:
            last = last.copy()
    if first is not None and last is not None:
        score = ssim_u8(first, last)
        _add(checks, "loop ssim", score >= 0.995, f"{score:.6f}")
    else:
        _add(checks, "loop ssim", False, "could not decode frames")
    if luma.size == 0:
        _add(checks, "black frames", False, "no decoded frames")
        _add(checks, "photosensitivity", False, "no luma")
        _add(checks, "pixel safe zone", False, "frame size is not 1080x1920")
    else:
        darkest = float(luma.min())
        _add(checks, "black frames", darkest > BLACK_LUMA, f"min mean luma {darkest:.4f}")
        hot = photosensitivity_hot_count(luma, 60)
        _add(checks, "photosensitivity", hot <= 3, f"max jumps >0.10 in 1s: {hot}")
        if width != 1080 or height != 1920:
            _add(checks, "pixel safe zone", False, f"frame size {width}x{height} is not 1080x1920")
            pixel_ok = False
        else:
            pixel_ok = worst_mean <= SAFE_MEAN_ABS + 1e-9 and worst_p99 <= SAFE_P99_LUMA + 1e-9
            detail = (
                f"mean abs RGB {worst_mean * 255:.2f}/255 (limit 8), "
                f"p99 luma delta {worst_p99:.4f} (limit 0.06), worst frame {worst_index}"
            )
            _add(checks, "pixel safe zone", pixel_ok, detail)
        if not pixel_ok and worst_frame is not None:
            dump = path.with_name(path.stem + ".safezone.png")
            import cv2

            cv2.imwrite(str(dump), worst_frame)
            print(f"saved safe-zone failure frame to {dump}", file=sys.stderr)

    if audio is not None:
        samples = _decode_audio(ffmpeg, path)
        meter = pyln.Meter(48000)
        lufs = float(meter.integrated_loudness(samples))
        peak = true_peak_db(samples)
        sample_peak = float(np.max(np.abs(samples))) if samples.size else 1.0
        _add(checks, "lufs", abs(lufs + 14.0) <= 1.0, f"{lufs:.2f} LUFS")
        _add(checks, "true peak", peak <= -1.0 + 1e-3, f"{peak:.2f} dBTP")
        _add(checks, "no clipping", sample_peak < 1.0, f"sample peak {sample_peak:.4f}")
        tail_n = int(round(0.300 * 48000))
        tail = samples[-tail_n:] if samples.shape[0] >= tail_n else samples
        rms = float(np.sqrt(np.mean(tail ** 2))) if tail.size else 0.0
        tail_db = 20.0 * math.log10(max(rms, 1e-12))
        _add(checks, "tail silence", tail_db < -50.0, f"{tail_db:.1f} dBFS")

    sim = load_or_simulate(cfg, use_cache=use_cache)
    finite = bool(np.isfinite(sim.positions).all()) and bool(np.isfinite(sim.hit_times).all())
    _add(checks, "finite state", finite, "positions and hit times are finite" if finite else "NaN or inf in state")
    first = sim.first_bounce
    _add(checks, "first bounce", 0.30 <= first <= 0.60, f"{first:.4f}s")
    _add(checks, "count at 0.14 Tg", 6 <= sim.count_at_014 <= 10, str(sim.count_at_014))
    within = abs(sim.final_count - cfg.cap) <= 0.03 * cfg.cap
    _add(checks, "final count", within, f"{sim.final_count} vs cap {cfg.cap}")
    monotonic = bool(np.all(np.diff(sim.counts.astype(np.int64)) >= 0))
    _add(checks, "counter monotonic", monotonic, "growth counts are non-decreasing" if monotonic else "count decreased")
    layout = assert_layout_safe(cfg)
    _add(checks, "layout safe zone", not layout, "inside" if not layout else "; ".join(layout))
    return checks


def onset_score(samples: np.ndarray, sr: int = 48000) -> tuple[np.ndarray, float]:
    """Positive first difference of a smoothed dB envelope. Hop is 2 ms, smooth about 10 ms."""
    mono = np.mean(np.asarray(samples, dtype=np.float64) ** 2, axis=1)
    hop = max(1, int(round(0.002 * sr)))
    n = len(mono) // hop
    if n < 4:
        return np.zeros(1, dtype=np.float64), hop / sr
    rms = np.sqrt(mono[: n * hop].reshape(n, hop).mean(axis=1) + 1e-12)
    db = 20.0 * np.log10(rms)
    win = max(1, int(round(0.010 / (hop / sr))))
    kernel = np.ones(win, dtype=np.float64) / win
    smooth = np.convolve(db, kernel, mode="same")
    diff = np.empty_like(smooth)
    diff[0] = smooth[0] - smooth[-1]
    diff[1:] = np.diff(smooth)
    return np.maximum(diff, 0.0), hop / sr


def circular_peak(score: np.ndarray, hop_seconds: float, center: float, half: float) -> float:
    """Time of the max score within ``half`` seconds of ``center``, wrapping the series."""
    n = len(score)
    if n == 0 or hop_seconds <= 0:
        return center
    center_i = center / hop_seconds
    half_i = half / hop_seconds
    i0 = int(math.floor(center_i - half_i))
    i1 = int(math.ceil(center_i + half_i))
    best = -1.0
    found = center
    for index in range(i0, i1 + 1):
        unwrapped = index * hop_seconds
        if abs(unwrapped - center) > half + 1e-9:
            continue
        value = float(score[index % n])
        if value > best:
            best = value
            found = unwrapped
    return found


def verify_hop_file(path: Path, cfg: HopConfig) -> list[Check]:
    """Container, loudness, loop seams, note sync, and the vignette safe zone."""
    path = Path(path)
    checks: list[Check] = []
    renderer = HopRenderer(cfg, preview=False)
    grid = renderer.dance.grid
    expected = grid.duration * cfg.repeats
    ffmpeg = find_ffmpeg()
    ffprobe = find_ffprobe(ffmpeg)
    info = _probe(ffprobe, path)
    video = _stream(info, "video")
    audio = _stream(info, "audio")
    if video is None:
        _add(checks, "video stream", False, "missing")
        return checks
    width = int(video.get("width", 0))
    height = int(video.get("height", 0))
    _add(checks, "resolution", width == 1080 and height == 1920, f"{width}x{height}")
    _add(checks, "fps", str(video.get("r_frame_rate")) == "60/1", str(video.get("r_frame_rate")))
    _add(checks, "h264", video.get("codec_name") == "h264", str(video.get("codec_name")))
    _add(checks, "yuv420p", video.get("pix_fmt") == "yuv420p", str(video.get("pix_fmt")))
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
    vdur = _duration(video, info)
    adur = _duration(audio, info) if audio else None
    if vdur is None:
        _add(checks, "duration", False, "video duration missing")
    else:
        _add(checks, "duration", abs(vdur - expected) <= 0.020, f"{vdur:.4f}s vs {expected:.4f}s")
        if adur is not None:
            _add(checks, "av sync", abs(vdur - adur) <= 0.020, f"video {vdur:.4f}s audio {adur:.4f}s")
        else:
            _add(checks, "av sync", False, "audio duration missing")
    size_mb = path.stat().st_size / (1024 * 1024)
    _add(checks, "file size", size_mb < 100.0, f"{size_mb:.2f} MB")

    pad_boxes = []
    for pad in renderer.dance.layout.pads:
        pad_boxes.append(
            (
                max(0, int(math.floor(pad.left))),
                max(0, int(math.floor(pad.top))),
                min(width, int(math.ceil(pad.right))),
                min(height, int(math.ceil(pad.top + pad.thickness))),
            )
        )
    luma_means: list[float] = []
    pad_luma: list[np.ndarray] = []
    consecutive: list[float] = []
    first = None
    last = None
    prev = None
    worst_mean = 0.0
    worst_p99 = -1.0
    worst_index = 0
    worst_frame = None
    vignette = vignette_bgr(width, height) if (width, height) == (1080, 1920) else None
    mask = None
    vig_pix = None
    vig_luma = None
    if vignette is not None:
        x_cut = int(math.ceil(width * 0.88))
        y_cut = int(math.ceil(height * 0.80))
        mask = np.zeros((height, width), dtype=bool)
        mask[:, x_cut:] = True
        mask[y_cut:, :] = True
        vig_pix = vignette[mask].astype(np.float32)
        vig_luma = _bgr_luma(vig_pix)
    if width > 0 and height > 0:
        for index, frame in enumerate(_iter_decoded(ffmpeg, path, width, height)):
            if first is None:
                first = frame.copy()
            last = frame
            luma_means.append(_luma_mean(frame))
            row = np.empty(len(pad_boxes), dtype=np.float64)
            plane = 0.0722 * frame[..., 0] + 0.7152 * frame[..., 1] + 0.2126 * frame[..., 2]
            for pad_index, (x0, y0, x1, y1) in enumerate(pad_boxes):
                if x1 <= x0 or y1 <= y0:
                    row[pad_index] = 0.0
                else:
                    row[pad_index] = float(plane[y0:y1, x0:x1].mean())
            pad_luma.append(row)
            if prev is not None:
                consecutive.append(float(np.mean(np.abs(frame.astype(np.float32) - prev))))
            prev = frame
            if mask is not None and vig_pix is not None and vig_luma is not None:
                pix = frame[mask].astype(np.float32)
                mean_abs = float(np.mean(np.abs(pix - vig_pix)) / 255.0)
                p99 = float(np.quantile(_bgr_luma(pix) - vig_luma, 0.99))
                if mean_abs >= worst_mean or p99 >= worst_p99:
                    worst_frame = frame.copy()
                    worst_index = index
                worst_mean = max(worst_mean, mean_abs)
                worst_p99 = max(worst_p99, p99)
        if last is not None:
            last = last.copy()
    if first is not None and last is not None and consecutive:
        seam = float(np.mean(np.abs(last.astype(np.float32) - first.astype(np.float32))))
        limit = float(np.percentile(np.asarray(consecutive), 99))
        _add(checks, "video seam", seam <= limit + 1e-6, f"mad {seam:.3f} vs p99 {limit:.3f}")
    else:
        _add(checks, "video seam", False, "could not decode frames")
    luma = np.array(luma_means, dtype=np.float64)
    if luma.size == 0:
        _add(checks, "black frames", False, "no decoded frames")
        _add(checks, "photosensitivity", False, "no luma")
        _add(checks, "pixel safe zone", False, "frame size is not 1080x1920")
        pixel_ok = False
    else:
        darkest = float(luma.min())
        _add(checks, "black frames", darkest > BLACK_LUMA, f"min mean luma {darkest:.4f}")
        hot = photosensitivity_hot_count(luma, 60)
        _add(checks, "photosensitivity", hot <= 3, f"max jumps >0.10 in 1s: {hot}")
        if width != 1080 or height != 1920:
            _add(checks, "pixel safe zone", False, f"frame size {width}x{height} is not 1080x1920")
            pixel_ok = False
        else:
            pixel_ok = worst_mean <= SAFE_MEAN_ABS + 1e-9 and worst_p99 <= SAFE_P99_LUMA + 1e-9
            detail = (
                f"mean abs RGB {worst_mean * 255:.2f}/255 (limit 8), "
                f"p99 luma delta {worst_p99:.4f} (limit 0.06), worst frame {worst_index}"
            )
            _add(checks, "pixel safe zone", pixel_ok, detail)
    if not pixel_ok and worst_frame is not None:
        dump = path.with_name(path.stem + ".safezone.png")
        import cv2

        cv2.imwrite(str(dump), worst_frame)
        print(f"saved safe-zone failure frame to {dump}", file=sys.stderr)
        print(
            f"pixel safe zone measured mean abs {worst_mean * 255:.2f}/255, p99 luma delta {worst_p99:.4f}",
            file=sys.stderr,
        )

    layout = layout_failures(cfg, renderer.dance)
    _add(checks, "layout safe zone", not layout, "inside" if not layout else "; ".join(layout[:4]))

    if audio is None:
        _add(checks, "lufs", False, "no audio")
        _add(checks, "sync", False, "no audio")
        return checks
    samples = _decode_audio(ffmpeg, path)
    meter = pyln.Meter(48000)
    lufs = float(meter.integrated_loudness(samples))
    peak_lin = true_peak_fft(samples)
    peak = -120.0 if peak_lin <= 1e-12 else 20.0 * math.log10(peak_lin)
    sample_peak = float(np.max(np.abs(samples))) if samples.size else 1.0
    _add(checks, "lufs", abs(lufs + 14.0) <= 1.0, f"{lufs:.2f} LUFS")
    _add(checks, "true peak", peak <= -1.0 + 1e-3, f"{peak:.2f} dBTP")
    _add(checks, "no clipping", sample_peak < 1.0, f"sample peak {sample_peak:.4f}")
    seam_ok = True
    seam_bits = []
    for channel in range(samples.shape[1]):
        column = samples[:, channel]
        if column.size < 3:
            seam_ok = False
            seam_bits.append(f"ch{channel} empty")
            continue
        seam = abs(float(column[-1] - column[0]))
        limit = 3.0 * float(np.percentile(np.abs(np.diff(column)), 99))
        seam_bits.append(f"ch{channel} {seam:.5f} vs {limit:.5f}")
        if seam > limit + 1e-9:
            seam_ok = False
    _add(checks, "audio seam", seam_ok, ", ".join(seam_bits))
    head = samples[: int(round(0.030 * 48000))]
    rms = float(np.sqrt(np.mean(head ** 2))) if head.size else 0.0
    head_db = 20.0 * math.log10(max(rms, 1e-12))
    _add(checks, "first 30 ms", head_db > -40.0, f"{head_db:.1f} dBFS")

    score, hop_seconds = onset_score(samples, 48000)
    pad_series = np.vstack(pad_luma) if pad_luma else np.zeros((1, len(pad_boxes)))
    targets = []
    for cycle in range(cfg.repeats):
        for onset in grid.onset_times:
            targets.append(onset + cycle * grid.duration)
    offsets = []
    lines = []
    for index, target in enumerate(targets):
        audio_t = circular_peak(score, hop_seconds, target, 0.040)
        pad_index = renderer.dance.pad_index(index % len(grid.onset_times))
        column = pad_series[:, pad_index]
        video_diff = np.empty(len(column), dtype=np.float64)
        if len(column) == 0:
            video_t = target
        else:
            video_diff[0] = column[0] - column[-1]
            video_diff[1:] = np.diff(column)
            video_diff = np.maximum(video_diff, 0.0)
            video_t = circular_peak(video_diff, 1.0 / 60.0, target, 0.040)
        offset = audio_t - video_t
        offsets.append(offset)
        lines.append(f"note {index}: {offset * 1000:.1f} ms (audio {audio_t:.3f}s video {video_t:.3f}s)")
    offset_arr = np.array(offsets, dtype=np.float64)
    median = float(np.median(offset_arr)) if offset_arr.size else 999.0
    p90 = float(np.percentile(np.abs(offset_arr), 90)) if offset_arr.size else 999.0
    sync_ok = abs(median) <= 0.020 and p90 <= 0.034
    _add(checks, "sync", sync_ok, f"median {median * 1000:.1f} ms, p90 |offset| {p90 * 1000:.1f} ms")
    if not sync_ok:
        print("sync offsets failed; fix the detectors before changing thresholds", file=sys.stderr)
        for line in lines:
            print(line, file=sys.stderr)
    return checks


def format_table(checks: list[Check]) -> str:
    lines = [f"{'PASS' if item.ok else 'FAIL'}  {item.name:<22} {item.detail}" for item in checks]
    failed = sum(1 for item in checks if not item.ok)
    lines.append(f"{len(checks) - failed} passed, {failed} failed")
    return "\n".join(lines)


def _config_has_song(path: Path) -> bool:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return isinstance(data, dict) and bool(data.get("song"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify a rendered Short")
    parser.add_argument("mp4")
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=("bounce", "hop"), default=None)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args(argv)
    config_path = Path(args.config)
    mode = args.mode
    if mode is None:
        mode = "hop" if _config_has_song(config_path) else "bounce"
    if mode == "hop":
        checks = verify_hop_file(Path(args.mp4), load_hop_config(config_path))
    else:
        cfg = load_config(config_path)
        checks = verify_file(Path(args.mp4), cfg, use_cache=not args.no_cache)
    print(format_table(checks))
    return 0 if all(item.ok for item in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
