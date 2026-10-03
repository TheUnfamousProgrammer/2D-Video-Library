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


def detect_mode(path: Path) -> str:
    """Arena if generator is arena, odd if generator is odd, hop if the file names a song, morph if it sets recolor_strength, else bounce."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and data.get("generator") == "arena":
        return "arena"
    if isinstance(data, dict) and data.get("generator") == "odd":
        return "odd"
    if isinstance(data, dict) and data.get("song"):
        return "hop"
    if isinstance(data, dict) and "recolor_strength" in data:
        return "morph"
    return "bounce"


def cell_oklab_error(
    frame_bgr: np.ndarray,
    grid_rgb: np.ndarray,
    origin: tuple[int, int],
    cell: int,
    cols: int,
    rows: int,
) -> float:
    """Mean OKLab distance between cell averages of a decoded box and the prepared grid."""
    from fc_sat.color import rgb_u8_to_oklab

    x0, y0 = origin
    box = frame_bgr[y0 : y0 + rows * cell, x0 : x0 + cols * cell]
    averaged = box.reshape(rows, cell, cols, cell, 3).astype(np.float64).mean(axis=(1, 3))
    lab = rgb_u8_to_oklab(np.clip(averaged[..., ::-1], 0, 255))
    target = rgb_u8_to_oklab(grid_rgb)
    return float(np.mean(np.linalg.norm(lab - target, axis=-1)))


def verify_morph_file(path: Path, cfg, *, image_a, image_b) -> list[Check]:
    """Container, loudness, loop, fidelity, and safe-zone checks for Pixel Morph.

    Fidelity is a pass/fail only when recolor_strength is at least 0.99.
    Below that the measured OKLab error is reported and does not fail the file.
    True peak uses the bounce resample_poly meter.
    """
    from fc_sat.morph_images import load_rgb, prepare_grid
    from fc_sat.morph_motion import phase_frames, total_frames
    from fc_sat.morph_render import MorphRenderer
    from fc_sat.morph_render import layout_failures as morph_layout_failures

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
    plan = phase_frames(cfg.timeline(), 60)
    n_frames = total_frames(plan)
    expected = n_frames / 60.0
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

    cursor = 0
    hold_index = 0
    for name, count in plan:
        if name == "hold_b":
            hold_index = cursor + (count - 1) // 2
            break
        cursor += count
    grid_a = prepare_grid(load_rgb(image_a), cfg.cols, cfg.rows, cfg.focus_a)
    grid_b = prepare_grid(load_rgb(image_b), cfg.cols, cfg.rows, cfg.focus_b)
    renderer = MorphRenderer(cfg, image_a, image_b, preview=False)
    layout = morph_layout_failures(cfg)
    _add(checks, "layout safe zone", not layout, "inside" if not layout else "; ".join(layout))

    spilled = 0
    sample_at = {0, n_frames - 1}
    for phase in ("morph_ab", "morph_ba"):
        for tau in (0.25, 0.5, 0.75):
            sample_at.add(renderer.frame_index(phase, tau))
    for index in sorted(sample_at):
        layer = renderer.particle_layer(index)
        mask = np.ones(layer.shape[:2], dtype=bool)
        mask[renderer.y0 : renderer.y0 + renderer.box_h, renderer.x0 : renderer.x0 + renderer.box_w] = False
        spilled = max(spilled, int(layer[mask].max()) if np.any(mask) else 0)
    _add(checks, "particles inside box", spilled == 0, f"max outside pixel {spilled}")

    luma = np.empty(0, dtype=np.float64)
    worst_mean = 0.0
    worst_p99 = -1.0
    worst_index = 0
    worst_frame = None
    first = None
    last = None
    hold_frame = None
    if width == 1080 and height == 1920:
        vignette = vignette_bgr(width, height)
        hx0, hy0, hx1, hy1 = renderer.hook_box
        compare = np.ones((height, width), dtype=bool)
        compare[cfg.origin_y : cfg.origin_y + cfg.rows * cfg.cell, cfg.origin_x : cfg.origin_x + cfg.cols * cfg.cell] = False
        compare[max(0, hy0) : min(height, hy1), max(0, hx0) : min(width, hx1)] = False
        vig_pix = vignette[compare].astype(np.float32)
        vig_luma = _bgr_luma(vig_pix)
        means = []
        for index, frame in enumerate(_iter_decoded(ffmpeg, path, width, height)):
            if first is None:
                first = frame.copy()
            if index == hold_index:
                hold_frame = frame.copy()
            last = frame
            means.append(_luma_mean(frame))
            pix = frame[compare].astype(np.float32)
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
        pixel_ok = False
    else:
        darkest = float(luma.min())
        _add(checks, "black frames", darkest > BLACK_LUMA, f"min mean luma {darkest:.4f}")
        hot = photosensitivity_hot_count(luma, 60)
        _add(checks, "photosensitivity", hot <= 3, f"max jumps >0.10 in 1s: {hot}")
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
        print(detail if luma.size else "pixel safe zone failed", file=sys.stderr)

    enforce = float(cfg.recolor_strength) >= 0.99
    if first is not None:
        err_a = cell_oklab_error(first, grid_a, (cfg.origin_x, cfg.origin_y), cfg.cell, cfg.cols, cfg.rows)
    else:
        err_a = 1.0
    if hold_frame is not None:
        err_b = cell_oklab_error(hold_frame, grid_b, (cfg.origin_x, cfg.origin_y), cfg.cell, cfg.cols, cfg.rows)
    else:
        err_b = 1.0
    limit = float(cfg.max_delta_e)
    fidelity_ok = (err_a <= limit and err_b <= limit) if enforce else True
    fidelity_detail = (
        f"A {err_a:.4f}, B {err_b:.4f}, limit {limit:.3f}"
        + ("" if enforce else " (report only, recolor_strength < 0.99)")
    )
    _add(checks, "fidelity", fidelity_ok, fidelity_detail)

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
    return checks


def _gray_series(ffmpeg: str, path: Path) -> tuple[np.ndarray, np.ndarray]:
    width, height = 96, 170
    proc = subprocess.Popen(
        [ffmpeg, "-v", "error", "-i", str(path), "-vf", f"scale={width}:{height}", "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.stdout is not None
    needed = width * height
    means: list[float] = []
    diffs: list[float] = []
    prev = None
    try:
        while True:
            buf = proc.stdout.read(needed)
            if len(buf) < needed:
                break
            frame = np.frombuffer(buf, dtype=np.uint8).astype(np.float64) / 255.0
            means.append(float(frame.mean()))
            if prev is not None:
                diffs.append(float(np.mean(np.abs(frame - prev))))
            prev = frame
    finally:
        if proc.stdout:
            proc.stdout.close()
        proc.wait()
    return np.asarray(means, dtype=np.float64), np.asarray(diffs, dtype=np.float64)


def verify_arena_file(path: Path, cfg, sidecar: dict | None = None) -> list[Check]:
    """Container, loudness, pacing, layout, and photosensitivity checks for Flag Arena."""
    import cv2

    from fc_sat.arena_render import PLATFORM_BOX, camera_view, layout_boxes, layout_clear

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
    vdur = _duration(video, info)
    adur = _duration(audio, info) if audio else None
    if vdur is None:
        _add(checks, "duration", False, "video duration missing")
    else:
        _add(checks, "duration", 33.0 <= vdur <= 43.0, f"{vdur:.3f}s")
        if adur is not None:
            _add(checks, "av sync", abs(vdur - adur) <= 0.020, f"video {vdur:.4f}s audio {adur:.4f}s")
    size_mb = path.stat().st_size / (1024 * 1024)
    _add(checks, "file size", size_mb < 100.0, f"{size_mb:.2f} MB")
    meta = sidecar if sidecar is not None else {}
    side_path = path.with_suffix(".json")
    if not meta and side_path.is_file():
        meta = json.loads(side_path.read_text(encoding="utf-8"))
    gates = meta.get("gates") or {}
    gates_ok = bool(gates) and all(bool(value) for value in gates.values())
    _add(checks, "pacing gates", gates_ok, ", ".join(f"{name}={value}" for name, value in gates.items()) or "missing sidecar")
    first_elim = meta.get("first_elim")
    _add(checks, "first elimination", first_elim is not None and float(first_elim) >= 1.5, f"{first_elim}s")
    duration = float(meta.get("duration") or vdur or 0.0)
    winner_video = meta.get("winner_video")
    celebration = meta.get("celebration_video")
    winner_ok = winner_video is not None and duration > 0 and float(winner_video) >= 0.70 * duration
    _add(checks, "winner late", winner_ok, f"winner {winner_video}s of {duration:.2f}s")
    cele_ok = celebration is not None and duration > 0 and float(celebration) >= 0.75 * duration
    _add(checks, "celebration late", cele_ok, f"celebration {celebration}s")
    boxes = meta.get("text_boxes") or []
    box_fail = [
        box
        for box in boxes
        if box[0] < cfg.safe_x[0] or box[2] > cfg.safe_x[1] or box[1] < cfg.safe_y[0] or box[3] > cfg.safe_y[1]
    ]
    _add(checks, "text safe zone", bool(boxes) and not box_fail, f"{len(boxes)} boxes, {len(box_fail)} outside")
    zone_ok = True
    zone_detail = "inside the platform slot"
    for moment, hw, hh in ((2.0, frame[1], frame[2]) for frame in cfg.platform_keyframes[:2]):
        view = camera_view(cfg, moment, hw, hh)
        if (
            view.left < PLATFORM_BOX[0] - 1
            or view.right > PLATFORM_BOX[1] + 1
            or view.top < PLATFORM_BOX[2] - 1
            or view.bottom > PLATFORM_BOX[3] + 1
            or hw * view.zoom > cfg.screen_hw + 1.0
            or hh * view.zoom > cfg.screen_hh + 1.0
        ):
            zone_ok = False
            zone_detail = (
                f"t={moment:.1f} box {view.left:.0f},{view.top:.0f},{view.right:.0f},{view.bottom:.0f} "
                f"zoom {view.zoom:.3f}"
            )
    _add(checks, "platform box", zone_ok, zone_detail)
    hitstop = meta.get("hitstop")
    hitstop_value = None if hitstop is None else float(hitstop)
    _add(
        checks,
        "hit-stop",
        hitstop_value is not None and hitstop_value <= 1.5 + 1e-6,
        "missing" if hitstop_value is None else f"{hitstop_value:.3f}s",
    )
    sources = [str(item) for item in (meta.get("impulse_sources") or [])]
    bad_sources = sorted({item for item in sources if item not in {"ball", "dash", "boss", "clash"}})
    _add(
        checks,
        "impulse sources",
        bool(sources) and not bad_sources,
        "missing" if not sources else ("ok" if not bad_sources else " ".join(bad_sources)),
    )
    windup = meta.get("dash_windup_min")
    windup_value = None if windup is None else float(windup)
    _add(
        checks,
        "dash windup",
        windup_value is not None and windup_value >= 15.0 / 60.0 - 1e-6,
        "missing" if windup_value is None else f"{windup_value:.3f}s",
    )
    layout_ok = True
    layout_detail = "clear"
    for video_t in (0.5, 2.0, 3.2, 4.5, 8.0):
        for phase in ("main", "final", "wait"):
            boxes_now = layout_boxes(cfg, video_t, alive=16, phase=phase)
            if not layout_clear(boxes_now):
                layout_ok = False
                layout_detail = f"overlap at {video_t:.1f}s phase {phase}"
    _add(checks, "text boxes", layout_ok, layout_detail)
    cast = [str(code) for code in meta.get("cast") or [country.iso2 for country in cfg.countries]]
    blocked = sorted(code for code in cast if code in cfg.guards.blocked)
    _add(checks, "guards", not blocked, "blocked " + " ".join(blocked) if blocked else f"{len(cast)} codes")
    missing = [code for code in cast if not (Path("assets/flags/png") / f"{code.lower()}.png").is_file()]
    _add(checks, "flag files", not missing, "missing " + " ".join(missing) if missing else "present")
    _add(checks, "memes", bool(cfg.memes.last_reviewed), cfg.memes.last_reviewed)
    frame0 = _extract_frame(ffmpeg, path, 0, width, height) if width and height else None
    if frame0 is not None:
        hook = frame0[int(cfg.hook_y) : int(cfg.hook_y) + 160, int(cfg.safe_x[0]) : int(cfg.safe_x[1])]
        _add(checks, "hook frame 0", int(hook.max()) > 180, f"max {int(hook.max())}")
        counter = frame0[int(cfg.counter_y) : int(cfg.counter_y) + 140, 300:780]
        _add(checks, "counter frame 0", int(counter.max()) > 180, f"max {int(counter.max())}")
    means, diffs = _gray_series(ffmpeg, path)
    hot = photosensitivity_hot_count(means, 60) if len(means) else 99
    _add(checks, "photosensitivity", hot <= 3, f"max hot frames in 1s: {hot}")
    slow = meta.get("slow_spans") or []
    motion_fail = _quiet_motion(diffs, slow, fps=60, limit=0.001)
    _add(checks, "retention motion", motion_fail is None, "moving" if motion_fail is None else f"quiet at {motion_fail:.2f}s")
    if audio is not None:
        samples = _decode_audio(ffmpeg, path)
        meter = pyln.Meter(48000)
        lufs = float(meter.integrated_loudness(samples))
        peak = true_peak_db(samples)
        sample_peak = float(np.max(np.abs(samples))) if samples.size else 1.0
        _add(checks, "lufs", abs(lufs + 14.0) <= 1.0, f"{lufs:.2f} LUFS")
        _add(checks, "true peak", peak <= -1.0 + 1e-3, f"{peak:.2f} dBTP")
        _add(checks, "no clipping", sample_peak < 1.0, f"sample peak {sample_peak:.4f}")
        drop = meta.get("pre_drop") or [0.0, 0.0]
        quiet = _quiet_rms(samples, float(drop[0]), float(drop[1]))
        _add(checks, "retention rms", quiet is None, "holds above -40 dBFS" if quiet is None else f"quiet at {quiet:.2f}s")
    if any(not item.ok for item in checks) and frame0 is not None:
        worst = path.with_name(path.stem + ".worst.png")
        cv2.imwrite(str(worst), frame0)
        print(f"wrote worst frame {worst}", file=sys.stderr)
    return checks


def _quiet_motion(diffs: np.ndarray, slow_spans: list, *, fps: int, limit: float) -> float | None:
    if diffs.size < fps:
        return None
    window = fps
    for start in range(0, len(diffs) - window + 1, max(1, fps // 2)):
        moment = start / fps
        if any(span[0] - 0.05 <= moment <= span[1] + 0.05 for span in slow_spans):
            continue
        if float(np.mean(diffs[start : start + window])) < limit:
            return moment
    return None


def _quiet_rms(samples: np.ndarray, drop0: float, drop1: float, sr: int = 48000) -> float | None:
    hop = sr // 2
    win = sr // 2
    for start in range(0, max(1, len(samples) - win), hop):
        moment = start / sr
        if drop0 - 0.05 <= moment <= drop1 + 0.05:
            continue
        chunk = samples[start : start + win]
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        db = 20.0 * math.log10(max(rms, 1e-12))
        if db < -40.0:
            return moment
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify a rendered Short")
    parser.add_argument("mp4")
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=("bounce", "hop", "morph", "arena", "odd"), default=None)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args(argv)
    config_path = Path(args.config)
    mode = args.mode or detect_mode(config_path)
    if mode == "arena":
        from fc_sat.arena_config import load_arena_config

        checks = verify_arena_file(Path(args.mp4), load_arena_config(config_path))
    elif mode == "odd":
        from fc_sat.odd_config import load_odd_config
        from fc_sat.odd_verify import verify_odd_file

        checks = verify_odd_file(Path(args.mp4), load_odd_config(config_path))
    elif mode == "hop":
        checks = verify_hop_file(Path(args.mp4), load_hop_config(config_path))
    elif mode == "morph":
        from fc_sat.config import _load_hooks, _sibling
        from fc_sat.morph_config import load_morph_config, morph_from_public

        sidecar = Path(args.mp4).with_suffix(".json")
        if not sidecar.is_file():
            print(f"morph verify needs the sidecar {sidecar.name} with image paths", file=sys.stderr)
            return 1
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        hooks = _load_hooks(_sibling(config_path, "morph_hooks.yaml"))
        if isinstance(meta.get("config"), dict):
            cfg = morph_from_public(meta["config"], hooks)
        else:
            cfg = load_morph_config(config_path)
        image_a = meta.get("image_a")
        image_b = meta.get("image_b")
        if not image_a or not image_b:
            print("morph sidecar is missing image_a or image_b", file=sys.stderr)
            return 1
        checks = verify_morph_file(Path(args.mp4), cfg, image_a=image_a, image_b=image_b)
    else:
        cfg = load_config(config_path)
        checks = verify_file(Path(args.mp4), cfg, use_cache=not args.no_cache)
    print(format_table(checks))
    return 0 if all(item.ok for item in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
