"""Checks for an Odd One Out master.

Pixel detectability runs on the file and again on a crf-30 720x1280 re-encode.
Thresholds stay where the spec put them. A failure prints the measured number
and writes the mid-timer frame for that level.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import numpy as np

from fc_sat.audio import true_peak_db
from fc_sat.color import rgb_u8_to_oklab
from fc_sat.encode import find_ffmpeg, find_ffprobe
from fc_sat.odd_config import OddConfig, build_timeline, timeline_frames
from fc_sat.odd_diff import (
    _brightest_angle,
    _patch_lab,
    center_luma,
    measure_size,
    pulse_peak_hz,
    spin_from_angles,
)
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
MOTION_MIN = 0.0004
LABEL_FOR = {"hue": "EASY", "size": "MEDIUM", "spin": "HARD", "pulse": "BRUTAL"}


def harsh_reencode(src: Path, dest: Path) -> None:
    """crf 30, veryfast, 720x1280. Pixel checks only; the audio is dropped."""
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


def quiet_window(diffs: np.ndarray, fps: int, limit: float = MOTION_MIN) -> float | None:
    """Start time, in seconds, of a 1 s window whose mean motion is under ``limit``."""
    window = max(1, int(fps))
    if diffs.size < window:
        return 0.0 if float(np.mean(diffs)) < limit else None
    kernel = np.ones(window, dtype=np.float64) / window
    rolled = np.convolve(diffs.astype(np.float64), kernel, mode="valid")
    worst = int(np.argmin(rolled))
    if float(rolled[worst]) < limit:
        return worst / float(fps)
    return None


def _bg_lab(cfg: OddConfig) -> np.ndarray:
    text = cfg.background.lstrip("#")
    rgb = np.array([[[int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16)]]], dtype=np.uint8)
    return rgb_u8_to_oklab(rgb)[0, 0]


def _others(n: int, odd: int, k: int, seed: int) -> list[int]:
    rng = np.random.Generator(np.random.PCG64(seed))
    pool = [i for i in range(n) if i != odd]
    rng.shuffle(pool)
    return pool[: min(k, len(pool))]


def sim_checks(cfg: OddConfig, show) -> list[Check]:
    """One odd item, the anti-bias numbers, clearance, and CVD. From the sim, not the pixels."""
    checks: list[Check] = []
    for level in cfg.levels:
        sim = show.levels[level.id]
        radii = np.asarray(sim.radii, dtype=np.float64)
        odd = int(sim.odd_index)
        unique = odd == int(sim.odd_index) and 0 <= odd < level.count
        if level.difference == "size":
            bigger = np.where(radii > cfg.radius + 1e-6)[0]
            unique = unique and list(bigger) == [odd]
        else:
            unique = unique and float(np.max(np.abs(radii - cfg.radius))) < 1e-6
        _add(checks, f"L{level.id} one odd", unique, f"odd index {odd} of {level.count}")
        lo, hi = sim.speed_band
        speed_ok = lo - 1e-6 <= sim.odd_speed <= hi + 1e-6
        _add(
            checks,
            f"L{level.id} speed band",
            speed_ok,
            f"odd {sim.odd_speed:.2f} px/s inside [{lo:.2f}, {hi:.2f}]",
        )
        span = 0.5 * (1.0 - cfg.position_middle)
        x_lo = cfg.field_x0 + span * (cfg.field_x1 - cfg.field_x0)
        x_hi = cfg.field_x1 - span * (cfg.field_x1 - cfg.field_x0)
        y_lo = cfg.field_y0 + span * (cfg.field_y1 - cfg.field_y0)
        y_hi = cfg.field_y1 - span * (cfg.field_y1 - cfg.field_y0)
        mx, my = float(sim.mean_pos[0]), float(sim.mean_pos[1])
        _add(
            checks,
            f"L{level.id} position",
            x_lo <= mx <= x_hi and y_lo <= my <= y_hi,
            f"mean ({mx:.1f}, {my:.1f})",
        )
        _add(
            checks,
            f"L{level.id} clearance",
            sim.min_clearance >= -0.35,
            f"min center gap beyond 2r is {sim.min_clearance:.2f} px",
        )
        if level.difference == "hue":
            cvd = sim.diff_params.get("cvd") or {}
            worst = min(float(cvd.get(kind, 0.0)) for kind in ("protan", "deutan", "tritan"))
            _add(
                checks,
                f"L{level.id} cvd",
                worst + 1e-9 >= cfg.tier.cvd_min_distance,
                " ".join(f"{kind}={float(cvd.get(kind, 0.0)):.3f}" for kind in ("protan", "deutan", "tritan")),
            )
    return checks


def sequence_checks(cfg: OddConfig) -> list[Check]:
    checks: list[Check] = []
    timeline = build_timeline(cfg)
    labels = []
    reveal_ok = True
    detail = []
    for level in cfg.levels:
        play = next(segment for segment in timeline if segment.kind == "play" and segment.level_id == level.id)
        reveal = next(segment for segment in timeline if segment.kind == "reveal" and segment.level_id == level.id)
        aligned = play.end_frame == reveal.start_frame and abs(play.end_s - reveal.start_s) < 1e-9
        reveal_ok = reveal_ok and aligned
        detail.append(f"L{level.id} reveal@{reveal.start_s:.3f}s")
        labels.append(level.label)
    _add(checks, "reveal at timer 0", reveal_ok, ", ".join(detail))
    _add(checks, "label sequence", labels == [LABEL_FOR[level.difference] for level in cfg.levels], " ".join(labels))
    return checks


def _sample_indexes(start: int, count: int, n: int) -> list[int]:
    if count <= 1:
        return [start]
    last = start + count - 1
    return [int(round(start + (last - start) * i / (n - 1))) for i in range(n)]


def _field_motion(frame: np.ndarray, prev: np.ndarray, box: tuple[int, int, int, int]) -> float:
    x0, y0, x1, y1 = box
    a = frame[y0:y1:4, x0:x1:4].astype(np.int16)
    b = prev[y0:y1:4, x0:x1:4].astype(np.int16)
    if a.size == 0:
        return 0.0
    return float(np.mean(np.abs(a - b)) / 255.0)


def measure_decoded(
    path: Path,
    cfg: OddConfig,
    show,
    *,
    width: int,
    height: int,
    prefix: str,
    dump_dir: Path,
) -> list[Check]:
    """Stream the file once. Positions come from the sim at each frame's sim time."""
    checks: list[Check] = []
    scale = width / float(cfg.width)
    ffmpeg = find_ffmpeg()
    bg = _bg_lab(cfg)
    timeline = build_timeline(cfg)
    fps = cfg.fps
    field = (
        int(round(cfg.field_x0 * scale)),
        int(round(cfg.field_y0 * scale)),
        int(round(cfg.field_x1 * scale)),
        int(round(cfg.field_y1 * scale)),
    )
    plans = []
    for level in cfg.levels:
        play = next(segment for segment in timeline if segment.kind == "play" and segment.level_id == level.id)
        reveal = next(segment for segment in timeline if segment.kind == "reveal" and segment.level_id == level.id)
        sim = show.levels[level.id]
        odd = int(sim.odd_index)
        hue_frames = set(_sample_indexes(play.start_frame, play.n_frames, 10)) if level.difference == "hue" else set()
        size_frame = play.start_frame + play.n_frames // 2 if level.difference == "size" else None
        plans.append(
            {
                "level": level,
                "play": play,
                "reveal": reveal,
                "sim": sim,
                "odd": odd,
                "hue_frames": hue_frames,
                "hue_odd": [],
                "hue_others": [],
                "other_ids": _others(level.count, odd, 12 if level.difference == "hue" else 10, cfg.seed + level.id),
                "size_frame": size_frame,
                "spin_odd": [],
                "spin_others": [],
                "pulse_odd": [],
                "pulse_others": [],
                "before": None,
                "at_reveal": None,
                "mid": None,
            }
        )
    by_frame: dict[int, list] = {}
    for plan in plans:
        play = plan["play"]
        reveal = plan["reveal"]
        for index in range(play.start_frame, play.end_frame):
            by_frame.setdefault(index, []).append(plan)
        by_frame.setdefault(max(0, reveal.start_frame - 1), []).append(plan)
        by_frame.setdefault(reveal.start_frame, []).append(plan)

    cap_y = int(round(cfg.caption_y * scale))
    cap_px = max(2, int(round(cfg.caption_px * scale)))
    cap_x0 = int(round(cfg.caption_x[0] * scale))
    cap_x1 = int(round(cfg.caption_x[1] * scale))
    hook_max = 0.0
    diffs: list[float] = []
    lumas: list[float] = []
    prev = None
    started = time.perf_counter()
    last_print = started
    for index, frame in enumerate(_iter_decoded(ffmpeg, path, width, height)):
        lumas.append(_luma_mean(frame))
        if index == 0:
            band = frame[max(0, cap_y - cap_px // 2) : cap_y + cap_px // 2, cap_x0:cap_x1]
            if band.size:
                luma = 0.0722 * band[..., 0] + 0.7152 * band[..., 1] + 0.2126 * band[..., 2]
                hook_max = float(luma.max() / 255.0)
        if prev is not None:
            diffs.append(_field_motion(frame, prev, field))
        prev = frame
        for plan in by_frame.get(index, ()):
            _collect(plan, frame, index, scale, cfg, bg)
            if plan["mid"] is None and index == plan["play"].start_frame + plan["play"].n_frames // 2:
                plan["mid"] = frame.copy()
        now = time.perf_counter()
        if now - last_print >= 30.0:
            print(f"{prefix}decode frame {index} ({now - started:.0f}s)", flush=True)
            last_print = now

    motion = np.asarray(diffs, dtype=np.float64)
    quiet = quiet_window(motion, fps) if motion.size else 0.0
    if motion.size:
        window = max(1, fps)
        kernel = np.ones(min(window, motion.size), dtype=np.float64) / min(window, motion.size)
        rolled = np.convolve(motion, kernel, mode="valid")
        motion_detail = f"min 1s mean {float(rolled.min()):.5f}"
        if quiet is not None:
            motion_detail += f" at {quiet:.2f}s"
    else:
        motion_detail = "no frame pairs"
    if prefix == "":
        _add(checks, "hook on frame 0", hook_max >= 0.45, f"caption-band max luma {hook_max:.3f}")
        _add(checks, "motion", quiet is None, motion_detail)
        hot = photosensitivity_hot_count(np.asarray(lumas, dtype=np.float64), fps)
        _add(checks, "photosensitivity", hot <= 3, f"max jumps >0.10 in 1s: {hot}")

    for plan in plans:
        level = plan["level"]
        name = f"{prefix}L{level.id} {level.difference}"
        ok, detail = _score_plan(plan, cfg, fps, require_dim=(prefix == ""))
        _add(checks, name, ok, detail)
        if not ok and plan["mid"] is not None:
            dest = dump_dir / f"{path.stem}.L{level.id}.worst.png"
            import cv2

            cv2.imwrite(str(dest), plan["mid"])
            print(f"saved worst frame {dest}", flush=True)
    return checks


def _xy(sim, sim_t: float, index: int, scale: float) -> tuple[float, float]:
    point = sim.at(sim_t)[index]
    return float(point[0] * scale), float(point[1] * scale)


def _collect(plan, frame, index: int, scale: float, cfg: OddConfig, bg: np.ndarray) -> None:
    play = plan["play"]
    reveal = plan["reveal"]
    sim = plan["sim"]
    sim_t = (index - play.start_frame) / cfg.fps
    odd = plan["odd"]
    kind = plan["level"].difference
    if play.start_frame <= index < play.end_frame:
        if kind == "hue" and index in plan["hue_frames"]:
            ox, oy = _xy(sim, sim_t, odd, scale)
            plan["hue_odd"].append(_patch_lab(frame, ox, oy))
            others = []
            for item in plan["other_ids"]:
                x, y = _xy(sim, sim_t, item, scale)
                others.append(_patch_lab(frame, x, y))
            plan["hue_others"].append(others)
        elif kind == "size" and index == plan["size_frame"]:
            odd_xy = _xy(sim, sim_t, odd, scale)
            other_xy = [_xy(sim, sim_t, item, scale) for item in range(plan["level"].count) if item != odd]
            window = cfg.radius * cfg.tier.size_ratio * scale * 2.4
            plan["size_ratio"] = measure_size(frame, odd_xy, other_xy, window, bg)["ratio"]
        elif kind == "spin":
            radius = cfg.radius * scale
            ox, oy = _xy(sim, sim_t, odd, scale)
            plan["spin_odd"].append(_brightest_angle(frame, ox, oy, radius))
            row = []
            for item in plan["other_ids"]:
                x, y = _xy(sim, sim_t, item, scale)
                row.append(_brightest_angle(frame, x, y, radius))
            plan["spin_others"].append(row)
        elif kind == "pulse":
            ox, oy = _xy(sim, sim_t, odd, scale)
            plan["pulse_odd"].append(center_luma(frame, ox, oy))
            row = []
            for item in plan["other_ids"][:8]:
                x, y = _xy(sim, sim_t, item, scale)
                row.append(center_luma(frame, x, y))
            plan["pulse_others"].append(row)
    if index == reveal.start_frame - 1:
        item = 0 if odd != 0 else 1
        x, y = _xy(sim, sim_t, item, scale)
        plan["before"] = center_luma(frame, x, y)
    if index == reveal.start_frame:
        item = 0 if odd != 0 else 1
        x, y = _xy(sim, sim_t, item, scale)
        plan["at_reveal"] = center_luma(frame, x, y)


def _score_plan(plan, cfg: OddConfig, fps: int, *, require_dim: bool) -> tuple[bool, str]:
    level = plan["level"]
    kind = level.difference
    before = plan["before"]
    after = plan["at_reveal"]
    dim_ok = (not require_dim) or (before is not None and after is not None and after < before * 0.75)
    dim = f"normal center {before if before is not None else -1:.3f} -> {after if after is not None else -1:.3f}"
    if kind == "hue":
        if len(plan["hue_odd"]) < 2:
            return False, f"not enough hue samples; {dim}"
        odd = np.mean(plan["hue_odd"], axis=0)
        distances = []
        others = np.asarray(plan["hue_others"], dtype=np.float64)
        for item in range(others.shape[1]):
            distances.append(float(np.linalg.norm(odd - others[:, item].mean(axis=0))))
        measured = float(np.mean(distances))
        nominal = float(plan["sim"].diff_params.get("distance", cfg.tier.hue_min_distance))
        ok = measured >= 0.70 * nominal and dim_ok
        return ok, f"oklab {measured:.3f} vs 70% of {nominal:.3f}; {dim}"
    if kind == "size":
        if "size_ratio" not in plan:
            return False, f"size frame missing; {dim}"
        ratio = float(plan["size_ratio"])
        ok = ratio >= 1.2 and dim_ok
        return ok, f"area ratio {ratio:.3f} (need >= 1.2); {dim}"
    if kind == "spin":
        odd_stats = spin_from_angles(
            plan["spin_odd"], fps, expect_sign=-1.0, nominal_rev_s=cfg.tier.spin_rev_s
        )
        other_ok = 0
        other_n = 0
        columns = list(zip(*plan["spin_others"])) if plan["spin_others"] else []
        for column in columns:
            stats = spin_from_angles(column, fps, expect_sign=1.0, nominal_rev_s=cfg.tier.spin_rev_s)
            other_n += 1
            mag_ok = abs(stats["omega"] - stats["nominal"]) <= 0.30 * stats["nominal"]
            if stats["sign_fraction"] >= 0.95 and mag_ok:
                other_ok += 1
        odd_mag = abs(odd_stats["omega"] - odd_stats["nominal"]) <= 0.30 * odd_stats["nominal"]
        ok = odd_stats["sign_fraction"] >= 0.95 and odd_mag and other_n > 0 and other_ok == other_n and dim_ok
        return (
            ok,
            f"odd sign {odd_stats['sign_fraction']:.2f} omega {odd_stats['omega']:.2f} "
            f"nominal {odd_stats['nominal']:.2f}; others {other_ok}/{other_n}; {dim}",
        )
    odd_hz = pulse_peak_hz(plan["pulse_odd"], fps)
    columns = list(zip(*plan["pulse_others"])) if plan["pulse_others"] else []
    other_hz = [pulse_peak_hz(column, fps) for column in columns]
    target_odd = cfg.tier.pulse_odd_hz
    target_norm = cfg.tier.pulse_normal_hz
    others_ok = all(abs(hz - target_norm) <= 0.2 for hz in other_hz) if other_hz else False
    closer = abs(odd_hz - target_odd) < abs(odd_hz - target_norm)
    ok = abs(odd_hz - target_odd) <= 0.2 and others_ok and closer and dim_ok
    shown = ", ".join(f"{hz:.2f}" for hz in other_hz[:4])
    return ok, f"odd {odd_hz:.2f} Hz (want {target_odd:.2f}); others {shown} (want {target_norm:.2f}); {dim}"


def layout_frame_checks(cfg: OddConfig, show) -> list[Check]:
    """Math for every caption state, plus rendered boxes on a handful of frames."""
    from fc_sat.odd_layout import field_box, layout_failures
    from fc_sat.odd_render import OddRenderer

    checks: list[Check] = []
    math_bad = math_layout_failures(cfg)
    _add(checks, "layout math", not math_bad, "inside" if not math_bad else "; ".join(math_bad))
    renderer = OddRenderer(cfg, show, preview=False)
    field = field_box(cfg)
    safe = (cfg.safe_x[0], cfg.safe_y[0], cfg.safe_x[1], cfg.safe_y[1])
    indexes = {0, renderer.n_frames - 1}
    for segment in show.timeline:
        if segment.kind in ("play", "reveal"):
            indexes.add(segment.start_frame)
            indexes.add(min(renderer.n_frames - 1, segment.start_frame + segment.n_frames // 2))
    rendered: list[str] = []
    for index in sorted(indexes):
        renderer.render(index)
        boxes = [(name, tuple(float(v) for v in box)) for name, box in renderer.boxes if name != "pill"]
        rendered.extend(layout_failures(boxes, field, safe))
        for name, box in renderer.boxes:
            if name == "pill":
                if box[0] < cfg.field_x0 - 2 or box[2] > cfg.field_x1 + 2 or box[1] < cfg.field_y0 - 2 or box[3] > cfg.field_y1 + 2:
                    rendered.append(f"frame {index} pill leaves the field")
    _add(
        checks,
        "layout frames",
        not rendered,
        f"{len(indexes)} frames inside" if not rendered else "; ".join(rendered[:6]),
    )
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
        _add(checks, "duration window", 30.0 <= vdur <= 42.0, f"{vdur:.3f}s")
        _add(checks, "duration vs timeline", abs(vdur - expected) <= 0.020, f"{vdur:.4f}s vs {expected:.4f}s")
        if adur is not None:
            _add(checks, "av sync", abs(vdur - adur) <= 0.020, f"video {vdur:.4f}s audio {adur:.4f}s")
        else:
            _add(checks, "av sync", False, "audio duration missing")
    size_mb = path.stat().st_size / (1024 * 1024)
    _add(checks, "file size", size_mb < 100.0, f"{size_mb:.2f} MB")
    return checks, video, audio


def verify_odd_file(path: Path, cfg: OddConfig, show=None) -> list[Check]:
    """Container, loudness, layout, sim constraints, and pixel detectability.

    The harsh re-encode is written next to the master as ``{stem}.harsh.mp4``.
    Quick-tier films are shorter than 30 s, so the duration window fails on purpose.
    The upload file is the normal four-level timeline.
    """
    from fc_sat.odd_sim import simulate_show

    path = Path(path)
    checks: list[Check] = []
    print(f"verify {path.name}", flush=True)
    box, video, audio = container_checks(path, cfg)
    checks.extend(box)
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
        import pyloudnorm as pyln

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
        print("detectability decode", flush=True)
        checks.extend(measure_decoded(path, cfg, show, width=width, height=height, prefix="", dump_dir=path.parent))
        harsh = path.with_name(path.stem + ".harsh.mp4")
        print(f"harsh re-encode -> {harsh.name}", flush=True)
        harsh_reencode(path, harsh)
        print("harsh detectability", flush=True)
        checks.extend(
            measure_decoded(harsh, cfg, show, width=HARSH_W, height=HARSH_H, prefix="harsh ", dump_dir=path.parent)
        )
    else:
        _add(checks, "detectability", False, f"need {cfg.width}x{cfg.height}, got {width}x{height}"        )
    return checks
