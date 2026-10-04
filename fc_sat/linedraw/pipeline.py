"""Load the fit, mix the score, and encode the short."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from fc_sat.beatkit.delivery import frame_indices, require_delivery
from fc_sat.linedraw.audio import render_score
from fc_sat.linedraw.choreo import N_FRAMES, build_plan
from fc_sat.linedraw.encode import pipe_bgr, pipe_pair
from fc_sat.linedraw.optimize import load_fit
from fc_sat.linedraw.picture import load_picture
from fc_sat.linedraw.post import lint_post, render_post, retention_map
from fc_sat.linedraw.qa import STILLS, contact_sheet, loudness_image, spectrogram_image, write_stills
from fc_sat.linedraw.render import LineRenderer, panel_fraction
from fc_sat.linedraw.verify import verify
from fc_sat.stage_log import StageLog

ROOT = Path(__file__).resolve().parents[2]


def load_state(out: Path, job: dict):
    meta = json.loads((out / "picture.json").read_text())
    picture = load_picture(out / "picture.npz")
    fit = load_fit(out / "lines.npz")
    plan = build_plan(
        int(meta["kept"]),
        int(meta["l_aha"]),
        float(meta.get("reveal_fraction", 0.55)),
        pre_drop=meta.get("pre_drop"),
    )
    return meta, picture, fit, plan


def write_timeline(out: Path) -> None:
    payload = {
        "bpm": 150,
        "fps": 60,
        "frames": N_FRAMES,
        "sample_rate": 48000,
        "hop": 800,
        "sections": [
            {"name": "hook", "start": 0, "end": 191},
            {"name": "rising", "start": 192, "end": 743},
            {"name": "silence", "start": 744, "end": 767},
            {"name": "drop", "start": 768, "end": 863},
            {"name": "twist", "start": 864, "end": 959},
            {"name": "rising_2", "start": 960, "end": 1439},
            {"name": "climax", "start": 1440, "end": 1823},
        ],
        "events": [
            {"frame": 0, "event": "kick_and_pluck"},
            {"frame": 384, "event": "riser"},
            {"frame": 576, "event": "snare_roll"},
            {"frame": 744, "event": "silence"},
            {"frame": 768, "event": "impact"},
            {"frame": 876, "event": "zoom_whoosh"},
            {"frame": 940, "event": "pitch_dip"},
            {"frame": 960, "event": "second_drop"},
            {"frame": 1344, "event": "breakdown"},
            {"frame": 1440, "event": "wipe_whoosh"},
            {"frame": 1536, "event": "wipe_whoosh_back"},
            {"frame": 1632, "event": "chord_stab"},
            {"frame": 1800, "event": "tape_stop"},
        ],
    }
    (out / "timeline.json").write_text(json.dumps(payload, indent=2) + "\n")


def write_audio(out: Path, fit, plan, seed: int):
    stems, mix, sfx, lufs, peak = render_score(fit, plan, seed)
    sf.write(out / "linedraw.wav", mix, 48000, subtype="FLOAT")
    sf.write(out / "linedraw.sfx.wav", sfx, 48000, subtype="FLOAT")
    stem_dir = out / "stems"
    stem_dir.mkdir(parents=True, exist_ok=True)
    for name in ("drums", "bass", "cowbell", "pad", "fx", "scratches", "whooshes", "impact"):
        sf.write(stem_dir / f"{name}.wav", stems[name], 48000, subtype="FLOAT")
    spectrogram_image(mix, out / "spectrogram.png")
    loudness_image(mix, out / "loudness.png")
    from fc_sat.linedraw.verify import audio_quality

    errors, stats = audio_quality(mix)
    rows = "\n".join(f"- {key}: {value:.3f}" for key, value in stats.items())
    verdict = "pass" if not errors else "FAIL " + "; ".join(errors)
    report = [
        "# Audio",
        "",
        f"Master {lufs:.2f} LUFS before the encode headroom check, wav true peak {peak:.2f} dBTP.",
        "The wav is 32-bit float, 48 kHz, stereo, exactly 1824 × 800 samples.",
        "Stems are in out/stems/. The spectrogram is out/spectrogram.png and the level timeline is out/loudness.png.",
        "",
        "## Measured",
        "",
        rows,
        "",
        f"Quality checks: {verdict}.",
        "The drop window is frames 768–948 against the build window frames 564–744.",
        "The silent beat is frames 744–767 and the loop tail is frames 1806–1823.",
        "Harmonic delta is 100–300 Hz energy minus 40–90 Hz energy on the drop, in dB. Within 18 dB means the 808 still speaks on a phone.",
        "",
    ]
    (out / "audio_report.md").write_text("\n".join(report))
    write_timeline(out)
    print(f"audio {lufs:.2f} LUFS peak {peak:.2f} dBTP", flush=True)
    return mix, sfx, lufs, peak


def _frames(renderer: LineRenderer, indices: list[int], luma: list[float] | None = None):
    for frame in indices:
        image = renderer.render(frame)
        if luma is not None:
            rgb = image[:, :, ::-1].astype(np.float64)
            luma.append(float((0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]).mean() / 255.0))
        yield image


def encode_pair(renderer: LineRenderer, out: Path, preset: str = "slow") -> None:
    spec = require_delivery("full", 1080, 1920, 60, N_FRAMES)
    luma: list[float] = []
    pipe_pair(
        _frames(renderer, list(range(spec.frames)), luma),
        out / "linedraw.mp4",
        out / "linedraw.music_off.mp4",
        width=spec.width,
        height=spec.height,
        fps=spec.fps,
        first_audio=out / "linedraw.wav",
        second_audio=out / "linedraw.sfx.wav",
        preset=preset,
        n_frames=spec.frames,
    )
    np.save(out / "luma.npy", np.array(luma))


def write_delivery(renderer: LineRenderer, mode: str, output: Path, audio: Path | None, preset: str) -> None:
    spec = require_delivery(mode, 540 if mode != "full" else 1080, 960 if mode != "full" else 1920, 30 if mode != "full" else 60, 912 if mode == "preview" else 105 if mode == "hooks" else N_FRAMES)
    indices = frame_indices(mode, N_FRAMES)
    pipe_bgr(
        _frames(renderer, indices),
        output,
        width=spec.width,
        height=spec.height,
        fps=spec.fps,
        audio_path=audio,
        preset=preset,
        n_frames=spec.frames,
    )


def write_hooks(out: Path, picture, fit, plan, job: dict) -> None:
    from fc_sat.linedraw.render import LineRenderer as Renderer

    strip = ImageStrip()
    full = Renderer(fit, picture, plan, job, width=1080, height=1920, hook="A")
    energies = []
    previous = None
    min_text = 10_000.0
    for frame in range(25):
        image = full.render(frame)
        min_text = min(min_text, full.last_caption_px or min_text)
        gray = image.mean(axis=2)
        if previous is not None:
            energies.append(float(np.mean(np.abs(gray - previous))))
        previous = gray
    print(
        f"style panel {panel_fraction():.3f} min_text {min_text:.1f}px motion_0_24 {float(np.mean(energies)):.4f}",
        flush=True,
    )
    for hook in ("A", "B", "C"):
        renderer = Renderer(fit, picture, plan, job, width=540, height=960, hook=hook)
        write_delivery(renderer, "hooks", out / f"hook_{hook}.mp4", out / "linedraw.wav", "veryfast")
        if hook == "A":
            tiles = []
            for frame in (0, 12, 30, 60, 120):
                rgb = renderer.render(frame)[:, :, ::-1]
                from PIL import Image

                image = Image.fromarray(rgb, mode="RGB")
                image.thumbnail((180, 320))
                tiles.append(image)
            sheet = Image.new("RGB", (180 * len(tiles), 320), (14, 17, 23))
            for index, tile in enumerate(tiles):
                sheet.paste(tile, (index * 180, 0))
            sheet.save(out / "hook_strip.png")
    _ = strip


class ImageStrip:
    pass


def write_motion(out: Path, renderer: LineRenderer) -> None:
    from fc_sat.linedraw.encode import pipe_bgr

    spans = ((0, 180, "motion_hook.mp4"), (700, 880, "motion_drop.mp4"), (860, 1040, "motion_zoom.mp4"))
    half = LineRenderer(renderer.fit, renderer.picture, renderer.plan, renderer.job, width=540, height=960, hook=renderer.hook)
    for start, end, name in spans:
        indices = list(range(start, end, 2))

        def frames(indices=indices):
            for frame in indices:
                yield half.render(frame)

        pipe_bgr(frames(), out / name, width=540, height=960, fps=30, audio_path=None, preset="veryfast", n_frames=len(indices))
        print(f"wrote {out / name}", flush=True)


def render_full(job: dict, log: StageLog) -> int:
    from make_linedraw import _optimize

    out = ROOT / "out"
    out.mkdir(parents=True, exist_ok=True)
    if not (out / "lines.npz").exists():
        _optimize(job, out)
        log.mark("optimize")
    meta, picture, fit, plan = load_state(out, job)
    if not (out / "linedraw.wav").exists():
        write_audio(out, fit, plan, job["seed"])
        log.mark("audio")
    elif not (out / "timeline.json").exists():
        write_timeline(out)
    renderer = LineRenderer(fit, picture, plan, job, width=1080, height=1920, hook=job.get("hook", "A"))
    encode_pair(renderer, out, preset="slow")
    log.mark("encode")
    write_stills(renderer, out / "stills", STILLS + (600, 768, 876))
    contact_sheet(renderer, out / "contact_sheet.png")
    write_motion(out, renderer)
    log.mark("stills")
    post = render_post(meta)
    errors = lint_post(post, meta)
    (out / "postkit.md").write_text(post)
    (out / "retention_map.md").write_text(retention_map(meta))
    if errors:
        print("post lint " + "; ".join(errors))
    code, _report = verify(out, out / "linedraw.mp4")
    log.mark("verify")
    return code
