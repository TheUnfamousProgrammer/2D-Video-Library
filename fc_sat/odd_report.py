"""Sidecars for Odd One Out: report, answers, post text, and JSON."""

from __future__ import annotations

import json
from pathlib import Path

from fc_sat.odd_config import OddConfig
from fc_sat.odd_diff import REVEAL_LABELS
from fc_sat.odd_sim import Show, location_phrase, reveal_position


def mmss(seconds: float) -> str:
    total = int(round(seconds))
    return f"{total // 60:02d}:{total % 60:02d}"


def answer_lines(cfg: OddConfig, show: Show) -> list[str]:
    lines = []
    for level in cfg.levels:
        sim = show.levels[level.id]
        play = next(segment for segment in show.timeline if segment.kind == "play" and segment.level_id == level.id)
        reveal = next(segment for segment in show.timeline if segment.kind == "reveal" and segment.level_id == level.id)
        pos = reveal_position(sim, level.timer)
        where = location_phrase(float(pos[0]), float(pos[1]), cfg)
        lines.append(
            f"L{level.id} {level.label}: starts {mmss(play.start_s)}, "
            f"reveal {mmss(reveal.start_s)} ({where}). {REVEAL_LABELS[level.difference]}"
        )
    return lines


def write_answers(path: Path, cfg: OddConfig, show: Show) -> None:
    body = ["# Answers", ""]
    body.extend(f"- {line}" for line in answer_lines(cfg, show))
    body.append("")
    path.write_text("\n".join(body), encoding="utf-8")


def write_post(path: Path, cfg: OddConfig, show: Show) -> None:
    pinned = "\n".join(answer_lines(cfg, show))
    path.write_text(
        f"{cfg.hook} #shorts\n\n"
        "Find the odd one. Comment the level where you gave up.\n\n"
        "#shorts #findtheodd #visualchallenge #satisfying #brainteaser\n\n"
        "Pinned comment:\n"
        f"{pinned}\n",
        encoding="utf-8",
    )


def write_report(path: Path, cfg: OddConfig, show: Show, *, timings: list[tuple[str, float]], critique: str) -> None:
    lines = [
        "# Odd One Out report",
        "",
        f"Seed {cfg.seed}, tier {cfg.tier_name}.",
        "",
        "Motion is the same law for every item, including the odd one. "
        "Speed is drawn once from Normal(mean, 0.20×mean) and clipped to "
        f"[{cfg.speed_clip[0]:.1f}, {cfg.speed_clip[1]:.1f}]×{cfg.speed_mean:.0f}. "
        "Heading starts uniform on the circle and omega starts at 0. "
        f"Each substep adds Normal(0, {cfg.angular_std}×sqrt(dt)) to omega, "
        f"clips it to ±{cfg.omega_clip}, and steps the heading. "
        "Walls reflect. Centers closer than "
        f"{cfg.repulse:.1f}×r (that is, {cfg.repulse / 2:.1f}×(ri+rj)) are pushed apart. "
        "The 1.0 s warm-up is discarded, so frame 0 is already moving.",
        "",
        "Pulse and spin use a per-item phase on a shared clock. "
        "Normals share a frequency, not one brightness, so a paused frame does not point at the odd item.",
        "",
    ]
    for level in cfg.levels:
        sim = show.levels[level.id]
        params = sim.diff_params
        lines.append(f"## Level {level.id} {level.label} ({level.difference})")
        lines.append("")
        lines.append(f"- Items: {level.count}")
        lines.append(f"- Odd item id: {sim.odd_index}")
        lines.append(f"- Reveal label: {REVEAL_LABELS[level.difference]}")
        lines.append(f"- Overlap minimum (center distance minus ri+rj): {sim.min_clearance:.3f} px")
        lines.append(
            f"- Odd speed {sim.odd_speed:.2f} px/s, others p{cfg.speed_low_pct:.0f}..p{cfg.speed_high_pct:.0f} "
            f"[{sim.speed_band[0]:.2f}, {sim.speed_band[1]:.2f}]"
        )
        lines.append(
            f"- Odd mean position during the timer: ({sim.mean_pos[0]:.1f}, {sim.mean_pos[1]:.1f})"
        )
        if params.get("cvd"):
            cvd = ", ".join(f"{name} {value:.3f}" for name, value in params["cvd"].items())
            lines.append(
                f"- Hue distance {params['distance']:.3f} (requested {params['requested_distance']:.3f}), "
                f"lightness delta {params['lightness_delta']:.3f}, CVD {cvd}, "
                f"rejected offsets {params['rejected_offsets']}"
            )
        if "size_ratio" in params:
            lines.append(f"- Size ratio {params['size_ratio']:.3f}, odd radius {params['odd_radius']:.2f} px")
        if "rev_s" in params:
            lines.append(
                f"- Spin {params['rev_s']} rev/s, normals {params['normal_direction']}, "
                f"odd {params['odd_direction']}"
            )
        if "odd_hz" in params:
            lines.append(
                f"- Pulse normals {params['normal_hz']} Hz, odd {params['odd_hz']} Hz, "
                f"amplitude ±{params['amplitude']:.0%} of lightness. {params['phase']}."
            )
        lines.append(f"- Reseeds: {len(sim.reseeds)}")
        for note in sim.reseeds:
            lines.append(f"  - attempt {note['attempt']} seed {note['seed']}: {note['reason']}")
        lines.append("")
        lines.append("Drivers (index, odd, speed, heading0, spin phase, pulse phase, radius):")
        lines.append("")
        for index in range(level.count):
            mark = "odd" if index == sim.odd_index else "normal"
            lines.append(
                f"- {index} {mark} speed {sim.speeds[index]:.2f} heading {sim.headings0[index]:.3f} "
                f"spin {sim.spin_phase[index]:.3f} pulse {sim.pulse_phase[index]:.3f} "
                f"r {sim.radii[index]:.2f}"
            )
        lines.append("")
    if timings:
        lines.append("## Timings")
        lines.append("")
        for name, seconds in timings:
            lines.append(f"- {name}: {seconds:.2f} s")
        lines.append("")
    lines.append("## Critique")
    lines.append("")
    lines.append(critique.strip() or "Not reviewed yet.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_json(path: Path, cfg: OddConfig, show: Show, *, timings: list[tuple[str, float]], loudness: dict | None) -> None:
    payload = {
        "seed": cfg.seed,
        "config": cfg.to_public_dict(),
        "timings": [{"stage": name, "seconds": seconds} for name, seconds in timings],
        "loudness": loudness,
        "timeline": [
            {
                "kind": segment.kind,
                "level": segment.level_id,
                "start": segment.start_s,
                "duration": segment.duration_s,
                "start_frame": segment.start_frame,
                "frames": segment.n_frames,
            }
            for segment in show.timeline
        ],
        "levels": {
            str(level.id): {
                "odd_index": show.levels[level.id].odd_index,
                "min_clearance": show.levels[level.id].min_clearance,
                "difference": show.levels[level.id].diff_params,
            }
            for level in cfg.levels
        },
    }
    path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _json_default(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(type(value).__name__)
