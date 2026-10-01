"""Sidecars for Odd One Out: report, answers, post text, and JSON."""

from __future__ import annotations

import json
from pathlib import Path

from fc_sat.odd_config import SKIPPED_MEASURE, OddConfig
from fc_sat.odd_diff import DIR_NAMES, reveal_label
from fc_sat.odd_sim import Show


def mmss(seconds: float) -> str:
    total = int(round(seconds))
    return f"{total // 60:02d}:{total % 60:02d}"


def magnitude_line(cfg: OddConfig, sim) -> str:
    size = sim.level.size
    kind = sim.level.difference
    rung = int(sim.diff_params.get("rung", 0))
    nominal = float(sim.diff_params.get("nominal", 0.0))
    if kind == "hue":
        distance = float(sim.diff_params.get("distance", 0.0))
        light = float(sim.diff_params.get("lightness_delta", 0.0))
        return (
            f"rung {rung}, OKLab distance {distance:.3f} (nominal {nominal:.2f}), "
            f"lightness delta {light:.3f}. This is a color gap on a {size:.0f}px item, not a size fraction."
        )
    if kind == "tilt":
        import math

        degrees = float(sim.diff_params.get("tilt_degrees", nominal))
        angle = math.radians(degrees)
        corner = 0.5 * math.hypot(math.cos(angle) - 1.0, math.sin(angle))
        return f"rung {rung}, tilt {degrees:.0f} deg, corner shift {corner:.2f} of the {size:.0f}px side"
    dot = float(sim.diff_params.get("dot_radius", cfg.dot_fraction * size / 2.0))
    mode = str(sim.diff_params.get("detail_mode", cfg.detail_mode))
    direction = int(sim.diff_params.get("direction", 0)) % 8
    offset = float(sim.diff_params.get("offset", nominal))
    return (
        f"rung {rung}, dot offset {offset:.2f} of the radius toward {DIR_NAMES[direction]}; "
        f"dot diameter {2 * dot:.1f}px is {2 * dot / size:.2f} of the {size:.0f}px item; mode {mode}"
    )


def answer_lines(cfg: OddConfig, show: Show) -> list[str]:
    lines = []
    for level in cfg.levels:
        sim = show.levels[level.id]
        play = next(segment for segment in show.timeline if segment.kind == "play" and segment.level_id == level.id)
        reveal = next(segment for segment in show.timeline if segment.kind == "reveal" and segment.level_id == level.id)
        lines.append(
            f"L{level.id}: starts {mmss(play.start_s)}, reveal {mmss(reveal.start_s)}, "
            f"row {sim.row + 1} column {sim.col + 1} ({sim.phrase}). {reveal_label(level.difference, cfg.detail_mode)}"
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


def write_report(
    path: Path,
    cfg: OddConfig,
    show: Show,
    *,
    timings: list[tuple[str, float]],
    critique: str,
    measured: dict | None = None,
    ladder_paths: list | None = None,
) -> None:
    lines = [
        "# Odd One Out report",
        "",
        f"Seed {cfg.seed}, tier {cfg.tier_name}.",
        "",
        "Every item is static on its cell center. The grid has no jitter. "
        "The odd item is one uniform draw. The only property that changes is the level's difference. "
        f"Items fade in from {cfg.pop_floor:.2f} opacity to full over {cfg.pop_seconds:.1f}s "
        "while the timer is already running, so frame 0 is level 1 with a full timer.",
        "",
    ]
    for level in cfg.levels:
        sim = show.levels[level.id]
        lines.append(f"## Level {level.id} ({level.difference})")
        lines.append("")
        lines.append(f"- Items: {level.count} on a {level.grid}x{level.grid} grid, {level.size:.0f}px")
        lines.append(f"- Odd item: index {sim.odd_index}, row {sim.row + 1}, column {sim.col + 1}, {sim.phrase}")
        lines.append(f"- Difference: {magnitude_line(cfg, sim)}")
        lines.append(f"- Reveal label: {reveal_label(level.difference, cfg.detail_mode)}")
        if int(sim.diff_params.get("rung", 0)) >= 5:
            lines.append(f"- Flag: {SKIPPED_MEASURE}")
        if sim.reseeds:
            lines.append(f"- Reseeds: {len(sim.reseeds)} ({'; '.join(sim.reseeds)})")
        else:
            lines.append("- Reseeds: 0")
        cvd = sim.diff_params.get("cvd")
        if cvd:
            lines.append(
                "- CVD: " + " ".join(f"{kind}={float(cvd[kind]):.3f}" for kind in ("protan", "deutan", "tritan"))
            )
        if measured and level.id in measured:
            lines.append(f"- After harsh re-encode: {measured[level.id]}")
        lines.append(
            f"- Drivers: {level.count} items share cell centers and one base color. "
            f"Odd cell only changes {level.difference}."
        )
        lines.append("")
    if ladder_paths:
        lines.append("## Ladder")
        lines.append("")
        lines.append("Stills only. No video. Open `ladder/index.html` to time yourself.")
        lines.append("")
        for item in ladder_paths:
            lines.append(f"- `{item}`")
        lines.append("")
    if timings:
        lines.append("## Timings")
        lines.append("")
        for name, seconds in timings:
            lines.append(f"- {name}: {seconds:.2f}s")
        lines.append("")
    lines.append("## Critique")
    lines.append("")
    lines.append(critique.strip())
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_json(path: Path, cfg: OddConfig, show: Show, *, timings: list[tuple[str, float]], loudness: dict | None) -> None:
    payload = {
        "seed": cfg.seed,
        "config": cfg.to_public_dict(),
        "timings": [{"stage": name, "seconds": seconds} for name, seconds in timings],
        "loudness": loudness,
        "levels": [
            {
                "id": level.id,
                "difference": level.difference,
                "odd_index": show.levels[level.id].odd_index,
                "row": show.levels[level.id].row,
                "col": show.levels[level.id].col,
                "phrase": show.levels[level.id].phrase,
            }
            for level in cfg.levels
        ],
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
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
