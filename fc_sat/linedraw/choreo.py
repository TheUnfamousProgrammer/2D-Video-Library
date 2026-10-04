"""Frame-exact line counts, zoom, and captions for the 1824-frame short."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

N_FRAMES = 1824
HOOK_LINES = 16
THROW_LINES = 64


def ease_out_cubic(t: float) -> float:
    t = min(1.0, max(0.0, t))
    return 1.0 - (1.0 - t) ** 3


def ease_in_out_cubic(t: float) -> float:
    t = min(1.0, max(0.0, t))
    if t < 0.5:
        return 4.0 * t * t * t
    return 1.0 - ((-2.0 * t + 2.0) ** 3) / 2.0


def _exp_in(u: np.ndarray, k: float) -> np.ndarray:
    u = np.clip(u, 0.0, 1.0)
    return (np.exp(k * u) - 1.0) / (math.exp(k) - 1.0)


def _cummax(values: np.ndarray) -> None:
    for index in range(1, len(values)):
        if values[index] < values[index - 1]:
            values[index] = values[index - 1]


@dataclass(frozen=True)
class Plan:
    n_lines: int
    l_aha: int
    burst: int
    pre_drop: int
    mid: int
    few: int
    reveal_fraction: float
    counts: np.ndarray

    def count_at(self, frame: int) -> int:
        frame = min(max(int(frame), 0), N_FRAMES - 1)
        if frame >= 1800:
            if frame >= 1812:
                return 0
            return int(round(self.n_lines * (1.0 - (frame - 1800) / 12.0)))
        return int(self.counts[frame])


def choose_hold(counts: np.ndarray, scores: np.ndarray, l_aha: int, ceiling: float = 0.30) -> int:
    """Pre-drop line count.

    The spec hold is 60% of L_aha. If that count is already past `ceiling` likeness,
    the face is readable and the drop would not be a reveal, so the hold moves back
    to the last logged count still under the ceiling.
    """
    aha = max(HOOK_LINES, int(l_aha))
    spec = aha - int(round(0.4 * aha))
    if len(scores) == 0:
        return spec
    index = int(np.searchsorted(counts, spec, side="right") - 1)
    if index < 0 or float(scores[index]) < ceiling:
        return max(HOOK_LINES, spec)
    under = np.flatnonzero(np.asarray(scores) < ceiling)
    if len(under) == 0:
        return HOOK_LINES
    return max(HOOK_LINES, int(counts[int(under[-1])]))


def build_plan(n_lines: int, l_aha: int, reveal_fraction: float = 0.55, pre_drop: int | None = None) -> Plan:
    """Monotone kept-line count through frame 1799. The fly-off is applied on read."""
    total = max(int(n_lines), 1)
    aha = max(16, min(int(l_aha), total))
    burst = int(round(0.4 * aha))
    pre = aha - burst
    if pre_drop is not None:
        pre = max(HOOK_LINES, min(int(pre_drop), aha))
        burst = aha - pre
    if pre < HOOK_LINES:
        pre = HOOK_LINES
        burst = aha - pre
    mid = min(total, max(aha, int(round(1.4 * aha))))
    few = min(total, max(mid, int(round(mid + 0.08 * (total - mid)))))
    counts = np.zeros(N_FRAMES, dtype=np.int32)
    for frame in range(192):
        counts[frame] = 0 if frame < 6 else min(HOOK_LINES, (frame - 6) // 12 + 1)
    index = np.arange(192, 744)
    u = (index - 192) / (743 - 192)
    rising = np.rint(HOOK_LINES + (pre - HOOK_LINES) * _exp_in(u, 5.0)).astype(np.int32)
    rising = np.clip(rising, HOOK_LINES, pre)
    counts[index] = rising
    _cummax(counts[192:744])
    counts[192] = HOOK_LINES
    counts[743] = pre
    counts[744:768] = pre
    counts[768:774] = aha
    index = np.arange(774, 864)
    u = (index - 774) / (863 - 774)
    smooth = u * u * (3.0 - 2.0 * u)
    mid_run = np.rint(aha + (mid - aha) * smooth).astype(np.int32)
    counts[index] = np.clip(mid_run, aha, mid)
    _cummax(counts[774:864])
    counts[863] = mid
    index = np.arange(864, 1057)
    u = (index - 864) / (1056 - 864)
    few_run = np.rint(mid + (few - mid) * u).astype(np.int32)
    counts[index] = np.clip(few_run, mid, few)
    _cummax(counts[864:1057])
    counts[1056] = few
    index = np.arange(1057, 1321)
    u = (index - 1057) / (1320 - 1057)
    race = np.rint(few + (total - few) * _exp_in(u, 4.0)).astype(np.int32)
    counts[index] = np.clip(race, few, total)
    _cummax(counts[1057:1321])
    counts[1320] = total
    counts[1321:1800] = total
    counts[1800:] = total
    return Plan(total, aha, burst, pre, mid, few, float(reveal_fraction), counts)


def zoom_at(frame: int) -> float:
    """1x, then a log-space move to 12x and back. The peak holds from 940 through 959."""
    frame = int(frame)
    if frame < 876 or frame > 1056:
        return 1.0
    if frame <= 940:
        eased = ease_in_out_cubic((frame - 876) / (940 - 876))
        return math.exp(math.log(12.0) * eased)
    if frame <= 959:
        return 12.0
    eased = ease_in_out_cubic((frame - 960) / (1056 - 960))
    return math.exp(math.log(12.0) * (1.0 - eased))


def hook_progress(frame: int, line_index: int) -> float:
    """0 is the start of the throw. Line 1 is at 0.4 on frame 0. Landing is 1.0, then a short settle."""
    start = int(line_index) * 12 - 4
    return (int(frame) - start) / 10.0


def loop_progress(frame: int) -> float:
    """Line 1 re-enters so frame 1823 matches frame 0 (progress 0.4)."""
    return (int(frame) - 1819) / 10.0


def throw_endpoints(x0: float, y0: float, x1: float, y1: float, progress: float, grid_w: int = 254, grid_h: int = 368) -> tuple[float, float, float, float]:
    """Fly in from outside: +25 degrees and 1.3x length, easing onto the stored segment."""
    landed = min(max(progress, 0.0), 1.0)
    settle = 0.0
    if progress > 1.0:
        settle = min(1.0, progress - 1.0)
    eased = ease_out_cubic(landed)
    cx = (x0 + x1) * 0.5
    cy = (y0 + y1) * 0.5
    angle = math.atan2(y1 - y0, x1 - x0)
    length = math.hypot(x1 - x0, y1 - y0)
    outward_x = cx - grid_w * 0.5
    outward_y = cy - grid_h * 0.5
    norm = math.hypot(outward_x, outward_y) or 1.0
    ux, uy = outward_x / norm, outward_y / norm
    push = (1.0 - eased) * grid_w * 0.75
    arc = math.sin(math.pi * landed) * min(40.0, 0.2 * max(length, 1.0))
    px, py = -uy, ux
    center_x = cx + ux * push + px * arc
    center_y = cy + uy * push + py * arc
    flown = angle + math.radians(25.0) * (1.0 - eased)
    flown_length = length * (1.3 - 0.3 * eased)
    if settle > 0.0:
        flown_length *= 1.0 + 0.02 * (1.0 - settle)
    half_x = math.cos(flown) * flown_length * 0.5
    half_y = math.sin(flown) * flown_length * 0.5
    return center_x - half_x, center_y - half_y, center_x + half_x, center_y + half_y


def flyoff_progress(frame: int, line_index: int, n_lines: int) -> float:
    """0 before the line leaves. The reverse throw starts on its stagger frame and finishes in 12 frames."""
    stagger = 0 if n_lines <= 1 else int(line_index) * 7 // max(int(n_lines), 1)
    local = int(frame) - (1800 + stagger)
    if local < 0:
        return 0.0
    return min(1.0, (local + 1) / 12.0)


def wrap_caption(text: str, limit: int = 18) -> list[str]:
    words = text.split()
    if not words:
        return []
    lines: list[str] = []
    current = ""
    for word in words:
        piece = word[:limit]
        trial = piece if not current else f"{current} {piece}"
        if len(trial) <= limit:
            current = trial
            continue
        if current:
            lines.append(current)
        current = piece
    if current:
        lines.append(current)
    return lines[:2]


def drop_caption(subject: str) -> list[str]:
    clean = " ".join(subject.split())
    if not clean:
        return ["THERE IT IS"]
    wrapped = wrap_caption(clean.upper(), 18)
    return ["IT'S", wrapped[0] if wrapped else "THERE IT IS"]


def ratio_caption(thrown: int, kept: int) -> list[str]:
    ratio = max(1, int(round(thrown / max(kept, 1))))
    return [f"1 IN {ratio}", "LINES KEPT"]


def caption_lines(frame: int, hook: str, subject: str, show_original: bool, thrown: int, kept: int) -> list[str]:
    frame = int(frame)
    hooks = {
        "A": ["GUESS WHAT", "THIS BECOMES"],
        "B": ["RANDOM LINES.", "WATCH CLOSELY."],
        "C": ["ONLY STRAIGHT", "LINES. NO CURVES."],
    }
    opening = hooks.get(hook, hooks["A"])
    if frame <= 191 or frame >= 1800:
        return opening
    if 768 <= frame <= 875:
        return drop_caption(subject)
    if 876 <= frame <= 939:
        return ["ZOOM IN"]
    if 940 <= frame <= 959:
        return ["JUST STRAIGHT", "LINES"]
    if 960 <= frame <= 1343:
        return ["THROWN AT RANDOM.", "MOST REJECTED."]
    if 1344 <= frame <= 1439:
        return ratio_caption(thrown, kept)
    if 1440 <= frame <= 1535:
        return ["THE ORIGINAL"] if show_original else ["THEN"]
    if 1536 <= frame <= 1631:
        return ["THE LINES"] if show_original else ["NOW"]
    if 1632 <= frame <= 1799:
        return ["NO CURVES.", "NO BRUSH."]
    return []


def caption_alpha(frame: int) -> float:
    frame = int(frame)
    if frame < 180:
        return 1.0
    if frame <= 191:
        return max(0.0, 1.0 - (frame - 180) / 11.0)
    if 1800 <= frame <= 1807:
        return (frame - 1800) / 7.0
    return 1.0


def show_thrown_counter(frame: int) -> bool:
    return 864 <= int(frame) <= 1343
