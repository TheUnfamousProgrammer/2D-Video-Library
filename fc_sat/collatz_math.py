"""Collatz step, trajectory, and records. Integer arithmetic only."""

from __future__ import annotations

MAX_ITERS = 10**7


class CollatzError(ValueError):
    """The start value is not a positive integer, or the run hit the cap."""


def check_start(n: object) -> int:
    if isinstance(n, bool) or not isinstance(n, int):
        raise CollatzError(f"start must be a positive integer, got {n!r}")
    if n < 1:
        raise CollatzError(f"start must be >= 1, got {n}")
    return n


def next_n(n: int) -> int:
    n = check_start(n)
    if n % 2 == 0:
        return n // 2
    return 3 * n + 1


def trajectory(n: int) -> list[int]:
    """Values from the start through the first 1, inclusive."""
    current = check_start(n)
    values = [current]
    seen = 0
    while current != 1:
        current = next_n(current)
        values.append(current)
        seen += 1
        if seen > MAX_ITERS:
            raise CollatzError(
                f"{n} did not reach 1 within {MAX_ITERS} steps; refusing to continue"
            )
    return values


def steps(n: int) -> int:
    return len(trajectory(n)) - 1


def peak(n: int) -> int:
    return max(trajectory(n))


def peak_index(n: int) -> int:
    values = trajectory(n)
    top = max(values)
    return values.index(top)


def first_step_above(n: int, threshold: int) -> int:
    for index, value in enumerate(trajectory(n)):
        if value > threshold:
            return index
    raise CollatzError(f"{n} never exceeds {threshold}")


def first_step_below_after_peak(n: int, threshold: int) -> int:
    values = trajectory(n)
    top_at = values.index(max(values))
    for index in range(top_at + 1, len(values)):
        if values[index] < threshold:
            return index
    raise CollatzError(f"{n} never drops below {threshold} after its peak")


def longest_below(limit: int) -> tuple[int, int]:
    """Start in 1..limit-1 with the most steps, and that step count."""
    if limit < 2:
        raise CollatzError("limit must be at least 2")
    best_n = 1
    best_steps = 0
    for n in range(1, limit):
        count = steps(n)
        if count > best_steps:
            best_n = n
            best_steps = count
    return best_n, best_steps


def highest_peak_below(limit: int) -> tuple[int, int]:
    """Return (peak, start) for the highest peak reached by a start below limit."""
    if limit < 2:
        raise CollatzError("limit must be at least 2")
    best_peak = 1
    best_n = 1
    for n in range(1, limit):
        top = peak(n)
        if top > best_peak:
            best_peak = top
            best_n = n
    return best_peak, best_n


def at_least_steps_below(limit: int, min_steps: int) -> list[tuple[int, int]]:
    found = []
    for n in range(1, limit):
        count = steps(n)
        if count >= min_steps:
            found.append((n, count))
    return found
