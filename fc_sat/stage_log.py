"""Per-stage timings and a heartbeat so a long job never stays silent."""

from __future__ import annotations

import threading
import time


class StageLog:
    def __init__(self) -> None:
        self.started = time.perf_counter()
        self.last = self.started
        self.rows: list[tuple[str, float]] = []

    def mark(self, name: str) -> float:
        now = time.perf_counter()
        elapsed = now - self.last
        self.rows.append((name, elapsed))
        print(f"stage {name} done in {elapsed:.2f}s", flush=True)
        self.last = now
        return elapsed

    def total(self) -> float:
        return time.perf_counter() - self.started


class Heartbeat:
    """Print a one-line pulse every `interval` seconds until stopped."""

    def __init__(self, message: str = "working", interval: float = 45.0) -> None:
        self.message = message
        self.interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="heartbeat", daemon=True)

    def __enter__(self) -> "Heartbeat":
        self._thread.start()
        return self

    def __exit__(self, *_) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            print(self.message, flush=True)
