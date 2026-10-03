"""Container probes. A 30.4 s file can be 30 fps; duration alone cannot tell."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


def count_video_frames(ffprobe: str, path: Path) -> tuple[str | None, int | None]:
    """Return (r_frame_rate, frame count). Frame count prefers the container, then a decode."""
    proc = subprocess.run(
        [ffprobe, "-v", "error", "-count_frames", "-print_format", "json", "-show_streams", str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise SystemExit(f"ffprobe failed:\n{proc.stderr}")
    info = json.loads(proc.stdout or "{}")
    video = next((stream for stream in info.get("streams", []) if stream.get("codec_type") == "video"), None)
    if video is None:
        return None, None
    rate = video.get("r_frame_rate")
    raw = video.get("nb_read_frames") or video.get("nb_frames")
    count = int(raw) if raw not in (None, "N/A") else None
    return (str(rate) if rate is not None else None), count


def frame_rate_errors(ffprobe: str, path: Path, *, rate: str = "60/1", frames: int = 1824) -> list[str]:
    found_rate, found_frames = count_video_frames(ffprobe, path)
    errors = []
    print(f"r_frame_rate {found_rate}, frames {found_frames}")
    if found_rate != rate:
        errors.append(f"r_frame_rate {found_rate}, expected {rate}")
    if found_frames != frames:
        errors.append(f"frame count {found_frames}, expected {frames}")
    return errors
