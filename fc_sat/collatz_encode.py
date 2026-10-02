"""Encode Collatz frames with the shared BT.709 ffmpeg command."""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from fc_sat.encode import assert_encoders, encode_command, find_ffmpeg


def write_mp4(
    frames: Iterable[np.ndarray],
    output: Path,
    *,
    width: int,
    height: int,
    fps: int,
    audio_path: Path | None,
    preset: str,
    n_frames: int | None = None,
) -> None:
    ffmpeg = find_ffmpeg()
    assert_encoders(ffmpeg)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem + ".partial.mp4")
    if temporary.exists():
        temporary.unlink()
    command = encode_command(
        ffmpeg,
        width=width,
        height=height,
        fps=fps,
        output=temporary,
        audio_path=audio_path,
        preset=preset,
    )
    proc = None
    success = False
    try:
        proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert proc.stdin is not None
        for index, frame in enumerate(frames):
            proc.stdin.write(np.ascontiguousarray(frame).tobytes())
            if n_frames and index and index % 60 == 0:
                print(f"render {index}/{n_frames}", flush=True)
        proc.stdin.close()
        proc.stdin = None
        stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
        code = proc.wait()
        if code != 0:
            raise SystemExit(f"ffmpeg failed with exit {code}:\n{stderr.strip()}")
        os.replace(temporary, output)
        success = True
    except KeyboardInterrupt:
        print("interrupted; stopping ffmpeg", file=sys.stderr)
        raise
    finally:
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
        if not success and temporary.exists():
            temporary.unlink()
