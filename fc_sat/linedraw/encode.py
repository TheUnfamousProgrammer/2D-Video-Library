"""ffmpeg pipe for the linedraw short. AAC is 256k; the shared pipe stays at 192k."""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from fc_sat.encode import BT709_FILTER, assert_encoders, find_ffmpeg


def encode_command(
    ffmpeg: str,
    *,
    width: int,
    height: int,
    fps: int,
    output: Path,
    audio_path: Path | None,
    preset: str,
    audio_bitrate: str = "256k",
) -> list[str]:
    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-framerate",
        str(fps),
        "-r",
        str(fps),
        "-i",
        "pipe:0",
    ]
    if audio_path is not None:
        command += ["-i", str(audio_path)]
    command += [
        "-vf",
        BT709_FILTER,
        "-c:v",
        "libx264",
        "-profile:v",
        "high",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "16",
        "-preset",
        preset,
        "-colorspace",
        "bt709",
        "-color_primaries",
        "bt709",
        "-color_trc",
        "bt709",
        "-color_range",
        "tv",
        "-bsf:v",
        "h264_metadata=colour_primaries=1:transfer_characteristics=1:matrix_coefficients=1:video_full_range_flag=0",
    ]
    if audio_path is not None:
        command += ["-c:a", "aac", "-b:a", audio_bitrate, "-ar", "48000", "-ac", "2", "-shortest"]
    else:
        command += ["-an"]
    command += ["-fps_mode", "cfr", "-r", str(fps), "-movflags", "+faststart", "-f", "mp4", str(output)]
    return command


def pipe_bgr(
    frames: Iterable[np.ndarray],
    output: Path,
    *,
    width: int,
    height: int,
    fps: int,
    audio_path: Path | None,
    preset: str = "slow",
    n_frames: int | None = None,
) -> None:
    ffmpeg = find_ffmpeg()
    assert_encoders(ffmpeg)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.stem}.partial.mp4")
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
    proc: subprocess.Popen | None = None
    success = False
    frame_iter = iter(frames)
    try:
        proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert proc.stdin is not None
        for index, frame in enumerate(frame_iter):
            proc.stdin.write(np.ascontiguousarray(frame).tobytes())
            if n_frames and index and index % 120 == 0:
                print(f"render: {index}/{n_frames}", flush=True)
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
        closer = getattr(frame_iter, "close", None)
        if closer is not None:
            closer()
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
        if not success and temporary.exists():
            temporary.unlink()


def pipe_pair(
    frames: Iterable[np.ndarray],
    first: Path,
    second: Path,
    *,
    width: int,
    height: int,
    fps: int,
    first_audio: Path,
    second_audio: Path,
    preset: str = "slow",
    n_frames: int | None = None,
) -> None:
    """Encode the same pictures twice, once with the mix and once with effects only."""
    ffmpeg = find_ffmpeg()
    assert_encoders(ffmpeg)
    first.parent.mkdir(parents=True, exist_ok=True)
    temps = []
    procs = []
    success = False
    frame_iter = iter(frames)
    try:
        for output, audio in ((first, first_audio), (second, second_audio)):
            temporary = output.with_name(f"{output.stem}.partial.mp4")
            if temporary.exists():
                temporary.unlink()
            temps.append((output, temporary))
            command = encode_command(
                ffmpeg, width=width, height=height, fps=fps, output=temporary, audio_path=audio, preset=preset
            )
            procs.append(subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE))
        for index, frame in enumerate(frame_iter):
            payload = np.ascontiguousarray(frame).tobytes()
            for proc in procs:
                assert proc.stdin is not None
                proc.stdin.write(payload)
            if n_frames and index and index % 60 == 0:
                print(f"render: {index}/{n_frames}", flush=True)
        for proc in procs:
            assert proc.stdin is not None
            proc.stdin.close()
            proc.stdin = None
        for proc, (output, temporary) in zip(procs, temps):
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
        closer = getattr(frame_iter, "close", None)
        if closer is not None:
            closer()
        for proc in procs:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)
        if not success:
            for _output, temporary in temps:
                if temporary.exists():
                    temporary.unlink()
