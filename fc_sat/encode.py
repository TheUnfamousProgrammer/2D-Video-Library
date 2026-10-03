"""ffmpeg pipe, mux, and cleanup.

Raw BGR frames are converted with scale=out_color_matrix=bt709:out_range=tv.
Input range flags are intentionally omitted. The bitstream is tagged BT.709
limited range. The file is written to a temporary path and renamed only after
ffmpeg exits 0, so a failed encode never leaves a partial mp4 at the final path.

The posted polycircle file was 30 fps. Preview, animatic, and hooks pass
``fps=30`` and every other timeline frame, and that file is still 30.4 s long,
so a duration check cannot tell it from the 60 fps master. The rawvideo
demuxer's frame-rate option is ``-framerate`` (its default is 25). This command
used to set only an input ``-r`` and never pinned the output, so a later
ffmpeg or a player could report a different rate than the one requested.
``-framerate``, ``-fps_mode cfr``, and an output ``-r`` lock the requested rate.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Iterable
from pathlib import Path

import cv2
import numpy as np

from fc_sat.config import Config
from fc_sat.render import Renderer
from fc_sat.sim import SimResult, read_sim_npz

BT709_FILTER = "scale=out_color_matrix=bt709:out_range=tv"
_WORKER: Renderer | None = None


def find_ffmpeg() -> str:
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise SystemExit(
            "ffmpeg was not found on PATH and imageio-ffmpeg is not installed. "
            "Install ffmpeg from https://ffmpeg.org or run: pip install imageio-ffmpeg"
        ) from exc
    try:
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise SystemExit(
            "ffmpeg was not found on PATH and the imageio-ffmpeg binary could not be loaded. "
            "Install ffmpeg from https://ffmpeg.org."
        ) from exc


def find_ffprobe(ffmpeg: str) -> str:
    found = shutil.which("ffprobe")
    if found:
        return found
    sibling = Path(ffmpeg).with_name("ffprobe")
    if sibling.exists():
        return str(sibling)
    return "ffprobe"


def assert_encoders(ffmpeg: str) -> None:
    proc = subprocess.run(
        [ffmpeg, "-hide_banner", "-encoders"],
        check=False,
        capture_output=True,
        text=True,
    )
    text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    missing = []
    names = set()
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0][:1] in {"V", "A", "."} or (len(parts) >= 2 and parts[0][:1] in "VAS"):
            names.add(parts[1])
    # The encoder listing looks like " V.S.. libx264 ..." or " A.... aac ...".
    if "libx264" not in text:
        missing.append("libx264")
    if not any(line.split()[1:2] == ["aac"] for line in text.splitlines() if line.split()):
        missing.append("aac")
    if missing:
        raise SystemExit(
            f"ffmpeg at {ffmpeg} does not provide {', '.join(missing)}. "
            "Install an ffmpeg build with libx264 and the native aac encoder."
        )


def encode_command(
    ffmpeg: str,
    *,
    width: int,
    height: int,
    fps: int,
    output: Path,
    audio_path: Path | None,
    preset: str = "slow",
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
        command += ["-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2", "-shortest"]
    else:
        command += ["-an"]
    command += ["-fps_mode", "cfr", "-r", str(fps), "-movflags", "+faststart", "-f", "mp4", str(output)]
    return command


def _init_worker(cfg: Config, npz_path: str, preview: bool) -> None:
    global _WORKER
    cv2.setNumThreads(1)
    _WORKER = Renderer(cfg, read_sim_npz(Path(npz_path)), preview=preview)


def _render_chunk(indices: list[int]) -> list[np.ndarray]:
    if _WORKER is None:
        raise RuntimeError("render worker was not initialized")
    return [_WORKER.render(index) for index in indices]


def _chunks(n_frames: int, size: int) -> list[list[int]]:
    return [list(range(start, min(n_frames, start + size))) for start in range(0, n_frames, size)]


def resolve_workers(requested: int) -> int:
    if requested and requested > 0:
        return int(requested)
    return max(1, (os.cpu_count() or 2) - 1)


def _terminate(proc: subprocess.Popen | None, pool) -> None:
    if pool is not None:
        pool.terminate()
        pool.join()
    if proc is not None and proc.poll() is None:
        proc.kill()
        proc.wait(timeout=5)


def pipe_raw_bgr(
    frames: Iterable[np.ndarray],
    output: Path,
    *,
    width: int,
    height: int,
    fps: int,
    audio_path: Path | None,
    preset: str = "slow",
    n_frames: int | None = None,
    pool=None,
) -> None:
    """Write raw BGR frames through the production ffmpeg command.

    The file lands at ``output`` only after ffmpeg exits 0. ``pool``, when
    given, is terminated if the pipe fails. ``workers == 1`` callers pass
    frames from the current process; ``workers > 1`` callers pass an ordered
    imap over a spawn pool.
    """
    ffmpeg = find_ffmpeg()
    assert_encoders(ffmpeg)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.partial.mp4")
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
    except BrokenPipeError as exc:
        stderr = ""
        if proc is not None and proc.stderr is not None:
            stderr = proc.stderr.read().decode("utf-8", errors="replace")
        raise SystemExit(f"ffmpeg closed the frame pipe:\n{stderr.strip()}") from exc
    except KeyboardInterrupt:
        print("interrupted; stopping workers and ffmpeg", file=sys.stderr)
        raise
    finally:
        closer = getattr(frame_iter, "close", None)
        if closer is not None:
            closer()
        _terminate(proc, pool)
        if not success and temporary.exists():
            temporary.unlink()


def iter_ordered_frames(
    n_frames: int,
    *,
    workers: int,
    render_one: Callable[[int], np.ndarray],
    initializer: Callable | None = None,
    initargs: tuple = (),
    render_chunk: Callable | None = None,
    chunk_size: int = 2,
):
    """Yield frames in order. One worker stays in-process; more use a spawn pool."""
    if workers <= 1:
        for index in range(n_frames):
            yield render_one(index)
        return
    if initializer is None or render_chunk is None:
        raise RuntimeError("workers > 1 requires an initializer and a render_chunk callable")
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    pool = ctx.Pool(workers, initializer=initializer, initargs=initargs)
    try:
        done = 0
        for frames in pool.imap(render_chunk, _chunks(n_frames, chunk_size)):
            for frame in frames:
                yield frame
                done += 1
            if done % 120 < chunk_size:
                print(f"render: {done}/{n_frames}", flush=True)
    finally:
        pool.terminate()
        pool.join()


def encode_frames(
    cfg: Config,
    sim: SimResult,
    npz_path: Path,
    output: Path,
    *,
    preview: bool,
    audio_path: Path | None,
    workers: int,
    preset: str = "slow",
) -> dict[str, float]:
    """Pipe frames to ffmpeg. Returns profile numbers. Raises SystemExit on encoder failure."""
    renderer = Renderer(cfg, sim, preview=preview)
    # Profile the last growth frame, which is the full-count pose.
    peak = dict(renderer.plan)["growth"] - 1
    started = time.perf_counter()
    profile_frame = renderer.render(peak)
    ms_frame = (time.perf_counter() - started) * 1000.0
    n_frames = renderer.n_frames
    worker_count = 1 if n_frames < 4 else resolve_workers(workers)
    eta_s = (ms_frame / 1000.0) * n_frames / worker_count
    print(
        f"profile: {ms_frame:.1f} ms/frame at {sim.final_count} balls "
        f"(single process); eta {eta_s / 60.0:.1f} min with {worker_count} workers"
    )
    if eta_s > 15 * 60:
        print(
            "warning: estimated render exceeds 15 minutes. "
            "Try fewer workers only if the machine is thrashing, or lower cap / bloom_strength.",
            file=sys.stderr,
        )

    def render_one(index: int) -> np.ndarray:
        if index == peak:
            return profile_frame
        return renderer.render(index)

    pipe_raw_bgr(
        iter_ordered_frames(
            n_frames,
            workers=worker_count,
            render_one=render_one,
            initializer=_init_worker,
            initargs=(cfg, str(npz_path), preview),
            render_chunk=_render_chunk,
        ),
        output,
        width=renderer.width,
        height=renderer.height,
        fps=renderer.fps,
        audio_path=audio_path,
        preset=preset,
        n_frames=n_frames if worker_count <= 1 else None,
    )
    return {"ms_per_frame": ms_frame, "eta_seconds": eta_s, "workers": float(worker_count)}


def encode_solid(
    color_bgr: tuple[int, int, int],
    output: Path,
    *,
    width: int = 64,
    height: int = 64,
    frames: int = 4,
    fps: int = 60,
    preset: str = "slow",
) -> None:
    """Encode a known solid color with the production filter and tags. Used by tests."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[..., 0] = color_bgr[0]
    frame[..., 1] = color_bgr[1]
    frame[..., 2] = color_bgr[2]
    pipe_raw_bgr(
        (frame for _ in range(frames)),
        output,
        width=width,
        height=height,
        fps=fps,
        audio_path=None,
        preset=preset,
    )
