"""What each mode is allowed to write.

The posted polycircle file was 1080x1920, 30 fps, 912 frames. That is the full
raster on the preview clock:

    python make_polycircle.py full --approved --hook A --out out/polycircle.mp4

The renderer was 1080x1920 and the mix is 30.4 s. The encode passed fps=30
(the preview rate) into the shared pipe, which muxes audio with -shortest.
1824 frames at 30 fps would run 60.8 s, so -shortest kept 30.4 * 30 = 912
frames. The container stayed 1080x1920 at r_frame_rate 30/1. It was not the
540x960 preview.

Full must be 1080x1920, 60 fps, 1824 frames. Preview, animatic, and hooks stay
540x960 at 30 fps.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Delivery:
    width: int
    height: int
    fps: int
    frames: int
    step: int


# step 1 writes every timeline frame. step 2 writes every other frame (30 fps preview of a 60 fps timeline).
DELIVERIES = {
    "full": Delivery(1080, 1920, 60, 1824, 1),
    "preview": Delivery(540, 960, 30, 912, 2),
    "animatic": Delivery(540, 960, 30, 912, 2),
    "hooks": Delivery(540, 960, 30, 105, 2),
}


def delivery(mode: str) -> Delivery:
    try:
        return DELIVERIES[mode]
    except KeyError as exc:
        raise SystemExit(f"unknown delivery mode {mode}") from exc


def frame_indices(mode: str, timeline_frames: int = 1824) -> list[int]:
    spec = delivery(mode)
    if spec.step == 1:
        if spec.frames != timeline_frames:
            raise SystemExit(f"{mode} frame count {spec.frames} does not match the timeline {timeline_frames}")
        return list(range(timeline_frames))
    return [min(timeline_frames - 1, index * spec.step) for index in range(spec.frames)]


def require_delivery(mode: str, width: int, height: int, fps: int, n_frames: int) -> Delivery:
    spec = delivery(mode)
    got = (width, height, fps, n_frames)
    want = (spec.width, spec.height, spec.fps, spec.frames)
    if got != want:
        raise SystemExit(
            f"{mode} would write {width}x{height} at {fps} fps, {n_frames} frames; "
            f"expected {spec.width}x{spec.height} at {spec.fps} fps, {spec.frames} frames"
        )
    return spec
