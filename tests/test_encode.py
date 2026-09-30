import json
import subprocess

import numpy as np

from fc_sat.encode import BT709_FILTER, encode_solid, find_ffmpeg, find_ffprobe


def test_solid_color_roundtrip(tmp_path):
    color = (20, 80, 220)  # BGR
    output = tmp_path / "solid.mp4"
    encode_solid(color, output, width=64, height=64, frames=4, fps=60, preset="ultrafast")
    ffmpeg = find_ffmpeg()
    ffprobe = find_ffprobe(ffmpeg)
    probe = subprocess.run(
        [ffprobe, "-v", "error", "-print_format", "json", "-show_streams", str(output)],
        check=True,
        capture_output=True,
        text=True,
    )
    info = json.loads(probe.stdout)
    video = info["streams"][0]
    assert video["codec_name"] == "h264"
    assert video["pix_fmt"] == "yuv420p"
    assert video["color_space"] == "bt709"
    assert video["color_transfer"] == "bt709"
    assert video["color_primaries"] == "bt709"
    assert video["color_range"] == "tv"
    raw = subprocess.run(
        [ffmpeg, "-v", "error", "-i", str(output), "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"],
        check=True,
        capture_output=True,
    )
    frame = np.frombuffer(raw.stdout, dtype=np.uint8).reshape(-1, 64, 64, 3)[0]
    # Sample the center so chroma subsampling at the edge does not dominate.
    decoded = frame[32, 32].astype(np.int16)
    assert np.max(np.abs(decoded - np.array(color))) <= 8
    assert BT709_FILTER == "scale=out_color_matrix=bt709:out_range=tv"
