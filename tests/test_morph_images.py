import hashlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from fc_sat.morph_assign import AssignmentError, assign, cache_key
from fc_sat.morph_images import ImageError, file_sha256, load_rgb, prepare_grid, reject_identical
from fc_sat.morph_motion import (
    arc_scales,
    arrival_tau,
    cell_centers,
    leg_pose,
    phase_at,
    phase_frames,
    rank_delays,
    rest_pose,
    smootherstep,
    total_frames,
)


def _save(path: Path, array: np.ndarray, mode: str = "RGB") -> None:
    Image.fromarray(array, mode=mode).save(path)


def test_loader_exif_rgba_gray_and_rejections(tmp_path: Path):
    wide = np.zeros((40, 80, 3), dtype=np.uint8)
    wide[:, :40] = (255, 0, 0)
    path = tmp_path / "wide.jpg"
    image = Image.fromarray(wide, mode="RGB")
    exif = image.getexif()
    exif[0x0112] = 6
    image.save(path, exif=exif)
    loaded = load_rgb(path)
    assert loaded.shape == (80, 40, 3)
    assert loaded[:20, :, 0].mean() > 200
    assert loaded[-20:, :, 0].mean() < 20

    rgba = np.zeros((32, 32, 4), dtype=np.uint8)
    rgba[..., 3] = 128
    rgba[..., 0] = 255
    rgba_path = tmp_path / "rgba.png"
    _save(rgba_path, rgba, "RGBA")
    composited = load_rgb(rgba_path)
    assert composited.shape[2] == 3
    assert composited[0, 0, 0] > composited[0, 0, 2]

    gray = np.full((32, 32), 200, dtype=np.uint8)
    gray_path = tmp_path / "gray.png"
    Image.fromarray(gray, mode="L").save(gray_path)
    assert load_rgb(gray_path).shape == (32, 32, 3)

    tiny = tmp_path / "tiny.png"
    _save(tiny, np.zeros((4, 4, 3), dtype=np.uint8))
    with pytest.raises(ImageError, match="smaller than"):
        prepare_grid(load_rgb(tiny), 8, 8)

    (tmp_path / "junk.bin").write_bytes(b"not an image")
    with pytest.raises(ImageError, match="unreadable"):
        load_rgb(tmp_path / "junk.bin")

    same = tmp_path / "same.png"
    _save(same, np.full((16, 16, 3), 40, dtype=np.uint8))
    twin = tmp_path / "twin.png"
    twin.write_bytes(same.read_bytes())
    with pytest.raises(ImageError, match="same sha256"):
        reject_identical(same, twin)
    assert file_sha256(same) == hashlib.sha256(same.read_bytes()).hexdigest()


def test_crop_respects_focus(tmp_path: Path):
    image = np.zeros((60, 120, 3), dtype=np.uint8)
    image[:, :30, 0] = 255
    image[:, -30:, 2] = 255
    left = prepare_grid(image, 8, 8, focus=(0.0, 0.5))
    right = prepare_grid(image, 8, 8, focus=(1.0, 0.5))
    assert left[..., 0].mean() > right[..., 0].mean()
    assert right[..., 2].mean() > left[..., 2].mean()
    with pytest.raises(ImageError, match="outside 0..1"):
        prepare_grid(image, 8, 8, focus=(1.2, 0.5))


def test_assignment_is_a_permutation_and_cache_ignores_hook(tmp_path: Path):
    rng = np.random.default_rng(3)
    grid_a = rng.integers(0, 255, size=(4, 4, 3), dtype=np.uint8)
    grid_b = rng.integers(0, 255, size=(4, 4, 3), dtype=np.uint8)
    first = assign(grid_a, grid_b, 0.3, cache_dir=tmp_path)
    assert sorted(first.permutation.tolist()) == list(range(16))
    second = assign(grid_a, grid_b, 0.3, cache_dir=tmp_path)
    assert second.from_cache
    assert np.array_equal(first.permutation, second.permutation)
    assert cache_key(grid_a, grid_b, 4, 4, 0.3) == cache_key(grid_a.copy(), grid_b.copy(), 4, 4, 0.3)
    assert cache_key(grid_a, grid_b, 4, 4, 0.3) != cache_key(grid_a, grid_b, 4, 4, 0.0)
    again = assign(grid_a, grid_b, 0.3, cache_dir=tmp_path)
    assert np.array_equal(again.permutation, first.permutation)
    with pytest.raises(AssignmentError, match="1 GB"):
        assign(np.zeros((120, 120, 3), dtype=np.uint8), np.zeros((120, 120, 3), dtype=np.uint8), 0.3, cache_dir=None)


def test_phase_holds_rest_and_loop():
    plan = phase_frames(
        {"hold_a_start": 0.4, "morph_ab": 6.0, "hold_b": 2.2, "morph_ba": 5.5, "hold_a_end": 0.4}
    )
    assert total_frames(plan) == 870
    cursor = 0
    for name, count in plan:
        if name.startswith("hold"):
            first_name, _, first_tau = phase_at(cursor, plan)
            last_name, _, last_tau = phase_at(cursor + count - 1, plan)
            assert first_name == last_name == name
            assert first_tau == 0.0
            assert last_tau == 1.0 or count == 1
        cursor += count
    assert phase_at(0, plan)[0] == "hold_a_start"
    assert phase_at(total_frames(plan) - 1, plan)[0] == "hold_a_end"


def test_motion_stays_inside_and_delays_from_center():
    cols, rows, cell = 6, 4, 10.0
    origin = (20.0, 30.0)
    centers = cell_centers(cols, rows, cell, *origin)
    delay = rank_delays(cols, rows, 0.30)
    center_index = int(np.argmin(np.hypot(np.arange(cols) - (cols - 1) / 2, 0)))
    # The geometric center cell is closer than a corner, so it leaves first.
    corner = 0
    assert delay[int(np.argmin(_center_distance(cols, rows)))] <= delay[corner]
    scales = arc_scales(cols * rows, seed=7)
    assert np.all(np.abs(scales) >= 0.5 - 1e-9)
    box = (origin[0], origin[1], origin[0] + cols * cell, origin[1] + rows * cell)
    end = centers[::-1].copy()
    lab = np.zeros((cols * rows, 3))
    target = np.ones((cols * rows, 3))
    previous = None
    for tau in np.linspace(0, 1, 9):
        pose = leg_pose(
            float(tau),
            centers,
            end,
            lab,
            target,
            delay,
            scales,
            delay_frac=0.30,
            arc_amp=0.15,
            recolor_strength=1.0,
            box=box,
            inset=0.75 * cell,
        )
        assert np.all(pose.position[:, 0] >= box[0] - 1e-6)
        assert np.all(pose.position[:, 0] <= box[2] + 1e-6)
        assert np.all(pose.position[:, 1] >= box[1] - 1e-6)
        assert np.all(pose.position[:, 1] <= box[3] + 1e-6)
        flying = (pose.ease > 1e-8) & (pose.ease < 1.0 - 1e-8)
        if np.any(flying):
            assert np.all(pose.position[flying, 0] >= box[0] + 0.75 * cell - 1e-6)
            assert np.all(pose.position[flying, 0] <= box[2] - 0.75 * cell + 1e-6)
        if previous is not None:
            assert np.max(np.abs(pose.position - previous)) < cell * 4
        previous = pose.position
    start = leg_pose(
        0.0, centers, end, lab, target, delay, scales,
        delay_frac=0.30, arc_amp=0.15, recolor_strength=1.0, box=box, inset=0.75 * cell,
    )
    finish = leg_pose(
        1.0, centers, end, lab, target, delay, scales,
        delay_frac=0.30, arc_amp=0.15, recolor_strength=1.0, box=box, inset=0.75 * cell,
    )
    assert np.allclose(start.lab, lab)
    assert np.allclose(finish.lab, target)
    assert np.allclose(start.position, centers)
    assert np.allclose(finish.position, end)
    assert np.allclose(smootherstep(0), 0)
    assert np.allclose(smootherstep(1), 1)
    assert np.allclose(arrival_tau(delay, 0.30)[np.argmin(delay)], 0.70)
    rest = rest_pose(centers, lab, "a")
    assert rest.settled and rest.showing == "a"
    assert center_index >= 0


def _center_distance(cols: int, rows: int) -> np.ndarray:
    ys, xs = np.mgrid[0:rows, 0:cols]
    return np.hypot(xs.ravel() - (cols - 1) / 2, ys.ravel() - (rows - 1) / 2)
