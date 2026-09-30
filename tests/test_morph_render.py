from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from fc_sat.config import ConfigError
from fc_sat.morph_audio import choose_ticks, synthesize_morph
from fc_sat.morph_config import load_morph_config, with_morph_overrides
from fc_sat.morph_motion import phase_at
from fc_sat.morph_render import MorphRenderer, nearest_block, particle_blocks
from make_morph import pair_tuple, unique_pairs


def _png(path: Path, color: tuple[int, int, int], mark: tuple[int, int, int]) -> None:
    image = np.zeros((48, 48, 3), dtype=np.uint8)
    image[..., 0] = color[0]
    image[..., 1] = color[1]
    image[..., 2] = color[2]
    image[:16, :16] = mark
    Image.fromarray(image, mode="RGB").save(path)


def _cfg():
    cfg = load_morph_config("configs/morph_default.yaml")
    return with_morph_overrides(
        cfg,
        cols=4,
        rows=4,
        cell=12,
        hold_a_start=0.05,
        morph_ab=0.2,
        hold_b=0.05,
        morph_ba=0.2,
        hold_a_end=0.05,
        workers=1,
    )


def test_blit_matches_per_particle_within_two_levels():
    grid = np.random.default_rng(1).integers(0, 255, size=(5, 6, 3), dtype=np.uint8)
    blit = nearest_block(grid, 8)
    particles = particle_blocks(grid, 8)
    error = float(np.mean(np.abs(blit.astype(np.float32) - particles.astype(np.float32))))
    assert error <= 2.0


def test_frames_are_deterministic_and_loop(tmp_path: Path):
    _png(tmp_path / "a.png", (20, 40, 200), (255, 0, 0))
    _png(tmp_path / "b.png", (200, 180, 20), (0, 255, 0))
    cfg = _cfg()
    first = MorphRenderer(cfg, tmp_path / "a.png", tmp_path / "b.png")
    second = MorphRenderer(cfg, tmp_path / "a.png", tmp_path / "b.png")
    assert np.array_equal(first.perm, second.perm)
    assert second.assignment.from_cache
    frame = first.render(3)
    assert np.array_equal(frame, second.render(3))
    assert np.array_equal(first.render(0), first.render(first.n_frames - 1))
    for name in ("hold_a_start", "hold_b", "hold_a_end"):
        start = first.frame_index(name, 0.0)
        end = first.frame_index(name, 1.0)
        assert phase_at(start, first.plan)[0] == name
        assert phase_at(end, first.plan)[0] == name
        _phase, pose_s = first.pose(start)
        _phase, pose_e = first.pose(end)
        assert pose_s.settled and pose_e.settled
    layer = first.particle_layer(first.frame_index("morph_ab", 0.5))
    mask = np.ones(layer.shape[:2], dtype=bool)
    mask[first.y0 : first.y0 + first.box_h, first.x0 : first.x0 + first.box_w] = False
    assert int(layer[mask].max()) == 0


def test_config_names_the_field():
    cfg = load_morph_config("configs/morph_default.yaml")
    with pytest.raises(ConfigError, match="recolor_strength"):
        with_morph_overrides(cfg, recolor_strength=1.5)
    with pytest.raises(ConfigError, match="smaller grid"):
        with_morph_overrides(cfg, max_particles=10)
    with pytest.raises(ConfigError, match="unchanged"):
        with_morph_overrides(cfg, hook="The pixels are unchanged")


def test_pairs_reject_a_repeated_tuple():
    hooks = ("Watch the pixels move", "Guess what it becomes")
    entries = [
        {"a": "mona.png", "b": "starry.png", "hook": hooks[0]},
        {"a": "mona.png", "b": "wave.png", "hook": hooks[1]},
        {"a": "mona.png", "b": "starry.png", "hook": hooks[0]},
    ]
    with pytest.raises(SystemExit, match="repeated pair"):
        unique_pairs(entries, hooks, hooks[0])
    identity = pair_tuple(entries[0], hooks, hooks[0])
    assert identity == ("mona", "starry", 0)


def test_audio_length_tail_ticks_and_loudness(tmp_path: Path):
    _png(tmp_path / "a.png", (20, 40, 200), (255, 0, 0))
    _png(tmp_path / "b.png", (220, 30, 40), (0, 0, 255))
    cfg = with_morph_overrides(
        _cfg(),
        hold_a_start=0.4,
        morph_ab=1.2,
        hold_b=0.5,
        morph_ba=1.2,
        hold_a_end=0.4,
    )
    renderer = MorphRenderer(cfg, tmp_path / "a.png", tmp_path / "b.png")
    audio, lufs, peak = synthesize_morph(
        cfg,
        delay_ab=renderer.delay_ab,
        delay_ba=renderer.delay_ba,
        x_b=renderer.centers_b[:, 0],
        x_a=renderer.centers_a[:, 0],
        n_frames=renderer.n_frames,
        fps=renderer.fps,
    )
    assert len(audio) == renderer.n_frames * 800
    assert np.all(audio[-int(0.300 * 48000) :] == 0)
    assert abs(lufs + 14.0) <= 1.0
    assert peak <= -1.0 + 1e-6
    crowded = np.full(30, 0.05)
    kept = choose_ticks(crowded, 12, seed=3)
    assert len(kept) == 12
    again = choose_ticks(crowded, 12, seed=3)
    assert np.array_equal(kept, again)
