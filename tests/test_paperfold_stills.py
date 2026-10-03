"""Hero stills: slate backing, scale, and stack-edge contrast."""

from __future__ import annotations

import numpy as np

from fc_sat.paperfold_scenes import FRAME_H, FRAME_W, GROUND
from fc_sat.paperfold_stills import (
    BACKING,
    BACKING_TOP,
    BACKING_W,
    DioramaStill,
    PaperBox,
    backing_box,
    edge_contrast,
    scene_at,
)
from fc_sat.paperfold_schedule import fold_frame


def test_backing_strip_is_240_px_and_clears_the_limit():
    x, y, w, h = backing_box()
    assert w == BACKING_W == 240
    assert x == (FRAME_W - 240) / 2
    assert y == BACKING_TOP * FRAME_H
    assert abs((y + h) - GROUND * FRAME_H) < 1e-6
    assert y < 0.30 * FRAME_H


def test_hero_frames_land_on_the_expected_plates():
    assert scene_at(0).name == "desk"
    assert scene_at(fold_frame(15)).name == "street"
    assert scene_at(fold_frame(23)).name == "city"
    assert scene_at(fold_frame(30)).name == "atmosphere"
    assert scene_at(fold_frame(42)).name == "deep_space"


def test_edge_contrast_reads_white_against_the_neighbour():
    image = np.zeros((200, 200, 3), np.uint8)
    image[:, :] = (77, 58, 43)  # BGR of #2B3A4D
    image[20:180, 80:120] = (248, 244, 242)
    ratios = edge_contrast(image, PaperBox(80, 20, 40, 160), outline=0)
    assert min(ratios) >= 3.0


def test_fold_stills_are_master_size_and_clear_3_to_1():
    drawer = DioramaStill()
    frames = [0, *(fold_frame(fold) for fold in (15, 23, 30, 42))]
    for frame in frames:
        still = drawer.render(frame)
        assert still.image.shape == (FRAME_H, FRAME_W, 3)
        assert min(still.ratios) >= 3.0, (frame, still.ratios, still.plate)
