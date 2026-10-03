"""Hero stills after the prism rework."""

from __future__ import annotations

import cv2
import numpy as np

from fc_sat.paperfold_scenes import FRAME_H, FRAME_W, scenes_from_manifest
from fc_sat.paperfold_stills import CARD_H, CARD_W, DioramaStill, STACK_W
from fc_sat.paperfold_schedule import fold_frame
from fc_sat.paperfold_text import load_script


def test_space_plates_use_the_earth_apex():
    rows = scenes_from_manifest()
    by_name = {row.name: row for row in rows}
    assert by_name["atmosphere"].screen_ground == 0.68
    assert by_name["orbit"].screen_ground == 0.58
    assert by_name["deep_space"].screen_ground == 0.58
    assert by_name["desk"].screen_ground == 0.68


def test_hook_copy_is_two_lines_without_a_space_before_the_period():
    script = load_script()
    assert script.hooks["A"] == ("42 FOLDS.", "THE MOON?")
    assert script.hooks["C"] == ("FOLD IT", "42 TIMES.")
    assert script.hooks["D"] == script.hooks["A"]
    assert script.hooks["B"] == ("CAN PAPER", "REACH THE", "MOON?")
    for lines in script.hooks.values():
        for line in lines:
            assert " ." not in line
            assert " ?" not in line


def test_hero_stills_are_master_size():
    drawer = DioramaStill()
    frames = [0, *(fold_frame(fold) for fold in (15, 23, 30, 32, 42))]
    seen = []
    for frame in frames:
        still = drawer.render(frame)
        assert still.image.shape == (FRAME_H, FRAME_W, 3)
        seen.append(still.scene)
    assert seen == ["desk", "street", "city", "atmosphere", "orbit", "deep_space"]


def test_hook_card_and_stack_widths():
    drawer = DioramaStill()
    hook = drawer.render(0)
    assert hook.paper.w == CARD_W / 2.0
    assert hook.paper.h == CARD_H
    tower = drawer.render(fold_frame(15))
    assert tower.paper.w == STACK_W


def test_moon_mask_clears_pixels_outside_the_disc():
    from fc_sat.paperfold_art import mask_moon_circle

    image = np.zeros((400, 300, 4), np.uint8)
    image[:, :, :3] = (90, 78, 40)
    image[:, :, 3] = 255
    cv2.circle(image, (150, 200), 80, (230, 220, 200, 255), -1)
    masked, (cx, cy, radius) = mask_moon_circle(image)
    assert masked[5, 5, 3] == 0
    assert masked[200, 150, 3] > 200
    yy, xx = np.mgrid[0:400, 0:300]
    outside = (xx - cx) ** 2 + (yy - cy) ** 2 > (radius + 1.5) ** 2
    assert int(masked[:, :, 3][outside].max()) == 0
