"""Art id matching and ingest edge cases."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from fc_sat.paperfold_art import art_id, key_magenta, place_plate, run_ingest


def test_manifest_id_ignores_the_upload_suffix():
    assert art_id("p01_hook_night_desk_moon.png_20261003125709.jpg") == "p01_hook_night_desk_moon"
    assert art_id("o02_mug_png_extra.png") == "o02_mug"


def test_small_plate_is_upscaled_and_wrong_aspect_is_logged():
    tall = np.zeros((356, 200, 3), dtype=np.uint8)
    tall[:] = (40, 30, 20)
    fitted, notes = place_plate(tall, 0.70)
    assert fitted.shape == (1920, 1080, 3)
    assert "UPSCALED" in notes
    square = np.zeros((400, 400, 3), dtype=np.uint8)
    _fitted, notes = place_plate(square, 0.70)
    assert any(note.startswith("WRONG ASPECT") for note in notes)


def test_keyed_cutout_drops_magenta_and_keeps_the_object():
    image = np.zeros((200, 160, 3), dtype=np.uint8)
    image[:] = (255, 0, 255)  # BGR magenta
    image[40:160, 50:110] = (240, 240, 240)
    keyed, fraction, box = key_magenta(image)
    assert fraction >= 0.90
    assert keyed.shape[2] == 4
    assert box[2] > box[0] and box[3] > box[1]
    assert keyed[0, 0, 3] < 16
    assert keyed[100, 80, 3] > 200


def test_missing_asset_is_unapproved(tmp_path: Path):
    art = tmp_path / "art"
    (art / "plates").mkdir(parents=True)
    (art / "objects").mkdir()
    manifest = {
        "plates": [{"id": "p01_hook_night_desk_moon", "ground_y": 0.68, "scene": "desk"}],
        "objects": [{"id": "o01_person"}],
    }
    (art / "MANIFEST.yaml").write_text(yaml.safe_dump(manifest))
    # A plate that exists, an object that does not.
    cv2.imwrite(str(art / "plates" / "p01_hook_night_desk_moon.png_1.jpg"), np.full((356, 200, 3), 30, np.uint8))
    out = tmp_path / "out"
    from fc_sat.paperfold_art import find_source

    assert find_source(art / "plates", "p01_hook_night_desk_moon") is not None
    assert find_source(art / "objects", "o01_person") is None
    code = run_ingest(art, out, art / "MANIFEST.yaml")
    assert code == 1
    text = (out / "art_report.md").read_text()
    assert "UNAPPROVED" in text
    assert "o01_person: MISSING" in text
