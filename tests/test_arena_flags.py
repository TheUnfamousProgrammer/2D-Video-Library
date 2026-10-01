import json
from pathlib import Path

import pytest
from PIL import Image

from tools.fetch_flags import PIN, confirm_pin, rasterize_svg


def test_missing_pin_reports_the_latest_tag():
    def fetch(url: str) -> tuple[int, bytes]:
        if url.endswith("/releases/latest"):
            return 200, json.dumps({"tag_name": "v9.9.9"}).encode()
        return 404, b""

    with pytest.raises(SystemExit, match="v9.9.9") as caught:
        confirm_pin("v0.0.1", fetch)
    assert "was not changed" in str(caught.value)
    assert "v0.0.1" in str(caught.value)


def test_existing_pin_is_accepted():
    def fetch(url: str) -> tuple[int, bytes]:
        assert url.endswith("/" + PIN)
        return 200, b"{}"

    assert confirm_pin(PIN, fetch) == PIN


def test_every_cast_flag_exists_and_rasterizes(tmp_path: Path):
    from fc_sat.arena_config import load_guards, load_cast

    guards = load_guards("configs/guards.yaml")
    codes: set[str] = set()
    for path in Path("configs/casts").glob("*.yaml"):
        countries, _warnings = load_cast(path, guards)
        codes.update(country.iso2.lower() for country in countries)
    for code in sorted(codes):
        svg = Path("assets/flags/svg") / f"{code}.svg"
        png = Path("assets/flags/png") / f"{code}.png"
        assert svg.is_file(), code
        assert png.is_file(), code
        with Image.open(png) as image:
            assert image.size == (512, 512)
    sample = Path("assets/flags/svg") / f"{sorted(codes)[0]}.svg"
    out = tmp_path / "again.png"
    rasterize_svg(sample, out)
    with Image.open(out) as image:
        assert image.size == (512, 512)
