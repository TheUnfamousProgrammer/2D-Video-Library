import pytest

from make_odd import unique_odd_variants


def test_full_is_refused_without_approval():
    with pytest.raises(SystemExit, match="--approved"):
        from make_odd import main

        main(["--full", "--out", "out/odd.mp4"])


def test_batch_pairs_are_unique():
    variants = unique_odd_variants(6, 7)
    keys = [(item["seed"], item["tier"]) for item in variants]
    assert len(keys) == len(set(keys)) == 6
    assert {item["tier"] for item in variants} == {"easy", "normal", "hard"}


def test_batch_full_is_refused():
    from make_odd import main

    with pytest.raises(SystemExit, match="--batch"):
        main(["--full", "--approved", "--batch", "2", "--out", "out/odd.mp4"])


def test_ladder_writes_stills_and_answers_without_video(tmp_path, monkeypatch):
    from pathlib import Path

    from make_odd import main

    config = Path("configs/odd_default.yaml").resolve()
    monkeypatch.chdir(tmp_path)
    code = main(["--ladder", "--config", str(config), "--out", str(tmp_path / "odd.mp4")])
    assert code == 0
    images = list((tmp_path / "ladder").glob("*.png"))
    assert len(images) == 15
    names = {path.name for path in images}
    assert "hue_rung1_25items.png" in names
    assert "hue_rung5_25items.png" in names
    assert "tilt_rung3_36items.png" in names
    assert "detail_rung4_49items.png" in names
    answers = (tmp_path / "ladder" / "ANSWERS.md").read_text(encoding="utf-8")
    assert "hue_rung2_25items.png" in answers
    assert "row " in answers
    html = (tmp_path / "ladder" / "index.html").read_text(encoding="utf-8")
    assert "<script" not in html.lower()
    assert "detail_rung5_49items.png" in html
    assert list(tmp_path.rglob("*.mp4")) == []
