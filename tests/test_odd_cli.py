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
    assert {item["tier"] for item in variants} == {"normal", "brutal", "quick"}
