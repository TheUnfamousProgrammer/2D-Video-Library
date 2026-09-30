import pytest

from make import unique_variants, variant_path
from pathlib import Path


def test_batch_names_are_unique():
    combos = unique_variants(
        8,
        seed=7,
        palette="sunset",
        hook_index=0,
        vary="seed,palette,hook",
        palette_names=["sunset", "ocean", "candy"],
        n_hooks=4,
    )
    assert len(combos) == 8
    assert len(set(combos)) == 8
    paths = [variant_path(Path("out/a.mp4"), seed, palette, hook) for seed, palette, hook in combos]
    assert len(set(paths)) == len(paths)
    assert paths[0].name.startswith("a_7_")


def test_batch_refuses_to_repeat_when_seed_is_fixed():
    with pytest.raises(SystemExit):
        unique_variants(
            5,
            seed=1,
            palette="sunset",
            hook_index=0,
            vary="palette",
            palette_names=["sunset", "ocean"],
            n_hooks=3,
        )
