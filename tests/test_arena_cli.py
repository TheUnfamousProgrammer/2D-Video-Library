from pathlib import Path

from make_arena import output_name
from fc_sat.verify import detect_mode


def test_batch_output_names_include_cast_and_seed():
    base = Path("out/arena.mp4")
    names = {
        output_name(base, "configs/casts/asia.yaml", 3),
        output_name(base, "configs/casts/africa.yaml", 3),
        output_name(base, "configs/casts/asia.yaml", 9),
    }
    assert len(names) == 3
    assert output_name(base, "configs/casts/asia.yaml", 3).name == "arena_asia_s3.mp4"


def test_detect_mode_reads_generator_arena():
    assert detect_mode(Path("configs/arena_default.yaml")) == "arena"
