from pathlib import Path

import numpy as np

from fc_sat.odd_config import load_odd_config
from fc_sat.odd_verify import quiet_window, sequence_checks, sim_checks
from fc_sat.verify import detect_mode


def test_detect_mode_odd_and_leaves_bounce():
    assert detect_mode(Path("configs/odd_default.yaml")) == "odd"
    assert detect_mode(Path("configs/default.yaml")) == "bounce"


def test_label_sequence_and_reveal_boundary():
    cfg = load_odd_config("configs/odd_default.yaml")
    checks = {item.name: item for item in sequence_checks(cfg)}
    assert checks["reveal at timer 0"].ok
    assert checks["label sequence"].ok
    assert checks["label sequence"].detail == "EASY MEDIUM HARD BRUTAL"


def test_quiet_window_flags_a_still_second():
    moving = np.full(180, 0.01)
    assert quiet_window(moving, 60) is None
    stalled = moving.copy()
    stalled[60:120] = 0.0
    assert quiet_window(stalled, 60) == 1.0


def test_sim_constraints_on_a_small_level():
    from dataclasses import replace

    from fc_sat.odd_sim import simulate_show

    cfg = load_odd_config("configs/odd_default.yaml", level_ids=[1])
    cfg = replace(cfg, levels=tuple(replace(level, count=12, timer=1.2) for level in cfg.levels), warmup_seconds=0.25)
    show = simulate_show(cfg)
    checks = sim_checks(cfg, show)
    failed = [item for item in checks if not item.ok]
    assert failed == [], [(item.name, item.detail) for item in failed]


def test_batch_full_is_refused():
    import pytest

    from make_odd import main

    with pytest.raises(SystemExit, match="--batch"):
        main(["--full", "--approved", "--batch", "2", "--out", "out/odd.mp4"])
