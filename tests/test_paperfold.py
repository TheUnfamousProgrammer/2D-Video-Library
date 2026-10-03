"""Fast checks for the paperfold short."""

from __future__ import annotations

import pytest

from make_paperfold import main


def test_full_render_is_refused_without_approval():
    with pytest.raises(SystemExit, match="approved"):
        main(["full"])
