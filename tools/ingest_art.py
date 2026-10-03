#!/usr/bin/env python3
"""Fit the supplied plates and key the magenta cutouts.

    python tools/ingest_art.py
"""

from __future__ import annotations

from fc_sat.paperfold_art import run_ingest

if __name__ == "__main__":
    raise SystemExit(run_ingest())
