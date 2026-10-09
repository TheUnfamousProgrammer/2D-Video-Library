#!/usr/bin/env python3
"""The 37% rule: how many people should you date before you commit?

    python make_rule37.py voice      # Eleven v4 take (cached) -> out/rule37/vo.wav + word times
    python make_rule37.py stills     # contact sheet + hero frames
    python make_rule37.py full       # 1080x1920 60 fps with the mix
    python make_rule37.py thumb      # out/rule37/thumbnail.png
    python make_rule37.py full --captions --out out/rule37/rule37_captions.mp4
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out" / "rule37"
SCRIPT = ROOT / "configs" / "rule37_script.txt"
VOICE_ID = "FGY2WhTYpPnrIDTdsKH5"  # Laura - sassy, quirky social-media voice
SEED = 21
STABILITY = 0.4

_painter = None


def _init(captions: bool = False) -> None:
    global _painter
    from fc_sat.rule37.captions import Captions
    from fc_sat.rule37.cues import load
    from fc_sat.rule37.draw import Painter

    cues = load(OUT / "vo.words.json")
    _painter = Painter(cues, Captions(OUT / "vo.words.json", cues.end) if captions else None)


def _frame(i: int) -> np.ndarray:
    return _painter.render(i)


def voice() -> None:
    from fc_sat.rule37.vo_edit import build
    from fc_sat.rule37.voice import synth

    text = SCRIPT.read_text().strip().replace("\n", " ")
    mp3, meta = synth(text, VOICE_ID, stability=STABILITY, seed=SEED)
    z, words = build(mp3, meta, OUT / "vo.wav")
    print(f"vo {len(z) / 48000:.2f}s, {len(words)} words")


def stills(frames: list[int] | None, captions: bool = False) -> None:
    from fc_sat.rule37.cues import load

    cues = load(OUT / "vo.words.json")
    _init(captions)
    folder = OUT / "stills"
    folder.mkdir(parents=True, exist_ok=True)
    picks = frames or list(range(0, cues.n_frames, 45))
    thumbs = []
    for i in picks:
        img = Image.fromarray(_frame(i)[:, :, ::-1])
        img.save(folder / f"f{i:04d}.png") if frames else None
        thumbs.append(img.resize((216, 384), Image.Resampling.BOX))
    cols = 10
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (216 * cols, 384 * rows))
    for k, th in enumerate(thumbs):
        sheet.paste(th, ((k % cols) * 216, (k // cols) * 384))
    sheet.save(OUT / "contact.jpg", quality=85)
    print(f"wrote {OUT / 'contact.jpg'} ({len(thumbs)} frames)")


def full(out: Path, preview: bool, captions: bool = False) -> None:
    from fc_sat.encode import pipe_raw_bgr
    from fc_sat.rule37.audio import render_mix
    from fc_sat.rule37.cues import FPS, load

    cues = load(OUT / "vo.words.json")
    wav = OUT / "mix.wav"
    render_mix(cues, OUT / "vo.wav", wav)
    frames = range(cues.n_frames)
    with mp.get_context("spawn").Pool(max(1, mp.cpu_count() - 1), initializer=_init, initargs=(captions,)) as pool:
        it = pool.imap(_frame, frames, chunksize=8)
        pipe_raw_bgr(it, out, width=1080, height=1920, fps=FPS, audio_path=wav, n_frames=cues.n_frames, pool=pool,
                     preset="medium" if preview else "slow")
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["voice", "stills", "full", "thumb"])
    ap.add_argument("--frames", default="")
    ap.add_argument("--out", default=str(OUT / "rule37.mp4"))
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--captions", action="store_true", help="burn in word-by-word captions")
    a = ap.parse_args()
    if a.mode == "voice":
        voice()
    elif a.mode == "stills":
        stills([int(x) for x in a.frames.split(",")] if a.frames else None, a.captions)
    elif a.mode == "thumb":
        from fc_sat.rule37.cues import load
        from fc_sat.rule37.draw import Painter
        from fc_sat.rule37.thumb import render

        render(Painter(load(OUT / "vo.words.json")), OUT / "thumbnail.png")
    else:
        full(Path(a.out), a.preview, a.captions)


if __name__ == "__main__":
    main()
